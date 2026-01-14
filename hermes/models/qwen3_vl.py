from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig, VisionConfig
from .block import Block
from ..layers import RMSNorm, RotaryEmbedding
from ..layers.vision import VisionEncoder


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embed,
        vision_embed=None,
        vision_residuals=None,
        vision_mask=None,
        position_ids=None,
        past_key_values=None,
    ):
        
        if vision_embed is not None and vision_mask is not None:
            input_embed[vision_mask] = vision_embed

        cos, sin = self.rotary_emb(input_embed, position_ids)
        
        present_key_values = []
        for layer_idx, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[layer_idx] if past_key_values else None
            
            input_embed, present_kv = layer(input_embed, cos, sin, layer_past_kv=layer_past_kv)
            present_key_values.append(present_kv)
            
            if vision_residuals and vision_mask is not None:
                vision_residual = vision_residuals.get(layer_idx)
                if vision_residual is not None:
                    input_embed[vision_mask] = (
                        input_embed[vision_mask] + vision_residual
                    )

        input_embed = self.norm(input_embed)
        return input_embed, present_key_values


class Qwen3VL(nn.Module):
    def __init__(
        self, config: ModelConfig, vision_config: Optional[VisionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, list, int]:
        
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        position_ids, next_position = self._get_position_ids(
            input_ids=input_ids, 
            d_image=d_image, 
            past_key_values=past_key_values,
            position_offset=position_offset,
        )

        if pixels is not None:
            pixels = pixels.to(input_embeds.dtype)
            vision_embed, vision_residuals = self.model.visual(
                pixels=pixels, d_image=d_image
            )
            image_pad_token = getattr(self.config, "image_token_id", 151655)
            vision_mask = input_ids == image_pad_token
            output, present_key_values = self.model.language_model(
                input_embed=input_embeds,
                vision_embed=vision_embed,
                vision_residuals=vision_residuals,
                vision_mask=vision_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        else:
            output, present_key_values = self.model.language_model(
                input_embed=input_embeds, 
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

        logits = (
            output @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(output)
        )
        return logits, present_key_values, next_position

    def _get_position_ids(
        self, 
        input_ids: torch.Tensor, 
        d_image: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            position_ids: (3, B, T) tensor of position IDs
            next_position: the next text position to use for subsequent tokens
        """
        B, T = input_ids.shape
        image_pad_token = getattr(self.config, "image_token_id", 151655)

        if d_image is None or past_key_values is not None:
            # Text-only or decode phase: use sequential positions starting from offset
            start_pos = position_offset
            position_ids = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(3, B, -1)
            return position_ids, start_pos + T

        # Prefill with images: compute M-RoPE positions
        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=input_ids.device)
        final_text_idx = 0
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            text_idx, image_idx, seq_idx = 0, 0, 0
            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == image_pad_token:
                    text_idx, image_idx, seq_idx = self._emit_image_block(
                        position_ids=position_ids,
                        batch_idx=batch_idx,
                        seq_idx=seq_idx,
                        text_idx=text_idx,
                        image_idx=image_idx,
                        d_image=d_image,
                    )
                else:
                    position_ids[:, batch_idx, seq_idx] = text_idx
                    text_idx, image_idx, seq_idx = text_idx + 1, image_idx, seq_idx + 1
            final_text_idx = max(final_text_idx, text_idx)

        return position_ids, final_text_idx

    def _emit_image_block(
        self,
        position_ids: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
        text_idx: int,
        image_idx: int,
        d_image: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int, int]:
        t_img, h_img, w_img = d_image[image_idx]
        t_img = int(t_img.item())
        h_img = int((h_img // spatial_merge_size).item())
        w_img = int((w_img // spatial_merge_size).item())

        image_token_count = h_img * w_img
        video_token_count = t_img * image_token_count
        for offset in range(video_token_count):
            target_idx = seq_idx + offset
            remaining = offset % image_token_count
            h_pos = remaining // w_img
            w_pos = remaining % w_img

            position_ids[:, batch_idx, target_idx] = text_idx
            position_ids[1, batch_idx, target_idx] = text_idx + h_pos
            position_ids[2, batch_idx, target_idx] = text_idx + w_pos

        return text_idx + 1, image_idx + 1, seq_idx + video_token_count

    def _generate_core(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_tokens: Optional[list],
    ):
       
        if stop_tokens is None:
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        generated_ids = input_ids
        past_key_values = None
        position_offset = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    # Prefill: process full input with vision
                    current_ids = generated_ids
                    current_pixels = pixels
                    current_d_image = d_image
                else:
                    # Decode: only process new token, no vision
                    current_ids = next_token
                    current_pixels = None
                    current_d_image = None
                
                logits, past_key_values, position_offset = self.forward(
                    input_ids=current_ids,
                    pixels=current_pixels,
                    d_image=current_d_image,
                    past_key_values=past_key_values,
                    position_offset=position_offset,
                )
                
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                token_id = next_token[0].item()
                yield token_id, generated_ids

                if token_id in stop_tokens:
                    break

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        generated_ids = input_ids

        for _, generated_ids in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            pass

        return generated_ids

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        for token_id, _ in self._generate_core(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
        ):
            yield token_id
