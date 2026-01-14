import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embed // config.n_heads
        self.qkv = nn.Linear(config.n_embed, config.n_embed * 3, bias=True)
        self.proj = nn.Linear(config.n_embed, config.n_embed)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb_vision(
        tensor: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        output = (tensor * cos) + (VisionAttention._rotate_half(tensor) * sin)
        output = output.to(orig_dtype)
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = x.shape[0]
        q, k, v = (
            self.qkv(x)
            .reshape(seq_length, 3, self.n_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = self._apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Process each image separately using cu_seqlens
        # This avoids creating O(n²) attention masks and enables flash attention
        outputs = []
        for i in range(1, len(cu_seqlens)):
            start, end = cu_seqlens[i - 1].item(), cu_seqlens[i].item()
            
            # Extract this image's patches: (seq_i, n_heads, head_dim)
            q_i = q[start:end]
            k_i = k[start:end]
            v_i = v[start:end]
            
            # Reshape to (1, n_heads, seq_i, head_dim) for SDPA
            q_i = q_i.transpose(0, 1).unsqueeze(0)
            k_i = k_i.transpose(0, 1).unsqueeze(0)
            v_i = v_i.transpose(0, 1).unsqueeze(0)
            
            # Use scaled_dot_product_attention - enables Flash Attention when available
            # No mask needed since all patches in same image can attend to each other
            attn_out = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)
            
            # Reshape back: (1, n_heads, seq_i, head_dim) -> (seq_i, n_heads, head_dim)
            attn_out = attn_out.squeeze(0).transpose(0, 1)
            outputs.append(attn_out)
        
        # Concatenate all image outputs
        attn_output = torch.cat(outputs, dim=0)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

