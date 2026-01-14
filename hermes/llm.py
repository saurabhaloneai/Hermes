import json
from pathlib import Path
from typing import Optional
from time import perf_counter
from dataclasses import dataclass
import torch
from accelerate import load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from .models import ModelConfig, VisionConfig, Qwen3VL
from .utlis import Processor


@dataclass
class GenerationStats:
    ttft: float
    total_time: float
    num_tokens: int
    tokens_per_second: float


class LLM:
    def __init__(
        self,
        model: Qwen3VL,
        processor: Processor,
    ):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_image_pixels: Optional[int] = 768 * 768,
    ):
        weights_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
        model_path = Path(weights_path)

        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)

        llm_config = hf_config["text_config"]
        config = ModelConfig(
            n_embed=llm_config["hidden_size"],
            n_heads=llm_config["num_attention_heads"],
            n_kv_heads=llm_config["num_key_value_heads"],
            n_layer=llm_config["num_hidden_layers"],
            n_mlp=llm_config["intermediate_size"],
            n_vocab=llm_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=llm_config["rope_theta"],
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
        )

        vision_config = None
        vision_config_data = hf_config.get("vision_config")
        if vision_config_data is not None:
            vision_config = VisionConfig(
                n_embed=vision_config_data["hidden_size"],
                n_layer=vision_config_data["depth"],
                n_heads=vision_config_data["num_heads"],
                n_output_embed=vision_config_data["out_hidden_size"],
                n_mlp=vision_config_data["intermediate_size"],
                deepstack_visual_indexes=vision_config_data["deepstack_visual_indexes"],
                num_position_embeddings=vision_config_data["num_position_embeddings"],
                in_channels=vision_config_data["in_channels"],
                temporal_patch_size=vision_config_data["temporal_patch_size"],
                patch_size=vision_config_data["patch_size"],
                spatial_merge_size=vision_config_data["spatial_merge_size"],
            )

        model = Qwen3VL(config, vision_config=vision_config)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block", "VisionBlock"],
            dtype=torch.bfloat16,
        )
        model.eval()

        processor = Processor.from_pretrained(model_id)
        if max_image_pixels is not None:
            processor.max_pixels = max_image_pixels

        return cls(model=model, processor=processor)

    def generate(
        self,
        text: str,
        image: Optional[str] = None,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        user_content = []
        if image:
            user_content.append({"type": "image", "url": image})
        user_content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": user_content})

        inputs = self.processor(messages, add_generation_prompt=True, device=self.device)

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": max_new_tokens,
        }

        if inputs["pixels"] is not None:
            generation_kwargs["pixels"] = inputs["pixels"]
        if inputs["d_image"] is not None:
            generation_kwargs["d_image"] = inputs["d_image"]

        output_ids = self.model.generate(**generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:].tolist()
        response = self.processor.tokenizer.decode(generated_ids)

        return response

    def generate_stream(
        self,
        text: str,
        image: Optional[str] = None,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        user_content = []
        if image:
            user_content.append({"type": "image", "url": image})
        user_content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": user_content})

        inputs = self.processor(messages, add_generation_prompt=True, device=self.device)

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": max_new_tokens,
        }

        if inputs["pixels"] is not None:
            generation_kwargs["pixels"] = inputs["pixels"]
        if inputs["d_image"] is not None:
            generation_kwargs["d_image"] = inputs["d_image"]

        generated_tokens = []
        previous_text = ""
        ttft = None
        start_time = perf_counter()

        for token_id in self.model.generate_stream(**generation_kwargs):
            if ttft is None:
                ttft = perf_counter() - start_time

            generated_tokens.append(token_id)
            current_text = self.processor.tokenizer.decode(generated_tokens)
            new_text = current_text[len(previous_text):]
            if new_text:
                previous_text = current_text
                yield new_text

        total_time = perf_counter() - start_time
        num_tokens = len(generated_tokens)
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0

        self.last_generation_stats = GenerationStats(
            ttft=ttft or 0,
            total_time=total_time,
            num_tokens=num_tokens,
            tokens_per_second=tokens_per_second,
        )

    def get_last_stats(self) -> Optional[GenerationStats]:
        return getattr(self, "last_generation_stats", None)

if __name__ == "__main__":
    llm = LLM.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # Test with image - UPDATE THIS PATH to your actual image
    print("--- Streaming Generation (with image) ---")
    for chunk in llm.generate_stream(
        text="What is in this image?",
        image="/path/to/your/real/image.jpg",  # <-- UPDATE THIS PATH
    ):
        print(chunk, end="", flush=True)
    print()

    stats = llm.get_last_stats()
    if stats:
        print(f"\n--- Performance Stats (image) ---")
        print(f"TTFT: {stats.ttft * 1000:.2f} ms")
        print(f"Total time: {stats.total_time:.2f} s")
        print(f"Tokens generated: {stats.num_tokens}")
        print(f"Tokens/second: {stats.tokens_per_second:.2f}")



