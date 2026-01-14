#!/usr/bin/env python3
"""
Hermes Model Profiler - Generates Perfetto traces for performance analysis.

This script profiles the Qwen3VL model to identify bottlenecks in:
- Vision encoder (patch embedding, attention blocks, merger)
- Language model forward pass
- Token generation loop

Usage:
    python profile_model.py --image /path/to/image.jpg
    python profile_model.py --text-only  # Profile without image

View traces:
    1. Open https://ui.perfetto.dev
    2. Drag and drop the generated trace_*.json file
"""

import argparse
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def profiler_context(output_path: str, with_stack: bool = True):
    """Context manager for profiling with Perfetto output."""
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        yield prof
    
    # Export to Chrome/Perfetto format
    prof.export_chrome_trace(output_path)
    print(f"\n✅ Trace saved to: {output_path}")
    print("📊 Open https://ui.perfetto.dev and drag-drop the trace file to view")


def print_stats_table(prof, sort_by: str = "cuda_time_total", row_limit: int = 25):
    """Print formatted profiler stats."""
    print(f"\n{'='*80}")
    print(f"Top {row_limit} operations by {sort_by}:")
    print(f"{'='*80}")
    
    # Handle MPS (no cuda_time)
    if not torch.cuda.is_available() and "cuda" in sort_by:
        sort_by = sort_by.replace("cuda", "cpu")
    
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def profile_vision_encoder_only(
    llm,
    image_path: str,
    output_path: str = "trace_vision_encoder.json",
):
    """Profile just the vision encoder in isolation."""
    from hermes.utlis import Processor
    
    print("\n🔬 Profiling Vision Encoder Only...")
    
    # Prepare inputs
    messages = [
        {"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "Describe this image."},
        ]}
    ]
    inputs = llm.processor(messages, add_generation_prompt=True, device=llm.device)
    
    if inputs["pixels"] is None:
        print("❌ No pixels found - image may not have loaded correctly")
        return
    
    pixels = inputs["pixels"]
    d_image = inputs["d_image"]
    
    print(f"   Pixels shape: {pixels.shape}")
    print(f"   d_image: {d_image}")
    
    # Warm up
    with torch.no_grad():
        for _ in range(2):
            _ = llm.model.vision(pixels, d_image)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Profile
    with profiler_context(output_path) as prof:
        with torch.no_grad():
            with record_function("vision_encoder_total"):
                vision_encoder = llm.model.vision
                
                with record_function("patch_embed"):
                    hidden_states = vision_encoder.patch_embed(pixels)
                
                with record_function("pos_embed_interpolate"):
                    pos_embeds = vision_encoder.fast_pos_embed_interpolate(d_image)
                    hidden_states = hidden_states + pos_embeds
                
                with record_function("rotary_pos_emb"):
                    rotary_pos_emb = vision_encoder.rot_pos_emb(d_image)
                    
                import torch.nn.functional as F
                cu_seqlens = torch.repeat_interleave(
                    d_image[:, 1] * d_image[:, 2], d_image[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                
                with record_function("vision_blocks"):
                    for layer_num, blk in enumerate(vision_encoder.blocks):
                        with record_function(f"vision_block_{layer_num}"):
                            hidden_states = blk(
                                hidden_states, 
                                cu_seqlens=cu_seqlens, 
                                rotary_pos_emb=rotary_pos_emb
                            )
                
                with record_function("patch_merger"):
                    output = vision_encoder.merger(hidden_states)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print_stats_table(prof)


def profile_full_forward_pass(
    llm,
    image_path: Optional[str],
    output_path: str = "trace_forward.json",
):
    """Profile a complete forward pass (prefill)."""
    print("\n🔬 Profiling Full Forward Pass...")
    
    # Prepare inputs
    user_content = []
    if image_path:
        user_content.append({"type": "image", "url": image_path})
    user_content.append({"type": "text", "text": "Describe this image in detail." if image_path else "Hello, how are you?"})
    
    messages = [{"role": "user", "content": user_content}]
    inputs = llm.processor(messages, add_generation_prompt=True, device=llm.device)
    
    input_ids = inputs["input_ids"]
    pixels = inputs["pixels"]
    d_image = inputs["d_image"]
    
    print(f"   Input IDs shape: {input_ids.shape}")
    if pixels is not None:
        print(f"   Pixels shape: {pixels.shape}")
    
    # Warm up
    with torch.no_grad():
        for _ in range(2):
            _ = llm.model(input_ids, pixels=pixels, d_image=d_image)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Profile
    with profiler_context(output_path) as prof:
        with torch.no_grad():
            with record_function("full_forward"):
                if pixels is not None:
                    with record_function("vision_encoding"):
                        # This happens inside model.forward, but we trace it here
                        pass
                
                with record_function("model_forward"):
                    logits, _ = llm.model(input_ids, pixels=pixels, d_image=d_image)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print_stats_table(prof)


def profile_generation(
    llm,
    image_path: Optional[str],
    max_tokens: int = 20,
    output_path: str = "trace_generation.json",
):
    """Profile token generation with KV cache."""
    print(f"\n🔬 Profiling Token Generation ({max_tokens} tokens)...")
    
    # Prepare inputs
    user_content = []
    if image_path:
        user_content.append({"type": "image", "url": image_path})
    user_content.append({"type": "text", "text": "Describe this image." if image_path else "Tell me a short story."})
    
    messages = [{"role": "user", "content": user_content}]
    inputs = llm.processor(messages, add_generation_prompt=True, device=llm.device)
    
    input_ids = inputs["input_ids"]
    pixels = inputs["pixels"]
    d_image = inputs["d_image"]
    
    print(f"   Input IDs shape: {input_ids.shape}")
    if pixels is not None:
        print(f"   Pixels shape: {pixels.shape}")
    
    # Warm up with a short generation
    with torch.no_grad():
        for _ in llm.model.generate_stream(input_ids, pixels=pixels, d_image=d_image, max_new_tokens=3):
            pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Profile
    with profiler_context(output_path) as prof:
        with torch.no_grad():
            token_count = 0
            with record_function("generation_total"):
                for i, token in enumerate(llm.model.generate_stream(
                    input_ids, pixels=pixels, d_image=d_image, max_new_tokens=max_tokens
                )):
                    with record_function(f"token_{i}"):
                        token_count += 1
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"   Generated {token_count} tokens")
    print_stats_table(prof)


def profile_detailed_generation(
    llm,
    image_path: Optional[str],
    max_tokens: int = 10,
    output_path: str = "trace_detailed_generation.json",
):
    """
    Profile generation with detailed breakdown of each component.
    This provides the most granular view of where time is spent.
    """
    print(f"\n🔬 Profiling Detailed Generation ({max_tokens} tokens)...")
    
    # Prepare inputs
    user_content = []
    if image_path:
        user_content.append({"type": "image", "url": image_path})
    user_content.append({"type": "text", "text": "Describe what you see." if image_path else "Hello!"})
    
    messages = [{"role": "user", "content": user_content}]
    inputs = llm.processor(messages, add_generation_prompt=True, device=llm.device)
    
    input_ids = inputs["input_ids"]
    pixels = inputs["pixels"]
    d_image = inputs["d_image"]
    
    # Warm up
    with torch.no_grad():
        _, _ = llm.model(input_ids, pixels=pixels, d_image=d_image)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # Profile with schedule for multi-step
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=schedule(wait=0, warmup=1, active=max_tokens + 1, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(output_path),
    ) as prof:
        with torch.no_grad():
            # Prefill
            with record_function("prefill"):
                logits, past_kv = llm.model(input_ids, pixels=pixels, d_image=d_image)
            prof.step()
            
            # Decode tokens
            next_token = logits[:, -1:].argmax(dim=-1)
            position_offset = input_ids.shape[1]
            
            for i in range(max_tokens):
                with record_function(f"decode_step_{i}"):
                    with record_function("model_forward"):
                        logits, past_kv = llm.model(
                            next_token, 
                            past_key_values=past_kv,
                            position_offset=position_offset,
                        )
                    
                    with record_function("sampling"):
                        next_token = logits[:, -1:].argmax(dim=-1)
                    
                    position_offset += 1
                
                prof.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"✅ Trace saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile Hermes model for Perfetto")
    parser.add_argument("--image", type=str, help="Path to image file for vision profiling")
    parser.add_argument("--text-only", action="store_true", help="Profile text-only generation")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--output-dir", type=str, default="./profiler_traces", help="Output directory for traces")
    parser.add_argument(
        "--mode", 
        choices=["vision", "forward", "generation", "detailed", "all"], 
        default="all",
        help="What to profile"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Validate args
    if not args.text_only and not args.image:
        print("⚠️  No image provided. Use --image /path/to/image.jpg or --text-only")
        print("   Defaulting to text-only mode...")
        args.text_only = True
    
    image_path = None if args.text_only else args.image
    
    if image_path and not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    print("🚀 Loading Hermes model...")
    from hermes import LLM
    llm = LLM.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    
    device = get_device()
    print(f"   Device: {device}")
    print(f"   Model loaded successfully!")
    
    # Run profiling based on mode
    if args.mode in ["vision", "all"] and image_path:
        profile_vision_encoder_only(
            llm, 
            image_path, 
            str(output_dir / "trace_vision_encoder.json")
        )
    
    if args.mode in ["forward", "all"]:
        suffix = "image" if image_path else "text"
        profile_full_forward_pass(
            llm, 
            image_path, 
            str(output_dir / f"trace_forward_{suffix}.json")
        )
    
    if args.mode in ["generation", "all"]:
        suffix = "image" if image_path else "text"
        profile_generation(
            llm, 
            image_path, 
            args.max_tokens,
            str(output_dir / f"trace_generation_{suffix}.json")
        )
    
    if args.mode in ["detailed", "all"]:
        suffix = "image" if image_path else "text"
        profile_detailed_generation(
            llm, 
            image_path, 
            min(args.max_tokens, 10),  # Limit for detailed
            str(output_dir / f"trace_detailed_{suffix}.json")
        )
    
    print(f"\n{'='*80}")
    print("📁 All traces saved to:", output_dir)
    print("📊 Open https://ui.perfetto.dev and drag-drop trace files to analyze")
    print(f"{'='*80}")
    
    # Print tips
    print("\n💡 Tips for analyzing traces in Perfetto:")
    print("   1. Look for long bars in the timeline - these are bottlenecks")
    print("   2. Check for gaps between GPU operations (indicates CPU bottleneck)")
    print("   3. Search for 'cudaMemcpy' - excessive copies hurt performance")
    print("   4. Compare vision_encoder time vs model_forward time")
    print("   5. In decode steps, each step should be ~same duration")


if __name__ == "__main__":
    main()
