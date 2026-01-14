# Hermes 🚀

> [!IMPORTANT]
> **WIP** - Building High-Speed Inference Engine for VLM

## TODO

### Memory Optimizations
- [ ] **Paged Attention** - Block-based KV cache storage, reduces memory waste from 60-80% to ~4%
- [ ] **KV Cache Quantization** - INT4/INT8/FP8 quantization of KV cache for longer contexts
- [ ] **KV Cache Offloading** - Offload to CPU/disk for memory-constrained scenarios

### Attention Optimizations
- [ ] **FlashAttention 2** - 2x faster attention with optimized memory access patterns
- [ ] **FlashAttention 3** - 1.5-2x over FA2, overlaps compute and data movement (Hopper)
- [ ] **FlashInfer Kernels** - Efficient customizable attention kernels

### Decoding Optimizations
- [ ] **Speculative Decoding** - Draft model predicts tokens, main model verifies in parallel
- [ ] **Continuous Batching** - Dynamic batch merging for maximized GPU utilization

### Quantization
- [ ] **Weight Quantization** - INT4/INT8 weight quantization (AWQ, GPTQ)
- [ ] **FP8 Inference** - Native FP8 support for Hopper/Blackwell GPUs
- [ ] **Activation Quantization** - Handle outliers with mixed precision

### Parallelism
- [ ] **Tensor Parallelism** - Shard layers across multiple GPUs
- [ ] **Pipeline Parallelism** - Split model stages across devices
- [ ] **Expert Parallelism** - Distribute MoE experts across GPUs

### Kernel Optimizations
- [ ] **Custom CUDA Kernels** - Fused kernels for RMSNorm, RoPE, etc.
- [ ] **Triton Kernels** - Python-native GPU kernels for flexibility
- [ ] **CUDA Graphs** - Reduce kernel launch overhead

### Scheduling
- [ ] **Chunked Prefill** - Break long prefills into chunks
- [ ] **Prefill/Decode Disaggregation** - Separate stages for optimal batching
