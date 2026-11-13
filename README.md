# llm-training-kernels

From-scratch Transformer LM training stack with a fused FlashAttention-2 Triton kernel and profiling-driven optimization.

The whole pipeline is implemented and tested end to end: tokenizer, model, optimizer, training loop, attention kernels, benchmarks and profiling. Every number below was produced by a script in this repo, and each row links to the JSON that script wrote.

## Results

Hardware: **NVIDIA A40 (48 GB, sm86)**, CUDA 12.8, PyTorch 2.8.0, Triton 3.4.0. The container had a CPU quota of about 7.65 vCPUs (Xeon Gold 6342). Run date: 2026-11-13.

| What | Result | Source |
|---|---|---|
| FA-2 Triton vs unfused PyTorch attention, fwd+bwd, bf16, causal, N=8192 (B=8, d=64) | **10.3× faster** (9.7 vs 100.0 ms); extra memory **56 MiB vs 6.2 GB** | [attention_bf16_causal.json](benchmarks/results/attention_bf16_causal.json) |
| Same, non-causal | **6.4× faster** (12.9 vs 82.5 ms) | [attention_bf16.json](benchmarks/results/attention_bf16.json) |
| Memory scaling, fwd+bwd, bf16 | Triton grows linearly: 1.8 MiB at N=256 to 225 MiB at N=32,768. Naive grows quadratically and **runs out of memory at N=32,768** on 48 GB | [attention_bf16.json](benchmarks/results/attention_bf16.json) |
| Softmax share of attention: FLOPs vs latency (B=4, H=12, N=512, d=64) | **1.9% of FLOPs, 42.7% of latency** (fp32); 44.1% in bf16 | [fp32](benchmarks/results/profile_attention_fp32_n512_causal.json), [bf16](benchmarks/results/profile_attention_bf16_n512_causal.json) |
| bf16 autocast vs fp32, forward (12 layers, d=768, ctx 512, B=4) | **1.74× faster, −22.1% peak memory**; fwd+bwd 1.71×, full train step 1.57× | [model_small_ctx512.json](benchmarks/results/model_small_ctx512.json) |
| Training throughput, 22.7M model (B=128, ctx 256) | **61.0K → 222.2K tokens/s (3.64×)** with bf16 + `torch.compile`; compile alone 1.65×, bf16 alone 1.47× | [fp32](benchmarks/results/train_throughput_fp32.json), [bf16+compile](benchmarks/results/train_throughput_bf16_compile.json) |
| TinyStories: 22,696,448 params, 327.7M tokens, bf16 + compile | **validation loss 1.498** (200 batches), 25.3 min wall time | [train_tinystories_22m.json](benchmarks/results/train_tinystories_22m.json) |
| Byte-level BPE, vocab 10k, TinyStories train (2.23 GB) | **169 s** (13.2 MB/s), 1.1 GiB peak RSS, on about 8 vCPUs | [tokenizer_tinystories_train.json](benchmarks/results/tokenizer_tinystories_train.json) |
| Correctness on GPU | 89 passed; 9 expected failures are known issues, listed below | [pytest_gpu.txt](benchmarks/results/pytest_gpu.txt) |

Notes:
- "Unfused PyTorch attention" is this repo's `scaled_dot_product_attention` (explicit QKᵀ, softmax, PV), not `torch.nn.functional.scaled_dot_product_attention`.
- Headline kernel comparisons use bf16, where both sides use the same precision. In fp32, Triton's `tl.dot` runs on TF32 tensor cores by default while the PyTorch baseline uses full FP32, so the fp32 speedups (7.1× / 11.6× at N=8192) are not apples-to-apples. Those numbers are in [attention_fp32.json](benchmarks/results/attention_fp32.json) and [attention_fp32_causal.json](benchmarks/results/attention_fp32_causal.json).
- The throughput runs use random tokens: the compute is the same, but no dataset is needed. The full training run uses real TinyStories tokens.
- More caveats are in [benchmarks/results/notes.md](benchmarks/results/notes.md).

## Highlights

- **FlashAttention-2 in Triton, forward and backward.** The fused forward keeps an online softmax per query tile and saves only the output and the logsumexp. The backward is two kernels, one accumulating dK/dV per key tile and one accumulating dQ per query tile, with no atomics. Both recompute S and P on-chip from the logsumexp, so neither pass stores the N×N attention matrix. Causal attention skips fully masked tiles. Inputs can be fp32 or bf16, with fp32 accumulation. Write-up: [docs/flash_attention_backward.md](docs/flash_attention_backward.md).
- **Transformer LM training stack from scratch.** RMSNorm, SwiGLU, RoPE, causal multi-head attention, AdamW, a cosine schedule with warmup, gradient clipping, checkpointing and top-p sampling. Layers, losses and optimizers are tested against PyTorch references, and the 22.7M config's parameter count is asserted exactly.
- **An instrumented training loop.** bf16 autocast, `torch.compile`, and tokens/s reported as the median over fixed windows, with warmup, evaluation and checkpoint time excluded. A divergence guard stops the run on a NaN/inf loss or a spike above k× the running median. Each run writes a summary JSON and a per-step metrics log.
- **Profiling-driven analysis.** Attention is split into QKᵀ, scale+mask, softmax and PV, each wrapped in an NVTX range for Nsight Systems and timed with CUDA events. This is paired with an analytic FLOPs and arithmetic-intensity model, so the latency-vs-FLOPs gap can be reproduced without a profiler.
- **A fast byte-level BPE trainer.** Pre-tokenization runs in parallel over chunks split at document boundaries, and merges use a max-heap with incremental pair-count updates. Its merges are identical to a naive reference implementation.

## What this covers for LLM engineering work

| Area | In this repo |
|---|---|
| GPU kernels and performance | Triton FA-2 forward + backward; memory and latency scaling against naive attention; bf16 vs fp32 |
| Pre-training infrastructure | Mixed precision, `torch.compile`, throughput accounting, divergence detection, checkpointing, config-driven runs |
| Profiling and analysis | NVTX + Nsight Systems, CUDA-event stage timing, roofline-style FLOPs/byte reasoning |
| Tokenization and data | Parallel BPE training, streaming dataset encoding, compression-ratio and throughput tools |
| Correctness and reproducibility | 100 tests, mostly against PyTorch or naive reference implementations (Triton cases run on CUDA); one script reproduces every measurement; results carry hardware and version metadata |

## Profiling: why softmax costs more time than FLOPs

<!-- Nsight Systems timeline of the NVTX ranges goes here: docs/images/nsys_attention.png -->

Per-stage GPU time is measured with CUDA events (median of 20 runs), and the same ranges are wrapped in NVTX for Nsight Systems ([NVTX summary](benchmarks/results/nsys_attention_nvtx_sum.txt)). The NVTX summary reports host-side range durations, so the GPU split comes from the CUDA events.

In non-fused attention, QKᵀ and PV are matrix multiplies. They do about 2d FLOPs for every element they write: 25.6 FLOP/byte at d=64 in fp32 by the analytic model. Softmax does about 5 FLOPs per score but has to read and write the entire B·H·N×N score matrix, about 0.6 FLOP/byte, roughly 40× less work per byte moved. That is far below what any modern GPU needs to be compute-bound, so its runtime is set by memory bandwidth, not arithmetic. In eager PyTorch it is also several separate kernels (max, subtract, exp, sum, divide), each one a full pass over N² elements. The measured result: softmax is **1.9% of the FLOPs but 42.7% of the latency** on an A40. Scale+mask, also elementwise over N², takes another 32%. Fusing softmax into the matmul tiles so scores never leave on-chip memory is exactly what FlashAttention does.

## Quickstart

```bash
pip install -e '.[dev]'
pytest -q                                   # Triton tests run only on a CUDA GPU
```

Reproduce the measurements (each stage writes JSON to `benchmarks/results/`):

```bash
bash scripts/reproduce.sh                   # tests, attention, model, profile, throughput (no dataset needed)
bash scripts/reproduce.sh data tokenizer encode train   # TinyStories: download, BPE, encode, 22.7M training
```

Or run the pieces directly:

```bash
python -m benchmarks.bench_attention --dtype bf16 --causal
python -m benchmarks.bench_model --size small --dtypes fp32 bf16
python -m benchmarks.profile_attention --seq-len 512
python -m benchmarks.bench_tokenizer --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --out-dir data/tok
python train.py --config configs/tinystories_22m.yaml --dtype bf16 --compile \
    --train_path data/ts_train.bin --val_path data/ts_valid.bin
```

A GPU notebook for Colab or a Jupyter-enabled cloud GPU is in [notebooks/colab_gpu.ipynb](notebooks/colab_gpu.ipynb).

## Project layout

```
llmtk/
  model/        layers (Linear, Embedding, RMSNorm, SwiGLU, RoPE), attention, transformer, functional ops
  kernels/      FlashAttention-2: tiled PyTorch reference, Triton forward + backward
  tokenizer/    byte-level BPE trainer and tokenizer
  optim/        AdamW, cosine schedule, gradient clipping
  training.py   throughput meter, divergence detector
  data.py  checkpoint.py  generate.py
train.py        config-driven training loop
configs/        tinystories_22m.yaml
benchmarks/     bench_attention, bench_model, profile_attention, bench_tokenizer, results/
scripts/        reproduce.sh, encode_dataset.py
tests/          layers, BPE, optimizer, FlashAttention, training
docs/           flash_attention_backward.md
```

## Known issues

These are tracked rather than hidden. The affected test cases are marked `xfail`.

- The Triton forward does not mask padded keys, so non-causal attention is wrong when the key length is not a multiple of 16.
- The tiled PyTorch reference (`FlashAttentionPytorch`) ignores `is_causal` in its forward pass.
- `generate.decode` does not truncate the context once it exceeds the model's `context_length`.
- `train.py` reads raw uint16 token files, but `scripts/encode_dataset.py encode` writes `.npy`. `reproduce.sh` converts to `.bin` before training.

## Acknowledgements

Built as a self-study project following Stanford CS336 (Language Modeling from Scratch). All implementations in this repository are my own.
