# Benchmark notes

## A40 run (2026-11-13): caveats

- **CPU quota.** The RunPod container had a cgroup CPU quota of 7.65 vCPUs on a 96-core host, so
  the `cpu_count: 96` in the result files is the host count. BPE training ran 95 worker processes
  on about 8 vCPUs, and its time (169 s) is an 8-vCPU number.
- **TF32.** In fp32, Triton's `tl.dot` uses TF32 by default and the PyTorch baseline uses full
  FP32; compare kernels in bf16 for a like-for-like comparison.
- **PyTorch-tiled reference.** `FlashAttentionPytorch` supports fp32 only, so it is skipped in the
  bf16 attention sweeps. It is also limited to N <= 2048, because its tile loop runs in Python.
- **NVTX summary.** `nsys_attention_nvtx_sum.txt` shows host-side NVTX range times, which include
  kernel launch latency; the GPU-side stage split comes from CUDA events in `profile_attention_*.json`.
- **Throughput runs** (`train_throughput_*.json`) use random tokens, 200 steps, with 20 warmup steps
  excluded, and take the median over 50-step windows.
- **Training data.** `scripts/reproduce.sh encode` writes both `.npy` and raw `.bin`, and training
  reads the `.bin` (see Known issues in the README). Encoding produced 540,796,778 train and
  5,461,210 validation tokens (`encode_tinystories.txt`).
- **Hyperparameters.** The 22.7M run used `configs/tinystories_22m.yaml` with bf16 + `torch.compile`
  (lr 3e-4 → 3e-5 cosine, 200 warmup steps, wd 0.1, batch 128 × 256, 10k steps).

## Historical: course reference model (not this repo's code)

These were measured during CS336 A2 §2.1.3 with the course-provided `BasicsTransformerLM`,
**not** with `llmtk.model.TransformerLM`. They are kept for context only and are not quoted in
the README; re-measure with `benchmarks/bench_model.py`.

small, ctx=512, fp32, warmup=5, steps=10 (8 GB GPU):

| mode    | mean ± std (ms)  | peak memory |
|---------|------------------|-------------|
| forward | 134.91 ± 14.99   | 3.82 GiB    |
| fwd_bwd | 373.15 ± 18.70   | 4.55 GiB    |
| train   | 431.54 ± 26.04   | 5.04 GiB    |

- backward ≈ 238 ms ≈ 1.77× forward
- AdamW step ≈ 58 ms ≈ 13.5% of a train step

Warmup sweep (small, fwd_bwd):

| warmup | mean ± std (ms)   | note                       |
|--------|-------------------|----------------------------|
| 0      | 483.26 ± 309.58   | step 0 = 1360.86 ms (3.6×) |
| 1      | 359.86 ± 26.32    |                            |
| 5      | 379.51 ± 17.88    |                            |

w=1 has the lowest mean, possibly due to thermal throttling later in the run; std decreases
monotonically with more warmup.

medium / large at ctx=512 need 10.33 / 20.03 GiB peak, above the 8 GB of VRAM. The measured
15.9 s / 65.7 s per step reflect memory oversubscription and are **invalid**.
