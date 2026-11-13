# Benchmark notes

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
