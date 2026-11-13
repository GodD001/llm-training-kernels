# FlashAttention-2 backward in Triton

Code: `llmtk/kernels/flash_attention_triton.py` (`flash_bwd_dkdv_kernel`, `flash_bwd_dq_kernel`,
`flash_backward_triton`). The previous PyTorch backward is kept in
`llmtk/kernels/flash_attention_torch.py` for comparison.

## The math

For one batch element with queries `Q` (N_q×d), keys `K` and values `V` (N_k×d), and scale `s = 1/√d`:

```
S = s · Q Kᵀ          P = softmax(S) = exp(S − L)        O = P V
```

`L` is the row-wise logsumexp of `S`. Given the output gradient `dO`, the chain rule gives

```
dV = Pᵀ dO
dP = dO Vᵀ
dS = P ∘ (dP − D)        where D_i = Σ_j P_ij dP_ij = rowsum(dO ∘ O)_i
dQ = s · dS K
dK = s · dSᵀ Q
```

The identity `D = rowsum(dO ∘ O)` is the key trick. The softmax Jacobian needs `Σ_j P_ij dP_ij` for
every row, and that equals `dO_i · O_i`, a length-d dot product. So it can be computed up front
from tensors we already have, without ever forming `P` or `dP`.

## What is saved vs recomputed

| | Size | Where |
|---|---|---|
| Saved by forward | `Q, K, V, O` (N×d) and `L` (N) | GPU memory, O(N·d) |
| Precomputed in backward | `D = rowsum(dO ∘ O)` (N) | GPU memory, O(N) |
| Recomputed per tile | `S`, `P`, `dP`, `dS` (tile×tile) | on-chip (registers / shared memory) |

A naive backward keeps `P` (and usually `S`) for the whole sequence: B×N×N floats, which is 4 GiB
at B=4 and N=16k in fp32. Here every N×N quantity exists only as a 32×32 tile inside a running
program, so memory grows linearly in N. The cost is recomputing `QKᵀ` and `exp` once more. That is
cheap, because attention at these sizes is limited by memory traffic, not by FLOPs.

## Tiling: two kernels, no atomics

`dK_j` and `dV_j` are sums over all query rows; `dQ_i` is a sum over all key columns. A single
kernel can only own one side, so the work is split into two kernels:

1. **`flash_bwd_dkdv_kernel`**: grid = (key tiles, batch). Each program loads one `K_j, V_j` tile
   and keeps `dK_j, dV_j` accumulators in fp32. It loops over query tiles `i`, loading
   `Q_i, dO_i, L_i, D_i`, and:
   - recomputes `S = s·Q_i K_jᵀ` and `P = exp(S − L_i)`
   - `dV_j += Pᵀ dO_i`
   - `dS = P ∘ (dO_i V_jᵀ − D_i)`
   - `dK_j += dSᵀ Q_i`

   It writes `s·dK_j` and `dV_j` once at the end.
2. **`flash_bwd_dq_kernel`**: grid = (query tiles, batch). Each program owns one `dQ_i`
   accumulator and loops over key tiles with the same recomputation: `dQ_i += dS K_j`, then
   writes `s·dQ_i`.

Each output tile is written by exactly one program, so no `tl.atomic_add` is needed and results are
deterministic. The price is recomputing `S` and `P` in both kernels. The fused alternative computes
`dQ` inside the dK/dV loop with atomic adds, which avoids the second pass but adds contention and
non-deterministic float summation order.

Precision: inputs may be fp32 or bf16. Every `tl.dot` accumulates in fp32, `P` and `dS` are cast
to the input dtype only as matmul operands, and `L`/`D` are always fp32. Tile sizes are
`tl.constexpr` (default 32×32). With d ≤ 128 each program holds only a few 32×d tiles, which fits
comfortably on any recent GPU, including 8 GB cards.

## Causal masking

Query `i` may attend to key `j` only if `i ≥ j` (absolute positions, the same rule as the
forward). Tiles fall into three cases:

- **Fully masked tiles are skipped.** In the dK/dV kernel, a key tile starting at `k_start` gets
  nothing from queries `< k_start`, so the query loop starts at the query tile containing
  `k_start`. In the dQ kernel, a query tile ending at `q_start + Q_TILE` sees no keys beyond that,
  so the key loop stops there. Roughly half the tiles are never touched, so causal backward does
  about half the work.
- **Diagonal tiles** get an element-wise mask `offs_q ≥ offs_k`.
- **Padding.** The last tile may run past N. Out-of-range rows and columns are loaded as zeros and
  also masked.

All masks are applied to `P` itself (`P = where(valid, exp(S − L), 0)`). That zeroes the masked
entries of `dV`, `dS`, `dK` and `dQ` in one place, with no −∞ arithmetic.

## How it was checked

- `tests/test_flash_attention.py` compares O, L, dQ, dK and dV against fp32 naive attention:
  - fp32 and bf16, causal and non-causal
  - lengths that are multiples of 16/32, and lengths that are neither (33, 37, 100, 200), plus N_q ≠ N_k
  - d ∈ {16, 32, 64, 128}
- A memory test asserts that fwd+bwd at N=4096 allocates less than ¼ of one fp32 B×N×N matrix.
- `benchmarks/bench_attention.py` measures latency and peak memory against naive attention across
  sequence lengths.

Known limitation, inherited from the forward kernel and left unfixed by design (bug B1): with
`is_causal=False` and `N_k % 16 ≠ 0`, the forward does not mask padded keys, so `O` and `L` are
wrong, and so is the backward, which depends on `L`. The backward kernels themselves mask padding
correctly.
