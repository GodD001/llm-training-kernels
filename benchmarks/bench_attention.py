"""Latency and peak memory of naive attention vs PyTorch-tiled FA-2 vs Triton FA-2 across sequence lengths.

Peak memory is measured above the inputs (Q, K, V, dO), so it shows what each implementation allocates
for itself: naive attention stores the N x N score/probability matrices, FlashAttention does not.

Example:
  python -m benchmarks.bench_attention --dtype fp32 --name attention_fp32
  python -m benchmarks.bench_attention --dtype bf16 --causal --name attention_bf16_causal
"""

import argparse
import gc

import torch

from benchmarks._common import cuda_time, save_results
from llmtk.kernels.flash_attention_torch import FlashAttentionPytorch
from llmtk.kernels.flash_attention_triton import FlashAttentionTriton
from llmtk.model.functional import scaled_dot_product_attention

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    """This repo's non-fused attention: materializes the full (B, N, N) scores and probabilities."""
    mask = None
    if is_causal:
        n = q.shape[-2]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
    return scaled_dot_product_attention(q, k, v, mask)


IMPLS = {
    "naive": naive,
    "torch_tiled": FlashAttentionPytorch.apply,
    "triton": FlashAttentionTriton.apply,
}


def known_issue(impl: str, n: int, is_causal: bool) -> str | None:
    """Known correctness bugs (see tests/test_flash_attention.py) that affect a configuration."""
    if impl == "torch_tiled" and is_causal:
        return "B2: forward ignores is_causal (timing is for unmasked attention)"
    if impl == "triton" and not is_causal and n % 16 != 0:
        return "B1: padded keys not masked when N % 16 != 0"
    return None


def measure(impl: str, mode: str, b: int, n: int, d: int, dtype: torch.dtype, is_causal: bool, args) -> dict:
    """Time one (impl, mode, N) point and record peak extra memory; returns a result row."""
    fn = IMPLS[impl]
    row = {"impl": impl, "mode": mode, "seq_len": n, "known_issue": known_issue(impl, n, is_causal)}
    gc.collect()
    torch.cuda.empty_cache()
    try:
        q, k, v, do = (torch.randn(b, n, d, device="cuda", dtype=dtype) for _ in range(4))
        if mode == "fwd_bwd":
            for t in (q, k, v):
                t.requires_grad_()

            def step():
                q.grad = k.grad = v.grad = None
                fn(q, k, v, is_causal).backward(do)
        else:

            def step():
                with torch.no_grad():
                    fn(q, k, v, is_causal)

        step()  # compile / autotune outside the memory measurement
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        step()
        torch.cuda.synchronize()
        row["peak_extra_mib"] = (torch.cuda.max_memory_allocated() - base) / 1024**2
        row.update(cuda_time(step, args.warmup, args.iters))
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    finally:
        q = k = v = do = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=8, help="batch x heads (FA-2 inputs are (B, N, d))")
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--impls", nargs="+", choices=list(IMPLS), default=list(IMPLS))
    p.add_argument("--modes", nargs="+", choices=["fwd", "fwd_bwd"], default=["fwd", "fwd_bwd"])
    p.add_argument(
        "--tiled-max-n", type=int, default=2048, help="skip torch_tiled above this N (its Python tile loop is slow)"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--name", default=None, help="results file name (default derived from args)")
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        p.error(f"bf16 needs an sm80+ GPU; this is {torch.cuda.get_device_name()}")
    name = args.name or f"attention_{args.dtype}{'_causal' if args.causal else ''}"
    rows = []
    print(f"{'N':>6} {'impl':>12} {'mode':>8} {'median ms':>10} {'peak extra MiB':>15}  note")
    for n in args.seq_lens:
        for impl in args.impls:
            if impl == "torch_tiled" and n > args.tiled_max_n:
                continue
            for mode in args.modes:
                row = measure(impl, mode, args.batch, n, args.head_dim, dtype, args.causal, args)
                rows.append(row)
                ms = f"{row['median_ms']:10.3f}" if row["status"] == "ok" else f"{row['status']:>10}"
                mem = f"{row['peak_extra_mib']:15.1f}" if row["status"] == "ok" else " " * 15
                print(f"{n:>6} {impl:>12} {mode:>8} {ms} {mem}  {row['known_issue'] or ''}", flush=True)

    config = {k: v for k, v in vars(args).items()}
    path = save_results(name, config, rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
