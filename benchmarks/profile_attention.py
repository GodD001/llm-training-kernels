"""Where does attention time go? Per-stage latency (QK^T, scale+mask, softmax, PV) vs analytic FLOPs.

Runs the model's own (non-fused) attention ops, the same ones as
`llmtk.model.functional.scaled_dot_product_attention`, split into stages:
  * each stage is wrapped in an NVTX range, for Nsight Systems;
  * each stage is also timed with CUDA events (GPU time, no host syncs between stages), so the
    latency split can be reproduced without Nsight;
  * the analytic FLOPs and minimum bytes moved per stage are computed, to show that softmax is
    memory-bound: a small share of the FLOPs but a large share of the time.

Example:
  python -m benchmarks.profile_attention                  # timing + FLOPs, writes JSON
  python -m benchmarks.profile_attention --flops-only     # analytic split only, no GPU needed
  nsys profile -t cuda,nvtx -o benchmarks/results/nsys_attention --force-overwrite true \\
      python -m benchmarks.profile_attention --iters 20
"""

import argparse
import math
import statistics

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from benchmarks._common import save_results
from llmtk.model.functional import softmax

STAGES = ["QK^T", "scale+mask", "softmax", "PV"]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def analytic_costs(b: int, h: int, n: int, d: int, causal: bool, elem_bytes: int) -> dict[str, dict]:
    """FLOPs and minimum bytes moved per stage for one attention call (B*H independent heads).

    FLOP conventions: a multiply-add is 2 FLOPs; elementwise ops are 1 FLOP per element; softmax is
    5 FLOPs per score (max, subtract, exp, sum, divide). Bytes are a lower bound: each operand read
    once and each result written once.
    """
    bh, nn = b * h, n * n
    costs = {
        "QK^T": {"flops": 2 * bh * nn * d, "bytes": bh * (2 * n * d + nn) * elem_bytes},
        "scale+mask": {"flops": bh * nn * (2 if causal else 1), "bytes": bh * 2 * nn * elem_bytes},
        "softmax": {"flops": 5 * bh * nn, "bytes": bh * 2 * nn * elem_bytes},
        "PV": {"flops": 2 * bh * nn * d, "bytes": bh * (nn + 2 * n * d) * elem_bytes},
    }
    total = sum(c["flops"] for c in costs.values())
    for c in costs.values():
        c["flops_share"] = c["flops"] / total
        c["flops_per_byte"] = c["flops"] / c["bytes"]
    return costs


def attention_stages(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, events=None):
    """The model's attention, one stage at a time; records CUDA events between stages if given."""

    def mark(i: int) -> None:
        if events is not None:
            events[i].record()

    mark(0)
    with nvtx.range("QK^T"):
        scores = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
    mark(1)
    with nvtx.range("scale+mask"):
        scores = scores / math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
    mark(2)
    with nvtx.range("softmax"):
        probs = softmax(scores, dim=-1)
    mark(3)
    with nvtx.range("PV"):
        out = einsum(probs, v, "... queries keys, ... keys d_v -> ... queries d_v")
    mark(4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--no-causal", action="store_true", help="disable the causal mask (on by default, as in the LM)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--flops-only", action="store_true", help="print the analytic split and exit (no GPU needed)")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    b, h, n, d = args.batch, args.heads, args.seq_len, args.head_dim
    causal = not args.no_causal
    dtype = DTYPES[args.dtype]
    costs = analytic_costs(b, h, n, d, causal, torch.finfo(dtype).bits // 8)

    print(f"attention B={b} H={h} N={n} d={d} {args.dtype} causal={causal}")
    print(f"{'stage':>11} {'GFLOP':>9} {'FLOP share':>11} {'FLOP/byte':>10}")
    for s in STAGES:
        c = costs[s]
        print(f"{s:>11} {c['flops'] / 1e9:9.3f} {100 * c['flops_share']:10.2f}% {c['flops_per_byte']:10.2f}")

    rows = [{"stage": s, **costs[s]} for s in STAGES]
    summary = {"softmax_flops_share": costs["softmax"]["flops_share"]}
    name = args.name or f"profile_attention_{args.dtype}_n{n}{'_causal' if causal else ''}"

    if args.flops_only:
        path = save_results(name + "_flops", vars(args), rows, summary)
        print(f"wrote {path}")
        return

    if torch.cuda.is_available() and args.dtype == "bf16" and torch.cuda.get_device_capability() < (8, 0):
        p.error(f"bf16 needs an sm80+ GPU; this is {torch.cuda.get_device_name()}")
    device = "cuda"
    q, k, v = (torch.randn(b, h, n, d, device=device, dtype=dtype) for _ in range(3))
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device)) if causal else None

    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            attention_stages(q, k, v, mask)
    torch.cuda.synchronize()

    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    totals = []
    for i in range(args.iters):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(STAGES) + 1)]
        with nvtx.range(f"attention_{i}"):
            attention_stages(q, k, v, mask, events)
        torch.cuda.synchronize()
        for j, s in enumerate(STAGES):
            per_stage[s].append(events[j].elapsed_time(events[j + 1]))
        totals.append(events[0].elapsed_time(events[-1]))

    total_med = statistics.median(totals)
    print(f"\n{'stage':>11} {'median ms':>10} {'time share':>11} {'FLOP share':>11}")
    for row in rows:
        s = row["stage"]
        row["median_ms"] = statistics.median(per_stage[s])
        row["std_ms"] = statistics.stdev(per_stage[s]) if len(per_stage[s]) > 1 else 0.0
        row["time_share"] = row["median_ms"] / total_med
        print(f"{s:>11} {row['median_ms']:10.3f} {100 * row['time_share']:10.2f}% {100 * row['flops_share']:10.2f}%")
    print(f"{'total':>11} {total_med:10.3f}")

    summary |= {
        "total_median_ms": total_med,
        "softmax_time_share": next(r["time_share"] for r in rows if r["stage"] == "softmax"),
        "timing_method": "CUDA events between stages, median over iterations",
    }
    path = save_results(name, vars(args), rows, summary)
    print(f"wrote {path}")
    print(
        "\nNsight Systems view of the same ranges:\n"
        "  nsys profile -t cuda,nvtx -o benchmarks/results/nsys_attention --force-overwrite true "
        f"python -m benchmarks.profile_attention --seq-len {n} --dtype {args.dtype} --iters {args.iters}\n"
        "  nsys stats --report nvtx_sum benchmarks/results/nsys_attention.nsys-rep"
    )


if __name__ == "__main__":
    main()
