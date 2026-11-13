"""End-to-end timing of TransformerLM: forward, forward+backward, full train step; fp32 vs bf16 autocast.

Each (dtype, mode) point is timed with warmup + per-step synchronization (median, std) and its
peak memory is recorded. Steps are wrapped in NVTX ranges for Nsight Systems.

Example:
  python -m benchmarks.bench_model --size small --context-length 512 --dtypes fp32 bf16
"""

import argparse
import gc
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import torch
import torch.cuda.nvtx as nvtx

from benchmarks._common import cuda_time, save_results
from llmtk.model import TransformerLM, cross_entropy
from llmtk.optim import AdamW

MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
    # the TinyStories training model (configs/tinystories_22m.yaml)
    "ts22m": dict(d_model=512, d_ff=1344, num_layers=4, num_heads=16),
}

VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
MODES = ["forward", "fwd_bwd", "train"]


def build_model(size: str, context_length: int, device: str) -> TransformerLM:
    """Instantiate a TransformerLM of the named size."""
    return TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        theta=ROPE_THETA,
        **MODEL_SIZES[size],
    ).to(device)


def make_batch(batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token batch: (inputs, next-token targets)."""
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, context_length + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


def make_step_fn(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: str,
    autocast_ctx: Callable[[], AbstractContextManager],
) -> Callable[[], None]:
    """Return a no-arg function that runs one step; mode in {forward, fwd_bwd, train}."""

    def step() -> None:
        with nvtx.range(mode):
            if mode == "forward":
                # autograd stays on (as in a training forward), so activations are kept
                with autocast_ctx():
                    model(inputs)
                return
            if mode == "train":
                optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(inputs)
            loss = cross_entropy(logits.float(), targets)
            loss.backward()
            if mode == "train":
                optimizer.step()
            else:
                model.zero_grad(set_to_none=True)

    return step


def measure(args: argparse.Namespace, dtype: str, mode: str) -> dict:
    """Build a fresh model and time one (dtype, mode) point; returns a result row."""
    row = {"size": args.size, "dtype": dtype, "mode": mode, "batch_size": args.batch_size, "ctx": args.context_length}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build_model(args.size, args.context_length, "cuda")
        row["parameters"] = sum(p.numel() for p in model.parameters())
        fwd_model = torch.compile(model) if args.compile else model
        inputs, targets = make_batch(args.batch_size, args.context_length, "cuda")
        optimizer = AdamW(model.parameters())

        def autocast_ctx():
            return torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()

        step_fn = make_step_fn(fwd_model, inputs, targets, optimizer, mode, autocast_ctx)
        row.update(cuda_time(step_fn, args.warmup, args.steps))
        row["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    finally:
        model = fwd_model = optimizer = inputs = targets = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="small", choices=list(MODEL_SIZES))
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--name", default=None, help="results file name (default derived from args)")
    args = p.parse_args()

    if "bf16" in args.dtypes and torch.cuda.get_device_capability() < (8, 0):
        p.error(f"bf16 needs an sm80+ GPU; this is {torch.cuda.get_device_name()} (use --dtypes fp32)")

    rows = []
    print(f"{'dtype':>5} {'mode':>8} {'median ms':>10} {'std ms':>8} {'peak GiB':>9}")
    for dtype in args.dtypes:
        for mode in args.modes:
            r = measure(args, dtype, mode)
            rows.append(r)
            if r["status"] == "ok":
                print(f"{dtype:>5} {mode:>8} {r['median_ms']:10.2f} {r['std_ms']:8.2f} {r['peak_memory_gib']:9.2f}")
            else:
                print(f"{dtype:>5} {mode:>8} {r['status']:>10}")

    # bf16 vs fp32 ratios for every mode measured in both dtypes
    ratios = {}
    by_key = {(r["dtype"], r["mode"]): r for r in rows if r["status"] == "ok"}
    for mode in args.modes:
        a, b = by_key.get(("fp32", mode)), by_key.get(("bf16", mode))
        if a and b:
            ratios[mode] = {
                "speedup_bf16_vs_fp32": a["median_ms"] / b["median_ms"],
                "peak_memory_change_pct": 100 * (b["peak_memory_gib"] / a["peak_memory_gib"] - 1),
            }
            print(f"{mode}: bf16 speedup {ratios[mode]['speedup_bf16_vs_fp32']:.2f}x, "
                  f"peak memory {ratios[mode]['peak_memory_change_pct']:+.1f}%")  # fmt: skip

    name = args.name or f"model_{args.size}_ctx{args.context_length}{'_compile' if args.compile else ''}"
    path = save_results(name, vars(args), rows, summary=ratios)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
