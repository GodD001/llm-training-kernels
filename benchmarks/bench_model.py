"""End-to-end timing of TransformerLM: forward, forward+backward, or full train step, with NVTX ranges.

Example:
  python benchmarks/bench_model.py --size small --context-length 512 --mode fwd_bwd --dtype bf16
"""

import argparse
import json
import os
import statistics
import timeit
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import torch
import torch.cuda.nvtx as nvtx

from llmtk.model import TransformerLM, cross_entropy
from llmtk.optim import AdamW

MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4
ROPE_THETA = 10000.0


def build_model(size: str, context_length: int, device: str) -> TransformerLM:
    """Instantiate a TransformerLM of the named size."""
    return TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        theta=ROPE_THETA,
        **MODEL_SIZES[size],
    ).to(device)


def make_batch(context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token batch: (inputs, next-token targets)."""
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


def make_step_fn(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: str,
    autocast_ctx: AbstractContextManager,
) -> Callable[[], None]:
    """Return a no-arg function that runs one step; mode in {forward, fwd_bwd, train}."""

    def step() -> None:
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(inputs)
            if mode != "forward":
                loss = cross_entropy(logits, targets)
        if mode != "forward":
            loss.backward()
        if mode == "train":
            optimizer.step()

    return step


def benchmark(step_fn: Callable[[], None], warmup_steps: int, measure_steps: int) -> tuple[float, float, list[float]]:
    """Run warmup then timed steps (synchronizing each); returns (mean s, std s, per-step s)."""
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            step_fn()
        torch.cuda.synchronize()

    times = []
    for i in range(measure_steps):
        with nvtx.range(f"step_{i}"):
            start = timeit.default_timer()
            step_fn()
            torch.cuda.synchronize()
            times.append(timeit.default_timer() - start)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, times


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="small", choices=list(MODEL_SIZES))
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--mode", default="fwd_bwd", choices=["forward", "fwd_bwd", "train"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--per-step", action="store_true")
    p.add_argument("--json", type=str, default=None, help="append the result record to this JSON list")
    args = p.parse_args()

    device = "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()

    header = f"{args.size:7s} ctx={args.context_length:5d} mode={args.mode:8s} dtype={args.dtype} warmup={args.warmup}"
    try:
        model = build_model(args.size, args.context_length, device)
        inputs, targets = make_batch(args.context_length, device)
        optimizer = AdamW(model.parameters())
        step_fn = make_step_fn(model, inputs, targets, optimizer, args.mode, autocast_ctx)
        mean, std, times = benchmark(step_fn, args.warmup, args.steps)
    except torch.cuda.OutOfMemoryError:
        print(f"{header}  ->  OOM")
        torch.cuda.empty_cache()
        return

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{header}  ->  {mean * 1000:8.2f} ms  ±{std * 1000:6.2f}  peak {peak:.2f} GiB")
    if args.json:
        rec = dict(
            size=args.size,
            ctx=args.context_length,
            mode=args.mode,
            dtype=args.dtype,
            warmup=args.warmup,
            steps=args.steps,
            mean_ms=mean * 1000,
            std_ms=std * 1000,
            peak_gib=peak,
        )
        data = []
        if os.path.exists(args.json):
            with open(args.json) as f:
                data = json.load(f)
        data.append(rec)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)

    if args.per_step:
        for i, t in enumerate(times):
            print(f"    step {i:2d}: {t * 1000:8.2f} ms")


if __name__ == "__main__":
    main()
