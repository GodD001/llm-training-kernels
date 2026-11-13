"""Train a TransformerLM on pre-tokenized uint16 token files.

Examples:
  python train.py --config configs/tinystories_22m.yaml --train_path data/ts_train.npy --val_path data/ts_valid.npy
  python train.py --config configs/tinystories_22m.yaml --dtype bf16 --compile --train_path ... --val_path ...
  # throughput only, random tokens, no dataset needed:
  python train.py --config configs/tinystories_22m.yaml --synthetic --max_iters 300 --dtype bf16 --compile

Every run writes benchmarks/results/train_<run_name>.json (summary + environment) and
benchmarks/results/train_<run_name>_metrics.jsonl (one line per log/eval point).
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks._common import RESULTS_DIR, env_info
from llmtk.checkpoint import save_checkpoint
from llmtk.data import get_batch
from llmtk.model import TransformerLM, cross_entropy
from llmtk.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from llmtk.training import DivergenceDetector, DivergenceError, ThroughputMeter


def build_parser() -> argparse.ArgumentParser:
    """All command-line options; a --config YAML file supplies defaults for any of them."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML file of defaults (keys = option names)")

    # Data and output
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--val_path", type=str, default=None)
    p.add_argument("--synthetic", action="store_true", help="random tokens instead of files (throughput runs)")
    p.add_argument("--ckpt_path", type=str, default="ckpt.pt", help="'' disables checkpointing")
    p.add_argument("--run_name", type=str, default=None, help="names the results files")
    p.add_argument("--results_dir", type=str, default=str(RESULTS_DIR), help="where summary/metrics files go")

    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer and LR schedule
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)

    # Precision / compilation
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast mixed precision")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")

    # Loop, logging, monitoring
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_interval", type=int, default=200, help="0 disables periodic evaluation")
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--final_eval_batches", type=int, default=100)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=50, help="steps per throughput window / log line")
    p.add_argument("--throughput_warmup", type=int, default=20, help="steps excluded from tokens/s (compile etc.)")
    p.add_argument("--div_k", type=float, default=3.0, help="stop if loss > k x running median (<=0 disables)")
    p.add_argument("--div_window", type=int, default=100, help="steps in the running median")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args, letting --config provide defaults that explicit flags override."""
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in p._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            p.error(f"unknown keys in {pre.config}: {unknown}")
        p.set_defaults(**cfg)
    args = p.parse_args(argv)
    if not args.synthetic and not (args.train_path and args.val_path):
        p.error("--train_path and --val_path are required unless --synthetic is given")
    if args.run_name is None:
        args.run_name = f"{args.dtype}{'_compile' if args.compile else ''}{'_synthetic' if args.synthetic else ''}"
    return args


def load_data(path: str, dtype: type = np.uint16) -> np.ndarray:
    """Memory-map a flat token file."""
    return np.memmap(path, dtype=dtype, mode="r")


def synthetic_data(vocab_size: int, n_tokens: int, seed: int) -> np.ndarray:
    """Uniform random token ids (for throughput measurements only)."""
    return np.random.default_rng(seed).integers(0, vocab_size, size=n_tokens, dtype=np.uint16)


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def estimate_val_loss(model, val_data: np.ndarray, args: argparse.Namespace, n_batches: int, autocast_ctx) -> float:
    """Mean loss over `n_batches` random validation batches."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, args.batch_size, args.context_length, args.device)
        with autocast_ctx():
            logits = model(x)
        losses.append(cross_entropy(logits.float(), y).item())
    model.train()
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.synthetic:
        n = max(2_000_000, 4 * args.batch_size * (args.context_length + 1))
        train_data = synthetic_data(args.vocab_size, n, args.seed)
        val_data = synthetic_data(args.vocab_size, n, args.seed + 1)
    else:
        train_data = load_data(args.train_path)
        val_data = load_data(args.val_path)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.context_length
    print(f"parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(
        f"tokens/step: {tokens_per_step:,} | steps: {args.max_iters:,} | "
        f"total tokens: {tokens_per_step * args.max_iters:,} | dtype: {args.dtype} | compile: {args.compile}"
    )

    fwd_model = torch.compile(model) if args.compile else model
    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    def autocast_ctx():
        if args.dtype == "bf16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    meter = ThroughputMeter(tokens_per_step, window=args.log_interval, warmup_steps=args.throughput_warmup)
    detector = DivergenceDetector(k=args.div_k, window=args.div_window)
    pending: list[tuple[int, torch.Tensor]] = []  # losses not yet copied to the host

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"train_{args.run_name}_metrics.jsonl"
    summary_path = results_dir / f"train_{args.run_name}.json"
    metrics_file = open(metrics_path, "w")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    latest = {"loss": float("nan")}

    def flush_losses() -> float:
        """Copy pending losses to the host and run the divergence check; returns the latest loss."""
        if pending:
            values = torch.stack([t for _, t in pending]).float().cpu().tolist()
            steps = [s for s, _ in pending]
            pending.clear()
            for s, v in zip(steps, values, strict=True):
                detector.check(s, v)
            latest["loss"] = values[-1]
        return latest["loss"]

    status, error = "completed", None
    last_val = None
    start = time.perf_counter()
    step = -1
    try:
        for step in range(args.max_iters):
            lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
            with autocast_ctx():
                logits = fwd_model(x)
            loss = cross_entropy(logits.float(), y)  # loss in fp32 even under bf16 autocast

            loss.backward()
            gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending.append((step, loss.detach()))

            if meter.should_sync():
                sync(device)
                last_loss = flush_losses()
                rate = meter.step()
                if rate is not None:
                    print(f"step {step:6d} | train {last_loss:.4f} | lr {lr:.2e} | {rate:,.0f} tok/s")
                    log({"step": step, "train_loss": last_loss, "lr": lr, "tokens_per_s": rate})
            else:
                meter.step()

            did_untimed_work = False
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                last_loss = flush_losses()
                last_val = estimate_val_loss(fwd_model, val_data, args, args.eval_batches, autocast_ctx)
                print(f"step {step:6d} | train {last_loss:.4f} | val {last_val:.4f} | lr {lr:.2e}")
                log({"step": step, "train_loss": last_loss, "val_loss": last_val, "lr": lr})
                did_untimed_work = True
            if args.ckpt_path and step > 0 and step % args.ckpt_interval == 0:
                save_checkpoint(model, optimizer, step, args.ckpt_path)
                did_untimed_work = True
            if did_untimed_work:
                sync(device)
                meter.reset_window()

        flush_losses()
        last_val = estimate_val_loss(fwd_model, val_data, args, args.final_eval_batches, autocast_ctx)
        print(f"final val loss ({args.final_eval_batches} batches): {last_val:.4f}")
        if args.ckpt_path:
            save_checkpoint(model, optimizer, args.max_iters, args.ckpt_path)
    except DivergenceError as e:
        status, error = "diverged", str(e)
        print(f"DIVERGENCE DETECTED: {e}", file=sys.stderr)
    finally:
        metrics_file.close()

    wall = time.perf_counter() - start
    summary = {
        "env": env_info(),
        "config": {k: v for k, v in vars(args).items()},
        "status": status,
        "error": error,
        "parameters": n_params,
        "steps_completed": step + 1,
        "tokens_seen": (step + 1) * tokens_per_step,
        "final_val_loss": last_val if status == "completed" else None,
        "last_val_loss": last_val,
        "median_tokens_per_s": meter.median,
        "throughput_windows": len(meter.rates),
        "wall_time_s": wall,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "metrics_file": str(Path(metrics_path).name),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    tps = f"{meter.median:,.0f}" if meter.median else "n/a"
    print(f"median tokens/s: {tps} | wall {wall:.1f}s | wrote {summary_path}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
