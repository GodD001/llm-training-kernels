"""Train a TransformerLM on pre-tokenized uint16 token files."""

import argparse

import numpy as np
import torch

from llmtk.checkpoint import save_checkpoint
from llmtk.data import get_batch
from llmtk.model import TransformerLM, cross_entropy
from llmtk.optim import AdamW, get_lr_cosine_schedule, gradient_clipping


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)

    # Data and output
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, default="ckpt.pt")

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

    # Loop
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=20)

    return p.parse_args()


def load_data(path: str, dtype: type = np.uint16) -> np.ndarray:
    """Memory-map a flat token file."""
    return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def estimate_val_loss(model: torch.nn.Module, val_data: np.ndarray, args: argparse.Namespace) -> float:
    """Mean loss over `args.eval_batches` random validation batches."""
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(val_data, args.batch_size, args.context_length, args.device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = get_args()
    torch.manual_seed(0)

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
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    for step in range(args.max_iters):
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args)
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | lr {lr:.2e}")

        if step % args.ckpt_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.ckpt_path)

    save_checkpoint(model, optimizer, args.max_iters, args.ckpt_path)


if __name__ == "__main__":
    main()
