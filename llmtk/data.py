"""Random-window batch sampling from a 1-D token array."""

import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows; returns (inputs, next-token targets), each (B, T)."""
    # Largest valid start is len(x) - context_length - 1; randint's upper bound is exclusive.
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)  # (B, 1) + (T,) -> (B, T)

    inputs = torch.tensor(x[idx], dtype=torch.long, device=device)
    targets = torch.tensor(x[idx + 1], dtype=torch.long, device=device)
    return inputs, targets
