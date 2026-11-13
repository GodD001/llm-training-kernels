"""Global gradient-norm clipping."""

from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Scale all gradients in place so their combined L2 norm is at most `max_l2_norm`."""
    # Parameters that did not take part in the forward pass have grad None.
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # Treat all gradients as one long vector.
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
