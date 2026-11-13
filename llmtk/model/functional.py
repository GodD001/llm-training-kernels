"""Stateless ops: softmax, scaled dot-product attention, cross-entropy."""

import math

import torch
from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along `dim`."""
    # Subtracting the max leaves the result unchanged but keeps exp() from overflowing.
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d_k)) V; `mask` is boolean with True = may attend.

    Shapes: Q (..., n_q, d_k), K (..., n_k, d_k), V (..., n_k, d_v) -> (..., n_q, d_v).
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over all positions; logits (..., vocab), targets (...)."""
    m = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    target_logit = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -target_logit + log_sum_exp
    return loss.mean()
