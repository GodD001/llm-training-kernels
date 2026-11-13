"""Basic Transformer building blocks: Linear, Embedding, RMSNorm, SwiGLU, RoPE."""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    """Bias-free linear layer, y = x W^T, with truncated-normal init."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm; computed in fp32 and cast back to the input dtype."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W2(SiLU(W1 x) * W3 x)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            # 8/3 * d_model, rounded to a multiple of 64 for hardware efficiency.
            d_ff = round((8 / 3) * d_model / 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.w1(x)
        silu_out = s * torch.sigmoid(s)
        gate = self.w3(x)
        return self.w2(gate * silu_out)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding over adjacent dimension pairs, with precomputed cos/sin tables."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = einsum(positions, inv_freq, "i, k -> i k")  # (max_seq_len, d_k/2)
        # Buffers, not parameters; non-persistent so they are not saved in the state_dict.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k); token_positions: (..., seq_len)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # Pairs are adjacent dims: (x0, x1), (x2, x3), ...
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        out = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rearrange(out, "... d two -> ... (d two)")
