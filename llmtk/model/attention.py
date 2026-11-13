"""Causal multi-head self-attention."""

import torch
from einops import rearrange

from llmtk.model.functional import scaled_dot_product_attention
from llmtk.model.layers import Linear, RotaryPositionalEmbedding


class MultiHeadSelfAttention(torch.nn.Module):
    """Causal multi-head self-attention with optional RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        # One Linear computes all heads at once: d_model -> num_heads * d_k.
        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads and move the head dim in front of seq so it acts as a batch dim.
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # RoPE rotates Q and K only (not V); every head uses the same rotation.
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attn_output = rearrange(attn_output, "... h seq d -> ... seq (h d)")
        return self.output_proj(attn_output)
