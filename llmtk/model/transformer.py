"""Pre-norm Transformer block and decoder-only Transformer language model."""

import torch
import torch.nn as nn

from llmtk.model.attention import MultiHeadSelfAttention
from llmtk.model.layers import Embedding, Linear, RMSNorm, RotaryPositionalEmbedding, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm block: x + MHA(RMSNorm(x)), then y + SwiGLU(RMSNorm(y))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=self.rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))


class TransformerLM(nn.Module):
    """Decoder-only LM: embedding -> N TransformerBlocks -> RMSNorm -> untied LM head.

    Module names (token_embeddings, layers.{i}.attn.q_proj, ln_final, lm_head, ...) are kept
    stable so state_dicts stay loadable.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) int64 token ids -> (batch, seq, vocab_size) logits."""
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
