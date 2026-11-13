"""Model components vs straightforward PyTorch references on random inputs."""

import pytest
import torch
import torch.nn.functional as F

from llmtk.generate import decode
from llmtk.model import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    cross_entropy,
    scaled_dot_product_attention,
    softmax,
)

ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def rope_reference(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """RoPE via complex multiplication: pair (x[2k], x[2k+1]) rotated by pos * theta^(-2k/d)."""
    d = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angles = positions.to(torch.float64)[..., None] * inv_freq  # (..., seq, d/2)
    xc = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], d // 2, 2).contiguous())
    out = torch.view_as_real(xc * torch.polar(torch.ones_like(angles), angles))
    return out.reshape(x.shape).to(x.dtype)


# --- single layers ----------------------------------------------------------


def test_linear_matches_matmul():
    lin = Linear(16, 24)
    x = torch.randn(2, 5, 16)
    assert lin(x).shape == (2, 5, 24)
    torch.testing.assert_close(lin(x), x @ lin.W.T, atol=ATOL, rtol=0)


def test_embedding_matches_f_embedding():
    emb = Embedding(50, 8)
    ids = torch.randint(0, 50, (3, 7))
    assert emb(ids).shape == (3, 7, 8)
    torch.testing.assert_close(emb(ids), F.embedding(ids, emb.weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_reference(dtype):
    norm = RMSNorm(32)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    x = (torch.randn(4, 6, 32) * 3).to(dtype)
    xf = x.float()
    ref = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    out = norm(x)
    assert out.dtype == dtype
    atol = 1e-2 if dtype == torch.bfloat16 else ATOL
    torch.testing.assert_close(out.float(), ref.to(dtype).float(), atol=atol, rtol=0)


def test_swiglu_matches_reference():
    ffn = SwiGLU(32, 64)
    x = torch.randn(2, 5, 32)
    ref = (F.silu(x @ ffn.w1.W.T) * (x @ ffn.w3.W.T)) @ ffn.w2.W.T
    assert ffn(x).shape == x.shape
    torch.testing.assert_close(ffn(x), ref, atol=ATOL, rtol=0)


def test_swiglu_default_dff_is_multiple_of_64():
    ffn = SwiGLU(512)
    assert ffn.w1.W.shape == (1344, 512)


def test_rope_matches_complex_reference():
    d, seq = 16, 12
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d, max_seq_len=64)
    x = torch.randn(2, 3, seq, d)
    pos = torch.arange(seq)
    torch.testing.assert_close(rope(x, pos), rope_reference(x, pos, 10000.0), atol=ATOL, rtol=0)
    # arbitrary (non-contiguous) positions
    pos2 = torch.tensor([5, 0, 63, 7, 7, 1, 2, 3, 40, 41, 10, 11])
    torch.testing.assert_close(rope(x, pos2), rope_reference(x, pos2, 10000.0), atol=ATOL, rtol=0)


def test_rope_preserves_norm_and_relative_position():
    d = 8
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d, max_seq_len=128)
    q, k = torch.randn(d), torch.randn(d)
    rq = lambda v, p: rope(v[None], torch.tensor([p]))[0]  # noqa: E731
    torch.testing.assert_close(rq(q, 9).norm(), q.norm())
    # <R_m q, R_n k> depends only on m - n
    torch.testing.assert_close(rq(q, 10) @ rq(k, 3), rq(q, 50) @ rq(k, 43), atol=1e-4, rtol=0)


# --- functional ---------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, -1, 1])
def test_softmax_matches_torch(dim):
    x = torch.randn(4, 5, 6)
    torch.testing.assert_close(softmax(x, dim), torch.softmax(x, dim), atol=ATOL, rtol=0)


def test_softmax_is_stable_for_large_inputs():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
    out = softmax(x, -1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, -1))


@pytest.mark.parametrize("use_mask", [False, True])
def test_sdpa_matches_torch(use_mask):
    q, k, v = torch.randn(2, 3, 10, 8), torch.randn(2, 3, 12, 8), torch.randn(2, 3, 12, 4)
    mask = torch.rand(10, 12) > 0.3 if use_mask else None
    if mask is not None:
        mask[:, 0] = True  # keep every row non-empty
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v, mask), ref, atol=ATOL, rtol=0)


def test_cross_entropy_matches_torch():
    logits = torch.randn(3, 7, 20) * 5
    targets = torch.randint(0, 20, (3, 7))
    ref = F.cross_entropy(logits.reshape(-1, 20), targets.reshape(-1))
    torch.testing.assert_close(cross_entropy(logits, targets), ref, atol=ATOL, rtol=0)


def test_cross_entropy_is_stable_for_large_logits():
    logits = torch.tensor([[1e4, 0.0, -1e4]])
    assert torch.isfinite(cross_entropy(logits, torch.tensor([2])))


# --- attention / blocks / LM ---------------------------------------------------


def mha_reference(mha: MultiHeadSelfAttention, x: torch.Tensor, theta: float | None) -> torch.Tensor:
    b, n, _ = x.shape
    h = mha.num_heads

    def split(t):
        return t.view(b, n, h, -1).transpose(1, 2)

    q, k, v = split(x @ mha.q_proj.W.T), split(x @ mha.k_proj.W.T), split(x @ mha.v_proj.W.T)
    if theta is not None:
        pos = torch.arange(n)
        q, k = rope_reference(q, pos, theta), rope_reference(k, pos, theta)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(b, n, -1) @ mha.output_proj.W.T


@pytest.mark.parametrize("with_rope", [False, True])
def test_mha_matches_reference(with_rope):
    d_model, heads, n = 32, 4, 9
    rope = RotaryPositionalEmbedding(10000.0, d_model // heads, 32) if with_rope else None
    mha = MultiHeadSelfAttention(d_model, heads, rope=rope)
    x = torch.randn(2, n, d_model)
    out = mha(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, mha_reference(mha, x, 10000.0 if with_rope else None), atol=1e-4, rtol=0)


def test_transformer_block_shape():
    block = TransformerBlock(d_model=32, num_heads=4, d_ff=64, max_seq_len=16, theta=10000.0)
    x = torch.randn(2, 10, 32)
    assert block(x).shape == x.shape


def small_lm() -> TransformerLM:
    return TransformerLM(vocab_size=50, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64, theta=1e4)


def test_lm_shape_and_causality():
    lm = small_lm()
    ids = torch.randint(0, 50, (2, 12))
    logits = lm(ids)
    assert logits.shape == (2, 12, 50)
    ids2 = ids.clone()
    ids2[:, 8:] = (ids2[:, 8:] + 1) % 50  # change only future tokens
    torch.testing.assert_close(lm(ids2)[:, :8], logits[:, :8])


def test_lm_param_count_tinystories_22m():
    lm = TransformerLM(
        vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, theta=10000.0
    )
    assert sum(p.numel() for p in lm.parameters()) == 22_696_448


# --- generation -----------------------------------------------------------------


def test_decode_respects_max_new_tokens_and_vocab():
    lm = small_lm()
    out = decode(lm, torch.tensor([1, 2, 3]), max_new_tokens=5, temperature=0.8, top_p=0.9)
    assert out.shape == (5,) and int(out.min()) >= 0 and int(out.max()) < 50


def test_decode_tiny_top_p_is_greedy():
    lm = small_lm()
    prompt = torch.tensor([4, 5, 6])
    out = decode(lm, prompt, max_new_tokens=4, top_p=1e-6)
    ids = prompt[None]
    for tok in out:
        assert int(tok) == int(lm(ids)[0, -1].argmax())
        ids = torch.cat([ids, tok.view(1, 1)], dim=1)


def test_decode_stops_at_eos():
    lm = small_lm()
    first = int(decode(lm, torch.tensor([7]), max_new_tokens=1, top_p=1e-6)[0])
    out = decode(lm, torch.tensor([7]), max_new_tokens=10, top_p=1e-6, eos_token_id=first)
    assert out.tolist() == [first]


def test_decode_rejects_bad_args():
    with pytest.raises(ValueError):
        decode(small_lm(), torch.tensor([1]), 1, temperature=0)
    with pytest.raises(ValueError):
        decode(small_lm(), torch.tensor([1]), 1, top_p=1.5)
