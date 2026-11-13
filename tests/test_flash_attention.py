"""FlashAttention-2 (PyTorch-tiled and Triton) forward + backward vs naive attention.

Known bugs are marked xfail rather than hidden (fixes are out of scope for the refactor):
  B1  Triton forward does not mask padded keys, so non-causal attention is wrong when
      N_KEYS % 16 != 0 (the backward then inherits the wrong logsumexp).
  B2  FlashAttentionPytorch.forward ignores is_causal (the backward applies the causal mask),
      so causal outputs and gradients are wrong.
"""

import importlib.util

import pytest
import torch

from llmtk.kernels.flash_attention_torch import FlashAttentionPytorch

TOL = {torch.float32: 1e-2, torch.bfloat16: 2e-2}  # atol = rtol; bf16 has ~3 significant digits
FWD_TILE = 16

HAS_TRITON = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton kernel needs CUDA + triton")
HAS_BF16_MMA = HAS_TRITON and torch.cuda.get_device_capability() >= (8, 0)
requires_bf16 = pytest.mark.skipif(not HAS_BF16_MMA, reason="bf16 tensor cores need sm80+ (e.g. not a T4)")

# (batch, n_queries, n_keys, head_dim). Lengths cover multiples of the forward (16) and backward (32)
# tiles, and lengths divisible by neither.
SQUARE_SHAPES = [(2, 128, 128, 64), (1, 100, 100, 32), (3, 33, 33, 16), (2, 200, 200, 64), (1, 256, 256, 128)]
RECT_SHAPES = [(2, 48, 80, 32), (1, 64, 37, 16)]  # non-causal only


def naive_attention(q, k, v, is_causal):
    """Reference attention in fp32; returns (O, logsumexp L)."""
    q, k, v = q.float(), k.float(), v.float()
    s = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if is_causal:
        nq, nk = s.shape[-2:]
        mask = torch.arange(nq, device=s.device)[:, None] >= torch.arange(nk, device=s.device)[None, :]
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v, torch.logsumexp(s, dim=-1)


def make_inputs(shape, device, dtype=torch.float32, seed=0):
    b, nq, nk, d = shape
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, nq, d, device=device, generator=g).to(dtype)
    k = torch.randn(b, nk, d, device=device, generator=g).to(dtype)
    v = torch.randn(b, nk, d, device=device, generator=g).to(dtype)
    do = torch.randn(b, nq, d, device=device, generator=g).to(dtype)
    return q, k, v, do


def check_against_naive(fn, shape, is_causal, device, dtype=torch.float32):
    """Compare O, saved logsumexp, dQ, dK, dV against fp32 naive attention on the same inputs."""
    tol = TOL[dtype]
    q, k, v, do = make_inputs(shape, device, dtype)
    q1, k1, v1 = (t.clone().requires_grad_() for t in (q, k, v))
    q2, k2, v2 = (t.float().requires_grad_() for t in (q, k, v))

    o = fn(q1, k1, v1, is_causal)
    o_ref, l_ref = naive_attention(q2, k2, v2, is_causal)
    assert o.dtype == dtype

    # The autograd Function must save exactly one (B, Nq) tensor: the logsumexp.
    saved_l = [t for t in o.grad_fn.saved_tensors if t.shape == q.shape[:2]]
    assert len(saved_l) == 1

    o.backward(do)
    o_ref.backward(do.float())

    torch.testing.assert_close(o.float(), o_ref, atol=tol, rtol=tol, msg=lambda m: f"O: {m}")
    torch.testing.assert_close(saved_l[0].float(), l_ref, atol=tol, rtol=tol, msg=lambda m: f"L: {m}")
    for got, ref, name in zip((q1, k1, v1), (q2, k2, v2), "QKV", strict=True):
        assert got.grad.dtype == dtype
        torch.testing.assert_close(got.grad.float(), ref.grad, atol=tol, rtol=tol, msg=lambda m, n=name: f"d{n}: {m}")


def _ids(shape):
    return "b{}-nq{}-nk{}-d{}".format(*shape)


# --- PyTorch tiled (CPU) -----------------------------------------------------------

B2 = pytest.mark.xfail(reason="B2: FlashAttentionPytorch.forward ignores is_causal", strict=True)


@pytest.mark.parametrize("shape", SQUARE_SHAPES[:3] + RECT_SHAPES, ids=_ids)
def test_pytorch_tiled_non_causal(shape):
    check_against_naive(FlashAttentionPytorch.apply, shape, is_causal=False, device="cpu")


@B2
@pytest.mark.parametrize("shape", SQUARE_SHAPES[:3], ids=_ids)
def test_pytorch_tiled_causal(shape):
    check_against_naive(FlashAttentionPytorch.apply, shape, is_causal=True, device="cpu")


# --- Triton (CUDA) -----------------------------------------------------------------


def _triton_apply(*args):
    from llmtk.kernels.flash_attention_triton import FlashAttentionTriton

    return FlashAttentionTriton.apply(*args)


def _triton_case(shape, is_causal, dtype):
    marks = [requires_triton]
    if dtype == torch.bfloat16:
        marks.append(requires_bf16)
    if not is_causal and shape[2] % FWD_TILE != 0:
        # Not run on a GPU yet, so not strict: an unexpected pass is reported, not failed.
        marks.append(pytest.mark.xfail(reason="B1: padded keys are not masked when N_KEYS % 16 != 0", strict=False))
    tag = f"{_ids(shape)}-{'causal' if is_causal else 'full'}-{str(dtype).removeprefix('torch.')}"
    return pytest.param(shape, is_causal, dtype, marks=marks, id=tag)


TRITON_CASES = [
    _triton_case(s, causal, dtype)
    for dtype in (torch.float32, torch.bfloat16)
    for causal, shapes in ((False, SQUARE_SHAPES + RECT_SHAPES), (True, SQUARE_SHAPES))
    for s in shapes
]


@pytest.mark.parametrize(("shape", "is_causal", "dtype"), TRITON_CASES)
def test_triton(shape, is_causal, dtype):
    check_against_naive(_triton_apply, shape, is_causal=is_causal, device="cuda", dtype=dtype)


@requires_triton
@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_does_not_materialize_attention_matrix(is_causal):
    """fwd+bwd extra memory must stay far below a single fp32 (B, N, N) attention matrix."""
    b, n, d = 4, 4096, 64
    q, k, v, do = make_inputs((b, n, n, d), "cuda")
    for t in (q, k, v):
        t.requires_grad_()
    _triton_apply(q, k, v, is_causal).backward(do)  # compile outside the measurement
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _triton_apply(q, k, v, is_causal).backward(do)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    one_attention_matrix = b * n * n * 4  # 256 MiB
    assert extra < one_attention_matrix / 4, f"extra {extra / 2**20:.0f} MiB"
