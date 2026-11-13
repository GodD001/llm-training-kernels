"""AdamW, cosine LR schedule, gradient clipping, and the training-loop utilities."""

import math

import numpy as np
import pytest
import torch

from llmtk.checkpoint import load_checkpoint, save_checkpoint
from llmtk.data import get_batch
from llmtk.optim import AdamW, get_lr_cosine_schedule, gradient_clipping


def toy_problem(seed: int = 0):
    """Small least-squares regression with a fixed dataset."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(64, 10, generator=g)
    y = X @ torch.randn(10, 1, generator=g) + 0.1 * torch.randn(64, 1, generator=g)
    W = torch.randn(10, 1, generator=g)
    return X, y, W


def run(opt_cls, steps: int = 100, **kw) -> tuple[torch.Tensor, list[float]]:
    X, y, W0 = toy_problem()
    W = torch.nn.Parameter(W0.clone())
    opt = opt_cls([W], **kw)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((X @ W - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return W.detach(), losses


# --- AdamW ----------------------------------------------------------------------


def test_adamw_matches_torch_without_weight_decay():
    # With wd=0 the only difference from torch is where eps enters the bias correction (eps vs
    # eps*sqrt(1-beta2^t)), which is negligible here.
    kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ours, _ = run(AdamW, **kw)
    ref, _ = run(torch.optim.AdamW, **kw)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_adamw_close_to_torch_with_weight_decay():
    # torch decays before the Adam step; ours decays after (handout order). Difference is O(lr^2 * wd).
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ours, ours_losses = run(AdamW, **kw)
    ref, ref_losses = run(torch.optim.AdamW, **kw)
    torch.testing.assert_close(ours, ref, atol=1e-3, rtol=1e-3)
    assert ours_losses[-1] < 0.1 * ours_losses[0]


def test_adamw_matches_explicit_reference():
    """Exact match against a straightforward re-statement of the algorithm."""
    lr, (b1, b2), eps, wd = 1e-2, (0.9, 0.99), 1e-8, 0.05
    ours, _ = run(AdamW, steps=20, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    X, y, W = toy_problem()
    W = W.clone().double()
    X, y = X.double(), y.double()
    m = torch.zeros_like(W)
    v = torch.zeros_like(W)
    for t in range(1, 21):
        grad = 2 * X.T @ (X @ W - y) / X.shape[0]
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        alpha = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
        W = W - alpha * m / (v.sqrt() + eps)
        W = W - lr * wd * W
    torch.testing.assert_close(ours, W.float(), atol=1e-5, rtol=1e-5)


def test_adamw_skips_params_without_grad_and_rejects_negative_lr():
    p = torch.nn.Parameter(torch.ones(3))
    AdamW([p], lr=0.1).step()
    assert torch.equal(p, torch.ones(3))
    with pytest.raises(ValueError):
        AdamW([p], lr=-1.0)


# --- cosine schedule ----------------------------------------------------------


SCHED = dict(max_learning_rate=1.0, min_learning_rate=0.1, warmup_iters=10, cosine_cycle_iters=110)


def lr(it: int) -> float:
    return get_lr_cosine_schedule(it, **SCHED)


def test_schedule_boundaries():
    assert lr(0) == 0.0
    assert lr(5) == pytest.approx(0.5)
    assert lr(10) == pytest.approx(1.0)  # end of warmup = max
    assert lr(60) == pytest.approx(0.55)  # halfway through cosine = (max + min) / 2
    assert lr(110) == pytest.approx(0.1)  # end of cosine = min
    assert lr(111) == 0.1 and lr(10_000) == 0.1


def test_schedule_is_monotone():
    warm = [lr(t) for t in range(0, 11)]
    cos = [lr(t) for t in range(10, 111)]
    assert all(a < b for a, b in zip(warm, warm[1:], strict=False))
    assert all(a >= b for a, b in zip(cos, cos[1:], strict=False))


# --- gradient clipping ----------------------------------------------------------


def _params_with_grads(scale: float):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(4)), torch.nn.Parameter(torch.randn(2))]
    for p in ps[:2]:
        p.grad = torch.randn_like(p) * scale
    return ps  # ps[2].grad is None


def test_clipping_matches_torch():
    ours, ref = _params_with_grads(10.0), _params_with_grads(10.0)
    gradient_clipping(ours, 1.0)
    torch.nn.utils.clip_grad_norm_([p for p in ref if p.grad is not None], 1.0)  # torch uses eps=1e-6 too
    for a, b in zip(ours[:2], ref[:2], strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    total = torch.sqrt(sum((p.grad**2).sum() for p in ours[:2]))
    assert total == pytest.approx(1.0, abs=1e-4)


def test_clipping_leaves_small_gradients_alone():
    ps = _params_with_grads(1e-3)
    before = [p.grad.clone() for p in ps[:2]]
    gradient_clipping(ps, 1.0)
    assert all(torch.equal(a, p.grad) for a, p in zip(before, ps[:2], strict=True))


# --- data + checkpoint ------------------------------------------------------------


def test_get_batch_targets_are_shifted_inputs():
    data = np.arange(100, dtype=np.uint16)
    x, y = get_batch(data, batch_size=16, context_length=8, device="cpu")
    assert x.shape == y.shape == (16, 8) and x.dtype == torch.long
    assert torch.equal(y, x + 1)
    assert int(y.max()) <= 99


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    opt = AdamW(model.parameters(), lr=1e-2)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    save_checkpoint(model, opt, 7, tmp_path / "ck.pt")

    model2 = torch.nn.Linear(4, 2)
    opt2 = AdamW(model2.parameters(), lr=1e-2)
    assert load_checkpoint(tmp_path / "ck.pt", model2, opt2) == 7
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all(torch.equal(s1[k]["m"], s2[k]["m"]) for k in s1)
