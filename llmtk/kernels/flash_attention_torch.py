"""FlashAttention-2 in plain PyTorch: tiled forward + torch.compile'd backward (reference version)."""

import math

import torch


class FlashAttentionPytorch(torch.autograd.Function):
    """Tiled online-softmax forward (Bq = Bk = 16); saves O and logsumexp L for the backward.

    Input shapes: Q (B, Nq, d), K/V (B, Nk, d).
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        Bq, Bk = 16, 16
        scale = 1.0 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        for i in range(0, Nq, Bq):
            Qi = Q[:, i : i + Bq, :]
            nq = Qi.shape[1]
            Oi = torch.zeros(B, nq, d, device=Q.device, dtype=torch.float32)
            li = torch.zeros(B, nq, device=Q.device, dtype=torch.float32)
            mi = torch.full((B, nq), float("-inf"), device=Q.device, dtype=torch.float32)

            for j in range(0, Nk, Bk):
                Kj = K[:, j : j + Bk, :]
                Vj = V[:, j : j + Bk, :]

                S = Qi @ Kj.transpose(-2, -1) * scale
                m_new = torch.maximum(mi, S.max(dim=-1).values)
                P = torch.exp(S - m_new[..., None])
                alpha = torch.exp(mi - m_new)
                li = li * alpha + P.sum(dim=-1)
                Oi = Oi * alpha[..., None] + P @ Vj
                mi = m_new

            O[:, i : i + Bq, :] = Oi / li[..., None]
            L[:, i : i + Bq] = mi + torch.log(li)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])
        dQ, dK, dV = _flash_backward(Q, K, V, O, L, dO, ctx.is_causal, scale)
        return dQ, dK, dV, None  # matches forward's (Q, K, V, is_causal)


def _flash_backward_core(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    L: torch.Tensor,
    dO: torch.Tensor,
    is_causal: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-2 backward from saved L (recomputes S and P; materializes full B x Nq x Nk).

    Shapes: Q (B,Nq,d), K/V (B,Nk,d), O/dO (B,Nq,d), L (B,Nq) -> dQ (B,Nq,d), dK/dV (B,Nk,d).
    """
    D = (O * dO).sum(dim=-1)  # (B, Nq)
    S = Q @ K.transpose(-2, -1) * scale

    if is_causal:
        # Mask the upper triangle; otherwise P is non-zero at masked positions and dV/dK are wrong.
        Nq, Nk = Q.shape[-2], K.shape[-2]
        q_idx = torch.arange(Nq, device=Q.device)
        k_idx = torch.arange(Nk, device=Q.device)
        S = torch.where(q_idx[:, None] >= k_idx[None, :], S, S - 1e6)

    P = torch.exp(S - L[..., None])
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D[..., None])
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV


_flash_backward = torch.compile(_flash_backward_core)
