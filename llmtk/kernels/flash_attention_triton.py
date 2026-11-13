"""FlashAttention-2 in Triton: fused forward kernel plus two backward kernels (dK/dV and dQ).

Neither pass materializes the N x N attention matrix: the forward saves only O and the logsumexp L,
and the backward recomputes S and P tile by tile from L.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):  # fmt: skip
    """One program = one query tile of one batch element; streams over all K/V tiles."""
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Block pointers, each already offset to this batch element.
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Accumulators in fp32.
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] >= k_idx[None, :]
            S = tl.where(mask, S, S - 1e6)
        m_new = tl.maximum(tl.max(S, axis=-1), mi)
        P = tl.exp(S - m_new[:, None])
        alpha = tl.exp(mi - m_new)
        li = li * alpha + tl.sum(P, axis=-1)
        Oi = Oi * alpha[:, None] + tl.dot(P.to(Vj.dtype), Vj)
        mi = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Normalize and store O and the logsumexp L.
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


# Backward tile sizes. 32x32 tiles with d <= 128 keep each program's working set small enough for
# any recent NVIDIA GPU (including 8 GB consumer cards); both are tl.constexpr in the kernels.
BWD_Q_TILE_SIZE = 32
BWD_K_TILE_SIZE = 32


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qn,
    stride_kb, stride_kn,
    stride_lb,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):  # fmt: skip
    """One program = one K/V tile of one batch element; loops over Q tiles to accumulate dK, dV.

    Q, O, dO share strides (stride_qb, stride_qn); K, V, dK, dV share (stride_kb, stride_kn);
    the last dim is contiguous. L and Delta are (B, N_QUERIES) fp32.
    """
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_start = key_tile_index * K_TILE_SIZE
    offs_k = k_start + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)
    k_valid = offs_k < N_KEYS

    kv_offsets = batch_index * stride_kb + offs_k[:, None] * stride_kn + offs_d[None, :]
    Kj = tl.load(K_ptr + kv_offsets, mask=k_valid[:, None], other=0.0)
    Vj = tl.load(V_ptr + kv_offsets, mask=k_valid[:, None], other=0.0)

    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # Causal: queries before k_start can't see any key in this tile, so start at the Q tile that
    # contains query index k_start.
    q_lo = 0
    if is_causal:
        q_lo = (k_start // Q_TILE_SIZE) * Q_TILE_SIZE

    for q_start in range(q_lo, N_QUERIES, Q_TILE_SIZE):
        offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
        q_valid = offs_q < N_QUERIES
        q_offsets = batch_index * stride_qb + offs_q[:, None] * stride_qn + offs_d[None, :]
        Qi = tl.load(Q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
        dOi = tl.load(dO_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
        Li = tl.load(L_ptr + batch_index * stride_lb + offs_q, mask=q_valid, other=0.0)
        Di = tl.load(Delta_ptr + batch_index * stride_lb + offs_q, mask=q_valid, other=0.0)

        # Recompute S and P on-chip from the saved logsumexp; padded rows/cols and (if causal)
        # future keys get P = 0, so they contribute nothing to any gradient.
        S = tl.dot(Qi, tl.trans(Kj)) * scale
        valid = q_valid[:, None] & k_valid[None, :]
        if is_causal:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])
        P = tl.where(valid, tl.exp(S - Li[:, None]), 0.0)

        dV += tl.dot(tl.trans(P.to(dOi.dtype)), dOi)
        dP = tl.dot(dOi, tl.trans(Vj))
        dS = P * (dP - Di[:, None])
        dK += tl.dot(tl.trans(dS.to(Qi.dtype)), Qi)

    dK = dK * scale
    tl.store(dK_ptr + kv_offsets, dK.to(dK_ptr.dtype.element_ty), mask=k_valid[:, None])
    tl.store(dV_ptr + kv_offsets, dV.to(dV_ptr.dtype.element_ty), mask=k_valid[:, None])


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr, dQ_ptr,
    stride_qb, stride_qn,
    stride_kb, stride_kn,
    stride_lb,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):  # fmt: skip
    """One program = one Q tile of one batch element; loops over K/V tiles to accumulate dQ (no atomics)."""
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE
    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
    offs_d = tl.arange(0, D)
    q_valid = offs_q < N_QUERIES

    q_offsets = batch_index * stride_qb + offs_q[:, None] * stride_qn + offs_d[None, :]
    Qi = tl.load(Q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    dOi = tl.load(dO_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    Li = tl.load(L_ptr + batch_index * stride_lb + offs_q, mask=q_valid, other=0.0)
    Di = tl.load(Delta_ptr + batch_index * stride_lb + offs_q, mask=q_valid, other=0.0)

    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Causal: keys after the last query of this tile are fully masked, so stop there.
    k_hi = N_KEYS
    if is_causal:
        k_hi = tl.minimum(N_KEYS, q_start + Q_TILE_SIZE)

    for k_start in range(0, k_hi, K_TILE_SIZE):
        offs_k = k_start + tl.arange(0, K_TILE_SIZE)
        k_valid = offs_k < N_KEYS
        kv_offsets = batch_index * stride_kb + offs_k[:, None] * stride_kn + offs_d[None, :]
        Kj = tl.load(K_ptr + kv_offsets, mask=k_valid[:, None], other=0.0)
        Vj = tl.load(V_ptr + kv_offsets, mask=k_valid[:, None], other=0.0)

        S = tl.dot(Qi, tl.trans(Kj)) * scale
        valid = q_valid[:, None] & k_valid[None, :]
        if is_causal:
            valid = valid & (offs_q[:, None] >= offs_k[None, :])
        P = tl.where(valid, tl.exp(S - Li[:, None]), 0.0)

        dP = tl.dot(dOi, tl.trans(Vj))
        dS = P * (dP - Di[:, None])
        dQ += tl.dot(dS.to(Kj.dtype), Kj)

    dQ = dQ * scale
    tl.store(dQ_ptr + q_offsets, dQ.to(dQ_ptr.dtype.element_ty), mask=q_valid[:, None])


def flash_backward_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    L: torch.Tensor,
    dO: torch.Tensor,
    is_causal: bool,
    q_tile_size: int = BWD_Q_TILE_SIZE,
    k_tile_size: int = BWD_K_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-2 backward in Triton; memory is O(N * d), the N x N matrices never leave the chip."""
    Q, K, V, O, dO = (t.contiguous() for t in (Q, K, V, O, dO))
    L = L.contiguous()
    B, Nq, d = Q.shape
    Nk = K.shape[1]
    scale = 1.0 / math.sqrt(d)

    # Delta_i = rowsum(dO_i * O_i): an O(N d) elementwise op, cheap enough to leave to PyTorch.
    Delta = (dO.float() * O.float()).sum(dim=-1).contiguous()

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    common = dict(
        stride_qb=Q.stride(0),
        stride_qn=Q.stride(1),
        stride_kb=K.stride(0),
        stride_kn=K.stride(1),
        stride_lb=L.stride(0),
        N_QUERIES=Nq,
        N_KEYS=Nk,
        scale=scale,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
        is_causal=is_causal,
    )
    flash_bwd_dkdv_kernel[(triton.cdiv(Nk, k_tile_size), B)](Q, K, V, dO, L, Delta, dK, dV, **common)
    flash_bwd_dq_kernel[(triton.cdiv(Nq, q_tile_size), B)](Q, K, V, dO, L, Delta, dQ, **common)
    return dQ, dK, dV


class FlashAttentionTriton(torch.autograd.Function):
    """Autograd wrapper: Triton forward (16x16 tiles) and Triton backward (32x32 tiles).

    Input shapes: Q (B, Nq, d), K/V (B, Nk, d), CUDA tensors.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        Bq, Bk = 16, 16

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(Nq, Bq), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )  # fmt: skip

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_triton(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None
