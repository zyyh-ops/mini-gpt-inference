"""Experimental FP32 Triton fused *decode attention only*.

Inputs match minivllm.paged_attention_batch_online_v1 exactly:
    q:               [B, H, D] (arbitrary strides)
    pool.k / pool.v:  [L, N_phys, S, H, D]
    block_tables:    [B, M] (logical -> physical; -1 only in unused tail)
    context_lengths: [B] (strictly positive, include just-written decode KV)

One Triton CTA/program per (request, head), looping logical KV blocks and
maintaining numerically stable online softmax statistics in FP32.

Limitations: experimental correctness-first implementation, no GQA, only
FP32 and CUDA, no backward, not a production-optimized warp layout.  Does
NOT perform KV writes and does NOT modify ModelRunner/Scheduler.
"""
from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _paged_decode_f32(
        Q, K, V, BLOCK_TABLE, CONTEXT_LENGTHS, OUT,
        Q_STRIDE_B: tl.constexpr, Q_STRIDE_H: tl.constexpr, Q_STRIDE_D: tl.constexpr,
        K_STRIDE_L: tl.constexpr, K_STRIDE_P: tl.constexpr,
        K_STRIDE_S: tl.constexpr, K_STRIDE_H: tl.constexpr, K_STRIDE_D: tl.constexpr,
        V_STRIDE_L: tl.constexpr, V_STRIDE_P: tl.constexpr,
        V_STRIDE_S: tl.constexpr, V_STRIDE_H: tl.constexpr, V_STRIDE_D: tl.constexpr,
        TABLE_STRIDE_B: tl.constexpr, TABLE_STRIDE_M: tl.constexpr,
        LEN_STRIDE: tl.constexpr,
        OUT_STRIDE_B: tl.constexpr, OUT_STRIDE_H: tl.constexpr, OUT_STRIDE_D: tl.constexpr,
        LAYER_IDX, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        D_PAD: tl.constexpr, SM_SCALE: tl.constexpr,
    ):
        request = tl.program_id(0)
        head = tl.program_id(1)
        d = tl.arange(0, D_PAD)
        s = tl.arange(0, BLOCK_SIZE)
        q = tl.load(
            Q + request * Q_STRIDE_B + head * Q_STRIDE_H + d * Q_STRIDE_D,
            mask=d < D,
            other=0.0,
        ).to(tl.float32)
        length = tl.load(CONTEXT_LENGTHS + request * LEN_STRIDE)

        running_m = -float('inf')
        running_l = 0.0
        running_acc = tl.full((D_PAD,), 0.0, tl.float32)

        # Only visit valid logical blocks for this particular request.  In
        # particular, never load a -1 padding entry from another request.
        for logical in range(tl.cdiv(length, BLOCK_SIZE)):
            physical = tl.load(
                BLOCK_TABLE + request * TABLE_STRIDE_B + logical * TABLE_STRIDE_M
            )
            valid = (logical * BLOCK_SIZE + s) < length
            # Last block may be partially populated; unused KV is uninitialized.
            # Mask loads BEFORE arithmetic (0 * NaN would remain NaN).
            k = tl.load(
                K + LAYER_IDX * K_STRIDE_L + physical * K_STRIDE_P
                  + s[:, None] * K_STRIDE_S + head * K_STRIDE_H
                  + d[None, :] * K_STRIDE_D,
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + LAYER_IDX * V_STRIDE_L + physical * V_STRIDE_P
                  + s[:, None] * V_STRIDE_S + head * V_STRIDE_H
                  + d[None, :] * V_STRIDE_D,
                mask=valid[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
            score = tl.where(valid, score, -float('inf'))
            block_m = tl.max(score, axis=0)
            new_m = tl.maximum(running_m, block_m)
            alpha = tl.exp(running_m - new_m)
            prob = tl.exp(score - new_m)
            new_l = running_l * alpha + tl.sum(prob, axis=0)
            running_acc = running_acc * alpha + tl.sum(prob[:, None] * v, axis=0)
            running_m = new_m
            running_l = new_l

        answer = running_acc / running_l
        tl.store(
            OUT + request * OUT_STRIDE_B + head * OUT_STRIDE_H + d * OUT_STRIDE_D,
            answer,
            mask=d < D,
        )


def paged_attention_batch_triton_v0(q, pool, block_tables, context_lengths, layer_idx):
    """Run one fused FP32 decode-attention launch; caller already wrote new KV.

    Contract excludes zero-length requests and requires valid block table entries
    for all logical blocks touched by context_lengths (padding only in tail).
    No device-data-dependent validation on hot path: avoid .item() GPU sync.
    """
    if triton is None:
        raise RuntimeError('Triton is not installed; use a CUDA PyTorch environment')
    if not q.is_cuda:
        raise ValueError('CUDA q required for Triton v0')
    if q.ndim != 3:
        raise ValueError('q must have shape [B,H,D]')
    b, h, d = q.shape
    if b == 0 or h == 0 or d == 0:
        raise ValueError('q dimensions must be nonzero')
    if d > 128:
        raise ValueError('v0 supports head_dim <= 128')
    s = pool.block_size
    if s <= 0 or s & (s - 1):
        raise ValueError('v0 requires power-of-two block_size')
    if s > 64:
        raise ValueError('v0 currently supports block_size <= 64')
    if pool.k.dtype != torch.float32 or pool.v.dtype != torch.float32 or q.dtype != torch.float32:
        raise ValueError('v0 supports float32 only')
    if pool.k.ndim != 5 or pool.v.shape != pool.k.shape:
        raise ValueError('KV tensors must have matching [L,P,S,H,D] shape')
    if pool.k.shape[2:] != (s, h, d):
        raise ValueError('KV head dimensions or block size do not match q')
    if block_tables.ndim != 2 or block_tables.shape[0] != b:
        raise ValueError('block_tables must have shape [B,M]')
    if context_lengths.shape != (b,):
        raise ValueError('context_lengths must have shape [B]')
    if block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError('block_tables must contain integer indices')
    if context_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError('context_lengths must be integer')
    if not (0 <= layer_idx < pool.k.shape[0]):
        raise ValueError('layer_idx outside KV pool')
    if any(x.device != q.device for x in (pool.k, pool.v, block_tables, context_lengths)):
        raise ValueError('all inputs must reside on same CUDA device')
    if block_tables.shape[1] < 1:
        raise ValueError('block_tables need at least one block column')
    # Valid context_lengths, physical ids and populated KV slots are an input
    # invariant, matching existing paged_attention_batch_online_v1.
    out = torch.empty((b, h, d), dtype=q.dtype, device=q.device)
    ks, vs = pool.k.stride(), pool.v.stride()
    _paged_decode_f32[(b, h)](
        q, pool.k, pool.v, block_tables, context_lengths, out,
        *q.stride(), *ks, *vs, *block_tables.stride(), context_lengths.stride(0),
        *out.stride(), layer_idx, d, s, triton.next_power_of_2(d), 1.0 / math.sqrt(d),
        num_warps=4,
    )
    return out
