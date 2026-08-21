import math

import torch

from paged_kv_storage_reference import (
    SequenceState,
    BlockManager,
    KVBlockPool,
    ContiguousKVReference,
    append_token,
    make_kv,
)


def dense_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    H, D = q.shape

    assert k.ndim == 3
    assert v.ndim == 3

    assert k.shape[1:] == (H, D)
    assert v.shape == k.shape

    scale = 1.0 / math.sqrt(D)

    scores = torch.einsum(
        "hd,thd->ht",
        q, k,
    )

    scores = scores * scale

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    out = torch.einsum(
        "ht,thd->hd",
        probs, v,
    )

    return out, scores, probs


def paged_attention_reference(
    q: torch.Tensor,
    pool: KVBlockPool,
    seq: SequenceState,
    layer_idx: int,
):
    H, D = q.shape

    assert H == pool.n_head
    assert D == pool.head_dim
    assert seq.length > 0

    scale = 1.0 / math.sqrt(D)

    score_chunks = []

    block_infos = []

    for logical_block, physical_block in enumerate(
        seq.block_table
    ):
        logical_start = logical_block * pool.block_size

        remaining = seq.length - logical_start

        if remaining <= 0:
            break

        valid_tokens = min(pool.block_size, remaining)

        k_block = pool.k[layer_idx, physical_block, :valid_tokens, :, :]

        score_block = torch.einsum(
            "hd,thd->ht",
            q, k_block,
        )