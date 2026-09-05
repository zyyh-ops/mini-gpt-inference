import math

import torch

from sequence import SequenceState
from block_manager import BlockManager
from kv_block_pool import KVBlockPool


def paged_attention_online(
    q: torch.Tensor,
    pool: KVBlockPool,
    seq: SequenceState,
    layer_idx: int,
    context_length: int,
):
    import math

    H, D = q.shape

    scale = 1.0 / math.sqrt(D)

    m = torch.full(
        (H, ),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )

    l = torch.zeros(
        H,
        device=q.device,
        dtype=q.dtype,
    )

    O = torch.zeros_like(
        q,
        device=q.device,
        dtype=q.dtype,
    )

    for logical_block, physical_block in enumerate(seq.block_table):
        logic_start = logical_block * pool.block_size

        remaining = context_length - logic_start

        valid_tokens = min(pool.block_size, remaining)

        if valid_tokens <= 0:
            break

        k_block = pool.k[layer_idx, physical_block, :valid_tokens]

        scores = torch.einsum(
            "hd,thd->ht",
            q, k_block,
        )

        scores *= scale

        block_max = scores.max(dim=-1).values

        m_new = torch.maximum(m, block_max)

        alpha = torch.exp(m - m_new)

        beta = torch.exp(scores - m_new[:, None])

        l_new = (alpha * l + beta.sum(dim=-1))

        v_block = pool.v[layer_idx, physical_block, :valid_tokens]

        block_O = torch.einsum(
            "ht,thd->hd",
            beta, v_block,
        )

        O = (alpha[:,None] * l[:,None] * O + block_O) / l_new[:,None]

        m = m_new
        l = l_new

    return O