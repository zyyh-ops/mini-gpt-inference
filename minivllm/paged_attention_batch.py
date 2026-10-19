import math
import torch
from kv_block_pool import KVBlockPool
from sequence import SequenceState


def paged_attention_batch_reference(
    q: torch.Tensor,
    pool: KVBlockPool,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    layer_idx: int,
):
    B, H, D = q.shape

    scale = 1.0 / math.sqrt(D)

    outputs = []

    for b in range(B):
        context_length = int(context_lengths[b].item())

        assert context_length > 0

        m = torch.full(
            (H,),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )

        l = torch.zeros(
            H,
            device=q.device,
            dtype=q.dtype,
        )

        O = torch.zeros(
            H, D,
            device=q.device,
            dtype=q.dtype,
        )

        num_blocks = (
            context_length
            + pool.block_size - 1
        ) // pool.block_size

        for logical_block in range(num_blocks):
            physical_block = int(
                block_tables[
                    b,
                    logical_block,
                ].item()
            )

            assert physical_block >= 0

            logical_start = (
                logical_block * pool.block_size
            )

            remaining = context_length - logical_start

            valid_tokens = min(pool.block_size, remaining)

            k_block = pool.k[
                layer_idx,
                physical_block,
                :valid_tokens,
            ]

            v_block = pool.v[
                layer_idx,
                physical_block,
                :valid_tokens,
            ]

            scores = torch.einsum(
                "hd,thd->ht",
                q[b],
                k_block,
            )

            scores *= scale

            block_max = scores.max(dim=-1).values

            m_new = torch.maximum(m, block_max)

            alpha = torch.exp(m - m_new)

            beta = torch.exp(scores - m_new[:, None])

            l_new = alpha * l + beta.sum(dim=-1)

            block_O = torch.einsum(
                "ht,thd->hd",
                beta,
                v_block,
            )

            O = (
                alpha[:, None]
                * l[:, None]
                * O
                + block_O
            ) / l_new[:, None]

            m = m_new
            l = l_new

        outputs.append(O)

    return torch.stack(
        outputs,
        dim=0,
    )

    
def paged_attention_batch_online_v1(
    q: torch.Tensor,
    pool: KVBlockPool,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    layer_idx: int,
):
    B, H, D = q.shape

    S = pool.block_size
    scale = 1.0 / math.sqrt(D)
    M = block_tables.shape[1]

    m = torch.full(
        (B, H),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )

    l = torch.zeros(
        B, H,
        device=q.device,
        dtype=q.dtype,
    )

    O = torch.zeros(
        B, H, D,
        device=q.device,
        dtype=q.dtype,
    )

    for logical_block in range(M):
        physical_blocks = block_tables[
            :,
            logical_block,
        ]

        safe_blocks = torch.clamp(
            physical_blocks,
            min=0,
        )

        k_block = pool.k[
            layer_idx,
            safe_blocks,
        ]

        v_block = pool.v[
            layer_idx,
            safe_blocks,
        ]

        logical_start = logical_block * S

        valid_counts = torch.clamp(
            context_lengths - logical_start,
            min=0,
            max=S,
        )

        scores = torch.einsum(
            "bhd,bshd->bhs",
            q,
            k_block,
        )

        token_offsets = torch.arange(
            S,
            device=q.device,
        )

        valid_mask = (
            token_offsets[
                None,
                None,
                :
            ]
            <
            valid_counts[
                :,
                None,
                None,
            ]
        )

        scores = scores.masked_fill(
            ~valid_mask,
            float("-inf"),
        )
        scores = scores * scale

        block_max = scores.max(
            dim=-1
        ).values

        m_new = torch.maximum(
            m,
            block_max,
        )

        alpha = torch.exp(m - m_new)

        beta = torch.exp(scores - m_new[:, :, None])

        beta = torch.where(
            valid_mask,
            beta,
            torch.zeros_like(beta),
        )

        l_new = alpha * l + beta.sum(dim=-1)

        # The KV pool is allocated with torch.empty. Mask invalid value
        # positions BEFORE multiplication: 0 * NaN is still NaN.
        safe_v = torch.where(
            valid_mask[:, 0, :, None, None],
            v_block,
            torch.zeros_like(v_block),
        )

        block_O = torch.einsum(
            "bhs,bshd->bhd",
            beta,
            safe_v,
        )

        O_new = (
            alpha[:, :, None]
            * l[:, :, None]
            * O
            +
            block_O
        ) / l_new[:, :, None]

        active = (valid_counts > 0)

        active_bh = active[:, None]

        m = torch.where(
            active_bh,
            m_new,
            m,
        )

        l = torch.where(
            active_bh,
            l_new,
            l,
        )

        O = torch.where(
            active[:, None, None],
            O_new,
            O,
        )

    return O