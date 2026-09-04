import math
import torch
from kv_block_pool import KVBlockPool


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