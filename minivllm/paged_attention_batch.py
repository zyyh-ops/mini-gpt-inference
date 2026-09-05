import math
import torch
from kv_block_pool import KVBlockPool
from sequence import SequenceState
from model_runner import ModelRunner
from model import MiniGPT
from block_manager import BlockManager


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

    outputs = []

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

        block_O = torch.einsum(
            "bhs,bshd->bhd",
            beta,
            v_block,
        )

        O_new = (
            alpha[:, :, None]
            * l[:, :, None]
            * O
            +
            block_O
        ) / l_new[:, :, None]

        m_new = m
        l_new = l

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

        outputs.append(O)

    return torch.stack(
        O,
        dim=1,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequences = []

    context_lengths = [3, 9, 17, 31]

    for i, context_length in enumerate(context_lengths):
        seq = SequenceState(
            seq_id=chr(ord('A') + i),
            prompt_tokens=[j for j in context_length],
        )

        sequences.append(seq)

    N_EMBD = 512
    N_HEAD = 8
    HEAD_DIM = 64

    model = MiniGPT(
        n_embd=N_EMBD,
        n_head=N_HEAD,
        vocab_size=1024,
        block_size=1024,
        dropout=0,
        n_layer=8,
    ).to(device)

    kv_pool = KVBlockPool(
        n_layer=8,
        num_blocks=4*8,
        block_size=16,
        n_head=N_HEAD,
        head_dim=HEAD_DIM,
        device=device,
    )

    runner = ModelRunner(
        model=model,
        kv_pool=kv_pool,
        block_manager=BlockManager(
            num_blocks=4*8,
            block_size=16,
        )
    )

    for seq in sequences:
        next_token = runner.prefill(seq)

        seq.last_token = next_token

    out_reference = runner.decode_batch(sequences)

    