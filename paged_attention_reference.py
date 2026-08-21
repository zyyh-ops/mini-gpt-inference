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

        score_block = score_block * scale

        score_chunks.append(score_block)

        block_infos.append(
            (
                physical_block,
                valid_tokens,
            )
        )

    scores = torch.cat(
        score_chunks,
        dim=-1,
    )

    assert scores.shape == (H, seq.length)

    probs = torch.softmax(scores, dim=-1)

    out = torch.zeros_like(q)

    cursor = 0

    for physical_block, valid_tokens in block_infos:
        v_block = pool.v[layer_idx, physical_block, :valid_tokens, :, :]

        prob_block = probs[:, cursor: cursor + valid_tokens]

        partial_out = torch.einsum(
            "ht,thd->hd",
            prob_block,
            v_block,
        )

        out += partial_out

        cursor += valid_tokens

    assert cursor == seq.length

    return out, scores, probs


def check_attention(
    seq: SequenceState,
    pool: KVBlockPool,
    q_all: torch.Tensor,
):
    dense_k, dense_v = pool.gather_sequence(seq)

    print()
    print(f"Checking sequence {seq.seq_id}")

    print("length      =", seq.length)

    print("block_table =", seq.block_table)

    for layer_idx in range(pool.n_layer):
        q = q_all[layer_idx]

        dense_out, dense_scores, dense_probs = (
            dense_attention_reference(
                q,
                dense_k[layer_idx],
                dense_v[layer_idx],
            )
        )

        paged_out, paged_scores, paged_probs = (
            paged_attention_reference(
                q,
                pool,
                seq,
                layer_idx,
            )
        )

        scores_max_abs = (
            paged_scores - dense_scores
        ).abs().max().item()

        probs_max_abs = (
            paged_probs - dense_probs
        ).abs().max().item()

        out_max_abs = (
            paged_out - dense_out
        ).abs().max().item()

        print(
            f"layer={layer_idx}: "
            f"scores max_abs="
            f"{scores_max_abs:.8e}, "
            f"probs max_abs="
            f"{probs_max_abs:.8e}, "
            f"out max_abs="
            f"{out_max_abs:.8e}"
        )

        assert torch.allclose(
            paged_scores,
            dense_scores,
            atol=1e-6,
            rtol=1e-6,
        )

        assert torch.allclose(
            paged_probs,
            dense_probs,
            atol=1e-6,
            rtol=1e-6,
        )

        assert torch.allclose(
            paged_out,
            dense_out,
            atol=1e-5,
            rtol=1e-5,
        )


def main():
    torch.manual_seed(42)

    n_layer = 2
    n_head = 2
    head_dim = 4

    num_blocks = 6
    block_size = 4

    manager = BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
    )

    pool = KVBlockPool(
        n_layer=n_layer,
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
    )

    A = SequenceState("A")
    B = SequenceState("B")
    C = SequenceState("C")

    for pos in range(5):
        k, v = make_kv(
            seq_tag=1,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        append_token(
            A, manager, pool, k, v
        )

    assert A.block_table == [0, 1]


    for pos in range(5):
        k, v = make_kv(
            seq_tag=2,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        append_token(
            B, manager, pool, k, v
        )

    assert B.block_table == [2, 3]


    released = manager.free(A)

    assert set(released) == {0, 1}


    k, v = make_kv(
        seq_tag=3,
        logical_pos=0,
        n_layer=n_layer,
        n_head=n_head,
        head_dim=head_dim,
    )

    append_token(
        C, manager, pool, k, v
    )

    assert C.block_table == [1]


    for pos in range(5, 9):
        k, v = make_kv(
            seq_tag=2,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        append_token(
            B, manager, pool, k, v
        )

    assert B.length == 9
    assert B.block_table == [2, 3, 0]

    print(
        "B fragmented block table:",
        B.block_table,
    )


    q_all = torch.randn(
        n_layer, n_head, head_dim, dtype=torch.float32,
    )

    check_attention(
        B, pool, q_all,
    )

    print()
    print("PagedAttention v0 correctness passed.")


if __name__ == "__main__":
    main()