import torch
from sequence import SequenceState


def make_kv(
    seq_tag: int,
    logical_pos: int,
    n_layer: int,
    n_head: int,
    head_dim: int,
):
    x = torch.arange(
        n_layer * n_head * head_dim,
        dtype=torch.float32,
    ).reshape(
        n_layer,
        n_head,
        head_dim,
    )

    base = (
        seq_tag * 10000
        + logical_pos * 100
    )

    k = x + base
    v = x + base + 0.5

    return k, v


class KVBlockPool:
    def __init__(
        self,
        n_layer: int,
        num_blocks: int,
        block_size: int,
        n_head: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        self.n_layer = n_layer
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.dtype = dtype

        shape = (
            n_layer,
            num_blocks,
            block_size,
            n_head,
            head_dim,
        )

        self.k = torch.empty(
            shape,
            dtype=dtype,
        )

        self.v = torch.empty(
            shape,
            dtype=dtype,
        )

    def write(
        self,
        physical_block: int,
        block_offset: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        expected_shape = (
            self.n_layer,
            self.n_head,
            self.head_dim,
        )

        assert k_new.shape == expected_shape
        assert v_new.shape == expected_shape

        self.k[:, physical_block, block_offset, :, :].copy_(k_new)

        self.v[:, physical_block, block_offset, :, :].copy_(v_new)


    def write_layer(
        self,
        layer_idx: int,
        physical_block: int,
        block_offset: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        self.k[
            layer_idx,
            physical_block,
            block_offset,
        ].copy_(k_new)

        self.v[
            layer_idx,
            physical_block,
            block_offset,
        ].copy_(v_new)


    def gather_sequence(
        self,
        seq: SequenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = seq.length

        k_out = torch.empty(
            (
                self.n_layer,
                T,
                self.n_head,
                self.head_dim,
            ),
            dtype=self.dtype,
        )

        v_out = torch.empty_like(k_out)

        for logical_block, physical_block in enumerate(
            seq.block_table
        ):
            logical_start = (
                logical_block * self.block_size
            )

            logical_end = min(
                logical_start + self.block_size,
                T,
            )

            valid_tokens = logical_end - logical_start

            k_out[:, logical_start:logical_end, :, :].copy_(
                self.k[:, physical_block, :valid_tokens, :, :]
            )

            v_out[:, logical_start:logical_end, :, :].copy_(
                self.v[:, physical_block, :valid_tokens, :, :]
            )

        return k_out, v_out