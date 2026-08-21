from dataclasses import dataclass, field

import torch


@dataclass
class SequenceState:
    seq_id: str
    length: int = 0

    block_table: list[int] = field(default_factory=list)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.free_blocks = list(
            range(num_blocks - 1, -1, -1)
        )

        self.block_owner: dict[int, str] = {}

    def _allocate_block(
        self,
        seq: SequenceState,
    ) -> int:

        if not self.free_blocks:
            raise RuntimeError("KV block pool exhausted")

        physical_block = self.free_blocks.pop()

        assert physical_block not in self.block_owner

        self.block_owner[physical_block] = seq.seq_id
        seq.block_table.append(physical_block)

        return physical_block

    def reserve_append_slot(
        self,
        seq: SequenceState,
    ) -> tuple[int, int, int]:
        logical_pos = seq.length

        logical_block = logical_pos // self.block_size

        block_offset = logical_pos % self.block_size

        if logical_block == len(seq.block_table):
            self._allocate_block(seq)

        assert logical_block < len(seq.block_table)

        physcial_block = seq.block_table[logical_block]

        physcial_slot = (
            physcial_block * self.block_size
            + block_offset
        )

        return (
            physcial_block,
            block_offset,
            physcial_slot,
        )

    def logical_to_slot(
        self,
        seq: SequenceState,
        logical_pos: int,
    ) -> int:
        if not (0 <= logical_pos < seq.length):
            raise IndexError(
                f"logical_pos={logical_pos}, "
                f"length={seq.length}"
            )

        logical_block = logical_pos // self.block_size

        block_offset = logical_pos % self.block_size

        physical_block = seq.block_table[logical_block]

        return physical_block * self.block_size + block_offset

    def free(
        self,
        seq: SequenceState,
    ) -> list[int]:
        released = list(seq.block_table)

        for block in released:
            owner = self.block_owner.pop(block)

            assert owner == seq.seq_id

            self.free_blocks.append(block)

        seq.block_table.clear()
        seq.length = 0

        return released

    def check_invariants(
        self,
        sequences: list[SequenceState],
    ):
        all_blocks = set(range(self.num_blocks))

        free = set(self.free_blocks)
        allocated = set(self.block_owner.keys())

        assert len(free) == len(self.free_blocks)

        assert free.isdisjoint(allocated)

        assert free | allocated == all_blocks

        for seq in sequences:
            assert len(seq.block_table) == len(set(seq.block_table))

            for block in seq.block_table:
                assert self.block_owner[block] == seq.seq_id

            expected_blocks = (
                seq.length + self.block_size - 1
            ) // self.block_size

            assert (
                len(seq.block_table)
                == expected_blocks
            )


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


class ContiguousKVReference:
    def __init__(
        self,
        n_layer: int,
        capacity: int,
        n_head: int,
        head_dim: int,
    ):
        self.k = torch.empty(
            (n_layer, capacity, n_head, head_dim),
            dtype=torch.float32,
        )

        self.v = torch.empty_like(self.k)

        self.length = 0

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        pos = self.length

        self.k[:, pos].copy_(k_new)
        self.v[:, pos].copy_(v_new)

        self.length += 1

    def get(self):
        return (
            self.k[:, :self.length],
            self.v[:, :self.length],
        )


def append_token(
    seq: SequenceState,
    manager: BlockManager,
    pool: KVBlockPool,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> int:
    (
        physical_block,
        block_offset,
        physical_slot,
    ) = manager.reserve_append_slot(seq)

    pool.write(
        physical_block,
        block_offset,
        k_new,
        v_new,
    )

    seq.length += 1

    return physical_slot


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


def check_sequence(
    seq: SequenceState,
    pool: KVBlockPool,
    reference: ContiguousKVReference,
):
    paged_k, paged_v = pool.gather_sequence(seq)

    ref_k, ref_v = reference.get()

    k_max_abs = (
        paged_k - ref_k
    ).abs().max().item()

    v_max_abs = (
        paged_v - ref_v
    ).abs().max().item()

    print(
        f"{seq.seq_id}: "
        f"length={seq.length}, "
        f"block_table={seq.block_table}, "
        f"K max_abs={k_max_abs:.8f} "
        f"V max_abs={v_max_abs:.8f}"
    )

    assert torch.equal(paged_k, ref_k)
    assert torch.equal(paged_v, ref_v)


def main():
    torch.manual_seed(0)

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

    sequences = [A, B, C]

    ref_A = ContiguousKVReference(
        n_layer,
        capacity=32,
        n_head=n_head,
        head_dim=head_dim,
    )

    ref_B = ContiguousKVReference(
        n_layer,
        capacity=32,
        n_head=n_head,
        head_dim=head_dim,
    )

    ref_C = ContiguousKVReference(
        n_layer,
        capacity=32,
        n_head=n_head,
        head_dim=head_dim,
    )


    print("=== Phase 1: A append 5 ===")

    for pos in range(5):
        k, v = make_kv(
            seq_tag=1,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        slot = append_token(
            A,
            manager,
            pool,
            k,
            v,
        )

        ref_A.append(k, v)

        print(
            f"A logical_pos={pos} "
            f"-> physical_slot={slot}"
        )

    check_sequence(A, pool, ref_A)
    manager.check_invariants(sequences)

    print()


    print("=== Phase 2: B append 5 ===")

    for pos in range(5):
        k, v = make_kv(
            seq_tag=2,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        slot = append_token(
            B, manager, pool, k, v
        )

        ref_B.append(k, v)

        print(
            f"B logical_pos={pos} "
            f"-> physcial_slot={slot}"
        )

    check_sequence(B, pool, ref_B)
    manager.check_invariants(sequences)

    print()


    print("=== Phase 3: free A ===")

    released_A = manager.free(A)

    print("released A blocks =", released_A)
    print("free blocks       =", manager.free_blocks)

    manager.check_invariants(sequences)

    print()


    print("=== Phase 4: C reuse ===")

    k, v = make_kv(
        seq_tag=3,
        logical_pos=0,
        n_layer=n_layer,
        n_head=n_head,
        head_dim=head_dim,
    )

    slot_C = append_token(
        C, manager, pool, k, v
    )

    ref_C.append(k, v)

    print(
        "C block_table = ",
        C.block_table
    )

    print(
        "C first slot = ",
        slot_C,
    )

    assert C.block_table[0] in released_A

    check_sequence(C, pool, ref_C)

    check_sequence(B, pool, ref_B)

    manager.check_invariants(sequences)

    print()

    print("=== Phase 5: fragments B ===")

    for pos in range(5, 9):
        k, v = make_kv(
            seq_tag=2,
            logical_pos=pos,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )

        slot = append_token(
            B, manager, pool, k, v
        )

        ref_B.append(k, v)

        print(
            f"B logical_pos={pos} "
            f"-> physical_slot={slot}"
        )

    print()

    print(
        "B final block_table = ",
        B.block_table,
    )

    assert B.block_table == [2, 3, 0]

    check_sequence(B, pool, ref_B)

    assert manager.logical_to_slot(B, 8) == 0

    manager.check_invariants(sequences)

    print()
    print("All paged KV storage correctness checks passed.")


if __name__ == "__main__":
    main()