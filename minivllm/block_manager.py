from sequence import SequenceState, Status
import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.free_blocks = list(
            range(num_blocks - 1, -1, -1)
        )

        self.block_owner: dict[int, str] = {}

    def can_allocate(
        self,
        seq: SequenceState,
    ) -> bool:
        need_blocks = (
            seq.max_tokens + self.block_size - 1
        ) // self.block_size

        return need_blocks <= len(self.free_blocks)

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

    def allocate(
        self,
        seq: SequenceState,
    ):  
        need_blocks = (
            seq.max_tokens + self.block_size - 1
        ) // self.block_size

        for _ in range(need_blocks):
            self._allocate_block(seq)

    def num_new_blocks_needed(
        self,
        seq: SequenceState,
        num_tokens: int,
    ) -> int:
        current_capacity = (
            self.num_blocks
            * self.block_size
        )

        target_length = seq.length + num_tokens

        target_blocks = (
            target_length
            + self.block_size - 1
        ) // self.block_size

        return max(
            0,
            target_blocks - len(seq.block_table)
        )

    def can_reserve(
        self,
        seq: SequenceState,
        num_tokens: int,
    ) -> bool:
        need = self.num_new_blocks_needed(
            seq,
            num_tokens,
        )

        return need <= len(self.free_blocks)

    def reserve_append_slot(
        self,
        seq: SequenceState,
    ):
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

        return physcial_block, block_offset, physcial_slot

    def reserve_slots(
        self,
        seq: SequenceState,
        num_tokens: int,
    ) -> list[int]:
        slots = []

        start_pos = seq.length

        for i in range(num_tokens):
            logical_pos = start_pos + i

            logical_block = logical_pos // self.block_size

            offset = logical_pos % self.block_size

            if logical_block == len(seq.block_table):
                self._allocate_block(seq)

            physical_block = seq.block_table[logical_block]

            physical_slot = physical_block * self.block_size + offset

            slots.append(physical_slot)

        return slots

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