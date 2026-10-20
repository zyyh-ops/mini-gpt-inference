import unittest

from minivllm.block_manager import BlockManager
from minivllm.sequence import SequenceState


class TestBlockManager(unittest.TestCase):
    def test_cross_block_multi_request_and_reuse(self):
        manager = BlockManager(num_blocks=5, block_size=4)
        a = SequenceState(seq_id="A", prompt_tokens=[1] * 7, max_tokens=12)
        b = SequenceState(seq_id="B", prompt_tokens=[2] * 4, max_tokens=12)
        c = SequenceState(seq_id="C", prompt_tokens=[3] * 8, max_tokens=12)
        d = SequenceState(seq_id="D", prompt_tokens=[4] * 9, max_tokens=12)
        seqs = [a, b, c, d]

        self.assertEqual(manager.num_new_blocks_needed(a, 3), 1)
        slots = manager.reserve_slots(a, 3)
        a.length += 3  # Simulate commit after successful execution.
        manager.check_invariants(seqs)

        self.assertEqual(manager.num_new_blocks_needed(a, 1), 0)
        slots += manager.reserve_slots(a, 1)
        a.length += 1
        self.assertEqual(len(a.block_table), 1)
        manager.check_invariants(seqs)

        self.assertEqual(manager.num_new_blocks_needed(a, 3), 1)
        slots += manager.reserve_slots(a, 3)
        a.length += 3
        self.assertEqual(len(a.block_table), 2)
        for pos, slot in enumerate(slots):
            self.assertEqual(manager.logical_to_slot(a, pos), slot)
        manager.check_invariants(seqs)

        manager.reserve_slots(b, 4)
        b.length += 4
        self.assertTrue(set(a.block_table).isdisjoint(b.block_table))
        manager.check_invariants(seqs)

        released = set(manager.free(a))
        self.assertEqual(len(released), 2)
        manager.check_invariants(seqs)

        manager.reserve_slots(c, 8)
        c.length += 8
        self.assertEqual(set(c.block_table), released)
        self.assertTrue(set(c.block_table).isdisjoint(b.block_table))
        manager.check_invariants(seqs)

        # Two free blocks remain, but D needs three blocks.
        self.assertEqual(manager.num_new_blocks_needed(d, 9), 3)
        self.assertFalse(manager.can_reserve(d, 9))


if __name__ == "__main__":
    unittest.main()
