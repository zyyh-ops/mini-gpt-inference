"""Stage-1 preemption tests: KV eviction and semantic state preservation.

Run from repository root:
  python -m unittest discover -s tests -p 'test_preemption_state.py' -v

This stage does NOT yet implement automatic victim selection or restoration.
"""
import unittest
from minivllm.block_manager import BlockManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


def setup_running(manager, scheduler):
    seq = SequenceState(seq_id="A", max_tokens=9, prompt_tokens=[1, 2, 3, 4])
    manager.reserve_slots(seq, 5)   # cached: prompt + output token 7
    seq.length = 5
    seq.prompt_pos = 4
    seq.generated_tokens[:] = [7, 8]
    seq.last_token = 8  # token 8 has not yet been appended to KV
    seq.status = Status.RUNNING
    scheduler.running.append(seq)
    return seq


class PreemptionStateTest(unittest.TestCase):
    def test_running_preserves_generated_tokens_and_pending_token(self):
        m = BlockManager(num_blocks=4, block_size=4)
        scheduler = Scheduler(m, max_running=2, max_num_batched_tokens=16)
        seq = setup_running(m, scheduler)
        scheduler.preempt(seq)

        self.assertEqual(seq.status, Status.WAITING)
        self.assertEqual(seq.recompute_tokens, [1, 2, 3, 4, 7])
        self.assertEqual(seq.generated_tokens, [7, 8])
        self.assertEqual(seq.last_token, 8)
        self.assertEqual(seq.length, 0)
        self.assertEqual(seq.prompt_pos, 0)
        self.assertEqual(seq.block_table, [])
        self.assertIs(scheduler.waiting[0], seq)
        self.assertEqual(len(m.free_blocks), 4)
        self.assertEqual(m.block_owner, {})
        m.check_invariants([seq])

    def test_partial_prefill_rewinds_to_start(self):
        m = BlockManager(num_blocks=4, block_size=4)
        scheduler = Scheduler(m, max_running=2, max_num_batched_tokens=16)
        seq = SequenceState(seq_id="B", max_tokens=10, prompt_tokens=list(range(1, 8)))
        m.reserve_slots(seq, 3)
        seq.length = seq.prompt_pos = 3
        seq.status = Status.PREFILLING
        scheduler.prefilling.append(seq)
        scheduler.preempt(seq)

        self.assertEqual(seq.status, Status.WAITING)
        self.assertIsNone(seq.recompute_tokens)
        self.assertEqual(seq.remaining_prompt_tokens, 7)
        self.assertEqual((seq.length, seq.prompt_pos), (0, 0))
        self.assertEqual(len(m.free_blocks), 4)

    def test_recompute_interrupted_can_restart(self):
        m = BlockManager(num_blocks=4, block_size=4)
        scheduler = Scheduler(m, max_running=2, max_num_batched_tokens=16)
        seq = setup_running(m, scheduler)
        scheduler.preempt(seq)
        scheduler.waiting.remove(seq)
        seq.status = Status.PREFILLING
        scheduler.prefilling.append(seq)
        m.reserve_slots(seq, 2)
        seq.length = seq.prompt_pos = 2
        scheduler.preempt(seq)
        self.assertEqual(seq.recompute_tokens, [1, 2, 3, 4, 7])
        self.assertEqual(seq.remaining_prompt_tokens, 5)
        self.assertEqual(seq.generated_tokens, [7, 8])
        self.assertEqual(seq.last_token, 8)
        self.assertEqual((seq.length, seq.prompt_pos), (0, 0))
        self.assertEqual(len(m.free_blocks), 4)

if __name__ == '__main__':
    unittest.main()
