"""CPU-only scheduling tests for MiniRuntime/Scheduler.

Run from repository root:
  python -m unittest discover -s tests -p test_scheduler_contention.py -v

The first four cases test admission, decode priority, token budget and KV budget.
The final case is an intentionally red regression test on the original scheduler:
partial KV allocation can admit more sequences than can ever finish when the
engine has no preemption.
"""
import contextlib
import io
import unittest
import torch

from minivllm.block_manager import BlockManager
from minivllm.mini_runtime import MiniRuntime
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


def make_seq(name, prompt_len, output_len):
    return SequenceState(
        seq_id=name,
        prompt_tokens=[1] * prompt_len,
        max_tokens=prompt_len + output_len,
    )


def commit_prefill(manager, work):
    seq = work.seq
    assert work.start_pos == seq.length == seq.prompt_pos
    manager.reserve_slots(seq, work.num_tokens)
    seq.length += work.num_tokens
    seq.prompt_pos += work.num_tokens


def make_running(manager, name, prompt_len, output_len):
    seq = make_seq(name, prompt_len, output_len)
    manager.reserve_slots(seq, prompt_len)
    seq.length = seq.prompt_pos = prompt_len
    seq.last_token = 0
    seq.generated_tokens.append(0)  # first token produced by prefill
    seq.status = Status.RUNNING
    return seq


class FakeModelRunner:
    def __init__(self, manager):
        self.manager = manager

    def prefill_chunk(self, seq, start_pos, num_tokens):
        assert seq.prompt_pos == seq.length == start_pos
        self.manager.reserve_slots(seq, num_tokens)
        seq.length += num_tokens
        return torch.zeros((1, num_tokens, 4))

    def decode_batch(self, seqs):
        for seq in seqs:
            self.manager.reserve_append_slot(seq)
            seq.length += 1
        return torch.zeros((len(seqs), 4)), [0] * len(seqs)


def make_cpu_runtime(num_blocks, block_size, max_running, token_budget):
    runtime = MiniRuntime.__new__(MiniRuntime)
    manager = BlockManager(num_blocks, block_size)
    runtime.block_manager = manager
    runtime.scheduler = Scheduler(manager, max_running, token_budget)
    runtime.runner = FakeModelRunner(manager)
    runtime.iteration = 0
    runtime.records = []
    return runtime


class TestSchedulerBoundaries(unittest.TestCase):
    def test_chunked_prefill_respects_token_budget_and_max_running(self):
        manager = BlockManager(num_blocks=16, block_size=4)
        scheduler = Scheduler(manager, max_running=2, max_num_batched_tokens=5)
        a, b, c = make_seq('A', 7, 2), make_seq('B', 4, 2), make_seq('C', 4, 2)
        for seq in (a, b, c):
            scheduler.add_request(seq)

        first = scheduler.schedule()
        self.assertEqual([(x.seq.seq_id, x.start_pos, x.num_tokens)
                          for x in first.prefill], [('A', 0, 5)])
        self.assertEqual(first.num_scheduled_tokens, 5)
        self.assertEqual([s.seq_id for s in scheduler.waiting], ['B', 'C'])
        for work in first.prefill:
            commit_prefill(manager, work)

        second = scheduler.schedule()
        self.assertEqual([(x.seq.seq_id, x.start_pos, x.num_tokens)
                          for x in second.prefill], [('A', 5, 2), ('B', 0, 3)])
        self.assertEqual(second.num_scheduled_tokens, 5)
        self.assertEqual([s.seq_id for s in scheduler.waiting], ['C'])
        for work in second.prefill:
            commit_prefill(manager, work)
        manager.check_invariants([a, b, c])

    def test_decode_is_scheduled_before_prefill(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        scheduler = Scheduler(manager, max_running=2, max_num_batched_tokens=3)
        a = make_running(manager, 'A', 4, 4)
        b = make_seq('B', 6, 2)
        scheduler.running.append(a)
        scheduler.add_request(b)

        plan = scheduler.schedule()
        self.assertEqual([s.seq_id for s in plan.decode], ['A'])
        self.assertEqual([(x.seq.seq_id, x.num_tokens) for x in plan.prefill], [('B', 2)])
        self.assertEqual(plan.num_scheduled_tokens, 3)
        self.assertTrue({s.seq_id for s in plan.decode}.isdisjoint(
            {x.seq.seq_id for x in plan.prefill}))

    def test_kv_budget_accounts_for_scheduled_decode_before_admission(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        scheduler = Scheduler(manager, max_running=2, max_num_batched_tokens=8)
        a = make_running(manager, 'A', 4, 3)
        b = make_seq('B', 4, 1)
        scheduler.running.append(a)
        scheduler.add_request(b)

        plan = scheduler.schedule()
        self.assertEqual([s.seq_id for s in plan.decode], ['A'])
        self.assertEqual(plan.prefill, [])
        self.assertEqual([s.seq_id for s in scheduler.waiting], ['B'])
        self.assertEqual(plan.num_scheduled_tokens, 1)

    def test_max_running_counts_prefilling_and_running(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        scheduler = Scheduler(manager, max_running=2, max_num_batched_tokens=4)
        a = make_running(manager, 'A', 4, 4)
        b = make_seq('B', 6, 2)
        c = make_seq('C', 4, 1)
        scheduler.running.append(a)
        scheduler.add_request(b)
        first = scheduler.schedule()
        self.assertEqual([x.seq.seq_id for x in first.prefill], ['B'])
        for work in first.prefill:
            commit_prefill(manager, work)
        scheduler.add_request(c)

        second = scheduler.schedule()
        self.assertEqual([x.seq.seq_id for x in second.prefill], ['B'])
        self.assertEqual([s.seq_id for s in scheduler.waiting], ['C'])
        self.assertLessEqual(len(scheduler.running) + len(scheduler.prefilling), 2)


class TestSchedulerLiveness(unittest.TestCase):
    def test_partial_kv_admission_must_not_deadlock_requests(self):
        """Both requests must eventually finish with 2 physical KV blocks.

        With no preemption, admitting both at once traps one block per request;
        neither can acquire the second block needed by the first decode step.
        """
        runtime = make_cpu_runtime(num_blocks=2, block_size=4,
                                   max_running=2, token_budget=8)
        a = make_seq('A', 4, 3)
        b = make_seq('B', 4, 3)
        runtime.add_request(a)
        runtime.add_request(b)

        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(10):
                runtime.step()

        self.assertEqual([a.status, b.status], [Status.FINISHED, Status.FINISHED],
                         msg=f"KV stall: status={[a.status, b.status]}, "
                             f"generated={[len(a.generated_tokens), len(b.generated_tokens)]}, "
                             f"free_blocks={len(runtime.block_manager.free_blocks)}")
        self.assertEqual(len(runtime.block_manager.free_blocks), 2)
        self.assertEqual(runtime.block_manager.block_owner, {})


if __name__ == '__main__':
    unittest.main()
