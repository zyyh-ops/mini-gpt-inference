"""Auto recompute-preemption tests for MiniRuntime (CPU only).

Run from repository root after applying preemption_stage3.patch:
  python -m unittest discover -s tests \
      -p 'test_auto_preemption.py' -v
"""
import contextlib
import io
import unittest
import random

import torch

from minivllm.block_manager import BlockManager
from minivllm.mini_runtime import MiniRuntime
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.kv_block_pool import KVBlockPool


def request(name, prompt, num_output):
    return SequenceState(seq_id=name, prompt_tokens=list(prompt),
                         max_tokens=len(prompt) + num_output)


class FakeRunner:
    def __init__(self, block_manager):
        self.manager = block_manager

    def prefill_chunk(self, seq, start_pos, num_tokens):
        assert seq.length == seq.prompt_pos == start_pos
        self.manager.reserve_slots(seq, num_tokens)
        seq.length += num_tokens
        return torch.zeros((1, num_tokens, 16))

    def decode_batch(self, sequences):
        for seq in sequences:
            self.manager.reserve_append_slot(seq)
            seq.length += 1
        return torch.zeros((len(sequences), 16)), [0] * len(sequences)


def cpu_runtime(*, policy, num_blocks=2, block_size=4,
                max_running=2, token_budget=8, model=None):
    runtime = MiniRuntime.__new__(MiniRuntime)
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    runtime.block_manager = manager
    runtime.scheduler = Scheduler(manager, max_running, token_budget,
                                  preemption_policy=policy)
    if model is None:
        runtime.runner = FakeRunner(manager)
    else:
        pool = KVBlockPool(n_layer=1, num_blocks=num_blocks,
                           block_size=block_size, n_head=2,
                           head_dim=8, device='cpu')
        pool.k.fill_(float('nan'))
        pool.v.fill_(float('nan'))
        runtime.runner = ModelRunner(model, pool, manager)
    runtime.iteration = 0
    runtime.records = []
    return runtime


def advance_to_finish(runtime, seqs, max_steps=25):
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        for _ in range(max_steps):
            if all(s.status == Status.FINISHED for s in seqs):
                break
            runtime.step()
    if not all(s.status == Status.FINISHED for s in seqs):
        raise AssertionError(
            f"scheduler stalled after {max_steps} steps: "
            f"{[(s.seq_id, s.status, s.prompt_pos, s.length, len(s.generated_tokens)) for s in seqs]}"
        )


class TestAutoPreemption(unittest.TestCase):
    def test_two_request_deadlock_is_resolved(self):
        rt = cpu_runtime(policy='recompute')
        a = request('A', [1, 2, 3, 4], 3)
        b = request('B', [5, 6, 7, 8], 3)
        rt.add_request(a); rt.add_request(b)
        # The dynamic policy must be permissive enough to reproduce contention.
        with contextlib.redirect_stdout(io.StringIO()):
            rt.step()
        self.assertEqual([a.status, b.status], [Status.RUNNING, Status.RUNNING])
        self.assertEqual(len(rt.block_manager.free_blocks), 0)
        advance_to_finish(rt, [a, b])
        self.assertGreater(rt.scheduler.num_preemptions, 0)
        self.assertEqual([len(a.generated_tokens), len(b.generated_tokens)], [3, 3])
        self.assertEqual(len(rt.block_manager.free_blocks), 2)
        self.assertEqual(rt.block_manager.block_owner, {})

    def test_planned_decode_is_protected_from_eviction(self):
        rt = cpu_runtime(policy='recompute')
        a = request('A', [1, 2, 3, 4], 3)
        b = request('B', [1, 2, 3, 4], 3)
        rt.add_request(a); rt.add_request(b)
        with contextlib.redirect_stdout(io.StringIO()):
            rt.step()
        plan = rt.scheduler.schedule()  # must preempt B to schedule A
        self.assertEqual([s.seq_id for s in plan.decode], ['A'])
        self.assertEqual(a.status, Status.RUNNING)
        self.assertTrue(a.block_table)
        self.assertEqual(b.status, Status.WAITING)
        self.assertEqual(b.block_table, [])
        self.assertNotIn('B', [s.seq_id for s in plan.decode])
        self.assertNotIn('B', [x.seq.seq_id for x in plan.prefill])
        # The protected A still owns its original physical block.
        self.assertTrue(all(rt.block_manager.block_owner[x] == 'A'
                            for x in a.block_table))
        rt.block_manager.reserve_append_slot(a)  # reserved headroom is real

    def test_prefilling_victim_restarts_from_original_prompt(self):
        rt = cpu_runtime(policy='recompute', token_budget=8)
        a = request('A', [1, 2, 3, 4], 3)
        b = request('B', [3, 4, 5, 6, 7, 8], 1)
        rt.add_request(a); rt.add_request(b)
        with contextlib.redirect_stdout(io.StringIO()):
            rt.step()  # A uses 1 block, B begins partial prefill in other block
        self.assertEqual(a.status, Status.RUNNING)
        self.assertEqual(b.status, Status.PREFILLING)
        self.assertEqual(b.prompt_pos, 4)
        advance_to_finish(rt, [a, b])
        self.assertGreater(b.num_preemptions, 0)
        self.assertEqual(len(b.generated_tokens), 1)
        self.assertEqual(b.status, Status.FINISHED)

    def test_prefill_chunk_shrinks_to_fit_free_blocks(self):
        rt = cpu_runtime(policy='recompute', token_budget=12,
                         num_blocks=2, block_size=4, max_running=2)
        a = request('A', [1], 2)  # uses one physical block
        b = request('B', list(range(7)), 1)  # needs two blocks eventually
        rt.add_request(a); rt.add_request(b)
        plan = rt.scheduler.schedule()
        # A consumes one projected block; B can only afford the other.
        self.assertEqual([(x.seq.seq_id, x.num_tokens) for x in plan.prefill],
                         [('A', 1), ('B', 4)])
        self.assertEqual(plan.num_scheduled_tokens, 5)
        # Continue from a fresh scheduler (plan() is not an execution step).
        rt2 = cpu_runtime(policy='recompute', token_budget=12,
                          num_blocks=2, block_size=4, max_running=2)
        c = request('A', [1], 2)
        d = request('B', list(range(7)), 1)
        rt2.add_request(c); rt2.add_request(d)
        advance_to_finish(rt2, [c, d])
        self.assertEqual(rt2.block_manager.block_owner, {})

    def test_deterministic_multi_request_stress(self):
        # Small, reproducible state-machine exploration (CPU + FakeRunner).
        for case in range(40):
            rng = random.Random(2026 + case)
            bs = rng.choice((2, 3, 4, 8))
            blocks = rng.choice((2, 3, 4, 5, 6))
            rt = cpu_runtime(policy='recompute', num_blocks=blocks,
                             block_size=bs, max_running=rng.randint(1, 4),
                             token_budget=rng.randint(1, 12))
            seqs = []
            for n in range(rng.randint(2, 5)):
                prompt_len = rng.randint(1, bs * blocks - 1)
                out_len = rng.randint(1, bs * blocks - prompt_len)
                seq = request(chr(65+n), [1]*prompt_len, out_len)
                rt.add_request(seq)
                seqs.append(seq)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(400):
                    if all(s.status == Status.FINISHED for s in seqs):
                        break
                    rt.step()
                    rt.block_manager.check_invariants(seqs)
                else:
                    self.fail(f"case {case} did not finish")
            self.assertEqual(rt.block_manager.block_owner, {})

    def test_real_tiny_model_matches_conservative_baseline(self):
        torch.manual_seed(2026)
        model = MiniGPT(n_embd=16, n_head=2, vocab_size=32,
                        block_size=64, dropout=0.0, n_layer=1).eval()
        baseline = cpu_runtime(policy='conservative', model=model)
        preemptive = cpu_runtime(policy='recompute', model=model)
        data = [('A', [1, 5, 3, 7]), ('B', [2, 6, 4, 8])]
        base_seqs = [request(n, p, 3) for n,p in data]
        preempt_seqs = [request(n, p, 3) for n,p in data]
        for seq in base_seqs: baseline.add_request(seq)
        for seq in preempt_seqs: preemptive.add_request(seq)
        advance_to_finish(baseline, base_seqs)
        advance_to_finish(preemptive, preempt_seqs)
        self.assertGreater(preemptive.scheduler.num_preemptions, 0)
        for a, b in zip(base_seqs, preempt_seqs):
            self.assertEqual(a.generated_tokens, b.generated_tokens,
                             f"generation changed after recompute for {a.seq_id}")
        self.assertEqual(preemptive.block_manager.block_owner, {})


if __name__ == '__main__':
    unittest.main()
