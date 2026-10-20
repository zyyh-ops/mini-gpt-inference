"""Step 2: manual recompute restoration, using a REAL tiny MiniGPT on CPU.

Run from repository root after applying stage-1 and stage-2 patches:
  python -m unittest discover \
      -s tests -p 'test_preemption_restore.py' -v

Tests restoration only; they do not test automatic victim selection / liveness.
"""

import contextlib
import io
import unittest

import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.mini_runtime import MiniRuntime
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


def tiny_runtime(model, *, token_budget=3):
    runtime = MiniRuntime.__new__(MiniRuntime)
    manager = BlockManager(num_blocks=24, block_size=2)
    runtime.block_manager = manager
    runtime.scheduler = Scheduler(manager, max_running=1,
                                  max_num_batched_tokens=token_budget)
    runtime.runner = ModelRunner(
        model=model,
        kv_pool=KVBlockPool(n_layer=1, num_blocks=24, block_size=2,
                            n_head=2, head_dim=8, device='cpu'),
        block_manager=manager,
    )
    # Poison unwritten KV slots. Using torch.empty without this test can
    # accidentally hide invalid-padding reads in batched PagedAttention.
    runtime.runner.kv_pool.k.fill_(float('nan'))
    runtime.runner.kv_pool.v.fill_(float('nan'))
    runtime.iteration = 0
    runtime.records = []
    return runtime


def advance(runtime, n=1):
    with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
        for _ in range(n):
            runtime.step()


class TestRecomputeRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        cls.model = MiniGPT(n_embd=16, n_head=2, vocab_size=32,
                            block_size=64, dropout=0.0, n_layer=1).eval()

    def make_pair(self):
        reference = tiny_runtime(self.model)
        recovered = tiny_runtime(self.model)
        a = SequenceState(seq_id='A', prompt_tokens=[1, 5, 3, 7, 2], max_tokens=9)
        b = SequenceState(seq_id='B', prompt_tokens=[1, 5, 3, 7, 2], max_tokens=9)
        reference.add_request(a)
        recovered.add_request(b)
        # Chunked prefill: 3+2 tokens; then one decode step:
        # KV = prompt(5) + first generated token(1), output count = 2.
        advance(reference, 3)
        advance(recovered, 3)
        self.assertEqual(a.status, Status.RUNNING)
        self.assertEqual(b.status, Status.RUNNING)
        self.assertEqual(a.length, 6)
        self.assertEqual(b.length, 6)
        self.assertEqual(len(b.generated_tokens), 2)
        self.assertEqual(a.generated_tokens, b.generated_tokens)
        return reference, recovered, a, b

    def verify_restore_and_finish(self, *, interrupt_mid_recompute):
        ref_rt, resume_rt, ref, seq = self.make_pair()
        original_history = list(seq.generated_tokens)
        original_last = seq.last_token
        original_first_iter = seq.first_token_iter
        expected_k, expected_v = ref_rt.runner.kv_pool.gather_sequence(ref)

        resume_rt.scheduler.preempt(seq)
        self.assertEqual(seq.status, Status.WAITING)
        self.assertEqual(seq.recompute_tokens,
                         seq.prompt_tokens + original_history[:-1])
        self.assertEqual((seq.length, seq.prompt_pos), (0, 0))
        self.assertEqual(len(resume_rt.block_manager.free_blocks), 24)

        # First recompute chunk consumes 3/6 KV tokens.
        advance(resume_rt)
        self.assertEqual(seq.status, Status.PREFILLING)
        self.assertEqual(seq.prompt_pos, 3)
        self.assertEqual(seq.generated_tokens, original_history)
        self.assertEqual(seq.last_token, original_last)

        if interrupt_mid_recompute:
            # Even if preempted again while recomputing, the entire original
            # cached prefix can be rebuilt without adding output tokens.
            resume_rt.scheduler.preempt(seq)
            self.assertEqual(seq.status, Status.WAITING)
            self.assertEqual(seq.recompute_tokens,
                             seq.prompt_tokens + original_history[:-1])
            self.assertEqual(seq.prompt_pos, 0)
            self.assertEqual(len(resume_rt.block_manager.free_blocks), 24)
            advance(resume_rt)
            self.assertEqual(seq.status, Status.PREFILLING)

        # Second recompute chunk completes prefix. It MUST NOT sample logits.
        advance(resume_rt)
        self.assertEqual(seq.status, Status.RUNNING)
        self.assertIsNone(seq.recompute_tokens)
        self.assertEqual(seq.length, 6)
        self.assertEqual(seq.prompt_pos, seq.prompt_length)
        self.assertEqual(seq.generated_tokens, original_history)
        self.assertEqual(seq.last_token, original_last)
        self.assertEqual(seq.first_token_iter, original_first_iter)
        restored_k, restored_v = resume_rt.runner.kv_pool.gather_sequence(seq)
        torch.testing.assert_close(restored_k, expected_k, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(restored_v, expected_v, atol=1e-5, rtol=1e-5)
        resume_rt.block_manager.check_invariants([seq])

        for _ in range(8):
            if ref.status != Status.FINISHED:
                advance(ref_rt)
            if seq.status != Status.FINISHED:
                advance(resume_rt)
            if ref.status == seq.status == Status.FINISHED:
                break
        self.assertEqual(ref.status, Status.FINISHED)
        self.assertEqual(seq.status, Status.FINISHED)
        self.assertEqual(len(seq.generated_tokens), 4)
        self.assertEqual(seq.generated_tokens, ref.generated_tokens)
        self.assertEqual(len(resume_rt.block_manager.free_blocks), 24)
        self.assertEqual(resume_rt.block_manager.block_owner, {})

    def test_restore_without_duplicate_output_token(self):
        self.verify_restore_and_finish(interrupt_mid_recompute=False)

    def test_recompute_interrupted_then_restored_again(self):
        self.verify_restore_and_finish(interrupt_mid_recompute=True)


if __name__ == '__main__':
    unittest.main()
