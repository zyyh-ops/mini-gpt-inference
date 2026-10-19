"""CPU-only lifecycle regression tests for MiniRuntime.

This substitutes a deterministic fake ModelRunner but exercises the REAL
MiniRuntime, Scheduler, BlockManager, and SequenceState state transitions.
Run from repo root:
  PYTHONPATH=./minivllm python -m unittest discover -s tests -p 'test_runtime_lifecycle.py' -v
"""
import unittest
import torch

from block_manager import BlockManager
from mini_runtime import MiniRuntime
from scheduler import Scheduler
from sequence import SequenceState, Status


class FakeModelRunner:
    """Use real block reservations, but no neural-network compute or GPU."""

    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.prefill_calls = []
        self.decode_calls = []

    def prefill_chunk(self, seq, start_pos, num_tokens):
        assert start_pos == seq.prompt_pos == seq.length
        assert num_tokens > 0
        self.prefill_calls.append((seq.seq_id, start_pos, num_tokens))
        self.block_manager.reserve_slots(seq, num_tokens)
        seq.length += num_tokens
        # Deterministic logits: argmax always yields token 0.
        return torch.zeros((1, num_tokens, 4), dtype=torch.float32)

    def decode_batch(self, sequences):
        self.decode_calls.append([seq.seq_id for seq in sequences])
        for seq in sequences:
            self.block_manager.reserve_append_slot(seq)
            seq.length += 1
        batch_size = len(sequences)
        return torch.zeros((batch_size, 4), dtype=torch.float32), [0] * batch_size


def make_cpu_runtime(num_blocks=4, block_size=2, max_running=1,
                     max_num_batched_tokens=2):
    # Keep the real MiniRuntime.step(), without allocating the default
    # 8-layer, 512-dimensional MiniGPT or a full KV tensor pool.
    runtime = MiniRuntime.__new__(MiniRuntime)
    runtime.block_manager = BlockManager(num_blocks, block_size)
    runtime.scheduler = Scheduler(
        runtime.block_manager, max_running, max_num_batched_tokens
    )
    runtime.runner = FakeModelRunner(runtime.block_manager)
    runtime.iteration = 0
    runtime.records = []
    return runtime


class TestRuntimeLifecycle(unittest.TestCase):
    def test_single_token_request_finishes_after_prefill(self):
        runtime = make_cpu_runtime()
        seq = SequenceState(seq_id="A", prompt_tokens=[1, 2], max_tokens=3)
        runtime.add_request(seq)

        runtime.step()

        self.assertEqual(len(seq.generated_tokens), 1)
        self.assertEqual(seq.status, Status.FINISHED)
        self.assertEqual(seq.finish_iter, 1)
        self.assertEqual(runtime.scheduler.running, [])
        self.assertEqual(runtime.block_manager.block_owner, {})
        self.assertEqual(len(runtime.block_manager.free_blocks), 4)

    def test_two_token_request_does_not_generate_extra_token(self):
        runtime = make_cpu_runtime()
        seq = SequenceState(seq_id="A", prompt_tokens=[1, 2], max_tokens=4)
        runtime.add_request(seq)

        runtime.step()  # Prefill generates the first output token.
        self.assertEqual(len(seq.generated_tokens), 1)
        runtime.step()  # Decode generates the second (and final) token.

        self.assertEqual(len(seq.generated_tokens), 2)
        self.assertEqual(seq.status, Status.FINISHED)
        self.assertEqual(seq.finish_iter, 2)
        self.assertEqual(runtime.block_manager.block_owner, {})

    def test_chunked_prefill_then_decode_and_finish(self):
        runtime = make_cpu_runtime(
            num_blocks=4, block_size=2, max_running=1,
            max_num_batched_tokens=2,
        )
        seq = SequenceState(seq_id="C", prompt_tokens=[1, 2, 3, 4, 5],
                            max_tokens=7)
        runtime.add_request(seq)

        runtime.step()
        self.assertEqual((seq.prompt_pos, seq.length), (2, 2))
        self.assertEqual(seq.status, Status.PREFILLING)
        self.assertEqual(seq.generated_tokens, [])

        runtime.step()
        self.assertEqual((seq.prompt_pos, seq.length), (4, 4))
        self.assertEqual(seq.status, Status.PREFILLING)

        runtime.step()
        self.assertEqual((seq.prompt_pos, seq.length), (5, 5))
        self.assertEqual(seq.status, Status.RUNNING)
        self.assertEqual(len(seq.generated_tokens), 1)
        self.assertEqual(seq.first_token_iter, 3)
        self.assertEqual(runtime.runner.prefill_calls,
                         [("C", 0, 2), ("C", 2, 2), ("C", 4, 1)])

        runtime.step()
        self.assertEqual(seq.status, Status.FINISHED)
        self.assertEqual(len(seq.generated_tokens), 2)
        self.assertEqual(seq.finish_iter, 4)
        self.assertEqual(runtime.block_manager.block_owner, {})
        self.assertEqual(len(runtime.block_manager.free_blocks), 4)

    def test_finished_request_releases_blocks_for_waiting_request(self):
        runtime = make_cpu_runtime(
            num_blocks=1, block_size=2, max_running=1, max_num_batched_tokens=2
        )
        a = SequenceState(seq_id="A", prompt_tokens=[1, 2], max_tokens=3)
        b = SequenceState(seq_id="B", prompt_tokens=[3, 4], max_tokens=3)
        runtime.add_request(a)
        runtime.add_request(b)

        runtime.step()  # A generates its only output token and finishes.
        self.assertEqual(a.status, Status.FINISHED)
        self.assertEqual(len(runtime.block_manager.free_blocks), 1)
        self.assertEqual([s.seq_id for s in runtime.scheduler.waiting], ["B"])

        runtime.step()  # B should now use the reclaimed KV block.
        self.assertEqual(b.status, Status.FINISHED)
        self.assertEqual(len(b.generated_tokens), 1)
        self.assertEqual(b.finish_iter, 2)
        self.assertEqual(len(runtime.block_manager.free_blocks), 1)
        self.assertEqual(runtime.block_manager.block_owner, {})


if __name__ == "__main__":
    unittest.main()
