"""Step 4: FP32 first-chunk SDPA vs paged online Attention reference.

Run: python -m unittest tests.test_prefill_fastpath -v
"""
import unittest
from unittest.mock import patch

import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.mini_runtime import MiniRuntime
from minivllm.scheduler import Scheduler
from minivllm.sequence import Status
from minivllm.sequence import SequenceState


def run_chunks(model, *, fast, pieces, block_size=4, dtype=torch.float32):
    model.set_prefill_fastpath(fast)
    model.eval()
    num_blocks = 32
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    # Non-monotonic physical allocation order validates Paged KV semantics.
    permutation = list(range(num_blocks))
    permutation = permutation[::2] + permutation[1::2]
    manager.free_blocks = list(reversed(permutation))
    kv = KVBlockPool(
        n_layer=model.n_layer, num_blocks=num_blocks, block_size=block_size,
        n_head=model.blocks[0].attn.n_head,
        head_dim=model.blocks[0].attn.head_size,
        device=next(model.parameters()).device,
        dtype=dtype,
    )
    kv.k.fill_(float('nan'))
    kv.v.fill_(float('nan'))
    runner = ModelRunner(model, kv, manager)
    total = sum(pieces)
    seq = SequenceState(
        seq_id='A', prompt_tokens=[1 + (i * 3) % (model.vocab_size - 1)
                                   for i in range(total)],
        max_tokens=total+3,
    )
    outputs = []
    with torch.inference_mode():
        for chunk in pieces:
            output = runner.prefill_chunk(seq, seq.prompt_pos, chunk)
            outputs.append(output.detach().clone())
            seq.prompt_pos += chunk
    assert seq.length == total
    manager.check_invariants([seq])
    k, v = kv.gather_sequence(seq)
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    return torch.cat(outputs, dim=1), k, v, list(seq.block_table)


class TestPrefillFastPathV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        cls.model = MiniGPT(n_embd=32, n_head=4, vocab_size=64,
                            block_size=128, dropout=0.0, n_layer=2).eval()

    def tearDown(self):
        self.model.set_prefill_fastpath(False)

    def check_chunks(self, chunks):
        ref_logits, ref_k, ref_v, ref_blocks = run_chunks(
            self.model, fast=False, pieces=chunks)
        fast_logits, fast_k, fast_v, fast_blocks = run_chunks(
            self.model, fast=True, pieces=chunks)
        self.assertEqual(ref_blocks, fast_blocks)
        torch.testing.assert_close(fast_logits, ref_logits, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(fast_k, ref_k, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(fast_v, ref_v, atol=2e-5, rtol=2e-4)
        self.assertTrue(torch.equal(fast_logits.argmax(-1), ref_logits.argmax(-1)))

    def test_single_chunk_crosses_multiple_blocks(self):
        for size in (1, 5, 17, 31):
            with self.subTest(size=size):
                self.check_chunks([size])

    def test_chunked_prefill_nonzero_offset_fallback(self):
        for chunks in ([5, 7, 19], [1, 1, 9], [16, 1, 15]):
            with self.subTest(chunks=chunks):
                self.check_chunks(chunks)

    def test_nonzero_start_does_not_call_sdpa(self):
        # First chunk uses SDPA, second chunk must use the reference.
        self.model.set_prefill_fastpath(True)
        manager = BlockManager(num_blocks=12, block_size=4)
        pool = KVBlockPool(n_layer=2, num_blocks=12, block_size=4,
                           n_head=4, head_dim=8, dtype=torch.float32)
        seq = SequenceState(seq_id='A', prompt_tokens=list(range(1, 10)), max_tokens=12)
        runner = ModelRunner(self.model, pool, manager)
        with torch.inference_mode():
            runner.prefill_chunk(seq, 0, 4)
            seq.prompt_pos += 4
            with patch('minivllm.attention.F.scaled_dot_product_attention',
                       side_effect=AssertionError('SDPA invoked on nonzero start_pos')):
                runner.prefill_chunk(seq, 4, 5)
            seq.prompt_pos += 5
        self.assertEqual(seq.length, 9)

    def test_recompute_preemption_end_to_end_parity(self):
        # Constrained two-block workload forces actual automatic preemption.
        model = MiniGPT(n_embd=16, n_head=2, vocab_size=64,
                        block_size=96, dropout=0.0, n_layer=1).eval()

        def run(fast):
            model.set_prefill_fastpath(fast)
            bm = BlockManager(num_blocks=2, block_size=16)
            pool = KVBlockPool(n_layer=1, num_blocks=2, block_size=16,
                               n_head=2, head_dim=8, device='cpu')
            rt = MiniRuntime.__new__(MiniRuntime)
            rt.block_manager = bm
            rt.scheduler = Scheduler(bm, max_running=4,
                                     max_num_batched_tokens=64,
                                     preemption_policy='recompute')
            rt.runner = ModelRunner(model, pool, bm)
            rt.iteration = 0
            rt.records = []
            rt.sync_cuda_timing = False
            rt.verbose = False
            rt._kv_high_watermark = {}
            requests = [
                SequenceState(seq_id=key,
                              prompt_tokens=[(i * 7 + j + 1) % 63 + 1 for j in range(p)],
                              max_tokens=p + out)
                for i, (key, p, out) in enumerate([
                    ('A', 16, 8), ('B', 16, 8), ('C', 12, 6), ('D', 16, 5)
                ])
            ]
            for seq in requests:
                rt.add_request(seq)
            with torch.inference_mode():
                for _ in range(100):
                    if all(s.status is Status.FINISHED for s in requests):
                        break
                    rt.step()
                else:
                    raise AssertionError('recompute workload failed to finish')
            bm.check_invariants(requests)
            self.assertEqual(len(bm.free_blocks), 2)
            return {s.seq_id: s.generated_tokens for s in requests}, rt.metrics_summary()

        ref, ref_stats = run(False)
        fast, fast_stats = run(True)
        self.assertEqual(ref, fast)
        self.assertGreater(ref_stats['preemptions'], 0)
        self.assertGreater(ref_stats['repeated_kv_tokens'], 0)
        for field in ('iterations', 'output_tokens', 'preemptions', 'repeated_kv_tokens',
                      'prefill_tokens', 'decode_tokens'):
            self.assertEqual(ref_stats[field], fast_stats[field], field)

    def test_default_is_reference_and_flag_validation(self):
        fresh = MiniGPT(n_embd=16, n_head=2, vocab_size=20,
                        block_size=32, n_layer=1)
        self.assertFalse(fresh.blocks[0].attn.prefill_fastpath_enabled)
        with self.assertRaises(TypeError):
            fresh.set_prefill_fastpath('yes')


if __name__ == '__main__':
    unittest.main()
