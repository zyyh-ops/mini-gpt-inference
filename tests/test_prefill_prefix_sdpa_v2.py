"""Prefix-aware Chunked SDPA v2 correctness and lifecycle coverage."""
import unittest
from unittest.mock import patch
import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.mini_runtime import MiniRuntime
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


def run_chunks(model, pieces, mode, *, block_size=4):
    model.set_prefill_fastpath_mode(mode)
    model.eval()
    n_blocks = 64
    bm = BlockManager(n_blocks, block_size)
    # Force non-monotonic physical allocation to catch logical/physical mixups.
    perm = list(range(n_blocks))[::2] + list(range(n_blocks))[1::2]
    bm.free_blocks = list(reversed(perm))
    pool = KVBlockPool(model.n_layer, n_blocks, block_size,
                       model.blocks[0].attn.n_head, model.blocks[0].attn.head_size,
                       device=next(model.parameters()).device)
    pool.k.fill_(float('nan'))
    pool.v.fill_(float('nan'))
    runner = ModelRunner(model, pool, bm)
    prompt = [1 + i * 7 % (model.vocab_size - 1) for i in range(sum(pieces))]
    seq = SequenceState(seq_id='seq', prompt_tokens=prompt, max_tokens=len(prompt)+3)
    outputs=[]
    with torch.inference_mode():
        for length in pieces:
            outputs.append(runner.prefill_chunk(seq, seq.prompt_pos, length).clone())
            seq.prompt_pos += length
    bm.check_invariants([seq])
    k,v = pool.gather_sequence(seq)
    return torch.cat(outputs, dim=1),k,v,list(seq.block_table)


class TestPrefixSDPAV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        torch.set_num_threads(1)
        cls.model = MiniGPT(n_embd=32, n_head=4, vocab_size=64,
                            block_size=192, dropout=0.0, n_layer=2).eval()

    def tearDown(self):
        self.model.set_prefill_fastpath_mode('reference')

    def test_offset_mask_with_irregular_chunks_and_paged_blocks(self):
        for pieces in ([5,7,19], [1,1,9], [16,1,15], [8,8,8,8],
                       [3,5,4,16,11], [16,16,16,16], [64,32,32]):
            with self.subTest(pieces=pieces):
                ref=run_chunks(self.model, pieces, 'reference')
                v1=run_chunks(self.model, pieces, 'sdpa_v1')
                v2=run_chunks(self.model, pieces, 'sdpa_v2')
                self.assertEqual(ref[3],v2[3])
                for got in (v1,v2):
                    torch.testing.assert_close(got[0],ref[0],atol=2e-5,rtol=2e-4)
                    torch.testing.assert_close(got[1],ref[1],atol=2e-5,rtol=2e-4)
                    torch.testing.assert_close(got[2],ref[2],atol=2e-5,rtol=2e-4)
                    self.assertTrue(torch.equal(got[0].argmax(-1), ref[0].argmax(-1)))

    def test_later_chunk_uses_sdpa(self):
        original = torch.nn.functional.scaled_dot_product_attention
        count = 0
        def instrument(*args,**kwargs):
            nonlocal count
            count += 1
            return original(*args,**kwargs)
        with patch('minivllm.attention.F.scaled_dot_product_attention',side_effect=instrument):
            run_chunks(self.model,[5,9,17],'sdpa_v2')
        self.assertEqual(count, self.model.n_layer * 3)

    def test_mode_selection_compatible_with_v1(self):
        self.model.set_prefill_fastpath_mode('sdpa_v2')
        self.assertTrue(self.model.blocks[0].attn.prefill_prefix_fastpath_enabled)
        self.model.set_prefill_fastpath(True)
        self.assertFalse(self.model.blocks[0].attn.prefill_prefix_fastpath_enabled)
        self.assertTrue(self.model.blocks[0].attn.prefill_fastpath_enabled)
        with self.assertRaises(ValueError):
            self.model.set_prefill_fastpath_mode('bad')

    def test_recompute_preemption_e2e_parity(self):
        model = MiniGPT(n_embd=16, n_head=2, vocab_size=64,
                        block_size=96, dropout=0.0, n_layer=1).eval()
        def run(mode):
            model.set_prefill_fastpath_mode(mode)
            bm = BlockManager(num_blocks=2, block_size=16)
            pool = KVBlockPool(n_layer=1, num_blocks=2, block_size=16,
                               n_head=2, head_dim=8, device='cpu')
            rt = MiniRuntime.__new__(MiniRuntime)
            rt.block_manager = bm
            rt.scheduler = Scheduler(bm,max_running=4,max_num_batched_tokens=8,
                                     preemption_policy='recompute')
            rt.runner = ModelRunner(model,pool,bm)
            rt.iteration=0;rt.records=[];rt.sync_cuda_timing=False;rt.verbose=False;rt._kv_high_watermark={}
            requests = [SequenceState(seq_id=key,
                          prompt_tokens=[(i*7+j+1)%63+1 for j in range(p)],
                          max_tokens=p+out)
                        for i,(key,p,out) in enumerate(
                            [('A',16,8),('B',16,8),('C',12,6),('D',16,5)])]
            for seq in requests:rt.add_request(seq)
            with torch.inference_mode():
                for _ in range(150):
                    if all(s.status is Status.FINISHED for s in requests):break
                    rt.step()
                else:raise AssertionError('not finished')
            bm.check_invariants(requests)
            self.assertEqual(len(bm.free_blocks),2)
            return {s.seq_id:s.generated_tokens for s in requests},rt.metrics_summary()
        ref,rs=run('reference')
        v1,vs=run('sdpa_v1')
        v2,ws=run('sdpa_v2')
        self.assertGreater(rs['preemptions'],0)
        self.assertGreater(rs['repeated_kv_tokens'],0)
        self.assertEqual(ref,v1)
        self.assertEqual(ref,v2)
        for key in ('iterations','preemptions','repeated_kv_tokens','prefill_tokens','decode_tokens'):
            self.assertEqual(rs[key],ws[key])

if __name__ == '__main__':unittest.main()
