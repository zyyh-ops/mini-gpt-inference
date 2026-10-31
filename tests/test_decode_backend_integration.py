"""CPU-safe backend-routing tests; real Triton runtime parity is GPU-only."""
from __future__ import annotations
from unittest import TestCase
from unittest.mock import patch
import torch

from minivllm.model import MiniGPT
from minivllm.kv_block_pool import KVBlockPool
from minivllm.decode_batch_metadata import DecodeBatchMetadata
from minivllm.paged_attention_batch import paged_attention_batch_online_v1


class TestDecodeBackendRouting(TestCase):
    def _model(self):
        torch.manual_seed(10)
        return MiniGPT(n_embd=16,n_head=4,vocab_size=32,block_size=32,
                       dropout=0.0,n_layer=2).eval()

    def test_default_and_explicit_validation(self):
        model=self._model()
        self.assertTrue(all(b.attn.decode_attention_backend=='reference' for b in model.blocks))
        with self.assertRaisesRegex(ValueError,'decode backend'):
            model.set_decode_attention_backend('unknown')
        model.set_decode_attention_backend('triton_v0')
        self.assertTrue(all(b.attn.decode_attention_backend=='triton_v0' for b in model.blocks))
        model.set_prefill_fastpath_mode('sdpa_v2')
        self.assertTrue(all(b.attn.decode_attention_backend=='triton_v0' for b in model.blocks))
        model.set_decode_attention_backend('reference')
        self.assertTrue(all(b.attn.decode_attention_backend=='reference' for b in model.blocks))

    def test_route_keeps_kv_write_before_attention_and_logits_identical(self):
        model=self._model()
        device='cpu'
        inputs=torch.tensor([[1],[2]],dtype=torch.long)
        metadata=DecodeBatchMetadata(
            positions=torch.tensor([0,0]),
            context_lengths=torch.tensor([1,1]),
            block_tables=torch.tensor([[1],[0]],dtype=torch.long),
            slot_mapping=torch.tensor([16,0],dtype=torch.long),
        )
        def make_pool():
            pool=KVBlockPool(n_layer=2,num_blocks=2,block_size=16,
                             n_head=4,head_dim=4,device=device)
            pool.k.fill_(float('nan'))
            pool.v.fill_(float('nan'))
            return pool
        pool0=make_pool();pool1=make_pool()
        with torch.inference_mode():
            model.set_decode_attention_backend('reference')
            expected=model.decode_batch(inputs,metadata,pool0)
            calls=[]
            def mock_triton(q,pool,block_tables,context_lengths,layer_idx):
                calls.append(layer_idx)
                self.assertTrue(torch.isfinite(pool.k[layer_idx,1,0]).all())
                self.assertTrue(torch.isfinite(pool.v[layer_idx,0,0]).all())
                return paged_attention_batch_online_v1(
                    q,pool,block_tables,context_lengths,layer_idx)
            model.set_decode_attention_backend('triton_v0')
            with patch('minivllm.triton_paged_decode.paged_attention_batch_triton_v0',
                       side_effect=mock_triton):
                actual=model.decode_batch(inputs,metadata,pool1)
        self.assertEqual(calls,[0,1])
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
        torch.testing.assert_close(pool0.k[:,1,0],pool1.k[:,1,0],atol=0,rtol=0)
        torch.testing.assert_close(pool0.v[:,0,0],pool1.v[:,0,0],atol=0,rtol=0)

    def test_unsupported_cpu_mode_never_silently_falls_back(self):
        model=self._model()
        model.set_decode_attention_backend('triton_v0')
        pool=KVBlockPool(n_layer=2,num_blocks=1,block_size=16,n_head=4,head_dim=4)
        meta=DecodeBatchMetadata(
            positions=torch.tensor([0]),context_lengths=torch.tensor([1]),
            block_tables=torch.tensor([[0]]),slot_mapping=torch.tensor([0]))
        with torch.inference_mode():
            with self.assertRaises((RuntimeError, ValueError)):
                model.decode_batch(torch.tensor([[1]]),meta,pool)
