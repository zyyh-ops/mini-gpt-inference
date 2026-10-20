"""Guard against uninitialized padded KV values leaking into Attention."""
import unittest

import torch
from minivllm.kv_block_pool import KVBlockPool
from minivllm.paged_attention_batch import (
    paged_attention_batch_reference, paged_attention_batch_online_v1,
)


class TestMaskedKVPadding(unittest.TestCase):
    def test_nan_in_unwritten_kv_slots_is_never_read(self):
        torch.manual_seed(11)
        pool = KVBlockPool(n_layer=1, num_blocks=3, block_size=4,
                           n_head=2, head_dim=8)
        pool.k.fill_(float('nan'))
        pool.v.fill_(float('nan'))
        # Mixed batch: request 0 has 2 tokens in one block,
        # request 1 has 5 tokens across two blocks.
        pool.k[0, 0, :2] = torch.randn(2, 2, 8)
        pool.v[0, 0, :2] = torch.randn(2, 2, 8)
        pool.k[0, 1, :4] = torch.randn(4, 2, 8)
        pool.v[0, 1, :4] = torch.randn(4, 2, 8)
        pool.k[0, 2, :1] = torch.randn(1, 2, 8)
        pool.v[0, 2, :1] = torch.randn(1, 2, 8)
        q = torch.randn(2, 2, 8)
        tables = torch.tensor([[0, -1], [1, 2]])
        lengths = torch.tensor([2, 5])
        reference = paged_attention_batch_reference(q, pool, tables, lengths, 0)
        actual = paged_attention_batch_online_v1(q, pool, tables, lengths, 0)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
