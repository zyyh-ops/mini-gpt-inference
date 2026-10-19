import unittest

import torch
from kv_block_pool import KVBlockPool
from paged_attention_batch import (
    paged_attention_batch_reference,
    paged_attention_batch_online_v1,
)


class TestPagedAttentionBatch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.pool = KVBlockPool(
            n_layer=1,
            num_blocks=4,
            block_size=4,
            n_head=2,
            head_dim=4,
            device="cpu",
        )
        self.pool.k.copy_(torch.randn_like(self.pool.k))
        self.pool.v.copy_(torch.randn_like(self.pool.v))
        self.q = torch.randn(2, 2, 4)

    def compare(self, q, tables, lengths):
        ref = paged_attention_batch_reference(q, self.pool, tables, lengths, 0)
        out = paged_attention_batch_online_v1(q, self.pool, tables, lengths, 0)
        self.assertTrue(torch.isfinite(out).all().item())
        diff = (out - ref).abs().max().item()
        self.assertTrue(
            torch.allclose(out, ref, rtol=1e-5, atol=1e-5),
            msg=f"Attention mismatch: max_abs={diff:.6g}",
        )

    def test_single_block_scaling(self):
        self.compare(self.q[:1], torch.tensor([[2]]), torch.tensor([3]))

    def test_cross_block_online_softmax(self):
        self.compare(self.q[:1], torch.tensor([[0, 1]]), torch.tensor([6]))

    def test_variable_lengths_with_padded_block_tables(self):
        self.compare(
            self.q,
            torch.tensor([[0, 1], [2, -1]]),
            torch.tensor([6, 3]),
        )


if __name__ == "__main__":
    unittest.main()
