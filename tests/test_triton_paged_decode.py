"""CPU semantic fixture tests plus optional real CUDA/Triton fused-kernel tests."""
from __future__ import annotations

import unittest

import torch
from experiments.gpu_day6_fused_decode_ab import build_case, dense_gold
from minivllm.paged_attention_batch import paged_attention_batch_online_v1
from minivllm.triton_paged_decode import paged_attention_batch_triton_v0, triton


class TestPagedDecodeFixtureCPU(unittest.TestCase):
    def test_dense_matches_pytorch_online_for_ragged_blocks(self):
        for bs, context in [(8, 17), (16, 1), (16, 15), (16, 16),
                            (16, 17), (16, 31), (16, 65), (16, 129)]:
            with self.subTest(block_size=bs, context=context):
                q,pool,tables,lens,layer = build_case(
                    torch.device('cpu'), 3, context, 4, 8, bs, 'mixed')
                a = dense_gold(q,pool,tables,lens,layer)
                b = paged_attention_batch_online_v1(q,pool,tables,lens,layer)
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
                self.assertTrue(bool(torch.isfinite(a).all()))

    def test_nontrivial_physical_mapping_and_nan_padding(self):
        q,pool,tables,lens,layer = build_case(
            torch.device('cpu'),4,65,8,32,16,'mixed')
        self.assertEqual(tables.shape[0],4)
        self.assertTrue(bool((tables == -1).any()))
        # Padding must NOT be read; unused physical slots intentionally NaN.
        self.assertTrue(bool(torch.isnan(pool.k).any()))
        self.assertNotEqual(q.stride(-1),1)
        expected = dense_gold(q,pool,tables,lens,layer)
        self.assertTrue(bool(torch.isfinite(expected).all()))

    def test_import_does_not_require_triton_for_cpu_tests(self):
        self.assertTrue(callable(paged_attention_batch_triton_v0))
        if triton is None:
            q,pool,tables,lens,layer = build_case(torch.device('cpu'),1,16,2,4)
            with self.assertRaisesRegex(RuntimeError,'Triton is not installed'):
                paged_attention_batch_triton_v0(q,pool,tables,lens,layer)


@unittest.skipUnless(torch.cuda.is_available() and triton is not None,
                     'requires CUDA and Triton')
class TestPagedDecodeKernelCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    def test_kernel_matches_independent_oracle(self):
        from experiments.gpu_day6_fused_decode_ab import compare
        dev=torch.device('cuda')
        for bs,length,batch,dim in [
            (16,1,1,32),(16,15,2,32),(16,16,2,32),
            (16,17,4,32),(16,65,4,32),(16,129,4,32),
            (16,257,2,32),(8,37,3,16),(32,129,3,64),
            (16,33,2,48),
        ]:
            with self.subTest(block_size=bs,length=length,batch=batch,dim=dim):
                q,pool,tables,lens,layer=build_case(
                    dev,batch,length,8,dim,bs,'mixed',scale=0.3)
                result=compare(q,pool,tables,lens,layer,atol=2e-4,rtol=2e-4)
                self.assertTrue(result['finite'])

    def test_large_logits_stable(self):
        from experiments.gpu_day6_fused_decode_ab import compare
        q,pool,tables,lens,layer=build_case(
            torch.device('cuda'),2,65,4,32,16,'mixed',scale=8.0)
        result=compare(q,pool,tables,lens,layer,atol=5e-4,rtol=5e-4)
        self.assertTrue(result['finite'])


if __name__=='__main__':
    unittest.main()
