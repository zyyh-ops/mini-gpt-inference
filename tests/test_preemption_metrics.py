"""CPU-only validation of the paired preemption benchmark harness.

Place under tests/, run with:
  PYTHONPATH=./minivllm python -m unittest discover \
    -s tests -p 'test_preemption_metrics.py' -v
"""
from types import SimpleNamespace
import unittest
import torch

from experiments.bench_preemption import run_once
from model import MiniGPT


class TestPreemptionMetrics(unittest.TestCase):
    def test_recompute_work_and_policy_equivalence(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        model = MiniGPT(n_embd=16, n_head=2, vocab_size=64,
                        block_size=256, dropout=0, n_layer=1).eval()
        args = SimpleNamespace(device='cpu', block_size=16,
            max_running=4, token_budget=64, layers=1, heads=2,
            dim=16, max_steps=200)

        low_cons = run_once(model, args, 'conservative', 2)
        low_rec = run_once(model, args, 'recompute', 2)
        high_rec = run_once(model, args, 'recompute', 8)
        self.assertEqual(low_cons['per_request_output'], low_rec['per_request_output'])
        self.assertEqual(low_cons['per_request_output'], high_rec['per_request_output'])
        self.assertEqual(low_cons['preemptions'], 0)
        self.assertEqual(low_cons['repeated_kv_tokens'], 0)
        self.assertGreater(low_rec['preemptions'], 0)
        self.assertGreater(low_rec['repeated_kv_tokens'], 0)
        self.assertEqual(high_rec['preemptions'], 0)
        self.assertEqual(high_rec['repeated_kv_tokens'], 0)
        self.assertEqual(low_cons['output_tokens'], 27)


if __name__ == '__main__':
    unittest.main()
