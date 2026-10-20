"""Pure contract checks for the release verifier (no full smoke recursion)."""
import copy
import io
import contextlib
import unittest

import torch

from scripts.release_smoke import verify_benchmark, verify_records
from experiments.bench_preemption import make_runtime, make_workload
from minivllm.model import MiniGPT


class ReleaseContractTests(unittest.TestCase):
    def test_benchmark_rejects_mismatched_outputs(self):
        outputs = {"A": [1, 2]}
        good = {"runs": [
            {"policy": "conservative", "iterations": 27, "preemptions": 0,
             "repeated_kv_tokens": 0, "output_tokens": 2, "per_request_output": outputs},
            {"policy": "recompute", "iterations": 27, "preemptions": 3,
             "repeated_kv_tokens": 44, "output_tokens": 2, "per_request_output": outputs},
        ]}
        expected = {"conservative": (27, 0, 0), "recompute": (27, 3, 44)}
        verify_benchmark(good, expected)
        bad = copy.deepcopy(good)
        bad["runs"][1]["per_request_output"] = {"A": [1, 3]}
        with self.assertRaises(AssertionError):
            verify_benchmark(bad, expected)

    def test_telemetry_rejects_bad_accounting(self):
        report = {"summary": {"iterations": 27, "preemptions": 3, "repeated_kv_tokens": 44,
                              "prefill_tokens": 104, "decode_tokens": 23,
                              "scheduled_tokens": 127, "output_tokens": 27},
                  "records": [{"iteration": i + 1, "preemptions": 3 if i == 1 else 0,
                               "repeated_kv_tokens": 44 if i == 8 else 0,
                               "prefill_tokens": 104 if i == 0 else 0,
                               "decode_tokens": 23 if i == 0 else 0,
                               "scheduled_tokens": 127 if i == 0 else 0,
                               "output_tokens": 27 if i == 0 else 0,
                               "used_blocks": 2, "free_blocks": 0}
                              for i in range(27)],
                  "metadata": {"request_outputs": {"A": [], "B": [], "C": [], "D": []}}}
        verify_records(report)
        report["records"][2]["used_blocks"] = 3
        with self.assertRaises(AssertionError):
            verify_records(report)

    def test_runtime_quiet_default_and_opt_in_logs(self):
        with torch.inference_mode():
            torch.set_num_threads(1)
            model = MiniGPT(n_embd=16, n_head=2, vocab_size=64,
                            block_size=256, n_layer=1, dropout=0.0).cpu().eval()
            class Cfg:
                device = 'cpu'
                block_size = 16
                layers = 1
                heads = 2
                dim = 16
                max_running = 4
                token_budget = 64
            rt = make_runtime(model, Cfg, 'conservative', 2)
            rt.add_request(make_workload('baseline')[0])
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rt.step()
            self.assertEqual(buf.getvalue(), '')
            self.assertEqual(len(rt.records), 1)
            rt.verbose = True
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rt.step()
            self.assertIn('Iteration 2', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
