"""CPU-only accounting tests for MiniRuntime.IterationRecord.

These use real Scheduler / BlockManager with a deterministic fake runner and
compare Runtime-owned totals with the independent benchmark CountingRunner.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.bench_preemption import CountingRunner
from minivllm.block_manager import BlockManager
from minivllm.mini_runtime import IterationRecord, MiniRuntime
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


class FakeRunner:
    def __init__(self, manager):
        self.manager = manager

    def prefill_chunk(self, seq, start_pos, num_tokens):
        assert start_pos == seq.length == seq.prompt_pos
        self.manager.reserve_slots(seq, num_tokens)
        seq.length += num_tokens
        return torch.zeros(1, num_tokens, 4)

    def decode_batch(self, seqs):
        for seq in seqs:
            self.manager.reserve_append_slot(seq)
            seq.length += 1
        return torch.zeros(len(seqs), 4), [0] * len(seqs)


def make_runtime(*, policy="conservative", num_blocks=4, block_size=2,
                 max_running=1, budget=2, count_runner=False):
    rt = MiniRuntime.__new__(MiniRuntime)
    rt.block_manager = BlockManager(num_blocks, block_size)
    rt.scheduler = Scheduler(rt.block_manager, max_running, budget, policy)
    base = FakeRunner(rt.block_manager)
    rt.runner = CountingRunner(base) if count_runner else base
    rt.iteration = 0
    rt.records = []
    return rt


def request(name, prompt_len, output_len):
    return SequenceState(
        seq_id=name,
        prompt_tokens=list(range(1, prompt_len + 1)),
        max_tokens=prompt_len + output_len,
    )


class TestRuntimeMetrics(unittest.TestCase):
    def test_chunked_prefill_and_finish_snapshots(self):
        rt = make_runtime()
        a = request("A", 3, 2)
        rt.add_request(a)
        for _ in range(3):
            rt.step()

        self.assertEqual(len(rt.records), 3)
        self.assertTrue(all(isinstance(r, IterationRecord) for r in rt.records))
        self.assertEqual([r.prefill_tokens for r in rt.records], [2, 1, 0])
        self.assertEqual([r.decode_tokens for r in rt.records], [0, 0, 1])
        self.assertEqual([r.output_tokens for r in rt.records], [0, 1, 1])
        self.assertEqual([r.finished_requests for r in rt.records], [0, 0, 1])
        self.assertEqual([r.prefilling for r in rt.records], [1, 0, 0])
        self.assertEqual([r.running for r in rt.records], [0, 1, 0])
        self.assertEqual([r.used_blocks for r in rt.records], [1, 2, 0])
        self.assertEqual([r.free_blocks for r in rt.records], [3, 2, 4])
        self.assertEqual([r.preemptions for r in rt.records], [0, 0, 0])
        self.assertEqual([r.repeated_kv_tokens for r in rt.records], [0, 0, 0])
        self.assertEqual(rt.metrics_summary()["output_tokens"], 2)
        for r in rt.records:
            self.assertGreaterEqual(r.schedule_ms, 0.0)
            self.assertGreaterEqual(r.prefill_ms, 0.0)
            self.assertGreaterEqual(r.decode_ms, 0.0)
            self.assertGreaterEqual(r.iteration_ms, r.schedule_ms)
            self.assertAlmostEqual(r.kv_utilization, r.used_blocks / 4)
            self.assertEqual(r.scheduled_tokens, r.prefill_tokens + r.decode_tokens)

    def test_preemption_and_recompute_match_independent_counter(self):
        rt = make_runtime(policy="recompute", num_blocks=2, block_size=4,
                          max_running=2, budget=8, count_runner=True)
        seqs = [request("A", 4, 3), request("B", 4, 3)]
        for seq in seqs:
            rt.add_request(seq)
        for _ in range(30):
            if all(seq.status == Status.FINISHED for seq in seqs):
                break
            rt.step()
        else:
            self.fail("automatic preemption stalled")
        summary = rt.metrics_summary()
        self.assertGreater(summary["preemptions"], 0)
        self.assertGreater(summary["repeated_kv_tokens"], 0)
        self.assertEqual(summary["preemptions"], rt.scheduler.num_preemptions)
        self.assertEqual(summary["repeated_kv_tokens"], rt.runner.repeated_kv_tokens)
        self.assertEqual(summary["prefill_tokens"], rt.runner.total_prefill_tokens)
        self.assertEqual(summary["decode_tokens"], rt.runner.total_decode_tokens)
        self.assertEqual(summary["output_tokens"], 6)
        self.assertEqual([s.status for s in seqs], [Status.FINISHED, Status.FINISHED])
        self.assertEqual(rt.block_manager.block_owner, {})

    def test_partial_prefill_preemption_counts_replayed_tokens(self):
        rt = make_runtime(policy="recompute", num_blocks=2, block_size=4,
                          max_running=2, budget=8, count_runner=True)
        a, b = request("A", 4, 3), request("B", 6, 1)
        rt.add_request(a)
        rt.add_request(b)
        rt.step()
        self.assertEqual(b.prompt_pos, 4)
        self.assertEqual(b.status, Status.PREFILLING)
        for _ in range(30):
            if a.status == b.status == Status.FINISHED:
                break
            rt.step()
        else:
            self.fail("partial prefill recovery stalled")
        summary = rt.metrics_summary()
        self.assertGreater(b.num_preemptions, 0)
        self.assertGreaterEqual(summary["repeated_kv_tokens"], 4)
        self.assertEqual(summary["repeated_kv_tokens"], rt.runner.repeated_kv_tokens)
        self.assertEqual(summary["preemptions"], rt.scheduler.num_preemptions)

    def test_empty_step_and_json_export(self):
        rt = make_runtime()
        rt.step()
        r = rt.records[0]
        self.assertEqual(r.scheduled_tokens, 0)
        self.assertEqual(r.output_tokens, 0)
        self.assertEqual(r.used_blocks, 0)
        self.assertEqual(rt.metrics_summary()["preemptions"], 0)
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "nested" / "metrics.json"
            rt.export_records_json(dest)
            data = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(data["timing_mode"], "host_wall")
            self.assertEqual(len(data["records"]), 1)
            self.assertEqual(data["records"][0]["iteration"], 1)
            self.assertEqual(data["summary"]["iterations"], 1)


if __name__ == "__main__":
    unittest.main()
