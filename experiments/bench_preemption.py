#!/usr/bin/env python3
"""Paired conservative-vs-recompute experiments for the educational MiniRuntime.

Place in project root as experiments/bench_preemption.py.
Usage (from repo root):
  python -m experiments.bench_preemption --device cpu \
      --num-blocks 2,4,8 --repeats 1 --output-json artifacts/preemption/cpu_smoke.json
  python -m experiments.bench_preemption --device cuda \
      --num-blocks 2,4,8 --repeats 3 --output-json artifacts/preemption/gpu.json

Not a vLLM serving benchmark. All requests arrive together, greedy decoding;
per-iteration device synchronization measures a synchronous mini-runtime loop.
"""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import statistics
import time

import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.mini_runtime import MiniRuntime
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


class DiscardOutput(io.TextIOBase):
    def write(self, text):
        return len(text)


class CountingRunner:
    """Count repeated KV-token executions, including partial-prefill restarts."""

    def __init__(self, inner):
        self.inner = inner
        self.cache_high_watermark = {}
        self.repeated_kv_tokens = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0

    def _track(self, seq, start, end):
        prev = self.cache_high_watermark.get(seq.seq_id, 0)
        # Reprocessing a token position already executed at least once counts
        # as recompute work; a normal first-time token does not.
        self.repeated_kv_tokens += max(0, min(end, prev) - start)
        self.cache_high_watermark[seq.seq_id] = max(prev, end)

    def prefill_chunk(self, seq, start_pos, num_tokens):
        result = self.inner.prefill_chunk(seq, start_pos, num_tokens)
        self._track(seq, start_pos, start_pos + num_tokens)
        self.total_prefill_tokens += num_tokens
        return result

    def decode_batch(self, sequences):
        starts = [(seq, seq.length) for seq in sequences]
        result = self.inner.decode_batch(sequences)
        for seq, start in starts:
            self._track(seq, start, start + 1)
        self.total_decode_tokens += len(sequences)
        return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, pct):
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lo = math.floor(k)
    hi = math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def make_workload(profile="baseline"):
    # baseline: contention causes KV preemption and duplicated computation.
    # mixed_short: optimistic admission can benefit short requests without preemption.
    profiles = {
        "baseline": [("A", 16, 8), ("B", 16, 8), ("C", 12, 6), ("D", 16, 5)],
        "mixed_short": [("A", 4, 32), ("B", 4, 2), ("C", 4, 2), ("D", 4, 2)],
        # Deliberate but cheap preemption: short C is evicted once.
        "preempt_benefit": [("A", 4, 32), ("B", 16, 8), ("C", 4, 2), ("D", 4, 2)],
    }
    entries = profiles[profile]
    return [SequenceState(seq_id=name,
            prompt_tokens=[((i + j) % 31) + 1 for j in range(prompt)],
            max_tokens=prompt + output)
            for i, (name, prompt, output) in enumerate(entries)]


def make_runtime(model, cfg, policy, num_blocks):
    manager = BlockManager(num_blocks=num_blocks, block_size=cfg.block_size)
    sched = Scheduler(manager, max_running=cfg.max_running,
                      max_num_batched_tokens=cfg.token_budget,
                      preemption_policy=policy)
    kv_pool = KVBlockPool(n_layer=cfg.layers, num_blocks=num_blocks,
        block_size=cfg.block_size, n_head=cfg.heads,
        head_dim=cfg.dim // cfg.heads, device=cfg.device)
    rt = MiniRuntime.__new__(MiniRuntime)
    rt.block_manager, rt.scheduler = manager, sched
    rt.runner = CountingRunner(ModelRunner(model, kv_pool, manager))
    rt.iteration, rt.records = 0, []
    rt.sync_cuda_timing = getattr(cfg, "sync_cuda_timing", False)
    return rt


def run_once(model, cfg, policy, num_blocks):
    device = torch.device(cfg.device)
    rt = make_runtime(model, cfg, policy, num_blocks)
    seqs = make_workload(getattr(cfg, "workload", "baseline"))
    for seq in seqs:
        rt.add_request(seq)
    initial_counts = {s.seq_id: 0 for s in seqs}
    ttft_ms, e2e_ms = {}, {}
    peak_used_blocks = 0
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_clock = time.perf_counter()
    with torch.inference_mode(), contextlib.redirect_stdout(DiscardOutput()):
        for _ in range(cfg.max_steps):
            if all(s.status == Status.FINISHED for s in seqs):
                break
            rt.step()
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start_clock) * 1e3
            peak_used_blocks = max(peak_used_blocks, len(rt.block_manager.block_owner))
            rt.block_manager.check_invariants(seqs)
            for seq in seqs:
                if seq.seq_id not in ttft_ms and seq.generated_tokens:
                    ttft_ms[seq.seq_id] = elapsed_ms
                if seq.seq_id not in e2e_ms and seq.status == Status.FINISHED:
                    e2e_ms[seq.seq_id] = elapsed_ms
                assert len(seq.generated_tokens) >= initial_counts[seq.seq_id]
                initial_counts[seq.seq_id] = len(seq.generated_tokens)
        else:
            raise RuntimeError(f"No convergence: policy={policy}, blocks={num_blocks}, "
                f"statuses={[(s.seq_id, s.status.value, len(s.generated_tokens)) for s in seqs]}")
    synchronize(device)
    duration_ms = (time.perf_counter() - start_clock) * 1e3
    if not all(s.status == Status.FINISHED for s in seqs):
        raise AssertionError("some requests unfinished")
    assert not rt.block_manager.block_owner
    assert len(rt.block_manager.free_blocks) == num_blocks
    # Reconcile built-in Runtime instrumentation against an independent
    # counting runner, outside the timed execution loop.
    accounting = rt.metrics_summary()
    assert accounting["preemptions"] == rt.scheduler.num_preemptions
    assert accounting["repeated_kv_tokens"] == rt.runner.repeated_kv_tokens
    assert accounting["prefill_tokens"] == rt.runner.total_prefill_tokens
    assert accounting["decode_tokens"] == rt.runner.total_decode_tokens
    expected = sum(s.max_tokens - s.prompt_length for s in seqs)
    actual = sum(len(s.generated_tokens) for s in seqs)
    assert actual == expected, (actual, expected)
    result = {
        "policy": policy, "num_blocks": num_blocks, "block_size": cfg.block_size,
        "iterations": rt.iteration, "total_wall_ms": round(duration_ms, 3),
        "output_tokens": actual,
        "throughput_output_tok_s": round(actual / (duration_ms / 1e3), 3),
        "avg_ttft_ms": round(statistics.mean(ttft_ms.values()), 3),
        "avg_e2e_ms": round(statistics.mean(e2e_ms.values()), 3),
        "p95_e2e_ms": round(percentile(list(e2e_ms.values()), 95), 3),
        "preemptions": rt.scheduler.num_preemptions,
        "per_request_preemptions": {s.seq_id: s.num_preemptions for s in seqs},
        "repeated_kv_tokens": rt.runner.repeated_kv_tokens,
        "prefill_token_executions": rt.runner.total_prefill_tokens,
        "decode_token_executions": rt.runner.total_decode_tokens,
        "peak_used_kv_blocks": peak_used_blocks,
        "peak_kv_utilization": round(peak_used_blocks / num_blocks, 4),
        "per_request_output": {s.seq_id: list(s.generated_tokens) for s in seqs},
        "per_request_ttft_ms": {k: round(v, 3) for k, v in ttft_ms.items()},
        "per_request_e2e_ms": {k: round(v, 3) for k, v in e2e_ms.items()},
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workload', choices=('baseline','mixed_short','preempt_benefit'), default='baseline')
    ap.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    ap.add_argument('--sync-cuda-timing', action='store_true',
                    help='Synchronize GPU at Runtime stage boundaries (changes timing behavior)')
    ap.add_argument('--num-blocks', default='2,4,8',
                    help='Comma-separated KV Pool block capacities')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--warmups', type=int, default=1,
                    help='Unmeasured complete warmup pairs for each KV capacity')
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--max-running', type=int, default=4)
    ap.add_argument('--token-budget', type=int, default=64)
    ap.add_argument('--dim', type=int, default=16)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=200)
    ap.add_argument('--output-json', default='artifacts/preemption/paired_results.json')
    cfg = ap.parse_args()
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA device unavailable; use --device cpu for correctness smoke test')
    if cfg.dim % cfg.heads != 0 or cfg.block_size < 1:
        ap.error('dim must be divisible by heads and block-size positive')
    if cfg.repeats < 1 or cfg.warmups < 0:
        ap.error('repeats must be >= 1 and warmups >= 0')
    caps = [int(x) for x in cfg.num_blocks.split(',')]
    if any(x <= 0 for x in caps):
        ap.error('all num-blocks must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = MiniGPT(n_embd=cfg.dim, n_head=cfg.heads, vocab_size=64,
                    block_size=256, dropout=0.0, n_layer=cfg.layers).to(cfg.device).eval()
    runs = []
    # Alternate first policy to reduce order effects across repeats.
    for cap in caps:
        for _ in range(cfg.warmups):
            for warm_policy in ('conservative','recompute'):
                run_once(model, cfg, warm_policy, cap)
        for rep in range(cfg.repeats):
            order = ('conservative','recompute') if rep % 2 == 0 else ('recompute','conservative')
            outputs = {}
            for policy in order:
                result = run_once(model, cfg, policy, cap)
                result['repeat'] = rep
                runs.append(result)
                outputs[policy] = result['per_request_output']
                print(f"blocks={cap:<2} repeat={rep} policy={policy:<12} "
                      f"time={result['total_wall_ms']:.1f}ms "
                      f"preempt={result['preemptions']} "
                      f"replay={result['repeated_kv_tokens']} "
                      f"avgE2E={result['avg_e2e_ms']:.1f}ms")
            if outputs['conservative'] != outputs['recompute']:
                raise AssertionError('Token mismatch between two policies')
    report = {
        'definition': 'All-at-once requests; one synchronized runtime step; greedy decode; educational mini-runtime only',
        'workload_profile': cfg.workload,
        'device': cfg.device, 'torch': torch.__version__,
        'model': {'dim':cfg.dim, 'heads':cfg.heads, 'layers':cfg.layers, 'dtype':'float32'},
        'workload': [{'id':s.seq_id,'prompt_len':s.prompt_length,
                      'output_len':s.max_tokens-s.prompt_length} for s in make_workload(cfg.workload)],
        'block_size':cfg.block_size,'max_running':cfg.max_running,
        'token_budget':cfg.token_budget,'repeats':cfg.repeats,'warmups':cfg.warmups,
        'runs':runs,
    }
    dest=Path(cfg.output_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PASS policy token equality; saved {dest}")


if __name__ == '__main__':
    main()
