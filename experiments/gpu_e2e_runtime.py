"""Day 4: end-to-end MiniRuntime closed-workload GPU benchmark.

Put in experiments/gpu_e2e_runtime.py, run with
    python -m experiments.gpu_e2e_runtime --device cuda \
        --batch-sizes 1,4,8,16 --prompt-lengths 32,128,256 \
        --output-len 32 --kv-factors 1.0 --warmups 1 --repeats 5 \
        --output-json artifacts/gpu/e2e_runtime.json

All requests arrive at t=0; NOT a network serving/arrival-rate benchmark.
The timed interval covers the whole Prefill -> Decode -> finish lifecycle,
including Python scheduling, PyTorch eager operations, sampling and host work.
Model/Pool construction, warmups and correctness assertions are outside it.
CUDA is synchronized at the boundaries only. Calls to .item()/.tolist() inside
MiniRuntime can *implicitly* synchronize. Per-request timestamps are when
completed outputs become visible to the host, not isolated GPU kernel timing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.mini_runtime import MiniRuntime
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceState, Status


def csv_ints(text: str) -> list[int]:
    try:
        xs = [int(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers") from exc
    if not xs or any(x <= 0 for x in xs):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return xs


def csv_floats(text: str) -> list[float]:
    try:
        xs = [float(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers") from exc
    if not xs or any(not math.isfinite(x) or x <= 0 for x in xs):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return xs


def percentile(xs: list[float], p: float) -> float:
    vals = sorted(xs)
    idx = (len(vals) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def stats(xs: list[float]) -> dict:
    return {
        "median": round(statistics.median(xs), 6),
        "mean": round(statistics.mean(xs), 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
        "p95": round(percentile(xs, 95), 6),
    }


def make_sequences(batch: int, prompt_len: int, output_len: int, vocab: int) -> list[SequenceState]:
    return [SequenceState(
        seq_id=f"req_{i:02d}",
        prompt_tokens=[1 + ((i * 31 + pos * 7) % (vocab - 1)) for pos in range(prompt_len)],
        max_tokens=prompt_len + output_len,
    ) for i in range(batch)]


def required_blocks(batch: int, prompt_len: int, output_len: int, block_size: int) -> int:
    # Final sampled token is not appended to KV; length required <= max_tokens - 1.
    return batch * math.ceil((prompt_len + output_len - 1) / block_size)


def make_runtime(model, device, *, batch, prompt_len, output_len, block_size,
                 token_budget, capacity_factor, policy, heads, dim):
    max_per_req = math.ceil((prompt_len + output_len - 1) / block_size)
    full = required_blocks(batch, prompt_len, output_len, block_size)
    capacity = max(max_per_req, math.ceil(full * capacity_factor))

    manager = BlockManager(num_blocks=capacity, block_size=block_size)
    scheduler = Scheduler(manager, max_running=batch,
                          max_num_batched_tokens=token_budget,
                          preemption_policy=policy)
    pool = KVBlockPool(
        n_layer=model.n_layer, num_blocks=capacity, block_size=block_size,
        n_head=heads, head_dim=dim // heads,
        device=device, dtype=next(model.parameters()).dtype,
    )
    rt = MiniRuntime.__new__(MiniRuntime)
    rt.block_manager = manager
    rt.scheduler = scheduler
    rt.runner = ModelRunner(model, pool, manager)
    rt.iteration = 0
    rt.records = []
    rt.sync_cuda_timing = False
    rt.verbose = False
    rt._kv_high_watermark = {}
    sequences = make_sequences(batch, prompt_len, output_len, model.vocab_size)
    for seq in sequences:
        rt.add_request(seq)
    return rt, sequences, capacity


def run_once(model, args, device, *, batch, prompt_len, factor, policy,
             with_trace=False):
    rt, seqs, num_blocks = make_runtime(
        model, device, batch=batch, prompt_len=prompt_len,
        output_len=args.output_len, block_size=args.block_size,
        token_budget=args.token_budget, capacity_factor=factor, policy=policy,
        heads=args.heads, dim=args.dim,
    )
    previous_counts = {s.seq_id: 0 for s in seqs}
    first_seen = {}
    finished_at = {}
    output_seen_at = defaultdict(list)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        steady_allocated = torch.cuda.memory_allocated(device)
        steady_reserved = torch.cuda.memory_reserved(device)
    else:
        steady_allocated = None
        steady_reserved = None

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.max_steps):
            if all(s.status == Status.FINISHED for s in seqs):
                break
            rt.step()
            now_ms = (time.perf_counter() - started) * 1e3
            for s in seqs:
                n = len(s.generated_tokens)
                prev = previous_counts[s.seq_id]
                if n < prev or n - prev > 1:
                    raise AssertionError("unexpected generated token count evolution")
                if n > prev:
                    output_seen_at[s.seq_id].append(now_ms)
                    if s.seq_id not in first_seen:
                        first_seen[s.seq_id] = now_ms
                if s.status == Status.FINISHED and s.seq_id not in finished_at:
                    finished_at[s.seq_id] = now_ms
                previous_counts[s.seq_id] = n
        else:
            raise RuntimeError(f"max_steps exceeded: B={batch}, prompt={prompt_len}, "
                               f"factor={factor}, policy={policy}, "
                               f"states={[(s.seq_id, s.status.value) for s in seqs]}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1e3
    # End timing region: all assertions and structural validation below are untimed.
    if not all(s.status == Status.FINISHED for s in seqs):
        raise AssertionError("some requests remain unfinished")
    if rt.block_manager.block_owner or len(rt.block_manager.free_blocks) != num_blocks:
        raise AssertionError("KV blocks were not all released")
    rt.block_manager.check_invariants(seqs)
    for s in seqs:
        if len(s.generated_tokens) != args.output_len or len(output_seen_at[s.seq_id]) != args.output_len:
            raise AssertionError(f"request {s.seq_id} generated an incorrect number of tokens")
    total_output = sum(len(s.generated_tokens) for s in seqs)
    summary = rt.metrics_summary()
    if total_output != summary["output_tokens"]:
        raise AssertionError("Runtime native output token count mismatch")
    if summary["iterations"] != rt.iteration:
        raise AssertionError("Runtime iteration count mismatch")

    ttfts = list(first_seen.values())
    e2es = list(finished_at.values())
    itls = [times[i] - times[i - 1]
            for times in output_seen_at.values() for i in range(1, len(times))]
    tpots = {key: (times[-1] - times[0]) / (len(times) - 1)
             for key, times in output_seen_at.items() if len(times) > 1}
    result = {
        "batch_size": batch,
        "prompt_len": prompt_len,
        "output_len": args.output_len,
        "num_blocks": num_blocks,
        "kv_capacity_factor_requested": factor,
        "policy": policy,
        "iterations": rt.iteration,
        "total_wall_ms": round(elapsed_ms, 6),
        "output_tokens": total_output,
        "output_tok_s": round(total_output * 1000 / elapsed_ms, 6),
        "avg_ttft_ms": round(statistics.mean(ttfts), 6),
        "avg_e2e_ms": round(statistics.mean(e2es), 6),
        "avg_tpot_ms": round(statistics.mean(tpots.values()), 6) if tpots else None,
        "mean_inter_token_latency_ms": round(statistics.mean(itls), 6) if itls else None,
        "p95_inter_token_latency_ms": round(percentile(itls, 95), 6) if itls else None,
        "p95_request_e2e_ms": round(percentile(e2es, 95), 6),
        "preemptions": summary["preemptions"],
        "repeated_kv_tokens": summary["repeated_kv_tokens"],
        "prefill_token_executions": summary["prefill_tokens"],
        "decode_token_executions": summary["decode_tokens"],
        "kv_peak_used_blocks_at_step_end": summary["peak_used_blocks_at_step_end"],
        "kv_mean_utilization_at_step_end": round(summary["average_kv_utilization_at_step_end"], 6),
        "kv_pool_bytes_theoretical": int(2 * model.n_layer * num_blocks * args.block_size
                                         * args.dim * next(model.parameters()).element_size()),
        "per_request_ttft_ms": {s.seq_id: round(first_seen[s.seq_id], 6) for s in seqs},
        "per_request_e2e_ms": {s.seq_id: round(finished_at[s.seq_id], 6) for s in seqs},
        "per_request_tpot_ms": {k: round(v, 6) for k, v in tpots.items()},
        "per_request_output": {s.seq_id: list(s.generated_tokens) for s in seqs},
        "runtime_metrics": summary,
    }
    if device.type == "cuda":
        result.update({
            "cuda_steady_allocated_bytes": steady_allocated,
            "cuda_steady_reserved_bytes": steady_reserved,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
    if with_trace:
        result["iteration_records"] = [asdict(r) for r in rt.records]
    # Drop references to model-independent objects so later scenarios do not stack pools.
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--batch-sizes", type=csv_ints, default=[1, 4, 8, 16])
    ap.add_argument("--prompt-lengths", type=csv_ints, default=[32, 128, 256])
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--kv-factors", type=csv_floats, default=[1.0],
                    help="KV blocks = max(one-request max, ceil(factor * all-request max)); "
                         "0.5 provides contention, 1.0 fits all max KV demands")
    ap.add_argument("--policies", default="conservative",
                    help="Comma-separated: conservative and/or recompute")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--include-iteration-records", action="store_true",
                    help="Include full per-iteration telemetry (larger JSON)")
    ap.add_argument("--output-json", type=Path, default=Path("artifacts/gpu/e2e_runtime.json"))
    args = ap.parse_args()
    policies = [s.strip() for s in args.policies.split(",") if s.strip()]
    if not policies or any(s not in ("conservative", "recompute") for s in policies):
        ap.error("--policies must be conservative,recompute or a subset")
    if len(set(policies)) != len(policies):
        ap.error("duplicate policies are not allowed")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; use --device cpu for smoke")
    if (args.dim <= 0 or args.layers <= 0 or args.heads <= 0
        or args.dim % args.heads != 0 or args.vocab_size < 2
        or args.block_size <= 0 or args.token_budget <= 0
        or args.output_len < 2 or args.warmups < 0 or args.repeats < 1
        or args.max_steps < 1):
        ap.error("invalid model or benchmark configuration")
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    # Same weights/position-embedding range for all configurations.
    max_positions = max(args.prompt_lengths) + args.output_len + 32
    model = MiniGPT(n_embd=args.dim, n_head=args.heads, vocab_size=args.vocab_size,
                    block_size=max_positions, dropout=0.0,
                    n_layer=args.layers).to(device).eval()
    runs = []
    print(f"torch={torch.__version__} device={device} "
          f"model=({args.layers} layers, dim={args.dim}, heads={args.heads}) "
          "mode=natural_host_observation / boundary_cuda_sync", flush=True)
    for B in args.batch_sizes:
        for prompt_len in args.prompt_lengths:
            outputs_by_policy = {}
            for factor in args.kv_factors:
                for policy in policies:
                    for _ in range(args.warmups):
                        run_once(model, args, device, batch=B, prompt_len=prompt_len,
                                 factor=factor, policy=policy)
                    for rep in range(args.repeats):
                        result = run_once(model, args, device, batch=B, prompt_len=prompt_len,
                                          factor=factor, policy=policy,
                                          with_trace=args.include_iteration_records and rep == 0)
                        result["repeat"] = rep
                        key = (B, prompt_len, policy)
                        if key in outputs_by_policy and result["per_request_output"] != outputs_by_policy[key]:
                            raise AssertionError(f"token mismatch between capacities/repeats: {key}")
                        outputs_by_policy[key] = result["per_request_output"]
                        runs.append(result)
                        print(f"PASS B={B:2} P={prompt_len:3} factor={factor:.2f} "
                              f"policy={policy:<12} rep={rep} steps={result['iterations']:3} "
                              f"wall={result['total_wall_ms']:.2f}ms "
                              f"avgTTFT={result['avg_ttft_ms']:.2f}ms "
                              f"avgE2E={result['avg_e2e_ms']:.2f}ms "
                              f"tok/s={result['output_tok_s']:.1f} "
                              f"peakKV={result['kv_peak_used_blocks_at_step_end']}/{result['num_blocks']}",
                              flush=True)
            if len(policies) > 1:
                baseline_output = outputs_by_policy[(B, prompt_len, policies[0])]
                for p in policies[1:]:
                    if outputs_by_policy[(B, prompt_len, p)] != baseline_output:
                        raise AssertionError(f"policy token mismatch B={B} P={prompt_len}")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["batch_size"], run["prompt_len"], run["kv_capacity_factor_requested"], run["policy"])].append(run)
    summaries = []
    for (B, plen, factor, policy), vals in grouped.items():
        summaries.append({
            "batch_size": B, "prompt_len": plen, "kv_factor": factor,
            "num_blocks": vals[0]["num_blocks"], "policy": policy,
            "wall_ms": stats([r["total_wall_ms"] for r in vals]),
            "output_tok_s": stats([r["output_tok_s"] for r in vals]),
            "avg_ttft_ms": stats([r["avg_ttft_ms"] for r in vals]),
            "avg_e2e_ms": stats([r["avg_e2e_ms"] for r in vals]),
            "avg_tpot_ms": stats([r["avg_tpot_ms"] for r in vals]),
            "peak_allocated_bytes": stats([r["cuda_peak_allocated_bytes"] for r in vals])
                if device.type == "cuda" else None,
            "preemptions": [r["preemptions"] for r in vals],
        })
    report = {
        "definition": "Same-arrival closed batch, full MiniRuntime Prefill+Decode+finish; no network service or arrival rate",
        "timing": "Host wall time with CUDA synchronization only at run boundaries; implicit GPU syncs remain in .item/.tolist",
        "token_timestamps": "Host-observed step-completion times; each request's generated token becomes visible at the end of a runtime step",
        "memory": "CUDA allocated/reserved includes model, KV pool, temporary tensors; KV step-end sample can miss in-step peak; pool bytes formula excludes other allocations",
        "device": str(device), "torch": torch.__version__,
        "model": {"dim": args.dim, "heads": args.heads, "layers": args.layers,
                  "dtype": "float32", "max_positions": max_positions},
        "config": {"batch_sizes": args.batch_sizes, "prompt_lengths": args.prompt_lengths,
                   "output_len": args.output_len, "kv_factors": args.kv_factors,
                   "policies": policies, "block_size": args.block_size,
                   "token_budget": args.token_budget, "warmups": args.warmups, "repeats": args.repeats},
        "summaries": summaries, "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"E2E RUNTIME BENCHMARK PASS; saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
