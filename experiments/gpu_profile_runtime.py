#!/usr/bin/env python3
"""GPU Day 5: targeted MiniRuntime CPU/CUDA profiling.

Copy to experiments/gpu_profile_runtime.py, run from repository root with
`python -m experiments.gpu_profile_runtime --help`.

Dependency: experiments.gpu_e2e_runtime from Day 4 (make_runtime helper).
No changes to Scheduler, ModelRunner, Attention or the core Runtime.

Two intentionally SEPARATE measurement modes:
  phase: synchronous host *completed* stage timing; intrusive diagnostics,
         not comparable directly with asynchronous Day 4 wall times.
  ops: torch.profiler CPU/CUDA activity for exactly ONE Runtime step;
       strongly intrusive; use for attribution and kernel launch investigation.
Neither mode measures pure GPU kernel execution with CUDA Events.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from experiments.gpu_e2e_runtime import make_runtime
from minivllm.model import MiniGPT
from minivllm.sequence import Status


def csv_ints(value: str) -> list[int]:
    try:
        nums = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer(s)") from exc
    if not nums or min(nums) < 1:
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return nums


def median(values):
    return round(statistics.median(values), 4)


def make_model(args, device):
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    max_positions = args.max_positions  # Keep model initialization identical across separate invocations.
    return MiniGPT(n_embd=args.dim, n_head=args.heads,
                   vocab_size=args.vocab_size, block_size=max_positions,
                   dropout=0.0, n_layer=args.layers).to(device).eval()


def new_runtime(model, args, device, prompt):
    rt, sequences, capacity = make_runtime(
        model, device, batch=args.batch, prompt_len=prompt,
        output_len=args.output_len, block_size=args.block_size,
        token_budget=args.token_budget, capacity_factor=1.0,
        policy="conservative", heads=args.heads, dim=args.dim,
    )
    # Day 4 creates this field; profiling must not auto-sync in Runtime.step().
    rt.sync_cuda_timing = False
    rt.verbose = False
    return rt, sequences, capacity


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def instrument(rt, callback, *, mark: bool = False, attention: bool = False):
    """Patch only this Runtime instance. Restore refs on function exit."""
    replacements = []

    def wrap(obj, name, label):
        original = getattr(obj, name)

        def wrapped(*a, **kw):
            if mark:
                with record_function("runtime/" + label):
                    return callback(label, original, *a, **kw)
            return callback(label, original, *a, **kw)

        setattr(obj, name, wrapped)
        replacements.append((obj, name, original))

    wrap(rt.scheduler, "schedule", "scheduler")
    wrap(rt.runner, "prefill_chunk", "prefill_chunk")
    wrap(rt.runner, "decode_batch", "decode_batch")
    if attention:
        # Important: forward_prefill_chunk imported the function into its OWN
        # module namespace, so patch minivllm.attention rather than the source
        # minivllm.paged_attention_v1_reference module.
        import minivllm.attention as attn
        wrap(attn, "paged_attention_online", "paged_attention_online")
    return replacements


def restore(replacements):
    for obj, name, original in reversed(replacements):
        setattr(obj, name, original)


def validate_run(rt, seqs, capacity, args):
    assert all(s.status == Status.FINISHED for s in seqs), "unfinished request"
    assert not rt.block_manager.block_owner, "blocks not released"
    assert len(rt.block_manager.free_blocks) == capacity, "KV block leakage"
    rt.block_manager.check_invariants(seqs)
    assert all(len(s.generated_tokens) == args.output_len for s in seqs)
    metrics = rt.metrics_summary()
    assert metrics["output_tokens"] == args.batch * args.output_len
    assert metrics["preemptions"] == 0
    assert metrics["repeated_kv_tokens"] == 0
    return metrics


def phase_run(model, args, device, prompt):
    rt, seqs, capacity = new_runtime(model, args, device, prompt)
    elapsed_by_phase = defaultdict(float)
    call_count = defaultdict(int)

    def measure(label, original, *a, **kw):
        if label != "scheduler":
            sync(device)
        start = time.perf_counter()
        result = original(*a, **kw)
        if label != "scheduler":
            sync(device)
        elapsed_by_phase[label] += (time.perf_counter() - start) * 1e3
        call_count[label] += 1
        return result

    replacements = instrument(rt, measure)
    sync(device)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for _ in range(args.max_steps):
                if all(s.status == Status.FINISHED for s in seqs):
                    break
                rt.step()
            else:
                raise RuntimeError("too many iterations")
        sync(device)
        wall_ms = (time.perf_counter() - t0) * 1e3
    finally:
        restore(replacements)

    metrics = validate_run(rt, seqs, capacity, args)
    stage_sum = sum(elapsed_by_phase.values())
    return {
        "prompt_len": prompt,
        "batch_size": args.batch,
        "iterations": rt.iteration,
        "total_wall_ms": round(wall_ms, 4),
        "phase_ms": {name: round(elapsed_by_phase[name], 4)
                     for name in ("scheduler", "prefill_chunk", "decode_batch")},
        "other_ms": round(wall_ms - stage_sum, 4),
        "calls": dict(call_count),
        "prefill_tokens": metrics["prefill_tokens"],
        "decode_tokens": metrics["decode_tokens"],
        "output_tokens": metrics["output_tokens"],
        "peak_kv_blocks_step_end": metrics["peak_used_blocks_at_step_end"],
        "output_digest": {s.seq_id: list(s.generated_tokens) for s in seqs},
    }


def run_phases(model, args, device):
    all_runs = []
    for prompt in args.prompts:
        for _ in range(args.warmups):
            phase_run(model, args, device, prompt)
        group = []
        for rep in range(args.repeats):
            r = phase_run(model, args, device, prompt)
            if group and r["output_digest"] != group[0]["output_digest"]:
                raise AssertionError("outputs differ among repeated runs")
            r["repeat"] = rep
            group.append(r)
        all_runs.extend(group)
        key = ("total_wall_ms", "other_ms")
        wall = median([r["total_wall_ms"] for r in group])
        prefill = median([r["phase_ms"]["prefill_chunk"] for r in group])
        decode = median([r["phase_ms"]["decode_batch"] for r in group])
        schedule = median([r["phase_ms"]["scheduler"] for r in group])
        other = median([r["other_ms"] for r in group])
        print(f"PHASE P={prompt:3d} B={args.batch} steps={group[0]['iterations']:3d} "
              f"total={wall:.2f} ms prefill={prefill:.2f} "
              f"decode={decode:.2f} schedule={schedule:.2f} other={other:.2f}", flush=True)
    grouped = defaultdict(list)
    for r in all_runs:
        grouped[r["prompt_len"]].append(r)
    summaries = []
    for plen, items in grouped.items():
        fields = ("scheduler", "prefill_chunk", "decode_batch")
        summaries.append({
            "prompt_len": plen,
            "batch_size": args.batch,
            "iterations": items[0]["iterations"],
            "wall_ms_median": median([r["total_wall_ms"] for r in items]),
            "phase_ms_median": {f: median([r["phase_ms"][f] for r in items]) for f in fields},
            "other_ms_median": median([r["other_ms"] for r in items]),
            "calls": items[0]["calls"],
        })
    return {"mode": "phase", "method": "sync before/after runner stage, host wall with instrumentation",
            "summaries": summaries, "runs": all_runs}


def first_value(row, *names):
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return float(value)
    return 0.0


def run_ops(model, args, device):
    if len(args.prompts) != 1:
        raise ValueError("--mode ops requires a single --prompts length")
    plen = args.prompts[0]
    if args.capture_step > args.max_steps:
        raise ValueError("capture-step exceeds max-steps")
    if args.detail_attention and plen > 32:
        raise ValueError("--detail-attention is only allowed with --prompts <=32 to limit trace size")
    for _ in range(args.warmups):
        rt, seqs, _ = new_runtime(model, args, device, plen)
        with torch.inference_mode():
            for i in range(args.capture_step):
                rt.step()
        sync(device)
    rt, seqs, capacity = new_runtime(model, args, device, plen)
    with torch.inference_mode():
        for _ in range(args.capture_step - 1):
            rt.step()
    sync(device)

    def pass_through(label, original, *a, **kw):
        return original(*a, **kw)

    replacements = instrument(rt, pass_through, mark=True, attention=args.detail_attention)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with torch.inference_mode():
            with profile(activities=activities, record_shapes=False,
                         profile_memory=False, with_stack=False) as prof:
                rt.step()
            sync(device)
    finally:
        restore(replacements)
    if args.capture_step == 1:
        assert rt.records[-1].prefill_tokens > 0, "expected Prefill at step 1"
    if args.capture_step == 2 and plen <= args.token_budget:
        assert rt.records[-1].decode_tokens > 0, "expected Decode at step 2"

    events = prof.key_averages()
    op_stats = []
    for item in events:
        cpu_us = first_value(item, "self_cpu_time_total")
        cuda_us = first_value(item, "self_device_time_total", "self_cuda_time_total")
        op_stats.append({"name": item.key, "count": item.count,
                         "self_cpu_ms": round(cpu_us / 1000, 5),
                         "self_cuda_ms": round(cuda_us / 1000, 5)})
    top_cpu = sorted(op_stats, key=lambda o: o["self_cpu_ms"], reverse=True)[:args.top]
    top_cuda = sorted(op_stats, key=lambda o: o["self_cuda_ms"], reverse=True)[:args.top]
    print(f"OPS P={plen} B={args.batch} step={args.capture_step} "
          f"prefill_tokens={rt.records[-1].prefill_tokens} "
          f"decode_tokens={rt.records[-1].decode_tokens}", flush=True)
    print("TOP CPU self time (ms):")
    for x in top_cpu:
        print(f"  {x['self_cpu_ms']:10.3f}  count={x['count']:6d}  {x['name']}")
    if device.type == "cuda":
        print("TOP CUDA self time (ms):")
        for x in top_cuda:
            print(f"  {x['self_cuda_ms']:10.3f}  count={x['count']:6d}  {x['name']}")
        if not any(x['self_cuda_ms'] > 0 for x in top_cuda):
            print("WARNING: no CUDA timings captured; check PyTorch/CUPTI availability")
    trace_path = None
    if args.save_trace:
        trace_path = args.trace_json or str(args.output_json.with_name(args.output_json.stem + "_trace.json"))
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"TRACE saved {trace_path} (can be very large)")
    return {"mode": "ops", "method": "torch.profiler one selected step, not ordinary wall-clock benchmark",
            "profiled_step": args.capture_step,
            "step_record": {"prefill_tokens": rt.records[-1].prefill_tokens,
                            "decode_tokens": rt.records[-1].decode_tokens,
                            "iteration": rt.iteration},
            "top_cpu": top_cpu, "top_cuda": top_cuda,
            "all_ops": op_stats,
            "trace_file": trace_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("phase", "ops"), default="phase")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompts", type=csv_ints, default=[32, 128, 256])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--max-positions", type=int, default=320,
                        help="Fixed model position table size (same across compared runs)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument("--detail-attention", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--trace-json", type=str)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-json", type=Path,
                        default=Path("artifacts/gpu/day5_profile.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available in this environment")
    if (args.dim < 1 or args.layers < 1 or args.heads < 1 or
        args.dim % args.heads or args.batch < 1 or args.output_len < 2 or
        args.token_budget < 1 or args.max_steps < 1 or args.block_size < 1 or
        args.warmups < 0 or args.repeats < 1 or args.capture_step < 1 or
        args.top < 1 or args.vocab_size < 2 or
        args.max_positions < max(args.prompts) + args.output_len):
        parser.error("invalid model / workload / measurement parameters")
    device = torch.device(args.device)
    print(f"DAY5 torch={torch.__version__} device={device} mode={args.mode} "
          f"layers={args.layers} dim={args.dim} heads={args.heads} "
          f"batch={args.batch} prompts={args.prompts}", flush=True)
    model = make_model(args, device)
    report = run_phases(model, args, device) if args.mode == "phase" else run_ops(model, args, device)
    report.update({"device": str(device), "torch": torch.__version__,
                   "model": {"dim": args.dim, "layers": args.layers, "heads": args.heads,
                             "max_positions": args.max_positions},
                   "configuration": {"batch_size": args.batch, "prompts": args.prompts,
                                     "output_len": args.output_len, "token_budget": args.token_budget,
                                     "warmups": args.warmups,
                                     "repeats": args.repeats if args.mode == "phase" else None}})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DAY5 {args.mode.upper()} PASS; saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
