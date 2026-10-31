#!/usr/bin/env python3
"""GPU Day 6: bottleneck shift after prefix-aware SDPA v2.

Standalone experiment module. Requires the *existing* Day 4/5 experiment
helpers and SDPA v2 already installed in the Mini-GPT Inference repository.
Does not alter Scheduler, ModelRunner or attention source.

Modes:
  baseline: natural closed-batch E2E with CUDA sync only at boundaries.
  phase: intrusive completed-stage host timing (sync at runner boundaries).
  ops: one torch.profiler-captured step, chosen by actual dry-run state.

A profiled step is NOT an uninstrumented performance measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from experiments.gpu_e2e_runtime import make_runtime, run_once, stats, csv_ints
from minivllm.model import MiniGPT
from minivllm.sequence import Status


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_model(args, device):
    torch.manual_seed(2026)
    model = MiniGPT(
        n_embd=args.dim, n_head=args.heads, vocab_size=64,
        block_size=args.max_positions, dropout=0.0, n_layer=args.layers,
    ).to(device).eval()
    model.set_prefill_fastpath_mode(args.backend)
    return model


def new_runtime(model, args, device, batch, prompt):
    rt, seqs, num_blocks = make_runtime(
        model, device, batch=batch, prompt_len=prompt,
        output_len=args.output_len, block_size=args.block_size,
        token_budget=args.token_budget, capacity_factor=args.kv_factor,
        policy=args.policy, heads=args.heads, dim=args.dim,
    )
    rt.sync_cuda_timing = False
    rt.verbose = False
    return rt, seqs, num_blocks


def is_finished(seqs):
    return all(s.status == Status.FINISHED for s in seqs)


def validate(rt, seqs, capacity, args):
    if not is_finished(seqs):
        raise AssertionError('unfinished requests')
    if rt.block_manager.block_owner or len(rt.block_manager.free_blocks) != capacity:
        raise AssertionError('KV block leak')
    rt.block_manager.check_invariants(seqs)
    if any(len(s.generated_tokens) != args.output_len for s in seqs):
        raise AssertionError('incorrect output length')
    met = rt.metrics_summary()
    if met['output_tokens'] != len(seqs) * args.output_len:
        raise AssertionError('output token accounting mismatch')
    if args.policy == 'conservative' and met['preemptions']:
        raise AssertionError('unexpected conservative preemption')
    return met


@contextmanager
def patched(obj, name, wrapper_builder):
    original = getattr(obj, name)
    setattr(obj, name, wrapper_builder(original))
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextmanager
def staged_timers(rt, device, sums, calls, chunks, decode_sizes, meta_host_ms):
    """Only runner boundaries sync; metadata timing nested inside decode."""
    original_schedule = rt.scheduler.schedule
    original_prefill = rt.runner.prefill_chunk
    original_decode = rt.runner.decode_batch
    original_metadata = rt.runner._prepare_decode_batch

    def schedule(*a, **kw):
        t0 = time.perf_counter()
        result = original_schedule(*a, **kw)
        sums['scheduler'] += (time.perf_counter() - t0) * 1e3
        calls['scheduler'] += 1
        return result

    def prefill(seq, start_pos, num_tokens):
        label = 'prefill_initial' if start_pos == 0 else 'prefill_prefix'
        sync(device)
        t0 = time.perf_counter()
        result = original_prefill(seq=seq, start_pos=start_pos, num_tokens=num_tokens)
        sync(device)
        sums[label] += (time.perf_counter() - t0) * 1e3
        calls[label] += 1
        chunks.append({'request_id': seq.seq_id, 'start_pos': start_pos,
                       'num_tokens': num_tokens, 'phase': label})
        return result

    def metadata(sequences):
        t0 = time.perf_counter()
        result = original_metadata(sequences)
        # No sync here: report ONLY host-call wall time; overlaps decode_batch.
        meta_host_ms.append((time.perf_counter() - t0) * 1e3)
        return result

    def decode(sequences):
        sync(device)
        t0 = time.perf_counter()
        result = original_decode(sequences)
        sync(device)
        sums['decode_batch'] += (time.perf_counter() - t0) * 1e3
        calls['decode_batch'] += 1
        decode_sizes.append(len(sequences))
        return result

    rt.scheduler.schedule = schedule
    rt.runner.prefill_chunk = prefill
    rt.runner.decode_batch = decode
    rt.runner._prepare_decode_batch = metadata
    try:
        yield
    finally:
        rt.scheduler.schedule = original_schedule
        rt.runner.prefill_chunk = original_prefill
        rt.runner.decode_batch = original_decode
        rt.runner._prepare_decode_batch = original_metadata


def run_phase_once(model, args, device, batch, prompt):
    rt, seqs, capacity = new_runtime(model, args, device, batch, prompt)
    sums = defaultdict(float)
    calls = Counter()
    chunks, decode_sizes, metadata_host = [], [], []
    sync(device)
    t0 = time.perf_counter()
    with staged_timers(rt, device, sums, calls, chunks, decode_sizes, metadata_host):
        with torch.inference_mode():
            for _ in range(args.max_steps):
                if is_finished(seqs):
                    break
                rt.step()
            else:
                raise RuntimeError('max steps exceeded')
    sync(device)
    wall = (time.perf_counter() - t0) * 1000
    met = validate(rt, seqs, capacity, args)
    top = sum(sums[k] for k in ('scheduler', 'prefill_initial', 'prefill_prefix', 'decode_batch'))
    result = {
        'batch': batch, 'prompt_len': prompt, 'backend': args.backend,
        'total_wall_ms': round(wall, 5),
        'phase_ms': {k: round(sums[k], 5) for k in
                     ('scheduler', 'prefill_initial', 'prefill_prefix', 'decode_batch')},
        'prefill_total_ms': round(sums['prefill_initial'] + sums['prefill_prefix'], 5),
        'other_ms': round(wall - top, 5),
        'metadata_host_ms_nested_in_decode': round(sum(metadata_host), 5),
        'calls': dict(calls),
        'decode_batch_sizes': dict(sorted(Counter(decode_sizes).items())),
        'prefill_chunks': chunks,
        'iterations': met['iterations'],
        'prefill_token_executions': met['prefill_tokens'],
        'decode_token_executions': met['decode_tokens'],
        'preemptions': met['preemptions'],
        'repeated_kv_tokens': met['repeated_kv_tokens'],
        'output_digest': {s.seq_id: list(s.generated_tokens) for s in seqs},
    }
    return result


def run_phases(model, args, device):
    runs, summaries = [], []
    for p in args.prompts:
        for b in args.batches:
            for _ in range(args.warmups):
                run_phase_once(model, args, device, b, p)
            group = []
            for rep in range(args.repeats):
                result = run_phase_once(model, args, device, b, p)
                if group and (result['output_digest'] != group[0]['output_digest'] or
                              result['prefill_token_executions'] != group[0]['prefill_token_executions'] or
                              result['decode_token_executions'] != group[0]['decode_token_executions']):
                    raise AssertionError('repeated run outputs / work counts changed')
                result['repeat'] = rep
                group.append(result)
                runs.append(result)
            field = lambda k: stats([r[k] for r in group])
            phase = lambda k: stats([r['phase_ms'][k] for r in group])
            pre = field('prefill_total_ms')['median']
            dec = phase('decode_batch')['median']
            wall = field('total_wall_ms')['median']
            row = {
                'batch': b, 'prompt_len': p, 'backend': args.backend,
                'wall_ms': field('total_wall_ms'),
                'prefill_ms': field('prefill_total_ms'),
                'prefill_initial_ms': phase('prefill_initial'),
                'prefill_prefix_ms': phase('prefill_prefix'),
                'decode_ms': phase('decode_batch'),
                'schedule_ms': phase('scheduler'),
                'other_ms': field('other_ms'),
                'decode_metadata_host_ms_nested': field('metadata_host_ms_nested_in_decode'),
                'prefill_pct': round(pre * 100 / wall, 2),
                'decode_pct': round(dec * 100 / wall, 2),
                'calls': group[0]['calls'],
                'decode_batch_sizes': group[0]['decode_batch_sizes'],
                'iterations': group[0]['iterations'],
                'prefill_token_executions': group[0]['prefill_token_executions'],
                'decode_token_executions': group[0]['decode_token_executions'],
                'preemptions': group[0]['preemptions'],
                'repeated_kv_tokens': group[0]['repeated_kv_tokens'],
            }
            summaries.append(row)
            print(f'PHASE B={b:2d} P={p:3d} total={wall:.3f}ms '
                  f'prefill={pre:.3f}ms ({row["prefill_pct"]:.1f}%) '
                  f'decode={dec:.3f}ms ({row["decode_pct"]:.1f}%) '
                  f'initial/prefix calls={row["calls"].get("prefill_initial",0)}/'
                  f'{row["calls"].get("prefill_prefix",0)} '
                  f'decode_calls={row["calls"].get("decode_batch",0)}', flush=True)
    return {'mode': 'phase', 'method': 'sync at each runner stage, invasive host-completed timing',
            'summaries': summaries, 'runs': runs}


def run_baseline(model, args, device):
    old_args = SimpleNamespace(output_len=args.output_len, block_size=args.block_size,
                               token_budget=args.token_budget, heads=args.heads,
                               dim=args.dim, max_steps=args.max_steps)
    rows, runs = [], []
    for p in args.prompts:
        for b in args.batches:
            for _ in range(args.warmups):
                run_once(model, old_args, device, batch=b, prompt_len=p,
                         factor=args.kv_factor, policy=args.policy)
            group = []
            for rep in range(args.repeats):
                r = run_once(model, old_args, device, batch=b, prompt_len=p,
                             factor=args.kv_factor, policy=args.policy)
                if group and r['per_request_output'] != group[0]['per_request_output']:
                    raise AssertionError('different generated tokens between repeats')
                r['repeat'] = rep
                r['backend'] = args.backend
                group.append(r)
                runs.append(r)
            row = {'batch': b, 'prompt_len': p, 'backend': args.backend,
                   'wall_ms': stats([r['total_wall_ms'] for r in group]),
                   'ttft_ms': stats([r['avg_ttft_ms'] for r in group]),
                   'tpot_ms': stats([r['avg_tpot_ms'] for r in group]),
                   'output_tok_s': stats([r['output_tok_s'] for r in group]),
                   'iterations': group[0]['iterations']}
            rows.append(row)
            print(f'BASELINE B={b:2d} P={p:3d} wall={row["wall_ms"]["median"]:.3f}ms '
                  f'TTFT={row["ttft_ms"]["median"]:.3f}ms '
                  f'tok/s={row["output_tok_s"]["median"]:.1f}', flush=True)
    return {'mode': 'baseline', 'method': 'boundary-only CUDA sync, host observed E2E',
            'summaries': rows, 'runs': runs}


def find_capture_step(model, args, device, b, p):
    """Dry-run to find a step with genuine nonzero-prefix prefill or pure decode."""
    rt, seqs, cap = new_runtime(model, args, device, b, p)
    captured = None
    original = rt.runner.prefill_chunk
    step_starts = []

    def track(seq, start_pos, num_tokens):
        step_starts.append((start_pos, num_tokens, seq.seq_id))
        return original(seq=seq, start_pos=start_pos, num_tokens=num_tokens)

    rt.runner.prefill_chunk = track
    with torch.inference_mode():
        for _ in range(args.max_steps):
            if is_finished(seqs):
                break
            step_starts.clear()
            rt.step()
            rec = rt.records[-1]
            if args.capture == 'prefix':
                eligible = any(start > 0 for start, _, _ in step_starts)
            elif args.capture == 'initial':
                eligible = any(start == 0 for start, _, _ in step_starts)
            elif args.capture == 'decode':
                eligible = rec.decode_tokens > 0 and rec.prefill_tokens == 0
            elif args.capture == 'mixed':
                eligible = rec.decode_tokens > 0 and rec.prefill_tokens > 0
            else:
                raise ValueError(args.capture)
            if eligible:
                captured = {'step': rt.iteration, 'record': {
                    'prefill_tokens': rec.prefill_tokens,
                    'decode_tokens': rec.decode_tokens,
                    'prefill_requests': rec.prefill_requests,
                    'decode_requests': rec.decode_requests},
                    'chunks': list(step_starts)}
                break
    rt.runner.prefill_chunk = original
    if captured is None:
        raise RuntimeError(f'No {args.capture} step for B={b} P={p}. Change workload or capture type.')
    sync(device)
    return captured


@contextmanager
def profile_ranges(rt):
    """Named profiler scopes; each is INCLUSIVE of nested operations."""
    from torch.profiler import record_function
    import torch.nn.functional as F
    import minivllm.attention as attention_module

    original_schedule = rt.scheduler.schedule
    original_prefill = rt.runner.prefill_chunk
    original_decode = rt.runner.decode_batch
    original_meta = rt.runner._prepare_decode_batch
    original_attn = attention_module.KVCacheCausalSelfAttention.forward_prefill_chunk
    original_sdpa = F.scaled_dot_product_attention

    def sch(*a, **kw):
        with record_function('day6/scheduler'):
            return original_schedule(*a, **kw)
    def prefill(*a, **kw):
        with record_function('day6/prefill_runner'):
            return original_prefill(*a, **kw)
    def decode(*a, **kw):
        with record_function('day6/decode_runner'):
            return original_decode(*a, **kw)
    def metadata(*a, **kw):
        with record_function('day6/decode_metadata'):
            return original_meta(*a, **kw)
    def layer_attn(self, *a, **kw):
        with record_function('day6/prefill_attention_layer'):
            return original_attn(self, *a, **kw)
    def sdpa(*a, **kw):
        with record_function('day6/sdpa'):
            return original_sdpa(*a, **kw)

    rt.scheduler.schedule = sch
    rt.runner.prefill_chunk = prefill
    rt.runner.decode_batch = decode
    rt.runner._prepare_decode_batch = metadata
    attention_module.KVCacheCausalSelfAttention.forward_prefill_chunk = layer_attn
    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        rt.scheduler.schedule = original_schedule
        rt.runner.prefill_chunk = original_prefill
        rt.runner.decode_batch = original_decode
        rt.runner._prepare_decode_batch = original_meta
        attention_module.KVCacheCausalSelfAttention.forward_prefill_chunk = original_attn
        F.scaled_dot_product_attention = original_sdpa


def run_ops(model, args, device):
    from torch.profiler import ProfilerActivity, profile
    if len(args.prompts) != 1 or len(args.batches) != 1:
        raise ValueError('ops mode requires one --prompts and one --batches value')
    p, b = args.prompts[0], args.batches[0]
    chosen = find_capture_step(model, args, device, b, p)
    step = chosen['step']
    for _ in range(args.warmups):
        rt, seqs, _ = new_runtime(model, args, device, b, p)
        with torch.inference_mode():
            for _ in range(step):
                rt.step()
        sync(device)

    rt, seqs, _ = new_runtime(model, args, device, b, p)
    with torch.inference_mode():
        for _ in range(step - 1):
            rt.step()
    sync(device)
    acts = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        acts.append(ProfilerActivity.CUDA)
    with profile_ranges(rt):
        with torch.inference_mode():
            with profile(activities=acts, record_shapes=False,
                         profile_memory=False, with_stack=False) as prof:
                rt.step()
            sync(device)
    rec = rt.records[-1]
    if rec.prefill_tokens != chosen['record']['prefill_tokens'] or rec.decode_tokens != chosen['record']['decode_tokens']:
        raise AssertionError('Dry-run and profiled schedule differ; check determinism')
    ops = []
    for x in prof.key_averages():
        ops.append({
            'name': x.key, 'count': x.count,
            'self_cpu_ms': round(float(getattr(x, 'self_cpu_time_total', 0.0)) / 1e3, 6),
            'self_cuda_ms': round(float(getattr(x, 'self_device_time_total',
                                                 getattr(x, 'self_cuda_time_total', 0.0))) / 1e3, 6),
        })
    lookup = {x['name']: x for x in ops}
    highlight = ('cudaLaunchKernel', 'aten::index_select', 'aten::cat',
                 'aten::scaled_dot_product_attention', 'aten::einsum',
                 'aten::where', 'aten::max', 'aten::exp',
                 'aten::bmm', 'aten::empty_strided', 'aten::copy_')
    counts = {key: lookup.get(key, {}).get('count', 0) for key in highlight}
    print(f'OPS capture={args.capture} B={b} P={p} step={step} '
          f'prefill_tokens={rec.prefill_tokens} decode_tokens={rec.decode_tokens}', flush=True)
    print('Counts:', counts, flush=True)
    top_cpu = sorted(ops, key=lambda x: x['self_cpu_ms'], reverse=True)[:args.top]
    top_cuda = sorted(ops, key=lambda x: x['self_cuda_ms'], reverse=True)[:args.top]
    for k, rows in [('CPU', top_cpu), ('CUDA', top_cuda)]:
        print(f'TOP {k} self time (ms):')
        for r in rows:
            print(f'  {r["self_cpu_ms" if k == "CPU" else "self_cuda_ms"]:9.3f} '
                  f'count={r["count"]:6d} {r["name"]}', flush=True)
    trace_file = None
    if args.save_trace:
        trace_file = str(args.output_json.with_name(args.output_json.stem + '_trace.json'))
        Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(trace_file)
        print(f'Saved Chrome Trace: {trace_file}', flush=True)
    return {'mode': 'ops', 'method': 'ONE instrumented step; avoid adding nested times',
            'capture': args.capture, 'chosen': chosen,
            'highlight_counts': counts,
            'top_cpu': top_cpu, 'top_cuda': top_cuda, 'ops': ops,
            'trace_file': trace_file}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('baseline', 'phase', 'ops'), default='phase')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--backend', choices=('reference', 'sdpa_v1', 'sdpa_v2'), default='sdpa_v2')
    ap.add_argument('--batches', type=csv_ints, default=[1,4])
    ap.add_argument('--prompts', type=csv_ints, default=[128,256])
    ap.add_argument('--output-len', type=int, default=8)
    ap.add_argument('--token-budget', type=int, default=64)
    ap.add_argument('--kv-factor', type=float, default=1.0)
    ap.add_argument('--policy', choices=('conservative', 'recompute'), default='conservative')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--block-size', type=int, default=16)
    ap.add_argument('--max-positions', type=int, default=296)
    ap.add_argument('--warmups', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=2000)
    ap.add_argument('--capture', choices=('initial', 'prefix', 'decode', 'mixed'), default='prefix')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--save-trace', action='store_true')
    ap.add_argument('--output-json', type=Path, default=Path('artifacts/gpu/day6_bottleneck.json'))
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA unavailable')
    if (args.dim < 1 or args.heads < 1 or args.dim % args.heads or args.layers < 1 or
        args.output_len < 2 or args.token_budget < 1 or args.block_size < 1 or
        args.warmups < 0 or args.repeats < 1 or args.max_steps < 1 or
        args.top < 1 or args.max_positions < max(args.prompts) + args.output_len or
        not 0 < args.kv_factor <= 1):
        ap.error('invalid parameters')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    model = make_model(args, device)
    print(f'DAY6 torch={torch.__version__} device={device} mode={args.mode} '
          f'backend={args.backend} B={args.batches} P={args.prompts} '
          f'budget={args.token_budget} TF32=False', flush=True)
    result = {'baseline': run_baseline, 'phase': run_phases, 'ops': run_ops}[args.mode](model, args, device)
    result.update({'device': str(device), 'torch': torch.__version__,
                   'model': {'layers': args.layers, 'dim': args.dim, 'heads': args.heads},
                   'configuration': {k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
                   'precision': {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                 'float32_matmul_precision': torch.get_float32_matmul_precision()}})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DAY6 {args.mode.upper()} PASS; saved {args.output_json}', flush=True)


if __name__ == '__main__':
    main()
