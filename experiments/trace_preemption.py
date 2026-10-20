#!/usr/bin/env python3
"""Trace scheduling state (not a timed benchmark) for preemption policies.

Place alongside experiments/bench_preemption.py and run from repo root:
PYTHONPATH=./minivllm:./experiments python experiments/trace_preemption.py \
  --blocks 2 --output-json artifacts/preemption/trace_blocks2.json
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bench_preemption import make_runtime, make_workload
from model import MiniGPT
from sequence import Status


def run_trace(model, cfg, policy, blocks):
    rt = make_runtime(model, cfg, policy, blocks)
    seqs = make_workload(getattr(cfg, "workload", "baseline"))
    for seq in seqs:
        rt.add_request(seq)
    records = []
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        for _ in range(cfg.max_steps):
            if all(s.status == Status.FINISHED for s in seqs):
                break
            before_preempt = {s.seq_id: s.num_preemptions for s in seqs}
            before_out = {s.seq_id: len(s.generated_tokens) for s in seqs}
            before_replays = rt.runner.repeated_kv_tokens
            before_prefill = rt.runner.total_prefill_tokens
            before_decode = rt.runner.total_decode_tokens
            rt.step()
            rt.block_manager.check_invariants(seqs)
            records.append({
                'iteration': rt.iteration,
                'newly_preempted': [s.seq_id for s in seqs if s.num_preemptions > before_preempt[s.seq_id]],
                'new_output_tokens': {s.seq_id: len(s.generated_tokens)-before_out[s.seq_id] for s in seqs
                                      if len(s.generated_tokens)>before_out[s.seq_id]},
                'prefill_tokens_executed': rt.runner.total_prefill_tokens - before_prefill,
                'decode_tokens_executed': rt.runner.total_decode_tokens - before_decode,
                'repeated_tokens_executed': rt.runner.repeated_kv_tokens - before_replays,
                'kv_blocks_used': len(rt.block_manager.block_owner),
                'waiting': [s.seq_id for s in rt.scheduler.waiting],
                'prefilling': [s.seq_id for s in rt.scheduler.prefilling],
                'running': [s.seq_id for s in rt.scheduler.running],
                'requests': {s.seq_id: {'status': s.status.value, 'kv_length': s.length,
                                        'prompt_pos': s.prompt_pos,
                                        'generated': len(s.generated_tokens),
                                        'block_count': len(s.block_table),
                                        'preemptions': s.num_preemptions,
                                        'recomputing': s.recompute_tokens is not None}
                             for s in seqs},
            })
        else:
            raise RuntimeError(f'No convergence after {cfg.max_steps} iterations: {policy}, blocks={blocks}')
    if not all(s.status == Status.FINISHED for s in seqs):
        raise AssertionError('unfinished requests')
    if rt.block_manager.block_owner:
        raise AssertionError('KV leaked after completion')
    return {
        'policy': policy, 'blocks': blocks,
        'iterations': rt.iteration, 'preemptions': rt.scheduler.num_preemptions,
        'repeated_kv_tokens': rt.runner.repeated_kv_tokens,
        'final_output': {s.seq_id: list(s.generated_tokens) for s in seqs},
        'trace': records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workload', choices=('baseline','mixed_short'), default='baseline')
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--output-json', default='artifacts/preemption/trace_blocks2.json')
    p.add_argument('--max-steps', type=int, default=200)
    cfg = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = MiniGPT(n_embd=16, n_head=2, vocab_size=64, block_size=256,
                    dropout=0.0, n_layer=1).cpu().eval()
    opts = SimpleNamespace(device='cpu', block_size=16, layers=1, heads=2, dim=16,
                           max_running=4, token_budget=64, max_steps=cfg.max_steps, workload=cfg.workload)
    a = run_trace(model, opts, 'conservative', cfg.blocks)
    b = run_trace(model, opts, 'recompute', cfg.blocks)
    assert a['final_output'] == b['final_output'], 'strategy output tokens differ'
    dest = Path(cfg.output_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'mode':'trace_only_not_a_benchmark','workload_profile':cfg.workload,'runs':[a,b]}, indent=2, ensure_ascii=False))
    print('POLICY        STEPS  PREEMPTS  REPLAYED_KV')
    for run in [a,b]:
        print(f"{run['policy']:<13} {run['iterations']:>5} {run['preemptions']:>9} {run['repeated_kv_tokens']:>11}")
    print('\nRECOMPUTE: events and work per iteration')
    for e in b['trace']:
        extra = f"preempt={','.join(e['newly_preempted']) or '-'}"
        print(f"step={e['iteration']:>2} used={e['kv_blocks_used']}/{cfg.blocks} "
              f"prefill={e['prefill_tokens_executed']:>2} decode={e['decode_tokens_executed']:>2} "
              f"repeated={e['repeated_tokens_executed']:>2} {extra} "
              f"running={e['running']} waiting={e['waiting']}")
    print(f'PASS identical outputs; saved {dest}')


if __name__ == '__main__':
    main()
