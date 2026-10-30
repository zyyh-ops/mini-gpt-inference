"""FP32 first-chunk SDPA vs paged online reference (E2E + optional ops profile).

Run after applying gpu_day5_prefill_sdpa_v1.patch and with the Day 4
experiments/gpu_e2e_runtime.py present. Results compare identical models,
closed arrivals, same scheduling and KV capacity. Fastpath starts disabled.

Examples:
  python -m experiments.gpu_prefill_fastpath_ab --device cuda --mode e2e \\
      --prompts 32,128,256 --batch 1 --warmups 1 --repeats 5 \\
      --output-json artifacts/gpu/prefill_sdpa_v1_ab.json
  python -m experiments.gpu_prefill_fastpath_ab --device cuda --mode ops \\
      --prompts 32,128 --batch 1 \\
      --output-json artifacts/gpu/prefill_sdpa_v1_ops.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.gpu_e2e_runtime import make_runtime, run_once, stats, csv_ints
from minivllm.model import MiniGPT
from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model_runner import ModelRunner
from minivllm.sequence import SequenceState


def make_model(args, device):
    torch.manual_seed(2026)
    return MiniGPT(
        n_embd=args.dim, n_head=args.heads,
        vocab_size=64, block_size=max(args.prompts) + args.output_len + 32,
        dropout=0.0, n_layer=args.layers,
    ).to(device).eval()


def e2e(args, device, model):
    runs = []
    old_args = SimpleNamespace(
        output_len=args.output_len, block_size=args.block_size,
        token_budget=args.token_budget, heads=args.heads, dim=args.dim,
        max_steps=args.max_steps,
    )
    for p in args.prompts:
        baseline_tokens = None
        baseline_counts = None
        # Alternating order is important for host/GPU state drift.
        for rep in range(args.warmups + args.repeats):
            order = ("reference", "sdpa_v1") if rep % 2 == 0 else ("sdpa_v1", "reference")
            for backend in order:
                model.set_prefill_fastpath(backend == 'sdpa_v1')
                run = run_once(
                    model, old_args, device, batch=args.batch, prompt_len=p,
                    factor=1.0, policy=args.policy, with_trace=False,
                )
                counts = (run['iterations'], run['output_tokens'],
                          run['preemptions'], run['repeated_kv_tokens'],
                          run['prefill_token_executions'], run['decode_token_executions'])
                if baseline_tokens is None:
                    baseline_tokens = run['per_request_output']
                    baseline_counts = counts
                if run['per_request_output'] != baseline_tokens:
                    raise AssertionError(f'Token mismatch P={p}, backend={backend}, rep={rep}')
                if counts != baseline_counts:
                    raise AssertionError(f'Lifecycle accounting mismatch P={p}, backend={backend}')
                if rep >= args.warmups:
                    run['backend'] = backend
                    run['repeat'] = rep - args.warmups
                    runs.append(run)
                    print(f'PASS P={p:3} B={args.batch:2} backend={backend:9} '
                          f'total={run["total_wall_ms"]:.3f}ms '
                          f'TTFT={run["avg_ttft_ms"]:.3f}ms '
                          f'tok/s={run["output_tok_s"]:.1f} '
                          f'preempt={run["preemptions"]}', flush=True)
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r['prompt_len'], r['backend'])].append(r)
    summary = []
    for p in args.prompts:
        ref = grouped[(p, 'reference')]
        fast = grouped[(p, 'sdpa_v1')]
        a = statistics.median(r['total_wall_ms'] for r in ref)
        b = statistics.median(r['total_wall_ms'] for r in fast)
        summary.append({
            'prompt_len': p, 'batch': args.batch,
            'reference_wall_ms': stats([r['total_wall_ms'] for r in ref]),
            'sdpa_v1_wall_ms': stats([r['total_wall_ms'] for r in fast]),
            'speedup_wall': round(a / b, 5),
            'reference_ttft_ms': stats([r['avg_ttft_ms'] for r in ref]),
            'sdpa_v1_ttft_ms': stats([r['avg_ttft_ms'] for r in fast]),
            'reference_tok_s': stats([r['output_tok_s'] for r in ref]),
            'sdpa_v1_tok_s': stats([r['output_tok_s'] for r in fast]),
            'token_equality': True,
            'lifecycle_count_equality': True,
        })
    return {
        'mode':'e2e',
        'scope':'First chunk start_pos=0 FP32 SDPA; later chunks use online reference',
        'timing':'Natural E2E wall; boundaries synchronized; host .item/.tolist may sync',
        'note':'Reuses exact Day4 run_once; output and iteration counts validated outside timed path',
        'torch':torch.__version__, 'device':str(device),
        'configuration':vars(args) | {'output_json':str(args.output_json)},
        'summaries':summary,'runs':runs,
    }


def correctness(args, device, model):
    """Numerical FP32 cross-path check including noncontiguous Paged KV writes."""
    def run_chunks(pieces, fast):
        model.set_prefill_fastpath(fast)
        total = sum(pieces)
        count = (total + args.block_size - 1) // args.block_size + 4
        manager = BlockManager(num_blocks=count, block_size=args.block_size)
        order = list(range(count))
        order = order[::2] + order[1::2]
        manager.free_blocks = list(reversed(order))
        pool = KVBlockPool(
            n_layer=args.layers, num_blocks=count, block_size=args.block_size,
            n_head=args.heads, head_dim=args.dim // args.heads,
            device=device, dtype=torch.float32,
        )
        pool.k.fill_(float('nan'))
        pool.v.fill_(float('nan'))
        seq = SequenceState(
            seq_id='numerical',
            prompt_tokens=[1 + (i * 7) % 63 for i in range(total)],
            max_tokens=total + args.output_len,
        )
        runner = ModelRunner(model, pool, manager)
        outputs = []
        with torch.inference_mode():
            for chunk_len in pieces:
                out = runner.prefill_chunk(seq, seq.prompt_pos, chunk_len)
                outputs.append(out.detach().clone())
                seq.prompt_pos += chunk_len
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        manager.check_invariants([seq])
        k, v = pool.gather_sequence(seq)
        return torch.cat(outputs, dim=1), k, v, list(seq.block_table)

    outcomes = []
    for p in args.prompts:
        cases = [(f'full_{p}', [p])]
        if p > 1:
            first = min(args.block_size, p-1)
            cases.append((f'split_{p}', [first, p-first]))
        for label, pieces in cases:
            ref_logits, ref_k, ref_v, ref_blocks = run_chunks(pieces, False)
            fast_logits, fast_k, fast_v, fast_blocks = run_chunks(pieces, True)
            torch.testing.assert_close(fast_logits, ref_logits, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(fast_k, ref_k, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(fast_v, ref_v, atol=2e-5, rtol=2e-4)
            if not torch.equal(fast_logits.argmax(-1), ref_logits.argmax(-1)):
                raise AssertionError(f'Argmax mismatch for {label}')
            if ref_blocks != fast_blocks:
                raise AssertionError(f'Physical block allocation mismatch for {label}')
            err = (fast_logits-ref_logits).abs().max().item()
            kv_err = max((fast_k-ref_k).abs().max().item(),
                         (fast_v-ref_v).abs().max().item())
            outcomes.append({'case': label, 'prompt_len': p,
                             'chunks': pieces, 'logits_max_abs': err,
                             'kv_max_abs': kv_err, 'token_equal': True})
            print(f'CORRECTNESS PASS {label:12} logits_max_abs={err:.3e} '
                  f'kv_max_abs={kv_err:.3e}', flush=True)
    return {'mode':'correctness', 'device':str(device), 'torch':torch.__version__,
            'atol':2e-5, 'rtol':2e-4, 'outcomes':outcomes}


def ops(args, device, model):
    if args.batch != 1:
        raise ValueError('--mode ops currently requires --batch 1')
    from torch.profiler import ProfilerActivity, profile
    summaries = []
    for p in args.prompts:
        for backend in ('reference','sdpa_v1'):
            model.set_prefill_fastpath(backend=='sdpa_v1')
            rt, seqs, _ = make_runtime(
                model, device, batch=1, prompt_len=p,
                output_len=args.output_len, block_size=args.block_size,
                token_budget=args.token_budget, capacity_factor=1.0,
                policy=args.policy, heads=args.heads, dim=args.dim,
            )
            # Profile the very first full Prefill step; no Decode here.
            if p > args.token_budget:
                raise ValueError('ops profiling expects prompt <= token budget')
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            activities = [ProfilerActivity.CPU]
            if device.type=='cuda':
                activities.append(ProfilerActivity.CUDA)
            with torch.inference_mode():
                with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
                    rt.step()
                    if device.type=='cuda':
                        torch.cuda.synchronize(device)
            events = prof.key_averages()
            lookup = {e.key:e for e in events}
            launches = lookup.get('cudaLaunchKernel')
            selections = ['aten::bmm', 'aten::einsum', 'aten::exp',
                          'aten::max', 'aten::sum','aten::scaled_dot_product_attention']
            row = {
                'prompt_len':p, 'backend':backend,
                'cudaLaunchKernel_calls': launches.count if launches else None,
                'operator_counts':{name:(lookup[name].count if name in lookup else 0)
                                   for name in selections},
                'prefill_tokens': rt.metrics_summary()['prefill_tokens'],
            }
            summaries.append(row)
            print(f'OPS P={p:3} backend={backend:9} '
                  f'launches={row["cudaLaunchKernel_calls"]} ops={row["operator_counts"]}', flush=True)
    return {'mode':'ops','torch':torch.__version__, 'device':str(device),
            'note':'Profiler instrumentation; counts only, not directly comparable latency',
            'summaries':summaries}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--mode', choices=['correctness','e2e','ops'], default='e2e')
    ap.add_argument('--prompts',type=csv_ints,default=[32,128,256])
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--output-len',type=int,default=8)
    ap.add_argument('--dim',type=int,default=256)
    ap.add_argument('--layers',type=int,default=4)
    ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--block-size',type=int,default=16)
    ap.add_argument('--token-budget',type=int,default=256)
    ap.add_argument('--policy',choices=['conservative','recompute'],default='conservative')
    ap.add_argument('--warmups',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--max-steps',type=int,default=10000)
    ap.add_argument('--output-json',type=Path,default=Path('artifacts/gpu/prefill_sdpa_v1_ab.json'))
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        ap.error('CUDA not available')
    if (args.dim<=0 or args.layers<=0 or args.heads<=0 or args.dim % args.heads
        or args.output_len<2 or args.token_budget<=0 or args.block_size<=0
        or args.batch<1 or args.warmups<0 or args.repeats<1):
        ap.error('invalid parameters')
    torch.set_num_threads(1)
    # Day 5: strict-FP32 correctness and performance A/B must use the same
    # matrix-multiplication precision across reference and SDPA fast paths.
    # Do NOT force the SDPA MATH backend: retain automatic CUDA selection.
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device=torch.device(args.device)
    model=make_model(args,device)
    result=(correctness(args,device,model) if args.mode=='correctness'
            else e2e(args,device,model) if args.mode=='e2e'
            else ops(args,device,model))
    result['precision'] = {
        'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'sdpa_backend': 'auto',
    }
    print(f'PRECISION {result["precision"]}', flush=True)
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'STEP4 {args.mode.upper()} PASS; saved {args.output_json}',flush=True)

if __name__=='__main__':
    main()
