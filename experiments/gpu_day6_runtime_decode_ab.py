"""Day 6 Step 5: integrated PyTorch vs Triton batched Decode A/B.

Reference and Triton use identical SDPA-v2 Prefill, weights, inputs, KV
capacity and scheduling. Triton ONLY replaces paged decode attention. Model
and Runtime are constructed outside timed E2E region. FP32/TF32 off.

Modes:
  correctness : lock-step Runtime comparisons of per-step logits, active
                logical KV and generated tokens. Run on CUDA for Triton.
  e2e         : alternating-backend boundary-synchronized E2E benchmarking.
  ops         : profile one real ModelRunner.decode_batch invocation, not the
                entire run; account for both cuLaunchKernel/cudaLaunchKernel.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from experiments.gpu_e2e_runtime import (csv_ints, make_runtime, run_once, stats)
from experiments.gpu_prefill_fastpath_ab import make_model
from minivllm.sequence import Status

BACKENDS=("reference","triton_v0")


def config(args, device):
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model=make_model(args,device)
    model.set_prefill_fastpath_mode('sdpa_v2')
    model.set_decode_attention_backend('reference')
    return model


def counts(run):
    return tuple(run[k] for k in (
        'iterations','output_tokens','preemptions','repeated_kv_tokens',
        'prefill_token_executions','decode_token_executions'))


def _attach_logits_probe(runtime):
    runner=runtime.runner
    original=runner.decode_batch
    runtime._captured_decode_logits=None
    def wrapped(seqs):
        logits,tokens=original(seqs)
        runtime._captured_decode_logits=logits.detach().clone()
        return logits,tokens
    runner.decode_batch=wrapped


def lockstep(args,device,model,batch,prompt):
    """Two independent KV pools and schedulers, identical weights.

    Alternating a single shared model is safe: it is in eval/inference_mode,
    all CUDA operations share the default stream, and each pool is distinct.
    """
    rt0,ss0,cap0=make_runtime(model,device,batch=batch,prompt_len=prompt,
        output_len=args.output_len,block_size=args.block_size,
        token_budget=args.token_budget,capacity_factor=args.kv_factor,
        policy=args.policy,heads=args.heads,dim=args.dim)
    rt1,ss1,cap1=make_runtime(model,device,batch=batch,prompt_len=prompt,
        output_len=args.output_len,block_size=args.block_size,
        token_budget=args.token_budget,capacity_factor=args.kv_factor,
        policy=args.policy,heads=args.heads,dim=args.dim)
    assert cap0==cap1
    _attach_logits_probe(rt0);_attach_logits_probe(rt1)
    max_logit_abs=0.0;max_kv_abs=0.0;decode_events=0
    with torch.inference_mode():
        for step in range(args.max_steps):
            if all(x.status == Status.FINISHED for x in ss0):
                break
            rt0._captured_decode_logits=None
            rt1._captured_decode_logits=None
            model.set_decode_attention_backend('reference')
            rt0.step()
            model.set_decode_attention_backend('triton_v0')
            rt1.step()
            a,b=rt0._captured_decode_logits,rt1._captured_decode_logits
            if (a is None)!=(b is None):
                raise AssertionError(f'decode dispatch differs at {step=}')
            if a is not None:
                torch.testing.assert_close(a,b,atol=args.atol,rtol=args.rtol,
                    msg=f'Decode Logits mismatch at {step=} {batch=} {prompt=}')
                max_logit_abs=max(max_logit_abs,(a-b).abs().max().item())
                decode_events+=1
            for x,y in zip(ss0,ss1):
                if (x.status != y.status or x.length!=y.length or
                    x.prompt_pos!=y.prompt_pos or x.generated_tokens!=y.generated_tokens or
                    x.block_table!=y.block_table):
                    raise AssertionError(f'sequence state mismatch {x.seq_id} at {step=}')
                if x.status!=Status.FINISHED and x.length>0:
                    k0,v0=rt0.runner.kv_pool.gather_sequence(x)
                    k1,v1=rt1.runner.kv_pool.gather_sequence(y)
                    for u,v in ((k0,k1),(v0,v1)):
                        torch.testing.assert_close(u,v,atol=args.atol,rtol=args.rtol,
                            msg=f'logical KV mismatch {x.seq_id} at {step=}')
                        max_kv_abs=max(max_kv_abs,(u-v).abs().max().item())
            if rt0.metrics_summary()!=rt1.metrics_summary():
                # Host timing differs, so compare lifecycle counts only below.
                s0,s1=rt0.metrics_summary(),rt1.metrics_summary()
                for key in ('iterations','scheduled_tokens','prefill_tokens','decode_tokens',
                            'output_tokens','preemptions','repeated_kv_tokens'):
                    if s0[key]!=s1[key]:
                        raise AssertionError(f'{key} mismatch at {step=}: {s0[key]} vs {s1[key]}')
        else:
            raise AssertionError(f'run exceeded max_steps={args.max_steps}')
    for ss in (ss0,ss1):
        if not all(x.status == Status.FINISHED for x in ss):
            raise AssertionError('unfinished requests')
    for rt in (rt0,rt1):
        if rt.block_manager.block_owner or len(rt.block_manager.free_blocks)!=cap0:
            raise AssertionError('KV capacity leak after completed workload')
        rt.block_manager.check_invariants(ss0 if rt is rt0 else ss1)
    s=rt0.metrics_summary()
    if decode_events==0:
        raise AssertionError('no Decode calls exercised')
    if args.require_preemption and s['preemptions']==0:
        raise AssertionError('preemption required but none occurred')
    return {'batch':batch,'prompt_len':prompt,'max_logit_abs':max_logit_abs,
        'max_logical_kv_abs':max_kv_abs,'decode_events':decode_events,
        'preemptions':s['preemptions'],'repeated_kv_tokens':s['repeated_kv_tokens'],
        'iterations':s['iterations'],'outputs_equal':True,'lifecycle_equal':True}


def correctness(args,device,model):
    rows=[]
    for p in args.prompts:
        for b in args.batches:
            row=lockstep(args,device,model,b,p)
            rows.append(row)
            print(f'CORRECTNESS PASS B={b} P={p} '
                f'logits_max_abs={row["max_logit_abs"]:.3e} '
                f'KV_max_abs={row["max_logical_kv_abs"]:.3e} '
                f'preempt={row["preemptions"]} replay={row["repeated_kv_tokens"]}',flush=True)
    return {'mode':'correctness','results':rows}


def e2e(args,device,model):
    runargs=SimpleNamespace(output_len=args.output_len,block_size=args.block_size,
        token_budget=args.token_budget,heads=args.heads,dim=args.dim,max_steps=args.max_steps)
    rows=[];summaries=[]
    for p in args.prompts:
        for b in args.batches:
            ref_outputs=None;ref_counts=None;group=defaultdict(list)
            for rep in range(args.warmups+args.repeats):
                order=BACKENDS if rep%2==0 else BACKENDS[::-1]
                for backend in order:
                    model.set_decode_attention_backend(backend)
                    with torch.inference_mode():
                        run=run_once(model,runargs,device,batch=b,prompt_len=p,
                            factor=args.kv_factor,policy=args.policy,with_trace=False)
                    if ref_outputs is None:
                        ref_outputs=run['per_request_output'];ref_counts=counts(run)
                    if run['per_request_output']!=ref_outputs:
                        raise AssertionError(f'generated token mismatch {b=} {p=} {backend=}')
                    if counts(run)!=ref_counts:
                        raise AssertionError(f'lifecycle mismatch {b=} {p=} {backend=}: '
                            f'{counts(run)} vs {ref_counts}')
                    if args.require_preemption and run['preemptions']==0:
                        raise AssertionError(f'no preemption {b=} {p=} {backend=}')
                    if rep>=args.warmups:
                        run.update(backend=backend,repeat=rep-args.warmups)
                        rows.append(run);group[backend].append(run)
                        print(f'PASS B={b:2} P={p:3} backend={backend:9} '
                            f'E2E={run["total_wall_ms"]:.3f}ms '
                            f'TTFT={run["avg_ttft_ms"]:.3f}ms '
                            f'tok/s={run["output_tok_s"]:.1f} '
                            f'preempt={run["preemptions"]}',flush=True)
            summary={'batch':b,'prompt_len':p,'token_equality':True,
                'lifecycle_equality':True,'preemptions':ref_counts[2]}
            for backend in BACKENDS:
                group_runs=group[backend]
                summary[backend]={
                    'wall_ms':stats([r['total_wall_ms'] for r in group_runs]),
                    'ttft_ms':stats([r['avg_ttft_ms'] for r in group_runs]),
                    'tpot_ms':stats([r['avg_tpot_ms'] for r in group_runs]),
                    'output_tok_s':stats([r['output_tok_s'] for r in group_runs]),
                }
            summary['e2e_speedup']=round(summary['reference']['wall_ms']['median'] /
                summary['triton_v0']['wall_ms']['median'],5)
            summaries.append(summary)
    return {'mode':'e2e','summaries':summaries,'runs':rows}


def ops(args,device,model):
    if device.type!='cuda':
        raise ValueError('ops mode requires CUDA')
    from torch.profiler import profile,ProfilerActivity
    results=[]
    for backend in BACKENDS:
        model.set_decode_attention_backend(backend)
        b,p=max(args.batches),max(args.prompts)
        rt,seqs,_=make_runtime(model,device,batch=b,prompt_len=p,
            output_len=args.output_len,block_size=args.block_size,
            token_budget=args.token_budget,capacity_factor=args.kv_factor,
            policy=args.policy,heads=args.heads,dim=args.dim)
        original=rt.runner.decode_batch
        captured=[]
        def probe(sequences):
            if not captured and len(sequences)>=args.capture_batch:
                # Instrument only one real decode invocation, and report the
                # actual batch size selected by Scheduler.
                torch.cuda.synchronize(device)
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                    result=original(sequences)
                    torch.cuda.synchronize(device)
                lookup=prof.key_averages()
                selected=('cudaLaunchKernel','cuLaunchKernel','aten::einsum','aten::bmm',
                    'aten::where','aten::max','aten::exp','aten::index',
                    'aten::scaled_dot_product_attention')
                counts={key:sum(e.count for e in lookup if e.key==key) for key in selected}
                kernels=[{'name':e.key,'count':e.count,'cuda_ms':e.self_device_time_total/1000.0}
                    for e in lookup if getattr(e,'device_type',None) is not None
                    and e.self_device_time_total>0]
                kernels=sorted(kernels,key=lambda x:-x['cuda_ms'])[:12]
                captured.append({'batch':len(sequences),'counts':counts,'top_cuda_events':kernels})
                return result
            return original(sequences)
        rt.runner.decode_batch=probe
        with torch.inference_mode():
            for _ in range(args.max_steps):
                if all(s.status==Status.FINISHED for s in seqs):break
                rt.step()
            else:raise AssertionError('max_steps exceeded')
        if not captured:
            raise AssertionError(f'No decode call with batch >= {args.capture_batch}')
        row={'backend':backend,'captured':captured[0]}
        results.append(row)
        print(f'OPS backend={backend} B={captured[0]["batch"]} '
            f'counts={captured[0]["counts"]}',flush=True)
    return {'mode':'ops','results':results,
        'note':'Profiler is intrusive. Kernel launches include cuLaunchKernel and cudaLaunchKernel.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--mode',choices=('correctness','e2e','ops'),default='correctness')
    ap.add_argument('--prompts',type=csv_ints,default=[128,256])
    ap.add_argument('--batches',type=csv_ints,default=[1,4])
    ap.add_argument('--token-budget',type=int,default=64)
    ap.add_argument('--output-len',type=int,default=8)
    ap.add_argument('--dim',type=int,default=256)
    ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--layers',type=int,default=4)
    ap.add_argument('--block-size',type=int,default=16)
    ap.add_argument('--policy',choices=('conservative','recompute'),default='conservative')
    ap.add_argument('--kv-factor',type=float,default=1.0)
    ap.add_argument('--require-preemption',action='store_true')
    ap.add_argument('--capture-batch',type=int,default=2)
    ap.add_argument('--warmups',type=int,default=2)
    ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--max-steps',type=int,default=10000)
    ap.add_argument('--atol',type=float,default=2e-5)
    ap.add_argument('--rtol',type=float,default=2e-4)
    ap.add_argument('--output-json',type=Path,default=Path('artifacts/gpu/day6_step5.json'))
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    if args.device!='cuda':
        ap.error('Triton v0 is CUDA only; use CPU unit tests for smoke')
    if (args.dim<1 or args.heads<1 or args.dim%args.heads or args.dim//args.heads>128
        or args.block_size<1 or args.block_size>64 or args.block_size&(args.block_size-1)
        or args.token_budget<1 or args.output_len<2 or args.layers<1
        or args.kv_factor<=0 or args.max_steps<1 or args.warmups<0 or args.repeats<1):
        ap.error('invalid dimensions or benchmark settings')
    if args.require_preemption and (args.policy!='recompute' or max(args.batches)<2):
        ap.error('require-preemption needs --policy recompute and batch>=2')
    device=torch.device(args.device)
    model=config(args,device)
    if args.mode=='correctness':result=correctness(args,device,model)
    elif args.mode=='e2e':result=e2e(args,device,model)
    else:result=ops(args,device,model)
    result.update({'torch':torch.__version__,'device':str(device),
        'gpu':torch.cuda.get_device_name(device),
        'precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision':torch.get_float32_matmul_precision()},
        'prefill_backend':'sdpa_v2',
        'decode_backends':BACKENDS,
        'configuration':vars(args)|{'output_json':str(args.output_json)}})
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DAY6 STEP5 '+args.mode.upper()+' PASS; saved '+str(args.output_json),flush=True)

if __name__=='__main__':main()
