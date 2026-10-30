"""Reference vs first-chunk SDPA v1 vs prefix-aware Chunked SDPA v2.

Always disables TF32 for the strict-FP32 cross-path correctness comparison.
The v2 implementation gathers prefix KV into temporary contiguous tensors;
this is a correctness-first transitional fastpath, not fused paged attention.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import torch

from experiments.gpu_e2e_runtime import run_once, stats, csv_ints
from experiments.gpu_prefill_fastpath_ab import make_model
from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model_runner import ModelRunner
from minivllm.sequence import SequenceState
from experiments.gpu_e2e_runtime import make_runtime
from types import SimpleNamespace

BACKENDS = ('reference', 'sdpa_v1', 'sdpa_v2')


def precision():
    return {
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'sdpa_backend':'auto',
    }


def e2e(args, device, model):
    old_args = SimpleNamespace(output_len=args.output_len, block_size=args.block_size,
                               token_budget=args.token_budget, heads=args.heads,
                               dim=args.dim, max_steps=args.max_steps)
    runs=[]
    for p in args.prompts:
        expected_tokens=None
        expected_counts=None
        for rep in range(args.warmups+args.repeats):
            order = BACKENDS[rep%3:]+BACKENDS[:rep%3]
            for backend in order:
                model.set_prefill_fastpath_mode(backend)
                run=run_once(model,old_args,device,batch=args.batch,
                             prompt_len=p,factor=args.kv_factor,policy=args.policy,
                             with_trace=False)
                counts=tuple(run[key] for key in (
                    'iterations','output_tokens','preemptions','repeated_kv_tokens',
                    'prefill_token_executions','decode_token_executions'))
                if expected_tokens is None:
                    expected_tokens=run['per_request_output']
                    expected_counts=counts
                if run['per_request_output'] != expected_tokens:
                    raise AssertionError(f'Output token mismatch {p=} {backend=} {rep=}')
                if counts!=expected_counts:
                    raise AssertionError(f'Lifecycle mismatch {p=} {backend=} {rep=}: {counts} vs {expected_counts}')
                if args.require_preemption and run['preemptions'] <= 0:
                    raise AssertionError(
                        f'Expected real preemption, but got preemptions=0; '
                        f'prompt={p}, batch={args.batch}, KV factor={args.kv_factor}. '
                        'Reduce --kv-factor or change the workload.'
                    )
                if rep>=args.warmups:
                    run['backend']=backend
                    run['repeat']=rep-args.warmups
                    runs.append(run)
                    print(f'PASS P={p:3} budget={args.token_budget:3} '
                          f'backend={backend:9} wall={run["total_wall_ms"]:.3f}ms '
                          f'TTFT={run["avg_ttft_ms"]:.3f}ms '
                          f'tok/s={run["output_tok_s"]:.1f} preempt={run["preemptions"]} '
                          f'replay={run["repeated_kv_tokens"]}',flush=True)
    groups=defaultdict(list)
    for r in runs:groups[(r['prompt_len'],r['backend'])].append(r)
    summaries=[]
    for p in args.prompts:
        row={'prompt_len':p,'token_budget':args.token_budget,'batch':args.batch,
             'kv_factor':args.kv_factor,'num_blocks':groups[(p,'reference')][0]['num_blocks'],
             'token_equality':True,'lifecycle_count_equality':True}
        for backend in BACKENDS:
            g=groups[(p,backend)]
            row[backend]={'wall_ms':stats([r['total_wall_ms'] for r in g]),
                          'ttft_ms':stats([r['avg_ttft_ms'] for r in g]),
                          'output_tok_s':stats([r['output_tok_s'] for r in g])}
        ref=row['reference']['wall_ms']['median']
        row['speedup_v1_vs_ref']=round(ref/row['sdpa_v1']['wall_ms']['median'],5)
        row['speedup_v2_vs_ref']=round(ref/row['sdpa_v2']['wall_ms']['median'],5)
        row['speedup_v2_vs_v1']=round(row['sdpa_v1']['wall_ms']['median']/row['sdpa_v2']['wall_ms']['median'],5)
        summaries.append(row)
    return {'mode':'e2e','torch':torch.__version__,'device':str(device),'precision':precision(),
            'note':'Closed-batch E2E, same weights, inputs and capacity; 3 alternating backend orders',
            'configuration':vars(args) | {'output_json':str(args.output_json)},
            'summaries':summaries,'runs':runs}


def correctness(args,device,model):
    if args.batch!=1:raise ValueError('correctness uses --batch 1')
    def run_chunks(chunks,backend):
        model.set_prefill_fastpath_mode(backend)
        total=sum(chunks)
        blocks=(total+args.block_size-1)//args.block_size+4
        manager=BlockManager(num_blocks=blocks,block_size=args.block_size)
        order=list(range(blocks));order=order[::2]+order[1::2]
        manager.free_blocks=list(reversed(order))
        pool=KVBlockPool(n_layer=args.layers,num_blocks=blocks,block_size=args.block_size,
                         n_head=args.heads,head_dim=args.dim//args.heads,
                         device=device,dtype=torch.float32)
        pool.k.fill_(float('nan'));pool.v.fill_(float('nan'))
        seq=SequenceState(seq_id='numerical',
                          prompt_tokens=[1+(i*7)%63 for i in range(total)],
                          max_tokens=total+args.output_len)
        runner=ModelRunner(model,pool,manager)
        outputs=[]
        with torch.inference_mode():
            for n in chunks:
                outputs.append(runner.prefill_chunk(seq,seq.prompt_pos,n).detach().clone())
                seq.prompt_pos+=n
            if device.type=='cuda':torch.cuda.synchronize(device)
        manager.check_invariants([seq])
        k,v=pool.gather_sequence(seq)
        return torch.cat(outputs,dim=1),k,v,list(seq.block_table)
    cases=[]
    for p in args.prompts:
        cases.append((f'full_{p}',[p]))
        if p>1:
            first=min(args.block_size,p-1)
            cases.append((f'split_{p}',[first,p-first]))
        if p>args.token_budget:
            chunks=[args.token_budget]*(p//args.token_budget)
            if p%args.token_budget:chunks.append(p%args.token_budget)
            cases.append((f'budget_{p}',chunks))
        if p>5:
            cases.append((f'offset5_{p}',[5,p-5]))
    results=[]
    for label,pieces in cases:
        ref=run_chunks(pieces,'reference')
        v1=run_chunks(pieces,'sdpa_v1')
        v2=run_chunks(pieces,'sdpa_v2')
        for backend,got in (('sdpa_v1',v1),('sdpa_v2',v2)):
            for tensor_idx,name in ((0,'logits'),(1,'K'),(2,'V')):
                torch.testing.assert_close(got[tensor_idx],ref[tensor_idx],atol=2e-5,rtol=2e-4,
                                            msg=f'{label} {backend} {name}')
            if not torch.equal(got[0].argmax(-1),ref[0].argmax(-1)):
                raise AssertionError(f'{label} {backend} argmax mismatch')
            if got[3]!=ref[3]:raise AssertionError(f'{label} {backend} physical layout mismatch')
            err=(got[0]-ref[0]).abs().max().item()
            kv_err=max((got[1]-ref[1]).abs().max().item(),
                       (got[2]-ref[2]).abs().max().item())
            results.append({'case':label,'chunks':pieces,'backend':backend,
                            'logits_max_abs':err,'kv_max_abs':kv_err})
            print(f'PASS {label:16} {backend:8} logits_err={err:.3e} KV_err={kv_err:.3e}',flush=True)
    return {'mode':'correctness','torch':torch.__version__,'device':str(device),
            'precision':precision(),'atol':2e-5,'rtol':2e-4,'outcomes':results}


def ops(args,device,model):
    if args.batch!=1:raise ValueError('ops requires batch 1')
    from torch.profiler import profile,ProfilerActivity
    rows=[]
    for p in args.prompts:
        if p<=args.token_budget:
            raise ValueError('ops requires prompt > token budget to capture second chunk')
        for backend in BACKENDS:
            model.set_prefill_fastpath_mode(backend)
            rt,seqs,_=make_runtime(model,device,batch=1,prompt_len=p,output_len=args.output_len,
                                   block_size=args.block_size,token_budget=args.token_budget,
                                   capacity_factor=1.0,policy=args.policy,heads=args.heads,dim=args.dim)
            with torch.inference_mode():
                rt.step() # First chunk, no profiler
                if device.type=='cuda':torch.cuda.synchronize(device)
                acts=[ProfilerActivity.CPU]
                if device.type=='cuda':acts.append(ProfilerActivity.CUDA)
                with profile(activities=acts,record_shapes=False,profile_memory=False) as prof:
                    rt.step() # Second chunk, nonzero prefix
                    if device.type=='cuda':torch.cuda.synchronize(device)
            lookup={e.key:e for e in prof.key_averages()}
            count=lambda key:lookup[key].count if key in lookup else 0
            row={'prompt_len':p,'backend':backend,'capture_step':2,
                 'cudaLaunchKernel_calls':count('cudaLaunchKernel'),
                 'operator_counts':{name:count(name) for name in ('aten::einsum','aten::max','aten::exp','aten::scaled_dot_product_attention', 'aten::index_select')},
                 'prefill_tokens_so_far':rt.metrics_summary()['prefill_tokens']}
            rows.append(row)
            print(f'OPS P={p} backend={backend} launches={row["cudaLaunchKernel_calls"]} '
                  f'ops={row["operator_counts"]}',flush=True)
    return {'mode':'ops','torch':torch.__version__,'device':str(device),'precision':precision(),
            'note':'Second chunk only, counts affected by profiler instrumentation',
            'summaries':rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--mode',choices=('correctness','e2e','ops'),default='e2e')
    ap.add_argument('--prompts',type=csv_ints,default=[128,256])
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--output-len',type=int,default=8)
    ap.add_argument('--dim',type=int,default=256)
    ap.add_argument('--layers',type=int,default=4)
    ap.add_argument('--heads',type=int,default=8)
    ap.add_argument('--block-size',type=int,default=16)
    ap.add_argument('--token-budget',type=int,default=64)
    ap.add_argument('--policy',choices=('conservative','recompute'),default='conservative')
    ap.add_argument('--kv-factor',type=float,default=1.0,
                    help='KV capacity factor; 1.0 fits all max demands, 0.5 creates contention')
    ap.add_argument('--require-preemption',action='store_true',
                    help='Fail if any compared backend does not actually preempt')
    ap.add_argument('--warmups',type=int,default=2)
    ap.add_argument('--repeats',type=int,default=7)
    ap.add_argument('--max-steps',type=int,default=10000)
    ap.add_argument('--output-json',type=Path,default=Path('artifacts/gpu/prefill_sdpa_v2_ab.json'))
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    if not (0 < args.kv_factor <= 1):ap.error('--kv-factor must be in (0,1]')
    if args.require_preemption and (args.mode != 'e2e' or args.policy != 'recompute'):
        ap.error('--require-preemption requires --mode e2e --policy recompute')
    if not (args.prompts and args.batch>0 and args.output_len>=2 and args.token_budget>0 and
            args.dim>0 and args.heads>0 and args.layers>0 and args.dim%args.heads==0):
        ap.error('invalid configuration')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device)
    model=make_model(args,device)
    result={'correctness':correctness,'e2e':e2e,'ops':ops}[args.mode](args,device,model)
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'V2 {args.mode.upper()} PASS; saved {args.output_json}',flush=True)

if __name__=='__main__':main()
