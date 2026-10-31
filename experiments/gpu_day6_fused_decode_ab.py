"""Day 6 Step 4: independent fused Triton paged decode numerical + performance A/B.

Run from the project root. No Runtime code changes or model integration.

Examples:
  python -m experiments.gpu_day6_fused_decode_ab --mode correctness --device cuda
  python -m experiments.gpu_day6_fused_decode_ab --mode bench --device cuda \\
    --batches 1,2,4 --contexts 32,128,256 --repeats 7 \\
    --output-json artifacts/gpu/day6_step4_kernel_ab.json
  python -m experiments.gpu_day6_fused_decode_ab --mode ops --device cuda \\
    --batches 2 --contexts 256
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from minivllm.kv_block_pool import KVBlockPool
from minivllm.paged_attention_batch import paged_attention_batch_online_v1
from minivllm.triton_paged_decode import paged_attention_batch_triton_v0


def ints(s):
    v = [int(part) for part in s.split(',') if part.strip()]
    if not v or min(v) < 1:
        raise argparse.ArgumentTypeError('expected positive comma-separated integers')
    return v


def build_case(device, batch=2, length=128, heads=8, dim=32, block_size=16,
               layout='mixed', seed=20261030, scale=0.3):
    """Synthetic *pre-filled* paged KV, with noncontiguous physical blocks.

    The last logical block has deliberately non-finite unfilled tokens, and
    unused block-table entries are -1.  FP32 and n_layer=2 layer_idx=1.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    if layout == 'mixed':
        lengths = [max(1, length - (i % 4) * max(1, length // 7)) for i in range(batch)]
    elif layout == 'fixed':
        lengths = [length] * batch
    else:
        raise ValueError('layout must be fixed/mixed')
    counts = [math.ceil(x / block_size) for x in lengths]
    n_phys = sum(counts) + 7
    layer = 1
    pool = KVBlockPool(2, n_phys, block_size, heads, dim, device='cpu', dtype=torch.float32)
    pool.k.fill_(float('nan'))
    pool.v.fill_(float('nan'))
    table = torch.full((batch, max(counts)), -1, dtype=torch.int64)
    order = torch.randperm(n_phys, generator=g).tolist()
    cur = 0
    for b, (length_b, blocks_b) in enumerate(zip(lengths, counts)):
        for logical in range(blocks_b):
            phys = order[cur]
            cur += 1
            table[b, logical] = phys
            remaining = length_b - logical * block_size
            valid = min(block_size, remaining)
            pool.k[layer, phys, :valid] = torch.randn(
                (valid, heads, dim), generator=g, dtype=torch.float32) * scale
            pool.v[layer, phys, :valid] = torch.randn(
                (valid, heads, dim), generator=g, dtype=torch.float32) * scale
    # Noncontiguous last dimension verifies actual q strides are respected.
    q = torch.randn((batch, heads, 2 * dim), generator=g)[..., ::2].mul_(scale)
    if device.type == 'cuda':
        kp = pool.k.to(device)
        vp = pool.v.to(device)
        pool.k, pool.v = kp, vp
        q, table = q.to(device), table.to(device)
    ctx = torch.tensor(lengths, dtype=torch.int64, device=device)
    return q, pool, table, ctx, layer


def dense_gold(q, pool, table, lengths, layer):
    """Independent dense softmax oracle, not the online recurrence."""
    bsz, heads, dim = q.shape
    out = []
    for b in range(bsz):
        length = int(lengths[b].item())
        parts_k, parts_v = [], []
        for j in range(math.ceil(length / pool.block_size)):
            phys = int(table[b, j].item())
            n_valid = min(pool.block_size, length - j * pool.block_size)
            parts_k.append(pool.k[layer, phys, :n_valid])
            parts_v.append(pool.v[layer, phys, :n_valid])
        k = torch.cat(parts_k, dim=0)
        v = torch.cat(parts_v, dim=0)
        scores = torch.einsum('hd,thd->ht', q[b], k) / math.sqrt(dim)
        probs = torch.softmax(scores, dim=-1)
        out.append(torch.einsum('ht,thd->hd', probs, v))
    return torch.stack(out, dim=0)


def compare(q, pool, table, ctx, layer, atol, rtol):
    with torch.inference_mode():
        expected = paged_attention_batch_online_v1(q, pool, table, ctx, layer)
        got = paged_attention_batch_triton_v0(q, pool, table, ctx, layer)
        # Independent oracle catches matching errors in online recurrences.
        dense = dense_gold(q, pool, table, ctx, layer)
        torch.testing.assert_close(expected, dense, atol=atol, rtol=rtol,
                                   msg='PyTorch paged reference disagrees with dense oracle')
        torch.testing.assert_close(got, dense, atol=atol, rtol=rtol,
                                   msg='Triton fused decode disagrees with dense oracle')
        diff = (got - expected).abs()
        return {'max_abs': float(diff.max().item()),
                'mean_abs': float(diff.mean().item()),
                'finite': bool(torch.isfinite(got).all().item())}


def timed(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def percentile(values, pct):
    x = sorted(values)
    return x[min(len(x)-1, max(0, math.ceil(pct*len(x))-1))]


def stats(times):
    return {'median_ms': statistics.median(times), 'mean_ms': statistics.mean(times),
            'p95_ms': percentile(times, 0.95), 'samples_ms': times}


def ops(fn, device):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        acts.append(ProfilerActivity.CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    with profile(activities=acts) as prof:
        with torch.inference_mode():
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
    keywords = ('cudaLaunchKernel', 'aten::einsum', 'aten::bmm', 'aten::where',
                'aten::max', 'aten::exp', 'aten::index', 'aten::scaled_dot_product_attention')
    events = prof.key_averages()
    return {name: sum(item.count for item in events if item.key == name) for name in keywords}


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    if args.device != 'cuda':
        raise SystemExit('Triton CUDA kernel requires --device cuda')
    # Keep same strict-FP32 policy as Day5 SDPA numerical validation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if args.heads < 1 or args.dim < 1 or args.dim > 128 or args.block_size > 64:
        raise SystemExit('heads>0, 1<=dim<=128, block_size<=64 required')
    result = {'mode':args.mode,'definition':'standalone attention only, no QKV/MLP/KV writes/Scheduler',
              'device':str(device),'gpu':torch.cuda.get_device_name(device),
              'torch':torch.__version__,'layout':args.layout,
              'fp32_tf32_disabled':True,'results':[]}
    if args.mode == 'correctness':
        cases = [(1, 1), (1, 15), (1, 16), (2, 17), (4, 33),
                 (4, 64), (4, 65), (2, 128), (4, 129), (2, 256), (4, 257)]
        for batch, length in cases:
            q, pool, table, ctx, layer = build_case(
                device, batch, length, args.heads, args.dim, args.block_size,
                'mixed', seed=20261030+batch+length)
            row = compare(q, pool, table, ctx, layer, args.atol, args.rtol)
            row.update(batch=batch, max_context=length, lengths=ctx.cpu().tolist())
            result['results'].append(row)
            print(f'CORRECTNESS PASS B={batch:2d} L={length:3d} '
                  f'max_abs={row["max_abs"]:.3e} mean_abs={row["mean_abs"]:.3e}', flush=True)
    else:
        for length in args.contexts:
            for batch in args.batches:
                q, pool, table, ctx, layer = build_case(
                    device, batch, length, args.heads, args.dim, args.block_size,
                    args.layout, seed=20261030+batch+length)
                corr = compare(q, pool, table, ctx, layer, args.atol, args.rtol)
                ref = lambda: paged_attention_batch_online_v1(q, pool, table, ctx, layer)
                tri = lambda: paged_attention_batch_triton_v0(q, pool, table, ctx, layer)
                # Compile + warm up OUTSIDE measurement and profile.
                with torch.inference_mode():
                    for _ in range(args.warmups):
                        ref()
                        tri()
                    torch.cuda.synchronize(device)
                    if args.mode == 'ops':
                        row = {'batch':batch, 'context':length, 'correctness':corr,
                               'reference_counts':ops(ref,device),
                               'triton_counts':ops(tri,device)}
                        print('OPS', row, flush=True)
                    else:
                        a, b = [], []
                        for i in range(args.repeats):
                            if i % 2:
                                b.append(timed(tri,device))
                                a.append(timed(ref,device))
                            else:
                                a.append(timed(ref,device))
                                b.append(timed(tri,device))
                        row = {'batch':batch, 'context':length, 'correctness':corr,
                               'reference':stats(a), 'triton':stats(b),
                               'speedup':statistics.median(a)/statistics.median(b)}
                        print(f'PASS B={batch:2d} L={length:3d} '
                              f'ref={row["reference"]["median_ms"]:.3f}ms '
                              f'triton={row["triton"]["median_ms"]:.3f}ms '
                              f'speedup={row["speedup"]:.2f}x',flush=True)
                    result['results'].append(row)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),encoding='utf-8')
        print(f'DAY6 STEP4 {args.mode.upper()} PASS; saved {args.output_json}',flush=True)
    else:
        print(f'DAY6 STEP4 {args.mode.upper()} PASS', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--mode',choices=['correctness','bench','ops'],default='correctness')
    p.add_argument('--batches',type=ints,default=[1,2,4])
    p.add_argument('--contexts',type=ints,default=[32,128,256])
    p.add_argument('--layout',choices=['fixed','mixed'],default='mixed')
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--dim',type=int,default=32,help='head dimension, not hidden dimension')
    p.add_argument('--block-size',type=int,default=16)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--atol',type=float,default=2e-4)
    p.add_argument('--rtol',type=float,default=2e-4)
    p.add_argument('--output-json',type=Path,default=Path('artifacts/gpu/day6_step4.json'))
    args=p.parse_args()
    if args.warmups<0 or args.repeats<1:
        p.error('warmups >= 0 and repeats >= 1 required')
    run(args)


if __name__=='__main__':
    main()
