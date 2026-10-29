#!/usr/bin/env python3
"""P0 GPU correctness checks for educational Mini-GPT Inference Runtime.

Copy to experiments/gpu_correctness_probe.py, then from the repository root run:
    python -m experiments.gpu_correctness_probe

This is a correctness probe, NOT a GPU performance benchmark.
"""
from __future__ import annotations

import argparse
import copy
import math
from types import SimpleNamespace

import torch

from minivllm.kv_block_pool import KVBlockPool
from minivllm.model import MiniGPT
from minivllm.paged_attention_batch import (
    paged_attention_batch_online_v1,
    paged_attention_batch_reference,
)
from experiments.bench_preemption import run_once


def close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    # These are small FP32 correctness fixtures; do not use BF16 tolerance here.
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4,
                               msg=lambda msg: f'{label}: {msg}')
    error = (actual - expected).abs().max().item()
    print(f'PASS {label}: max_abs={error:.8g}')


def attention_case(device: str):
    """Varied lengths + cross-block + padded block tables + NaN in unused KV."""
    torch.manual_seed(2026)
    k = torch.full((1, 4, 4, 2, 8), float('nan'))
    v = torch.full_like(k, float('nan'))
    # Logical request 0 uses physical blocks 0 (4 tokens), 1 (2 tokens).
    # Logical request 1 uses physical block 2 (3 tokens), padding block -1.
    for physical_block, valid_tokens in ((0, 4), (1, 2), (2, 3)):
        k[0, physical_block, :valid_tokens] = torch.randn(valid_tokens, 2, 8)
        v[0, physical_block, :valid_tokens] = torch.randn(valid_tokens, 2, 8)
    q = torch.randn(2, 2, 8)
    tables = torch.tensor([[0, 1], [2, -1]], dtype=torch.long)
    lengths = torch.tensor([6, 3], dtype=torch.long)

    pool = KVBlockPool(n_layer=1, num_blocks=4, block_size=4,
                       n_head=2, head_dim=8, device=device, dtype=torch.float32)
    pool.k.copy_(k.to(device))
    pool.v.copy_(v.to(device))
    query = q.to(device)
    table_d = tables.to(device)
    length_d = lengths.to(device)

    online = paged_attention_batch_online_v1(query, pool, table_d, length_d, 0)
    reference = paged_attention_batch_reference(query, pool, table_d, length_d, 0)
    if not bool(torch.isfinite(online).all().item()):
        raise AssertionError(f'Non-finite PagedAttention output on {device}')
    close(online, reference, f'{device} online vs reference')

    dense_outputs = []
    for b, num_tokens in enumerate(lengths.tolist()):
        gathered_k = []
        gathered_v = []
        for position in range(num_tokens):
            physical_block = int(tables[b, position // pool.block_size])
            offset = position % pool.block_size
            gathered_k.append(pool.k[0, physical_block, offset])
            gathered_v.append(pool.v[0, physical_block, offset])
        K = torch.stack(gathered_k, dim=0)  # [T, H, D]
        V = torch.stack(gathered_v, dim=0)
        scores = torch.einsum('hd,thd->ht', query[b], K) / math.sqrt(query.shape[-1])
        probs = torch.softmax(scores, dim=-1)
        dense_outputs.append(torch.einsum('ht,thd->hd', probs, V))
    dense = torch.stack(dense_outputs, dim=0)
    close(online, dense, f'{device} online vs independent dense attention')
    return online.detach().cpu()


def runtime_case(cpu_model, gpu_model, workload: str, blocks: int, policy: str) -> None:
    common = dict(block_size=16, layers=1, heads=2, dim=16,
                  max_running=4, token_budget=64, max_steps=200, workload=workload)
    cpu = run_once(cpu_model, SimpleNamespace(device='cpu', **common), policy, blocks)
    gpu = run_once(gpu_model, SimpleNamespace(device='cuda', **common), policy, blocks)
    if cpu['per_request_output'] != gpu['per_request_output']:
        raise AssertionError(f'CPU/GPU token mismatch: {workload=} {blocks=} {policy=}')
    for key in ('iterations', 'preemptions', 'repeated_kv_tokens',
                'prefill_token_executions', 'decode_token_executions'):
        if cpu[key] != gpu[key]:
            raise AssertionError(f'CPU/GPU {key} mismatch: {workload=} {blocks=} '
                                 f'{policy=}, cpu={cpu[key]}, gpu={gpu[key]}')
    print(f'PASS GPU/CPU E2E: {workload=} {blocks=} {policy=}, '
          f"iterations={gpu['iterations']}, preemptions={gpu['preemptions']}, "
          f"replayed={gpu['repeated_kv_tokens']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true',
                        help='Validate CPU attention fixture only (for script debugging)')
    args = parser.parse_args()
    print(f'torch={torch.__version__}; cuda_available={torch.cuda.is_available()}')
    cpu_attention = attention_case('cpu')
    if args.cpu_only:
        print('CPU-only probe PASS (GPU checks skipped by request)')
        return
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: GPU correctness has NOT been tested')
    # Avoid TF32 to keep this FP32 small-model cross-device comparison interpretable.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_attention = attention_case('cuda')
    close(gpu_attention, cpu_attention, 'CPU vs GPU PagedAttention')

    torch.manual_seed(2026)
    model_cpu = MiniGPT(n_embd=16, n_head=2, vocab_size=64, block_size=256,
                        dropout=0.0, n_layer=1).cpu().eval()
    model_gpu = copy.deepcopy(model_cpu).cuda().eval()
    with torch.inference_mode():
        for workload, blocks in (('baseline', 2), ('mixed_short', 3)):
            for policy in ('conservative', 'recompute'):
                runtime_case(model_cpu, model_gpu, workload, blocks, policy)
    torch.cuda.synchronize()
    print('GPU CORRECTNESS PROBE PASS')


if __name__ == '__main__':
    main()
