"""Export core MiniRuntime IterationRecord telemetry as JSON.

Examples:
  python -m experiments.record_runtime --policy recompute --blocks 2 \
    --output-json artifacts/observability/recompute_2blocks.json
  python -m experiments.record_runtime --policy conservative --blocks 3 \
    --workload mixed_short --output-json artifacts/observability/conservative_3blocks.json

This is a deterministic functional trace, not a performance benchmark.
"""
import argparse
import contextlib
import io
from types import SimpleNamespace

import torch

from .bench_preemption import make_runtime, make_workload
from minivllm.model import MiniGPT
from minivllm.sequence import Status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=('conservative', 'recompute'), default='recompute')
    p.add_argument('--workload', choices=('baseline', 'mixed_short'), default='baseline')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--sync-cuda-timing', action='store_true',
                   help='Force CUDA synchronization around timed stages')
    p.add_argument('--output-json', default='artifacts/observability/runtime_records.json')
    cfg = p.parse_args()
    if cfg.blocks < 1 or cfg.max_steps < 1:
        p.error('blocks and max-steps must be positive')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; choose --device cpu')
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = MiniGPT(n_embd=16, n_head=2, vocab_size=64,
                    block_size=256, dropout=0.0, n_layer=1).to(cfg.device).eval()
    options = SimpleNamespace(device=cfg.device, block_size=16, layers=1,
        heads=2, dim=16, max_running=4, token_budget=64,
        sync_cuda_timing=cfg.sync_cuda_timing)
    rt = make_runtime(model, options, cfg.policy, cfg.blocks)
    seqs = make_workload(cfg.workload)
    for seq in seqs:
        rt.add_request(seq)
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        for _ in range(cfg.max_steps):
            if all(seq.status == Status.FINISHED for seq in seqs):
                break
            rt.step()
            rt.block_manager.check_invariants(seqs)
        else:
            raise RuntimeError('Runtime did not finish the workload')
    if rt.block_manager.block_owner:
        raise AssertionError('KV blocks leaked')
    rt.export_records_json(cfg.output_json, metadata={
        'device': cfg.device,
        'policy': cfg.policy,
        'workload': cfg.workload,
        'num_blocks': cfg.blocks,
        'block_size': options.block_size,
        'request_outputs': {s.seq_id: list(s.generated_tokens) for s in seqs},
    })
    stats = rt.metrics_summary()
    print(f"PASS {cfg.policy} blocks={cfg.blocks} steps={stats['iterations']} "
          f"preemptions={stats['preemptions']} repeated_kv={stats['repeated_kv_tokens']}")
    print(f"Saved {cfg.output_json} ({len(rt.records)} iteration records)")


if __name__ == '__main__':
    main()
