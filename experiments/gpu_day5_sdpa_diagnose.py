"""Diagnose GPU FP32 SDPA/reference mismatches in mini-gpt-inference.

Copy this file into experiments/gpu_day5_sdpa_diagnose.py and run
    python -m experiments.gpu_day5_sdpa_diagnose --device cuda --prompts 32,128

No project files are changed. Compares both paths under the same precision
settings, first-layer through final logits, and verifies Paged KV outputs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.gpu_prefill_fastpath_ab import make_model
from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model_runner import ModelRunner
from minivllm.sequence import SequenceState


def run_chunks(model, device, prompt, pieces, fast):
    model.set_prefill_fastpath(fast)
    block_size = 16
    count = (prompt + block_size - 1) // block_size + 4
    manager = BlockManager(num_blocks=count, block_size=block_size)
    order = list(range(count))
    order = order[::2] + order[1::2]
    manager.free_blocks = list(reversed(order))
    pool = KVBlockPool(
        n_layer=model.n_layer, num_blocks=count, block_size=block_size,
        n_head=model.blocks[0].attn.n_head,
        head_dim=model.blocks[0].attn.head_size,
        device=device, dtype=torch.float32,
    )
    pool.k.fill_(float('nan'))
    pool.v.fill_(float('nan'))
    seq = SequenceState(
        seq_id='numerical',
        prompt_tokens=[1 + (i * 7) % 63 for i in range(prompt)],
        max_tokens=prompt + 8,
    )
    runner = ModelRunner(model, pool, manager)
    captures = {}
    original_methods = [block.forward_prefill_chunk for block in model.blocks]
    for layer_index, (block, original) in enumerate(zip(model.blocks, original_methods)):
        def capture(x, context, *, _index=layer_index, _original=original):
            out = _original(x, context)
            captures[_index] = out.detach().clone()
            return out
        block.forward_prefill_chunk = capture
    try:
        outputs = []
        with torch.inference_mode():
            for chunk_len in pieces:
                logits = runner.prefill_chunk(seq, seq.prompt_pos, chunk_len)
                outputs.append(logits.detach().clone())
                seq.prompt_pos += chunk_len
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        manager.check_invariants([seq])
        kv_k, kv_v = pool.gather_sequence(seq)
        return torch.cat(outputs, dim=1), kv_k, kv_v, captures
    finally:
        for block, original in zip(model.blocks, original_methods):
            block.forward_prefill_chunk = original


def compare(ref, fast, atol, rtol):
    rlogits, rk, rv, rlayers = ref
    flogits, fk, fv, flayers = fast
    diff = (flogits - rlogits).abs()
    threshold = atol + rtol * rlogits.abs()
    layer_error = [float((flayers[i]-rlayers[i]).abs().max().item())
                   for i in sorted(rlayers)]
    return dict(
        logits_max_abs=float(diff.max().item()),
        logits_mismatched=int((diff > threshold).sum().item()),
        logits_elements=rlogits.numel(),
        argmax_equal=bool(torch.equal(flogits.argmax(-1), rlogits.argmax(-1))),
        k_max_abs=float((fk-rk).abs().max().item()),
        v_max_abs=float((fv-rv).abs().max().item()),
        block_outputs_max_abs=layer_error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prompts', default='32,128')
    parser.add_argument('--output-json', default='artifacts/gpu/day5_sdpa_diagnose.json')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a CPU smoke test')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    prompts = [int(s.strip()) for s in args.prompts.split(',')]
    model = make_model(SimpleNamespace(dim=256, heads=8, layers=4,
                                        prompts=prompts, output_len=8), device)
    flags = {
        'matmul_tf32_before': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32_before': torch.backends.cudnn.allow_tf32,
        'matmul_precision_before': torch.get_float32_matmul_precision(),
    }
    print(f'torch={torch.__version__}, device={device}, flags={flags}', flush=True)
    records = []
    for prompt in prompts:
        for layout in ['full', 'split']:
            pieces = [prompt] if layout == 'full' else [min(16,prompt-1), prompt-min(16,prompt-1)]
            # The reference and fast path are measured under identical flags.
            for label, force_math, ieee in (
                ('auto_current', False, False),
                ('auto_no_tf32', False, True),
                ('math_current', True, False),
                ('math_no_tf32', True, True),
            ):
                prev_matmul = torch.backends.cuda.matmul.allow_tf32
                prev_cudnn = torch.backends.cudnn.allow_tf32
                try:
                    if ieee:
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False
                    ctx = sdpa_kernel(SDPBackend.MATH) if force_math else nullcontext()
                    with ctx:
                        reference = run_chunks(model, device, prompt, pieces, False)
                        fast = run_chunks(model, device, prompt, pieces, True)
                    rec = compare(reference, fast, atol=2e-5, rtol=2e-4)
                    rec.update(prompt=prompt, layout=layout, pieces=pieces,
                               case=label, status='PASS' if rec['logits_mismatched']==0 else 'FAIL')
                    records.append(rec)
                    print(f"{rec['status']:4} P={prompt:3d} {layout:5s} {label:15s} "
                          f"logits_max={rec['logits_max_abs']:.3e} "
                          f"bad={rec['logits_mismatched']}/{rec['logits_elements']} "
                          f"k_max={rec['k_max_abs']:.3e} "
                          f"layer_err={[round(e,7) for e in rec['block_outputs_max_abs']]}",
                          flush=True)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = prev_matmul
                    torch.backends.cudnn.allow_tf32 = prev_cudnn
    output = {'torch': torch.__version__, 'device': str(device), 'initial_flags': flags,
              'model': {'dim': 256, 'layers': 4, 'heads': 8}, 'results': records}
    dest=Path(args.output_json)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(f'Saved {dest}',flush=True)


if __name__ == '__main__':
    main()
