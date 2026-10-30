"""Mini-GPT Runtime: decode-only serial-vs-batched scaling experiment.

Place at experiments/gpu_batch_decode_scaling.py and run from repository root:
  python -m experiments.gpu_batch_decode_scaling --device cuda \
      --batch-sizes 1,2,4,8,16,32 --context-lengths 128,256,512 \
      --dim 256 --layers 4 --heads 8 --warmups 3 --repeats 10 \
      --output-json artifacts/gpu/batch_decode_scaling.json

This is a synthetic-KV, one-step ModelRunner microbenchmark, NOT a serving or
full-prefix Prefill+Decode benchmark. It measures synchronized host wall time,
including Python, host/device synchronization, .item/.tolist and kernel launches.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from minivllm.block_manager import BlockManager
from minivllm.kv_block_pool import KVBlockPool
from minivllm.model import MiniGPT
from minivllm.model_runner import ModelRunner
from minivllm.sequence import SequenceState


def csv_ints(text: str) -> list[int]:
    items = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not items or any(v <= 0 for v in items):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return items


def length_for_request(context_len: int, idx: int, layout: str) -> int:
    if layout == "fixed":
        return context_len
    # Ragged lengths keep real per-request padding and mixed block-table sizes.
    return max(1, context_len - (idx % 4) * max(1, context_len // 8))


def build_runner(model, device, lengths, block_size, heads, dim, vocab_size):
    blocks_per_seq = [math.ceil((length + 1) / block_size) for length in lengths]
    n_blocks = sum(blocks_per_seq)
    manager = BlockManager(num_blocks=n_blocks, block_size=block_size)
    pool = KVBlockPool(
        n_layer=model.n_layer, num_blocks=n_blocks, block_size=block_size,
        n_head=heads, head_dim=dim // heads, device=device, dtype=torch.float32,
    )
    sequences = []
    for i, length in enumerate(lengths):
        prompt = [(i * 17 + j) % vocab_size for j in range(length)]
        seq = SequenceState(
            seq_id=f"request_{i}", prompt_tokens=prompt, max_tokens=length + 2,
        )
        # Preallocate the next decode slot, excluding allocation from timing.
        # Historical K/V values are injected below rather than computed via Prefill.
        manager.reserve_slots(seq, length + 1)
        seq.length = length
        seq.last_token = prompt[-1]
        sequences.append(seq)
    return ModelRunner(model, pool, manager), sequences


def reset_lengths(sequences, lengths):
    for seq, length in zip(sequences, lengths):
        seq.length = length


def serial_decode(runner, sequences):
    return [runner.decode_one(seq) for seq in sequences]


def batch_decode(runner, sequences):
    return runner.decode_batch(sequences)


def check_correctness(model, serial_runner, batch_runner, serial_seq, batch_seq,
                      lengths, atol, rtol, device):
    captured = []
    def hook(_module, _inputs, output):
        captured.append(output[:, -1, :].detach().clone())
    reset_lengths(serial_seq, lengths)
    reset_lengths(batch_seq, lengths)
    with model.language_model_head.register_forward_hook(hook):
        serial_tokens = serial_decode(serial_runner, serial_seq)
    # No hook on the batch path: it returns logits directly.
    batch_logits, batch_tokens = batch_decode(batch_runner, batch_seq)
    ref_logits = torch.cat(captured, dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diff = float((ref_logits - batch_logits).abs().max().item())
    if serial_tokens != batch_tokens:
        raise AssertionError(
            f"Token mismatch: serial={serial_tokens}, batch={batch_tokens}, max_abs={diff}"
        )
    if not torch.allclose(ref_logits, batch_logits, atol=atol, rtol=rtol):
        raise AssertionError(f"Logits mismatch: max_abs={diff}, atol={atol}, rtol={rtol}")
    return diff


def timed_call(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000


def summarize(values):
    xs = sorted(values)
    return {
        "median_ms": round(statistics.median(xs), 6),
        "mean_ms": round(statistics.mean(xs), 6),
        "min_ms": round(xs[0], 6),
        "max_ms": round(xs[-1], 6),
        "p95_ms": round(xs[min(len(xs) - 1, math.ceil(len(xs) * 0.95) - 1)], 6),
    }


def one_config(model, args, device, batch_size, context_len):
    lengths = [length_for_request(context_len, i, args.layout) for i in range(batch_size)]
    serial_runner, serial_seq = build_runner(
        model, device, lengths, args.block_size, args.heads, args.dim, args.vocab_size,
    )
    batch_runner, batch_seq = build_runner(
        model, device, lengths, args.block_size, args.heads, args.dim, args.vocab_size,
    )
    with torch.no_grad():
        # Same synthetic historical KV cache in both runners.
        serial_runner.kv_pool.k.normal_(mean=0, std=0.2)
        serial_runner.kv_pool.v.normal_(mean=0, std=0.2)
        batch_runner.kv_pool.k.copy_(serial_runner.kv_pool.k)
        batch_runner.kv_pool.v.copy_(serial_runner.kv_pool.v)
    with torch.inference_mode():
        diff = check_correctness(
            model, serial_runner, batch_runner, serial_seq, batch_seq,
            lengths, args.atol, args.rtol, device,
        )
        def run_serial():
            reset_lengths(serial_seq, lengths)
            return serial_decode(serial_runner, serial_seq)
        def run_batch():
            reset_lengths(batch_seq, lengths)
            return batch_decode(batch_runner, batch_seq)
        for _ in range(args.warmups):
            run_serial()
            run_batch()
        serial_times = []
        batch_times = []
        for rep in range(args.repeats):
            # Alternate measurement order to mitigate time-dependent bias.
            if rep % 2 == 0:
                serial_times.append(timed_call(run_serial, device))
                batch_times.append(timed_call(run_batch, device))
            else:
                batch_times.append(timed_call(run_batch, device))
                serial_times.append(timed_call(run_serial, device))

    serial = summarize(serial_times)
    batched = summarize(batch_times)
    t_s = serial["median_ms"]
    t_b = batched["median_ms"]
    entry = {
        "batch_size": batch_size,
        "context_len_target": context_len,
        "context_lengths": lengths,
        "layout": args.layout,
        "max_abs_logits_error": diff,
        "token_equality": True,
        "logits_allclose": True,
        "serial": serial,
        "batched": batched,
        "speedup_vs_serial": round(t_s / t_b, 4),
        "serial_output_tok_s": round(batch_size * 1000.0 / t_s, 4),
        "batched_output_tok_s": round(batch_size * 1000.0 / t_b, 4),
        "serial_samples_ms": [round(x, 6) for x in serial_times],
        "batched_samples_ms": [round(x, 6) for x in batch_times],
    }
    print(
        f"PASS B={batch_size:2d} L={context_len:4d} layout={args.layout:5s} "
        f"logits_err={diff:.3g} serial={t_s:.3f}ms batch={t_b:.3f}ms "
        f"speedup={entry['speedup_vs_serial']:.2f}x "
        f"batch_tok/s={entry['batched_output_tok_s']:.1f}",
        flush=True,
    )
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-sizes", type=csv_ints, default=[1,2,4,8,16,32])
    parser.add_argument("--context-lengths", type=csv_ints, default=[128,256,512])
    parser.add_argument("--layout", choices=["fixed", "mixed"], default="fixed")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/gpu/batch_decode_scaling.json"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.dim <= 0 or args.heads <= 0 or args.layers <= 0 or args.dim % args.heads:
        parser.error("Require positive dim/layers/heads and dim divisible by heads")
    if min(args.block_size, args.vocab_size) < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("Invalid block size/vocab/warmups/repeats")
    torch.manual_seed(20261029)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Stabilize the FP32 serial-vs-batched numerical comparison.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    context_cap = max(args.context_lengths) + 1
    model = MiniGPT(
        n_embd=args.dim, n_head=args.heads, vocab_size=args.vocab_size,
        block_size=context_cap, dropout=0.0, n_layer=args.layers,
    ).to(device).eval()
    print(f"torch={torch.__version__} device={device} "
          f"model=(layers={args.layers}, dim={args.dim}, heads={args.heads}) "
          f"block_size={args.block_size}; timing=synchronized host wall time", flush=True)
    result = {
        "definition": "Synthetic prepopulated KV; one ModelRunner decode step for B requests, "
                      "serial B x decode_one vs one decode_batch; shared model weights; "
                      "preallocation; no prefill; synchronized host wall time including host overhead "
                      "and per-request token extraction; not serving throughput or pure kernel time",
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": {"dim": args.dim, "layers": args.layers, "heads": args.heads,
                  "vocab_size": args.vocab_size, "dtype": "float32"},
        "block_size": args.block_size,
        "layout": args.layout,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rtol": args.rtol,
        "atol": args.atol,
        "results": [],
    }
    for ctx in args.context_lengths:
        for batch in args.batch_sizes:
            row = one_config(model, args, device, batch, ctx)
            result["results"].append(row)
    # Compare batch throughput with B=1 at equal context length and layout.
    for row in result["results"]:
        b1 = next((r for r in result["results"]
                   if r["context_len_target"] == row["context_len_target"]
                   and r["batch_size"] == 1), None)
        if b1:
            row["relative_batch_throughput_vs_b1"] = round(
                row["batched_output_tok_s"] / b1["batched_output_tok_s"], 4
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DECODE SCALING PASS; saved {args.output_json}")


if __name__ == "__main__":
    main()
