import time
import torch

from minigpt.model import MiniGPT
from minigpt.attention import KVCache, create_kv_cache


@torch.inference_mode()
def benchmark_length(
    model: MiniGPT,
    T: int,
    vocab_size: int,
    kv_caches: list[KVCache],
    device,
):
    B = 1

    prompt = torch.randint(
        0, vocab_size,
        (B, T),
        device=device,
    ).to(device)

    next_token = torch.randint(
        0, vocab_size,
        (B, 1),
        device=device,
    ).to(device)

    full_tokens = torch.cat([prompt, next_token], dim=1)

    model(prompt, kv_caches=kv_caches)

    def prefill_fn():
        model(prompt)

    def naive_fn():
        model(full_tokens)

    def decode_fn():
        for cache in kv_caches:
            cache.length = T

        model(next_token, kv_caches=kv_caches)

    prefill_ms = benchmark(prefill_fn)
    naive_ms = benchmark(naive_fn)
    decode_ms = benchmark(decode_fn)

    return {
        "T": T,
        "prefill_ms": prefill_ms,
        "naive_ms": naive_ms,
        "decode_ms": decode_ms,
        "speedup": naive_ms / decode_ms,
    }


def benchmark(
    fn,
    warmup=20,
    repeats=100,
):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(repeats):
        fn()

    torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1000 / repeats


def main():
    
    for T in [8, 16, 32, 64, 128, 256]:
        model = MiniGPT(
            n_embd=512,
            n_head=8,
            vocab_size=1024,
            block_size=1000,
            dropout=0.0,
            n_layer=8,
        )

        model = model.to("cuda")

        kv_caches = []

        for _ in range(8):
            kv_cache = create_kv_cache(1, 8, 1000, 64, "cuda", torch.float32)
            kv_caches.append(kv_cache)

        result = benchmark_length(
            model=model,
            T=T,
            vocab_size=model.vocab_size,
            kv_caches=kv_caches,
            device="cuda",
        )

        print(
            f"T={result['T']:4d} "
            f"prefill={result['prefill_ms']:8.4f} ms "
            f"naive={result['naive_ms']:8.4f} ms "
            f"decode={result['decode_ms']:8.4f} ms "
            f"speedup={result['speedup']:8.2f}x "
            f"ratio={T} "
        )


if __name__ == "__main__":
    main()