from minigpt.attention import KVCache, create_kv_cache
import torch
from torch import Tensor

from minigpt.model import MiniGPT

def kv_cache_bytes(
    batch_size,
    n_layer,
    capacity,
    n_embd,
    bytes_per_element,
):
    return (
        2 * batch_size * n_layer * capacity * n_embd * bytes_per_element
    )


@torch.inference_mode()
def measure_decode_memory(
    model: MiniGPT,
    kv_caches: list[KVCache],
    next_token: Tensor,
):
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    result = model(
        next_token,
        kv_caches,
    )

    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    live_with_output = torch.cuda.memory_allocated()

    del result

    torch.cuda.synchronize()

    after_cleanup = torch.cuda.memory_allocated()

    return {
        "baseline": baseline,
        "peak": peak,
        "temporary_peak": peak - baseline,
        "live_output": live_with_output - baseline,
        "net_after_cleanup": after_cleanup - baseline,
    }


def mib(x):
    return x / 1024**2


def main():
    model = MiniGPT(
        n_embd=512,
        n_head=8,
        vocab_size=1024,
        block_size=1024,
        dropout=0.0,
        n_layer=8,
    ).to("cuda")

    physical = kv_cache_bytes(
        batch_size=1,
        n_layer=8,
        capacity=1024,
        n_embd=512,
        bytes_per_element=4,
    )

    print(
        f"Physical KV Cache: "
        f"{physical / 1024**2:.2f} MiB"
    )

    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()

    kv_caches = [
        create_kv_cache(
            batch_size=1,
            n_head=8,
            block_size=1024,
            head_size=64,
            device="cuda",
            dtype=torch.float32,
        )
        for _ in range(8)
    ]

    torch.cuda.synchronize()

    after_cache = torch.cuda.memory_allocated()

    delta = after_cache - baseline

    print(
        f"KV allocated delta: "
        f"{delta / 1024**2:.2f} MiB"
    )

    capcacity = 1024

    with torch.inference_mode():

        for T in [8, 16, 32, 64, 128, 256, 512, 768]:
            # logical = (2 * 1 * 8 * T * 512 * 4)

            # utilization = T / capcacity

            # print(
            #     f"T={T:4d} "
            #     f"logical={logical / 1024**2:8.2f} MiB "
            #     f"physcial={physical / 1024**2:8.2f} MiB "
            #     f"util={utilization * 100:6.2f}%"
            # )


            for cache in kv_caches:
                cache.length = 0

            prompt = torch.randint(
                0, 1024,
                (1, T),
                device="cuda",
            )

            next_token = torch.randint(
                0, 1024,
                (1, 1),
                device="cuda",
            )

            prefill_result = model(
                prompt,
                kv_caches=kv_caches,
            )

            del prefill_result

            torch.cuda.synchronize()

            assert all(
                cache.length == T
                for cache in kv_caches
            )

            stats = measure_decode_memory(
                model,
                kv_caches,
                next_token,
            )

            print(
                f"T={T:4d} "
                f"baseline={mib(stats['baseline']):8.2f} MiB "
                f"temp_peak={mib(stats['temporary_peak']):8.4f} MiB "
                f"live_output={mib(stats['live_output']):8.4f} MiB "
                f"net_after_cleanup={mib(stats['net_after_cleanup']):8.4f} MiB "
            )


if __name__ == "__main__":
    main()