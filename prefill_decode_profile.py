from minigpt.model import MiniGPT

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)


def profile_fn(name, fn, warmup=10, repeats=10):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with record_function(name):
            for _ in range(repeats):
                fn()

    torch.cuda.synchronize()

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(
        prof.key_averages(
            group_by_input_shape=True
        ).table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        )
    )

    prof.export_chrome_trace(
        f"proflie_{name}.json"
    )


def main():
    B, T = 1, 256

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniGPT(
        n_embd=512,
        n_head=8,
        vocab_size=1024,
        block_size=1024,
        dropout=0.0,
        n_layer=8,
    ).to(device)

    prompt = torch.randint(
        0, 1024,
        (B, T),
        device=device
    ).to(device)

    next_token = torch.randint(
        0, 1024,
        (B, 1),
        device=device,
    ).to(device)

    full_tokens = torch.cat(
        [prompt, next_token],
        dim=1,
    )

    with torch.inference_mode():

        _, _, kv_cache = model(
            prompt,
            use_cache=True,
        )

        def naive_fn():
            model(
                full_tokens,
                use_cache=False,
            )

        def decode_fn():
            model(
                next_token,
                past_key_values=kv_cache,
                use_cache=True,
            )

        profile_fn(
            "naive_t256",
            naive_fn,
        )

        profile_fn(
            "cached_decode_t256",
            decode_fn
        )


if __name__ == "__main__":
    main()