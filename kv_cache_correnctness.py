import torch
from minigpt.model import MiniGPT
from minigpt.attention import KVCache, create_kv_cache

from torch import Tensor


def compare(name, a, b, atol=1e-5, rtol=1e-5):
    diff = (a - b).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    ok = torch.allclose(
        a,
        b,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"{name:<20} "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"allclose={ok}"
    )

    return ok


@torch.inference_mode()
def check_prefill(
    model: MiniGPT, 
    input_ids: Tensor,
    kv_caches: list[KVCache] | None = None,
):
    model.eval()

    naive_logits, _ = model(input_ids)

    cached_logits, _ = model(
        input_ids, 
        kv_caches=kv_caches,
    )

    print("=== Prefill ===")

    print("naive logits :", naive_logits.shape)
    print("cache logits:", cached_logits.shape)

    prefill_ok = compare(
        "prefill full logits",
        naive_logits,
        cached_logits,
    )

    print(prefill_ok)

    return kv_caches


@torch.inference_mode()
def check_generation_correnctness(
    model: MiniGPT,
    input_ids: Tensor,
    kv_caches: list[KVCache] | None = None,
    max_new_tokens=8,
    atol=1e-5,
    rtol=1e-5,
):
    model.eval()

    tokens = input_ids.clone()

    naive_logits, _ = model(
        tokens,
    )

    cached_logits, _ = model(
        tokens,
        kv_caches=kv_caches,
    )

    print("=" * 80)
    print("Prefill correctness")
    print("=" * 80)

    print(f"input shape           : {tokens.shape}")
    print(f"naive logits shape    : {naive_logits.shape}")
    print(f"cached logits shape:  : {cached_logits.shape}")

    prefill_ok = compare(
        "prefill",
        naive_logits,
        cached_logits,
        atol=atol,
        rtol=rtol,
    )

    if not prefill_ok:
        raise RuntimeError("Prefill correctness failed")


    print()
    print("=" * 80)
    print("Decode correctness")
    print("=" * 80)

    for step in range(max_new_tokens):

        naive_last = naive_logits[:, -1, :]
        cached_last = cached_logits[:, -1, :]

        diff = (naive_last - cached_last).abs()

        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        logits_ok = torch.allclose(
            naive_last,
            cached_last,
            atol=atol,
            rtol=rtol,
        )

        naive_next = torch.argmax(
            naive_last,
            dim=-1,
            keepdim=True,
        )

        cached_next = torch.argmax(
            cached_last,
            dim=-1,
            keepdim=True,
        )

        token_ok = torch.equal(
            naive_next,
            cached_next,
        )

        print(
            f"step={step:02d} "
            f"seq_len={input_ids.size(1):4d} "
            f"max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} "
            f"allclose={logits_ok} "
            f"token_equal={token_ok} "
        )

        print(
            f"     naive logits shape  = {tuple(naive_logits.shape)}"
        )

        print(
            f"     cached logits shape = {tuple(cached_logits.shape)}"
        )

        if not logits_ok:
            raise RuntimeError(
                f"logits correctness failed at step {step}"
            )

        if not token_ok:
            raise RuntimeError(
                f"token correnctness failed at step {step}"
            )

        next_token = naive_next

        tokens = torch.cat(
            [tokens, next_token],
            dim=-1,
        )

        if step == max_new_tokens - 1:
            break

        naive_logits, _ = model(
            tokens,
        )

        cached_logits, _ = model(
            next_token,
            kv_caches=kv_caches,
        )

        check_cache_length(kv_caches, tokens.size(-1))

    print()
    print("All correctness checks passed.")

    return tokens


def check_cache_length(
    kv_caches,
    expected_length,
):
    for idx, kv_cache in enumerate(kv_caches):
        assert kv_cache.k.shape == kv_cache.v.shape

        assert kv_cache.length == expected_length, (
            f"layer {idx}: "
            f"expected cache length "
            f"{expected_length}, "
            f"got {kv_cache.length}"
        )


def main():
    B, T = 1, 8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randint(
        0, 1000,
        (B, T),
        device="cuda",
    )

    model = MiniGPT(
        n_embd=512,
        n_head=8,
        vocab_size=1024,
        block_size=1000,
        dropout=0.0,
        n_layer=8,
    ).to(device)

    kv_caches = []

    for _ in range(8):
        kv_cache = create_kv_cache(
            B, 8, 1000, 64, device, torch.float32,
        )
        kv_caches.append(kv_cache)

    # check_prefill(model, x, kv_caches)

    check_generation_correnctness(model, x, kv_caches)


if __name__ == "__main__":
    main()