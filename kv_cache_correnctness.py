import torch
from minigpt.model import MiniGPT
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
):
    model.eval()

    naive_logits, _, _ = model(
        input_ids,
        use_cache=False,
    )

    cached_logits, _, kv_cache = model(
        input_ids,
        use_cache=True,
    )

    print("=== Prefill ===")

    print("naive logits :", naive_logits.shape)
    print("cache logits:", cached_logits.shape)

    compare(
        "prefill full logits",
        naive_logits,
        cached_logits,
    )

    return kv_cache


@torch.inference_mode()
def check_generation_correnctness(
    model: MiniGPT,
    input_ids: Tensor,
    max_new_tokens=8,
    atol=1e-5,
    rtol=1e-5,
):
    model.eval()

    tokens = input_ids.clone()

    naive_logits, _, _ = model(
        tokens,
        use_cache=False,
    )

    cached_logits, _, kv_cache = model(
        tokens,
        use_cache=True,
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
            f"seq_len={tokens.size(1):4d} "
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

        naive_logits, _, _ = model(
            tokens,
            use_cache=False,
        )

        cached_logits, _, kv_cache = model(
            next_token,
            past_key_values=kv_cache,
            use_cache=True,
        )

        check_cache_length(kv_cache, tokens.size(-1))

    print()
    print("All correctness checks passed.")

    return tokens


def check_cache_length(
    kv_cache,
    expected_length,
):
    for idx, (k, v) in enumerate(kv_cache):
        assert k.shape == v.shape

        assert k.size(2) == expected_length, (
            f"layer {idx}: "
            f"expected cache length "
            f"{expected_length}, "
            f"got {k.size(2)}"
        )


def main():
    B, T = 1, 8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randint(
        0, 1000,
        (B, T),
        device="cuda",
    )

    x = x.to(device)

    model = MiniGPT(
        n_embd=512,
        n_head=8,
        vocab_size=1024,
        block_size=1000,
        dropout=0.0,
        n_layer=8,
    )

    model = model.to(device)

    check_prefill(model, x)

    check_generation_correnctness(model, x)


if __name__ == "__main__":
    main()