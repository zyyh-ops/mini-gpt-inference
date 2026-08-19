from dataclasses import dataclass
from copy import deepcopy

from minigpt.attention import KVCache, create_kv_cache
from minigpt.model import MiniGPT

import torch
from torch import Tensor


N_EMBD = 512
N_HEAD = 8
HEAD_SIZE = 64
N_LAYER = 8

VOCAB_SIZE = 1024
BLOCK_SIZE = 1024

MAX_BATCH_SIZE = 2


@dataclass
class SequenceState:
    request_id: str
    prompt_ids: Tensor
    currect_token: Tensor
    kv_caches: list[KVCache]

    generated_tokens: list[int]
    remaining_decode_tokens: int

    finished: bool = False


def create_request_kv_caches(
    device: str,
    dtype: torch.dtype,
) -> list[KVCache]:
    return [
        create_kv_cache(
            batch_size=1,
            n_head=N_HEAD,
            block_size=BLOCK_SIZE,
            head_size=HEAD_SIZE,
            device=device,
            dtype=dtype,
        )
        for _ in range(N_LAYER)
    ]


def get_cache_length(state: SequenceState) -> int:
    lengths = [
        cache.length
        for cache in state.kv_caches
    ]

    assert len(set(lengths)) == 1, (
        f"{state.request_id}: "
        f"inconsistent cache lengths: {lengths}"
    )

    return lengths[0]


def check_cache_storage_isolation(
    states: list[SequenceState],
):
    k_ptrs = []
    v_ptrs = []

    for state in states:
        for cache in state.kv_caches:
            k_ptrs.append(cache.k.data_ptr())
            v_ptrs.append(cache.v.data_ptr())

    assert len(k_ptrs) == len(set(k_ptrs)), (
        "Different KV caches share K storage"
    )

    assert len(v_ptrs) == len(set(v_ptrs)), (
        "Different KV caches share V storage"
    )


@torch.inference_mode()
def prefill_request(
    model: MiniGPT,
    request_id: str,
    prompt_ids: Tensor,
    max_new_tokens: int,
    device: str,
) -> SequenceState:

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    kv_caches = create_request_kv_caches(
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    logits, _, = model(
        prompt_ids,
        kv_caches=kv_caches,
    )

    first_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    state = SequenceState(
        request_id=request_id,
        prompt_ids=prompt_ids,
        kv_caches=kv_caches,
        currect_token=first_token,
        generated_tokens=[first_token.item()],
        remaining_decode_tokens=max_new_tokens - 1,
        finished=(max_new_tokens == 1),
    )

    assert get_cache_length(state) == prompt_ids.size(1)

    return state


@torch.inference_mode()
def decode_one_step(
    model: MiniGPT,
    state: SequenceState,
):
    if state.finished:
        raise RuntimeError(
            f"{state.request_id} is already finished"
        )

    before_len = get_cache_length(state)

    logits, _ = model(
        state.currect_token,
        kv_caches=state.kv_caches,
    )

    after_len = get_cache_length(state)

    assert after_len == before_len + 1, (
        f"{state.request_id}: "
        f"cache should grow "
        f"{before_len} -> {before_len + 1}, "
        f"but got {after_len}"
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    state.currect_token = next_token

    state.generated_tokens.append(next_token.item())

    state.remaining_decode_tokens -= 1

    if state.remaining_decode_tokens == 0:
        state.finished = True


@torch.inference_mode()
def generate_standalone(
    model: MiniGPT,
    prompt_ids: Tensor,
    max_new_tokens: int,
    device: str,
) -> list[int]:
    state = prefill_request(
        model=model,
        request_id="standalone",
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    while not state.finished:
        decode_one_step(model, state)

    return state.generated_tokens


@torch.inference_mode()
def run_multi_request_reference(
    model: MiniGPT,
    states: list[SequenceState],
    max_batch_size: int,
):
    waiting = [
        state
        for state in states
        if not state.finished
    ]

    active: list[SequenceState] = []

    iteration = 0

    def admit_requests():
        while(
            len(active) < max_batch_size and waiting
        ):
            active.append(waiting.pop(0))

    admit_requests()

    while active:
        print()
        print("=" * 80)
        print(
            f"iteration={iteration:02d} "
            f"active="
            f"{[s.request_id for s in active]}"
        )
        print("=" * 80)

        before_lengths = {
            state.request_id:
            get_cache_length(state)
            for state in states
        }

        for state in active:
            decode_one_step(model, state)

            print(
                f"{state.request_id}: "
                f"token={state.generated_tokens[-1]:4d} "
                f"cache_len={get_cache_length(state):4d} "
                f"remaining={state.remaining_decode_tokens:2d}"
            )

        active_ids = {
            state.request_id
            for state in active
        }

        for state in states:
            before = before_lengths[state.request_id]

            after = get_cache_length(state)

            if state.request_id in active_ids:
                assert after == before + 1, (
                    f"{state.request_id}: "
                    f"active cache didn't grow correctly"
                )

            else:
                assert after == before, (
                    f"{state.request_id}: "
                    f"inactive cache was modified "
                    f"{before} -> {after}" 
                )

        finished_now = [
            state
            for state in states
            if state.finished
        ]

        if finished_now:
            print(
                "finished:",
                [state.request_id for state in finished_now],
            )

        active = [
            state
            for state in active
            if not state.finished
        ]

        admit_requests()

        iteration += 1


def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniGPT(
        n_embd=N_EMBD,
        n_head=N_HEAD,
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        dropout=0.0,
        n_layer=N_LAYER,
    ).to(device)

    model.eval()

    prompts = {
        "A": torch.tensor(
            [[10, 11, 12, 13]],
            device=device,
        ),

        "B": torch.tensor(
            [[20, 21, 22, 23]],
            device=device,
        ),

        "C": torch.tensor(
            [[30, 31, 32]],
            device=device,
        ),

        "D": torch.tensor(
            [[40, 41, 42, 43, 44, 45]],
            device=device,
        ),
    }

    max_new_tokens = {
        "A": 4,
        "B": 8,
        "C": 3,
        "D": 5,
    }

    standalone_results = {}

    print()
    print("=" * 80)
    print("Standalone Ground Truth")
    print("=" * 80)

    for request_id, prompt in prompts.items():

        tokens = generate_standalone(
            model=model,
            prompt_ids=prompt,
            max_new_tokens=max_new_tokens[request_id],
            device=device,
        )

        standalone_results[request_id] = tokens

        print(
            f"{request_id}: "
            f"{tokens}"
        )


    states = []

    for request_id, prompt in prompts.items():

        state = prefill_request(
            model=model,
            request_id=request_id,
            prompt_ids=prompt,
            max_new_tokens=max_new_tokens[request_id],
            device=device,
        )

        states.append(state)

    check_cache_storage_isolation(states)

    print()
    print("KV storage isolation: PASS")

    run_multi_request_reference(
        model=model,
        states=states,
        max_batch_size=MAX_BATCH_SIZE,
    )

    print()
    print("=" * 80)
    print("Final Correctness")
    print("=" * 80)

    all_ok = True

    for state in states:

        reference = standalone_results[state.request_id]

        scheduled = state.generated_tokens

        ok = (reference == scheduled)

        all_ok &= ok

        print(
            f"{state.request_id}: "
            f"standalone={reference} "
            f"scheduled={scheduled} "
            f"equal={ok}"
        )

    print()

    if all_ok:
        print("All multi-request correctness checks passed.")
    else:
        raise RuntimeError(
            "Multi-request correctness failed"
        )


if __name__ == "__main__":
    main()