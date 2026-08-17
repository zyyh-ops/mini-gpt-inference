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
        deivce=device,
        dtyp=next(model.parameters()).dtype,
    )

    logits, _, _ = model(
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

    logits, _, _ = model(
        state.currect_token,
        kv_caches=state.kv_caches,
    )

    after_len = get_cache_length(state)


def simulate_continuous_batch(
    requests,
    max_batch_size,
):
    future = sorted(
        deepcopy(requests),
        key=lambda r: r.arrival_time,
    )

    waiting = []
    active = []

    currect_time = 0

    total_active_slots = 0
    total_capacity_slots = 0

    finished_requests = []

    while future or waiting or active:

        while(
            future
            and future[0].arrival_time <= currect_time
        ):
            waiting.append(future.pop(0))

        while(
            len(active) < max_batch_size
            and waiting
        ):
            req = waiting.pop(0)

            if req.first_scheduled_time is None:
                req.first_scheduled_time = currect_time

            active.append(req)

        if not active:
            if future:
                currect_time = future[0].arrival_time
                continue

            break

        print(
            f"t={currect_time:02d} "
            f"active="
            f"{[r.request_id for r in active]}"
        )

        total_active_slots += len(active)
        total_capacity_slots += max_batch_size

        for req in active:
            req.remaining_tokens -= 1

            if req.first_token_time is None:
                req.first_token_time = currect_time + 1

        still_active = []

        for req in active:
            if req.remaining_tokens == 0:
                req.finish_time = currect_time + 1
                finished_requests.append(req)
            else:
                still_active.append(req)

        active = still_active

        currect_time += 1

    return {
        "total_time": currect_time,
        "utilization": (
            total_active_slots
            / total_capacity_slots
        ),
        "requests": finished_requests,
    }


def print_serving_summary(name, result):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    for req in sorted(
        result["requests"],
        key=lambda r: r.request_id,
    ):
        queue_delay = (
            req.first_scheduled_time
            - req.arrival_time
        )

        ttft = (
            req.first_token_time
            - req.arrival_time
        )

        e2e = (
            req.finish_time
            - req.arrival_time
        )

        print(
            f"{req.request_id}: "
            f"arrival={req.arrival_time:2d} "
            f"first_schedule={req.first_scheduled_time:2d} "
            f"first_token={req.first_token_time:2d} "
            f"finish={req.finish_time:2d} "
            f"queue={queue_delay:2d} "
            f"TTFT={ttft:2d} "
            f"E2E={e2e:2d}"
        )

    print()
    print(
        f"total time      : "
        f"{result['total_time']}"
    )

    print(
        f"slot utilization : "
        f"{result['utilization'] * 100:.2f}%"
    )


def main():
    requests = [
        Request("A", arrival_time=0, remaining_tokens=4),
        Request("B", arrival_time=0, remaining_tokens=8),
        Request("C", arrival_time=2, remaining_tokens=3),
        Request("D", arrival_time=4, remaining_tokens=5),
        Request("E", arrival_time=6, remaining_tokens=2),
    ]

    max_batch_size = 2

    continuous = simulate_continuous_batch(
        requests, max_batch_size,
    )
    
    print_serving_summary(
        "Serving Summary",
        continuous,
    )


if __name__ == "__main__":
    main()