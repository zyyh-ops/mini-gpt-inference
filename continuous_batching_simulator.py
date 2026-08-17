from dataclasses import dataclass
from copy import deepcopy

from torch import Tensor
import torch

from minigpt.attention import KVCache, create_kv_cache


@dataclass
class Request:
    request_id: str

    remaining_tokens: int
    arrival_time: int

    first_scheduled_time: int | None = None
    first_token_time: int | None = None
    finish_time: int | None = None


@dataclass
class SequenceState:
    request_id: str
    currect_token: Tensor
    kv_caches: list[KVCache]
    finished: bool = False



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


def simulate_static_batch(
    requests,
    max_batch_size,
):
    waiting = deepcopy(requests)

    active = []

    iteration = 0

    total_active_slots = 0
    total_capacity_slots = 0

    request_finish_time = {}

    def admit_requests():
        while(
            len(active) < max_batch_size
            and waiting
        ):
            active.append(waiting.pop(0))

    admit_requests()

    while active:

        iteration += 1

        print(
            f"iter={iteration:02d} "
            f"active={[r.request_id for r in active]}"
        )

        total_active_slots += len(active)
        total_capacity_slots += max_batch_size

        for req in active:
            req.remaining_tokens -= 1

        finished = [
            req
            for req in active
            if req.remaining_tokens == 0
        ]

        for req in finished:
            request_finish_time[
                req.request_id
            ] = iteration

        active = [
            req
            for req in active
            if req.remaining_tokens > 0
        ]

        admit_requests()

    utilization = (
        total_active_slots
        / total_capacity_slots
    )

    return {
        "iterations": iteration,
        "utilization": utilization,
        "requests": finished,
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