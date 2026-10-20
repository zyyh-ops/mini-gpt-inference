from .scheduler import Scheduler
from .sequence import SequenceState, Status
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .model import MiniGPT
from .kv_block_pool import KVBlockPool
import torch
from dataclasses import dataclass


@dataclass
class IterationRecord:
    iteration: int

    prefill_requests: int
    decode_requests: int

    prefill_tokens: int
    decode_tokens: int

    scheduled_tokens: int

    waiting: int
    prefilling: int
    running: int

    used_blocks: int
    free_blocks: int

    kv_utilization: float

    prefill_ms: float
    decode_ms: float
    iteration_ms: float


class MiniRuntime:
    def __init__(
        self,
        num_blocks,
        block_size,
        max_running,
        max_num_batched_tokens,
        preemption_policy="conservative",
    ):
        self.block_manager = (
            BlockManager(
                num_blocks,
                block_size,
            )
        )

        self.scheduler = (
            Scheduler(
                self.block_manager,
                max_running,
                max_num_batched_tokens,
                preemption_policy=preemption_policy,
            )
        )

        self.runner = ModelRunner(
            model=MiniGPT(
                n_embd=512,
                n_head=8,
                vocab_size=1024,
                block_size=1024,
                dropout=0.0,
                n_layer=8,
            ),
            kv_pool=KVBlockPool(
                n_layer=8,
                num_blocks=num_blocks,
                block_size=block_size,
                n_head=8,
                head_dim=64,
            ),
            block_manager=self.block_manager,
        )

        self.iteration = 0
        self.records: list[IterationRecord] = []

    def add_request(
        self,
        seq: SequenceState,
    ):
        self.scheduler.add_request(seq)

    def step(self):
        import time
        self.iteration += 1
        iter_begin = time.perf_counter()
        plan = self.scheduler.schedule()

        assert (
            len(self.scheduler.running) + len(plan.prefill)
            <= self.scheduler.max_running
        ), f"error {len(self.scheduler.running)} + {len(plan.prefill)}, max_running={self.scheduler.running}"

        print(f"\nIteration {self.iteration}")
        print("prefill:", [x.seq.seq_id for x in plan.prefill])
        print("decode:", [x.seq_id for x in plan.decode])
        print("scheduled tokens:", plan.num_scheduled_tokens)

        decode_begin = time.perf_counter()

        if plan.decode:
            logits, next_tokens = self.runner.decode_batch(plan.decode)

            for seq, token in zip(plan.decode, next_tokens):
                seq.generated_tokens.append(token)
                seq.last_token = token

        decode_ms = (time.perf_counter() - decode_begin) * 1000

        prefill_begin = time.perf_counter()

        for item in plan.prefill:
            seq = item.seq

            if seq.first_schedule_iter is None:
                seq.first_schedule_iter = self.iteration

            logits = self.runner.prefill_chunk(
                seq=seq,
                start_pos=item.start_pos,
                num_tokens=item.num_tokens,
            )
            seq.prompt_pos += item.num_tokens
            assert seq.prompt_pos == seq.length

            if seq.prompt_finished:
                if seq.recompute_tokens is not None:
                    # The rebuilt KV excludes the pending last_token.
                    # Never sample logits again: that output was already
                    # generated before preemption.
                    assert seq.generated_tokens
                    assert seq.last_token == seq.generated_tokens[-1]
                    assert seq.length == len(seq.recompute_tokens)
                    seq.recompute_tokens = None
                    # prompt_pos tracks the temporary recompute prefix during
                    # restoration; restore its ordinary prompt-only meaning.
                    seq.prompt_pos = seq.prompt_length
                else:
                    first_token = torch.argmax(
                        logits[:, -1],
                        dim=-1,
                    ).item()

                    seq.generated_tokens.append(first_token)
                    seq.last_token = first_token
                    seq.first_token_iter = self.iteration

                seq.status = Status.RUNNING
                self.scheduler.prefilling.remove(seq)
                self.scheduler.running.append(seq)
                
        prefill_ms = (time.perf_counter() - prefill_begin) * 1000

        prefill_tokens = sum(
            item.num_tokens
            for item in plan.prefill
        )

        decode_tokens = len(plan.decode)

        assert (
            prefill_tokens + decode_tokens
            == plan.num_scheduled_tokens
        )

        assert (
            plan.num_scheduled_tokens
            <= self.scheduler.max_num_batched_tokens
        )

        prefill_ids = {
            x.seq.seq_id
            for x in plan.prefill
        }

        decode_ids = {
            x.seq_id
            for x in plan.decode
        }

        assert prefill_ids.isdisjoint(decode_ids)

        for seq in self.scheduler.running:
            if seq.is_finished():
                seq.finish_iter = self.iteration
        self.scheduler.retire_finished()
        
        print("the number of free blocks: ", len(self.block_manager.free_blocks))


def main():
    workloads: list[tuple[int, int]] = [
        (32, 32),
        (64, 32),
        (128, 16),
        (256, 16),
        (48, 32),
        (192, 16),
    ]

    for max_num_batched_tokens in [32, 64, 128, 256]:
        print("=" * 10 + f" {max_num_batched_tokens} " + "=" * 10)

        runtime = MiniRuntime(
            num_blocks=8,
            block_size=16,
            max_running=8,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        seq_list: list[SequenceState] = []

        for i, (prompt, output) in enumerate(workloads):
            seq = SequenceState(
                seq_id=chr(ord('A') + i),
                prompt_tokens=[i for i in range(prompt)],
                max_tokens=prompt+output,
            )
            seq_list.append(seq)

        runtime.add_request(seq_list[0])

        runtime.add_request(seq_list[1])

        for iteration in range(15):
            if iteration == 2:
                runtime.add_request(seq_list[2])

            if iteration == 4:
                runtime.add_request(seq_list[3])

            if iteration == 8:
                runtime.add_request(seq_list[4])

            if iteration == 10:
                runtime.add_request(seq_list[5])

            runtime.step()


if __name__ == "__main__":
    main()