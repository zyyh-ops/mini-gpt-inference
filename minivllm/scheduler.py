from block_manager import BlockManager
from sequence import SequenceState, Status
from dataclasses import dataclass


@dataclass
class ScheduleOutput:
    prefill: list[SequenceState]
    decode: list[SequenceState]
    num_scheduled_tokens: int


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_running: int,
        max_num_batched_tokens: int,
    ):
        self.waiting: list[SequenceState] = []
        self.running: list[SequenceState] = []

        self.block_manager = block_manager

        self.max_running = max_running

        self.max_num_batched_tokens = max_num_batched_tokens

    def add_request(
        self,
        seq: SequenceState
    ):
        self.waiting.append(seq)

    def retire_finished(self):
        alive = []

        for seq in self.running:

            if seq.is_finished():
                seq.status = Status.FINISHED

                self.block_manager.free(seq)

            else:
                alive.append(seq)

        self.running = alive

    def admit(self):
        while(len(self.running) < self.max_batch_size and self.waiting):
            seq = self.waiting[0]

            if not self.block_manager.can_allocate(seq):
                break

            self.waiting.pop(0)

            self.block_manager.allocate(seq)

            # seq.initialize()

            # seq.status = Status.RUNNING

            self.running.append(seq)

    def schedule(self):

        prefill_batch = []
        decode_batch = []

        token_budget = self.max_num_batched_tokens

        for seq in self.running:
            if token_budget < 1:
                break

            if not self.block_manager.can_reserve(
                seq,
                num_tokens=1,
            ):
                continue

            decode_batch.append(seq)

            token_budget -= 1

        while self.waiting:
            active_after_admit = len(self.running) + len(prefill_batch)

            if active_after_admit >= self.max_running:
                break

            seq = self.waiting[0]

            prompt_len = len(seq.prompt_tokens)

            if prompt_len > token_budget:
                break

            if not self.block_manager.can_reserve(
                seq,
                num_tokens=prompt_len,
            ):
                break

            self.waiting.pop(0)

            prefill_batch.append(seq)

            token_budget -= prompt_len

        num_scheduled_tokens = (
            self.max_num_batched_tokens - token_budget
        )

        return ScheduleOutput(
            prefill=prefill_batch,
            decode=decode_batch,
            num_scheduled_tokens=num_scheduled_tokens,
        )