from block_manager import BlockManager
from sequence import SequenceState, Status
from dataclasses import dataclass


@dataclass
class PrefillWorkItem:
    seq: SequenceState

    start_pos: int

    num_tokens: int

    @property
    def end_pos(self):
        return self.start_pos + self.num_tokens


@dataclass
class ScheduleOutput:
    prefill: list[PrefillWorkItem]
    decode: list[SequenceState]
    num_scheduled_tokens: int


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_running: int,
        max_num_batched_tokens: int,
    ):
        self.waiting: list[PrefillWorkItem] = []
        self.prefilling: list[PrefillWorkItem] = []
        self.running: list[SequenceState] = []

        self.block_manager = block_manager

        self.max_running = max_running

        self.max_num_batched_tokens = max_num_batched_tokens

    def add_request(
        self,
        seq: SequenceState
    ):
        self.waiting.append(seq)

    def preempt(self, seq: SequenceState) -> None:
        """Evict KV and queue for recomputation. Call only between steps."""
        if seq in self.running:
            assert seq.generated_tokens, "RUNNING request has no output token"
            assert seq.last_token == seq.generated_tokens[-1]
            seq.recompute_tokens = seq.prompt_tokens + seq.generated_tokens[:-1]
            assert len(seq.recompute_tokens) == seq.length
            self.running.remove(seq)
        elif seq in self.prefilling:
            # Discard partial prefill KV. For an interrupted recompute,
            # keep recompute_tokens; for an ordinary prefill, keep None.
            self.prefilling.remove(seq)
        else:
            raise ValueError("Only active requests may be preempted")

        self.block_manager.free(seq)  # also resets seq.length and block_table
        seq.prompt_pos = 0
        seq.status = Status.WAITING
        self.waiting.insert(0, seq)

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

        free_block_budget = len(self.block_manager.free_blocks)

        for seq in self.running:
            if token_budget < 1:
                break

            needed_blocks = (
                self.block_manager.num_new_blocks_needed(
                    seq,
                    num_tokens=1,
                )
            )

            if needed_blocks > free_block_budget:
                continue

            decode_batch.append(seq)

            token_budget -= 1

            free_block_budget -= needed_blocks

        for seq in list(self.prefilling):
            if token_budget <= 0:
                break

            remaining = seq.remaining_prompt_tokens

            if remaining <= 0:
                continue

            chunk_size = min(remaining, token_budget)

            needed_blocks = (
                self.block_manager.num_new_blocks_needed(
                    seq,
                    num_tokens=chunk_size,
                )
            )

            if needed_blocks > free_block_budget:
                continue

            prefill_batch.append(
                PrefillWorkItem(
                    seq=seq,
                    start_pos=seq.prompt_pos,
                    num_tokens=chunk_size,
                )
            )

            token_budget -= chunk_size

            free_block_budget -= needed_blocks

        while (
            self.waiting
            and token_budget > 0
        ):
            num_active = len(self.running) + len(self.prefilling)

            if num_active >= self.max_running:
                break

            seq = self.waiting[0]

            remaining = seq.remaining_prompt_tokens

            chunk_size = min(remaining, token_budget)

            needed_blocks = (
                self.block_manager.num_new_blocks_needed(
                    seq,
                    num_tokens=chunk_size,
                )
            )

            if needed_blocks > free_block_budget:
                break

            self.waiting.pop(0)

            seq.status = Status.PREFILLING

            self.prefilling.append(seq)

            prefill_batch.append(
                PrefillWorkItem(
                    seq=seq,
                    start_pos=seq.prompt_pos,
                    num_tokens=chunk_size,
                )
            )

            token_budget -= chunk_size

            free_block_budget -= needed_blocks

        num_scheduled_tokens = (
            self.max_num_batched_tokens - token_budget
        )

        return ScheduleOutput(
            prefill=prefill_batch,
            decode=decode_batch,
            num_scheduled_tokens=num_scheduled_tokens,
        )