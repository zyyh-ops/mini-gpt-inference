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
        preemption_policy: str = "conservative",
    ):
        self.waiting: list[PrefillWorkItem] = []
        self.prefilling: list[PrefillWorkItem] = []
        self.running: list[SequenceState] = []

        self.block_manager = block_manager
        self.max_running = max_running
        self.max_num_batched_tokens = max_num_batched_tokens

        if preemption_policy not in ("conservative", "recompute"):
            raise ValueError("preemption_policy must be 'conservative' or 'recompute'")
        self.preemption_policy = preemption_policy
        self.num_preemptions = 0

    def _max_kv_blocks(self, seq: SequenceState) -> int:
        tokens = seq.max_tokens - 1
        bs = self.block_manager.block_size
        return (tokens + bs - 1) // bs

    def _can_admit_without_preemption(self, seq: SequenceState) -> bool:
        active = self.running + self.prefilling
        committed = sum(self._max_kv_blocks(s) for s in active)
        return committed + self._max_kv_blocks(seq) <= self.block_manager.num_blocks

    def add_request(
        self,
        seq: SequenceState
    ):
        if not seq.prompt_tokens or seq.max_tokens <= seq.prompt_length:
            raise ValueError("prompt must be nonempty and output token must be positive")
        if self._max_kv_blocks(seq) > self.block_manager.num_blocks:
            raise ValueError("request's maximum KV demand exceeds total block capacity")
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

        self.num_preemptions += 1
        seq.num_preemptions += 1
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
            self.running.append(seq)

    def _choose_victim(
        self,
        target: SequenceState,
        protected_ids: set[int],
    ) -> SequenceState | None:
        """
        Prefer newest partial-prefill, then newest decode, but never protected work.
        This is a deliberately simple policy, not production-grade fairness.
        """
        for seq in [*reversed(self.prefilling), *reversed(self.running)]:
            if seq is target or id(seq) in protected_ids:
                continue
            if seq.block_table:
                return seq
        return None

    def _free_for(
        self,
        target: SequenceState,
        needed: int,
        available: int,
        protected_ids: set[int]
    ) -> tuple[int, bool]:
        """Reclaim physical blocks, preserving scheduled requests and the target."""
        any_preempted = False
        if self.preemption_policy != "recompute":
            return available, any_preempted
        while available < needed:
            victim = self._choose_victim(target, protected_ids)
            if victim is None:
                break
            released = len(victim.block_table)
            self.preempt(victim)
            available += released
            any_preempted = True
        return available, any_preempted

    def _prefill_chunk_size(
        self,
        seq: SequenceState,
        remaining_budget: int,
        available_blocks: int,
    ) -> int:
        desired = min(seq.remaining_prompt_tokens, remaining_budget)
        if self.preemption_policy != "recompute":
            return desired
        # Limit the work to already-held capacity plus uncommited blocks;
        # this makes progress instead of demanding a too-large chunk.
        fit = ((len(seq.block_table) + available_blocks)
                * self.block_manager.block_size - seq.length)
        return min(desired, max(0, fit))

    def schedule(self):
        prefill_batch: list[PrefillWorkItem] = []
        decode_batch: list[SequenceState] = []
        token_budget = self.max_num_batched_tokens
        free_block_budget = len(self.block_manager.free_blocks)

        # Protected work has only been planned, not executed yet: victim
        # selection must not free its KV / invalidate its block table.
        protected_ids: set[int] = set()
        preempted_this_round = False

        for seq in list(self.running):
            if token_budget < 1:
                break
            if seq.status != Status.RUNNING:
                continue # preempted by an earlier admission in this round
            needed = self.block_manager.num_new_blocks_needed(seq, 1)
            if needed > free_block_budget:
                free_block_budget, did_preempt = self._free_for(
                    seq, needed, free_block_budget, protected_ids)
                preempted_this_round |= did_preempt
            if needed > free_block_budget:
                continue

            decode_batch.append(seq)
            protected_ids.add(id(seq))
            token_budget -= 1
            free_block_budget -= needed

        for seq in list(self.prefilling):
            if token_budget <= 0:
                break
            if seq.status != Status.PREFILLING:
                continue # preempted earlier in this round
            if seq.remaining_prompt_tokens <= 0:
                continue

            chunk_size = self._prefill_chunk_size(seq, token_budget, free_block_budget)
            if chunk_size == 0 and self.preemption_policy == "recompute":
                # Need at least one token worth of KV capacity to progress.
                free_block_budget, did_preempt = self._free_for(
                    seq, 1, free_block_budget, protected_ids)
                preempted_this_round |= did_preempt
                chunk_size = self._prefill_chunk_size(seq, token_budget, free_block_budget)

            if chunk_size <= 0:
                continue

            needed = self.block_manager.num_new_blocks_needed(seq, chunk_size)
            if needed > free_block_budget:
                continue
            prefill_batch.append(PrefillWorkItem(seq, seq.prompt_pos, chunk_size))
            protected_ids.add(id(seq))
            token_budget -= chunk_size

            free_block_budget -= needed

        # After a preemption, defer *all* fresh admissions to the next step.
        # Otherwise the evicted sequence could immediately reclaim the KV
        # budget and start a ping-pong preempt/re-admit cycle.
        while self.waiting and token_budget > 0 and not preempted_this_round:
            active = len(self.running) + len(self.prefilling)
            if active >= self.max_running:
                break

            seq = self.waiting[0]
            if self.preemption_policy == "conservative" and not self._can_admit_without_preemption(seq):
                break

            chunk_size = self._prefill_chunk_size(seq, token_budget, free_block_budget)

            if chunk_size <= 0:
                break # do not preempt active work for a new admission
            needed = self.block_manager.num_new_blocks_needed(seq, chunk_size)
            if needed > free_block_budget:
                break

            self.waiting.pop(0)
            seq.status = Status.PREFILLING
            self.prefilling.append(seq)

            prefill_batch.append(PrefillWorkItem(seq, seq.prompt_pos, chunk_size))
            protected_ids.add(id(seq))
            token_budget -= chunk_size

            free_block_budget -= needed

        return ScheduleOutput(
            prefill=prefill_batch,
            decode=decode_batch,
            num_scheduled_tokens=self.max_num_batched_tokens - token_budget,
        )