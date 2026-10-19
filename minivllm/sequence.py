from dataclasses import dataclass, field
from enum import Enum

class Status(Enum):
    WAITING = "waiting"

    PREFILLING = "prefilling"

    RUNNING = "running"

    FINISHED = "finished"


@dataclass
class SequenceState:
    seq_id: str

    max_tokens: int

    prompt_tokens: list[int]

    prompt_pos: int = 0

    last_token: int = None

    generated_tokens: list[int] = field(default_factory=list)

    status: Status = Status.WAITING

    length: int = 0

    block_table: list[int] = field(default_factory=list)

    # None for first prefill; cached token prefix when restoring an
    # interrupted RUNNING request. The pending last_token is not included.
    recompute_tokens: list[int] | None = None

    @property
    def prefill_input_tokens(self) -> list[int]:
        return self.prompt_tokens if self.recompute_tokens is None else self.recompute_tokens

    @property
    def prompt_length(self):
        return len(self.prompt_tokens)

    @property
    def remaining_prompt_tokens(self):
        return len(self.prefill_input_tokens) - self.prompt_pos

    @property
    def prompt_finished(self):
        return self.prompt_pos == len(self.prefill_input_tokens)
    
    def is_finished(self):
        max_output_tokens = self.max_tokens - self.prompt_length
        return len(self.generated_tokens) >= max_output_tokens

    def initialize(self):
        self.last_token = self.prompt_tokens[-1]

    arrival_iter: int = 0

    first_schedule_iter: int | None = None

    first_token_iter: int | None = None

    finish_iter: int | None = None