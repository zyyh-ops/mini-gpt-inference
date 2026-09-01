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

    @property
    def prompt_length(self):
        return len(self.prompt_tokens)

    @property
    def remaining_prompt_tokens(self):
        return self.prompt_length - self.prompt_pos

    @property
    def prompt_finished(self):
        return self.prompt_pos == self.prompt_length
    
    def is_finished(self):
        return self.length >= self.max_tokens

    def initialize(self):
        self.last_token = self.prompt_tokens[-1]

    arrival_iter: int = 0

    first_schedule_iter: int | None = None

    first_token_iter: int | None = None

    finish_iter: int | None = None