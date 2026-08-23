from dataclasses import dataclass, field
from enum import Enum

class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SequenceState:
    seq_id: str

    max_tokens: int

    prompt_tokens: list[int]

    last_token: int

    generated_tokens: list[int] = field(default_factory=list)

    status: Status = Status.WAITING

    length: int = 0

    block_table: list[int] = field(default_factory=list)

    def is_finished(self):
        return self.length >= self.max_tokens