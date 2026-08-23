from block_manager import BlockManager
from sequence import SequenceState, Status


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_size: int,
    ):
        self.waiting: list[SequenceState] = []
        self.running: list[SequenceState] = []

        self.block_manager = block_manager

        self.max_batch_size = max_batch_size

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

            seq.status = Status.RUNNING

            self.running.append(seq)

    def schedule(self):

        self.retire_finished()

        self.admit()

        return self.running


