from scheduler import Scheduler
from sequence import SequenceState
from block_manager import BlockManager


class MiniRuntime:
    def __init__(
        self,
        num_blocks,
        block_size,
        max_running
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
            )
        )

        self.iteration = 0

    def add_request(
        self,
        seq: SequenceState,
    ):
        self.scheduler.add_request(seq)

    def step(self):
        print(
            f"\n=== Iteration "
            f"{self.iteration} ==="
        )

        active = self.scheduler.schedule()

        print(
            "running:",
            [
                x.seq_id
                for x in active
            ]
        )

        for seq in active:
            slot = self.block_manager.append_slot(seq)

            seq.generated_tokens.append(self.iteration)

            print(
                seq.seq_id,
                "length=",
                seq.length,
                "slot=",
                slot,
                "blocks=",
                seq.block_table
            )

        self.iteration += 1

        allocated = set(self.block_manager.block_owner.keys())

        free = set(self.block_manager.free_blocks)

        assert allocated.isdisjoint(free)


def main():
    runtime = MiniRuntime(
        num_blocks=8,
        block_size=4,
        max_running=2,
    )

    A = SequenceState(
        "A",
        max_tokens=10,
    )

    B = SequenceState(
        "B",
        max_tokens=7,
    )

    C = SequenceState(
        "C",
        max_tokens=12,
    )

    runtime.add_request(A)

    runtime.add_request(B)

    runtime.add_request(C)

    for _ in range(15):
        runtime.step()


if __name__ == "__main__":
    main()