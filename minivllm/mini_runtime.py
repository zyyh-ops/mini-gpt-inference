from scheduler import Scheduler
from sequence import SequenceState
from block_manager import BlockManager
from model_runner import ModelRunner
from model import MiniGPT
from kv_block_pool import KVBlockPool


class MiniRuntime:
    def __init__(
        self,
        num_blocks,
        block_size,
        max_running,
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
            _, _, slot = self.block_manager.reserve_append_slot(seq)

            next_token = self.runner.decode_one(seq)

            seq.generated_tokens.append(next_token)

            seq.last_token = next_token

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
        seq_id="A",
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
    )

    B = SequenceState(
        seq_id="B",
        prompt_tokens=[5, 6, 7, 8],
        max_tokens=7,
    )

    C = SequenceState(
        seq_id="C",
        prompt_tokens=[9, 10, 11],
        max_tokens=12,
    )

    runtime.add_request(A)

    runtime.add_request(B)

    runtime.add_request(C)

    for _ in range(15):
        runtime.step()


if __name__ == "__main__":
    main()