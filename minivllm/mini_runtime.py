from scheduler import Scheduler
from sequence import SequenceState, Status
from block_manager import BlockManager
from model_runner import ModelRunner
from model import MiniGPT
from kv_block_pool import KVBlockPool
import torch


class MiniRuntime:
    def __init__(
        self,
        num_blocks,
        block_size,
        max_running,
        max_num_batched_tokens,
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

        self.iteration += 1

        plan = self.scheduler.schedule()

        assert (
            len(self.scheduler.running)
            + len(plan.prefill)
            <= self.scheduler.max_running
        ), f"error {len(self.scheduler.running)} + {len(plan.prefill)}, max_running={self.scheduler.running}"


        print(f"\nIteration {self.iteration}")

        print("prefill:", [x.seq.seq_id for x in plan.prefill])

        print("decode:", [x.seq_id for x in plan.decode])

        print("scheduled tokens:", plan.num_scheduled_tokens)

        for seq in plan.decode:
            next_token = self.runner.decode_one(seq)

            seq.generated_tokens.append(next_token)

            seq.last_token = next_token

            if seq.length >= seq.max_tokens:
                seq.status = Status.FINISHED

                self.block_manager.free(seq)

                self.scheduler.running.remove(seq)

        for item in plan.prefill:
            seq = item.seq

            logits = self.runner.prefill_chunk(
                seq=seq,
                start_pos=item.start_pos,
                num_tokens=item.num_tokens,
            )

            seq.prompt_pos += item.num_tokens

            assert seq.prompt_pos == seq.length

            if seq.prompt_finished:
                first_token = torch.argmax(
                    logits[:, -1],
                    dim=-1,
                ).item()

                seq.generated_tokens.append(first_token)

                seq.last_token = first_token

                seq.status = Status.RUNNING

                self.scheduler.prefilling.remove(seq)

                self.scheduler.running.append(seq)

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


def main():
    runtime = MiniRuntime(
        num_blocks=8,
        block_size=4,
        max_running=4,
        max_num_batched_tokens=16,
    )

    A = SequenceState(
        seq_id="A",
        prompt_tokens=[i for i in range(0, 6)],
        max_tokens=8,
    )

    B = SequenceState(
        seq_id="B",
        prompt_tokens=[i for i in range(6, 11)],
        max_tokens=9,
    )

    C = SequenceState(
        seq_id="C",
        prompt_tokens=[i for i in range(11, 21)],
        max_tokens=13,
    )

    D = SequenceState(
        seq_id="D",
        prompt_tokens=[i for i in range(21, 24)],
        max_tokens=12,
    )

    runtime.add_request(A)

    runtime.add_request(B)

    runtime.add_request(C)

    runtime.add_request(D)

    for _ in range(15):
        runtime.step()


if __name__ == "__main__":
    main()