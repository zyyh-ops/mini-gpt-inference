from .scheduler import Scheduler
from .sequence import SequenceState, Status
from .block_manager import BlockManager
from .model_runner import ModelRunner
from .model import MiniGPT
from .kv_block_pool import KVBlockPool
import torch
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class IterationRecord:
    iteration: int

    prefill_requests: int
    decode_requests: int

    prefill_tokens: int
    decode_tokens: int

    scheduled_tokens: int

    waiting: int
    prefilling: int
    running: int

    used_blocks: int
    free_blocks: int

    kv_utilization: float

    prefill_ms: float
    decode_ms: float
    iteration_ms: float

    # Host wall-clock durations unless sync_cuda_timing=True.
    schedule_ms: float = 0.0
    preemptions: int = 0
    repeated_kv_tokens: int = 0
    output_tokens: int = 0
    finished_requests: int = 0


class MiniRuntime:
    # Per-iteration scheduling logs are opt-in. A class-level default keeps
    # runtimes assembled via MiniRuntime.__new__ (tests, experiments) quiet.
    verbose = False

    def __init__(
        self,
        num_blocks,
        block_size,
        max_running,
        max_num_batched_tokens,
        preemption_policy="conservative",
        sync_cuda_timing=False,
        verbose=False,
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
                preemption_policy=preemption_policy,
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
        self.records: list[IterationRecord] = []
        # An optional *synchronous* measurement mode for CUDA. The default
        # host timers measure dispatch as well as CPU work, not GPU kernel time.
        self.sync_cuda_timing = sync_cuda_timing
        self.verbose = verbose
        self._kv_high_watermark: dict[str, int] = {}

    def _timing_sync(self):
        # getattr supports tiny-test runtimes constructed via __new__.
        if getattr(self, "sync_cuda_timing", False) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _track_executed_kv(self, seq: SequenceState, start: int, count: int) -> int:
        """Count repeated logical KV positions, including prefill restarts.

        Called only after successful execution. The logical high-watermark
        survives KV eviction; physical block reuse does not reset it.
        """
        if not hasattr(self, "_kv_high_watermark"):
            self._kv_high_watermark = {}
        previous = self._kv_high_watermark.get(seq.seq_id, 0)
        end = start + count
        repeated = max(0, min(end, previous) - start)
        self._kv_high_watermark[seq.seq_id] = max(previous, end)
        return repeated

    def metrics_summary(self) -> dict:
        """Aggregated per-iteration accounting, not a serving benchmark."""
        records = self.records
        return {
            "iterations": len(records),
            "scheduled_tokens": sum(r.scheduled_tokens for r in records),
            "prefill_tokens": sum(r.prefill_tokens for r in records),
            "decode_tokens": sum(r.decode_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "preemptions": sum(r.preemptions for r in records),
            "repeated_kv_tokens": sum(r.repeated_kv_tokens for r in records),
            "peak_used_blocks_at_step_end": max((r.used_blocks for r in records), default=0),
            "average_kv_utilization_at_step_end": (
                sum(r.kv_utilization for r in records) / len(records)
                if records else 0.0
            ),
            "total_iteration_ms": sum(r.iteration_ms for r in records),
        }

    def export_records_json(self, path, metadata: dict | None = None) -> None:
        """Write a portable, explicit per-iteration trace to JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timing_mode": (
                "cuda_synchronized"
                if getattr(self, "sync_cuda_timing", False) and torch.cuda.is_available()
                else "host_wall"
            ),
            "summary": self.metrics_summary(),
            "records": [asdict(rec) for rec in self.records],
        }
        if metadata is not None:
            payload["metadata"] = metadata
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_request(
        self,
        seq: SequenceState,
    ):
        self.scheduler.add_request(seq)

    def step(self):
        self.iteration += 1
        self._timing_sync()
        iter_begin = time.perf_counter()
        preemptions_before = self.scheduler.num_preemptions
        plan = self.scheduler.schedule()
        schedule_ms = (time.perf_counter() - iter_begin) * 1000
        repeated_kv_tokens = 0
        output_tokens = 0

        assert (
            len(self.scheduler.running) + len(plan.prefill)
            <= self.scheduler.max_running
        ), f"error {len(self.scheduler.running)} + {len(plan.prefill)}, max_running={self.scheduler.running}"

        if self.verbose:
            print(f"\nIteration {self.iteration}")
            print("prefill:", [x.seq.seq_id for x in plan.prefill])
            print("decode:", [x.seq_id for x in plan.decode])
            print("scheduled tokens:", plan.num_scheduled_tokens)

        decode_begin = time.perf_counter()

        if plan.decode:
            decode_starts = [(seq, seq.length) for seq in plan.decode]
            logits, next_tokens = self.runner.decode_batch(plan.decode)
            for seq, start in decode_starts:
                repeated_kv_tokens += self._track_executed_kv(seq, start, 1)
            output_tokens += len(next_tokens)

            for seq, token in zip(plan.decode, next_tokens):
                seq.generated_tokens.append(token)
                seq.last_token = token

        self._timing_sync()
        decode_ms = (time.perf_counter() - decode_begin) * 1000

        prefill_begin = time.perf_counter()

        for item in plan.prefill:
            seq = item.seq

            if seq.first_schedule_iter is None:
                seq.first_schedule_iter = self.iteration

            logits = self.runner.prefill_chunk(
                seq=seq,
                start_pos=item.start_pos,
                num_tokens=item.num_tokens,
            )
            repeated_kv_tokens += self._track_executed_kv(
                seq, item.start_pos, item.num_tokens
            )
            seq.prompt_pos += item.num_tokens
            assert seq.prompt_pos == seq.length

            if seq.prompt_finished:
                if seq.recompute_tokens is not None:
                    # The rebuilt KV excludes the pending last_token.
                    # Never sample logits again: that output was already
                    # generated before preemption.
                    assert seq.generated_tokens
                    assert seq.last_token == seq.generated_tokens[-1]
                    assert seq.length == len(seq.recompute_tokens)
                    seq.recompute_tokens = None
                    # prompt_pos tracks the temporary recompute prefix during
                    # restoration; restore its ordinary prompt-only meaning.
                    seq.prompt_pos = seq.prompt_length
                else:
                    first_token = torch.argmax(
                        logits[:, -1],
                        dim=-1,
                    ).item()

                    seq.generated_tokens.append(first_token)
                    seq.last_token = first_token
                    seq.first_token_iter = self.iteration
                    output_tokens += 1

                seq.status = Status.RUNNING
                self.scheduler.prefilling.remove(seq)
                self.scheduler.running.append(seq)
                
        self._timing_sync()
        prefill_ms = (time.perf_counter() - prefill_begin) * 1000

        prefill_tokens = sum(
            item.num_tokens
            for item in plan.prefill
        )

        decode_tokens = len(plan.decode)

        assert (
            prefill_tokens + decode_tokens
            == plan.num_scheduled_tokens
        )

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

        finished_requests = 0
        for seq in self.scheduler.running:
            if seq.is_finished():
                seq.finish_iter = self.iteration
                finished_requests += 1
        self.scheduler.retire_finished()

        self._timing_sync()
        iteration_ms = (time.perf_counter() - iter_begin) * 1000
        free_blocks = len(self.block_manager.free_blocks)
        used_blocks = self.block_manager.num_blocks - free_blocks
        self.records.append(IterationRecord(
            iteration=self.iteration,
            prefill_requests=len(plan.prefill),
            decode_requests=len(plan.decode),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            scheduled_tokens=plan.num_scheduled_tokens,
            waiting=len(self.scheduler.waiting),
            prefilling=len(self.scheduler.prefilling),
            running=len(self.scheduler.running),
            used_blocks=used_blocks,
            free_blocks=free_blocks,
            kv_utilization=used_blocks / self.block_manager.num_blocks,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            iteration_ms=iteration_ms,
            schedule_ms=schedule_ms,
            preemptions=self.scheduler.num_preemptions - preemptions_before,
            repeated_kv_tokens=repeated_kv_tokens,
            output_tokens=output_tokens,
            finished_requests=finished_requests,
        ))
        if self.verbose:
            print("the number of free blocks: ", free_blocks)


def main():
    workloads: list[tuple[int, int]] = [
        (32, 32),
        (64, 32),
        (128, 16),
        (256, 16),
        (48, 32),
        (192, 16),
    ]

    for max_num_batched_tokens in [32, 64, 128, 256]:
        print("=" * 10 + f" {max_num_batched_tokens} " + "=" * 10)

        runtime = MiniRuntime(
            num_blocks=8,
            block_size=16,
            max_running=8,
            max_num_batched_tokens=max_num_batched_tokens,
            verbose=True,  # the demo exists to show these per-iteration logs
        )

        seq_list: list[SequenceState] = []

        for i, (prompt, output) in enumerate(workloads):
            seq = SequenceState(
                seq_id=chr(ord('A') + i),
                prompt_tokens=[i for i in range(prompt)],
                max_tokens=prompt+output,
            )
            seq_list.append(seq)

        runtime.add_request(seq_list[0])

        runtime.add_request(seq_list[1])

        for iteration in range(15):
            if iteration == 2:
                runtime.add_request(seq_list[2])

            if iteration == 4:
                runtime.add_request(seq_list[3])

            if iteration == 8:
                runtime.add_request(seq_list[4])

            if iteration == 10:
                runtime.add_request(seq_list[5])

            runtime.step()


if __name__ == "__main__":
    main()