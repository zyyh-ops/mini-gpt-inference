import time
import torch
from sequence import SequenceState
from model_runner import ModelRunner
from model import MiniGPT
from kv_block_pool import KVBlockPool
from block_manager import BlockManager
from torch.profiler import ProfilerActivity, profile, record_function


def cuda_time_ms(fn):
    torch.cuda.synchronize()

    begin = time.perf_counter()

    fn()

    torch.cuda.synchronize()

    return (
        time.perf_counter() - begin
    ) * 1000


def reset_length(
    sequences: list[SequenceState],
    fixed_length: int,
):
    for seq in sequences:
        seq.length = fixed_length


def benchmark_serial(
    runner: ModelRunner,
    sequences: list[SequenceState],
    fixed_length: int,
    warmup: int = 30,
    repeats: int = 200,
):
    times = []

    with torch.inference_mode():
        for _ in range(warmup):

            reset_length(
                sequences=sequences,
                fixed_length=fixed_length,
            )

            for seq in sequences:
                runner.decode_one(seq)

        for _ in range(repeats):

            reset_length(
                sequences=sequences,
                fixed_length=fixed_length,
            )

            ms = cuda_time_ms(
                lambda: [
                    runner.decode_one(seq)
                    for seq in sequences
                ]
            )

            times.append(ms)

    return times


def benchmark_batch(
    runner: ModelRunner,
    sequences: list[SequenceState],
    fixed_length: int,
    warmup: int = 30,
    repeats: int = 200,
):
    times = []

    with torch.inference_mode():
        for _ in range(warmup):

            reset_length(
                sequences=sequences,
                fixed_length=fixed_length,
            )

            runner.decode_batch(sequences)

        for _ in range(repeats):

            reset_length(
                sequences=sequences,
                fixed_length=fixed_length,
            )

            ms = cuda_time_ms(
                lambda: runner.decode_batch(sequences)
            )

            times.append(ms)

    return times


def summarize(times):
    import statistics

    xs = sorted(times)

    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p95": xs[int(len(xs) * 0.95)],
    }


def profile_serial(
    runner: ModelRunner,
    sequences: list[SequenceState],
    fixed_length: int = 120,
    warmup: int = 20,
):
    with torch.inference_mode():

        for _ in range(warmup):
            reset_length(
                sequences,
                fixed_length,
            )

            for seq in sequences:
                runner.decode_one(seq)

            
        reset_length(
            sequences,
            fixed_length,
        )

        torch.cuda.synchronize()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            
            with record_function(
                "serial_decode"
            ):
                for seq in sequences:
                    runner.decode_one(seq)

        torch.cuda.synchronize()

    return prof


def profile_batch(
    runner: ModelRunner,
    sequences: list[SequenceState],
    fixed_length: int = 120,
    warmup: int = 20,
):
    with torch.inference_mode():

        for _ in range(warmup):
            reset_length(
                sequences,
                fixed_length,
            )

            runner.decode_batch(sequences)

        reset_length(
            sequences,
            fixed_length,
        )

        torch.cuda.synchronize()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:

            with record_function(
                "batch_decode"
            ):
                runner.decode_batch(sequences)

        torch.cuda.synchronize()

    return prof


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N_EMBD, N_HEAD, VOCAB_SIZE, BLOCK_SIZE = 512, 8, 1024, 1024

    BLOCK_SIZE_1 = 16

    for B in [16]:
        NUM_BLOCKS = 8 * B

        model = MiniGPT(
            n_embd=512,
            n_head=8,
            vocab_size=1024,
            block_size=1024,
            dropout=0,
            n_layer=8,
        ).to(device)

        kv_pool = KVBlockPool(
            n_layer=8,
            num_blocks=NUM_BLOCKS,
            block_size=BLOCK_SIZE_1,
            n_head=8,
            head_dim=64,
            device=device,
        )

        serial_runner = ModelRunner(
            model=model,
            kv_pool=kv_pool,
            block_manager=BlockManager(
                num_blocks=NUM_BLOCKS,
                block_size=16,
            )
        )

        batch_runner = ModelRunner(
            model=model,
            kv_pool=kv_pool,
            block_manager=BlockManager(
                num_blocks=NUM_BLOCKS,
                block_size=BLOCK_SIZE_1,
            )
        )

        sequences = []

        for i in range(B):
            seq = SequenceState(
                seq_id=chr(ord('A') + i),
                prompt_tokens=[i for i in range(120)],
                max_tokens=128,
            )

            sequences.append(seq)

        import copy

        serial_sequences = copy.deepcopy(sequences)
        batch_sequences = copy.deepcopy(sequences)

        for seq in serial_sequences:
            next_token = serial_runner.prefill(seq)

            seq.last_token = next_token

        for seq in batch_sequences:
            next_token = batch_runner.prefill(seq)

            seq.last_token = next_token

        # serial_times = benchmark_serial(
        #     runner=serial_runner,
        #     sequences=serial_sequences,
        #     fixed_length=120,
        # )

        # batch_times = benchmark_batch(
        #     runner=batch_runner,
        #     sequences=batch_sequences,
        #     fixed_length=120,
        # )

        # statis_serial_times = summarize(serial_times)

        # statis_batch_times = summarize(batch_times)

        # serial_time = statis_serial_times["mean"]

        # batch_time = statis_batch_times["mean"]

        # speedup = serial_time / batch_time

        # throughput_serial = B * 1000 // serial_time

        # throughput_batch = B * 1000 // batch_time

        # batch_efficiency = speedup / B

        # print("=" * 50 + f" B_active={B} " + "=" * 50)

        # print(f"statis_serial_times={statis_serial_times}")

        # print(f"statis_batch_times={statis_batch_times}")

        # print(f"speedup={speedup:.2f}")

        # print(f"serial tok/s={throughput_serial} tokens/s")

        # print(f"batch tok/s={throughput_batch} tokens/s")

        serial_prof = profile_serial(
            runner=serial_runner,
            sequences=serial_sequences,
        )

        batch_prof = profile_batch(
            runner=batch_runner,
            sequences=batch_sequences,
        )

        for prof in (serial_prof, batch_prof):
            print(
                prof.key_averages(
                    group_by_input_shape=True
                ).table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            )

            print("=" * 80)

            print(
                prof.key_averages(
                    group_by_input_shape=True
                ).table(
                    sort_by="self_cpu_time_total",
                    row_limit=30,
                )
            )

            print("=" * 80)

            for event in prof.key_averages():
                if (
                    "item" in event.key
                    or
                    "local_scalar" in event.key
                ):
                    print(
                        event.key,
                        "count=",
                        event.count,
                        "cpu=",
                        event.cpu_time_total,
                    )

            print()
            print()


if __name__ == "__main__":
    main()