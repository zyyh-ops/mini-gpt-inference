# Runtime Observability：每轮指标与统计口径

Phase 2 · Step 2 为 `MiniRuntime.step()` 接入 `IterationRecord`。
它是**教学 Runtime 的执行统计**，不等于 vLLM/SGLang 的线上 Serving 指标。

## 快速复现

从仓库根目录执行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v

python -m experiments.record_runtime --policy recompute --blocks 2 \
  --output-json artifacts/observability/recompute_blocks2.json

python -m experiments.record_runtime --policy conservative --blocks 3 \
  --workload mixed_short \
  --output-json artifacts/observability/conservative_mixed3.json
```

第一条 `baseline + recompute + 2 Blocks` 应得到 **27 iterations、3 次 Preemption、44 个重复 KV Token**；第二条 `mixed_short + conservative + 3 Blocks` 应得到 **34 iterations、0 次 Preemption**。同一输入下的最终 Token 需要与之前实验一致。

每次调用 `runtime.step()` 会向 `runtime.records` 追加一个 `IterationRecord`。

```python
from dataclasses import asdict
record = runtime.records[-1]
print(record.preemptions, record.repeated_kv_tokens, record.used_blocks)
print(runtime.metrics_summary())
runtime.export_records_json('artifacts/observability/records.json')
```

## 计数字段的定义

- `scheduled_tokens`: 当轮 Prefill 和 Decode 总输入 Token 数；必须等于 `prefill_tokens + decode_tokens`。
- `prefill_requests` / `decode_requests`: 当轮执行的请求（或 Prefill Work Item）数量。
- `output_tokens`: 当轮新采样的 Token 数；**重计算完成时不重复采样**。
- `preemptions`: 当轮 `Scheduler.num_preemptions` 的增量，而非累计总值。
- `repeated_kv_tokens`: 当前轮重复执行的**逻辑 KV Token 位置**，包含抢占后重计算以及已执行部分 Prefill 被中断后的重做。通过每个请求的历史高水位（highest executed logical KV position）判断，包含 Decode 历史，以独立 `CountingRunner` 交叉验证。它不等于重计算 FLOPs。
- `finished_requests`: 当轮由 RUNNING 转移为 FINISHED 的请求数。
- `waiting` / `prefilling` / `running`: **本轮结束、回收完成后**的队列长度。
- `used_blocks` / `free_blocks` / `kv_utilization`: **本轮结束、回收完成后**的 Block 快照。`peak_used_blocks_at_step_end` 不代表执行轮内部的瞬时峰值。
- `schedule_ms`: Scheduler 调度阶段耗时。
- `decode_ms`, `prefill_ms`: 执行阶段耗时（默认 Host Wall，注意下方口径）。
- `iteration_ms`: 整轮调用的耗时，包括调度、执行、状态更新及统计。不能简单等同于上面三个阶段的和。

### 计时模式

默认 `sync_cuda_timing=False`，使用 `time.perf_counter()`：

- CPU：是主机墙钟时间，包含 Python 调用与 CPU 算子执行。
- CUDA：仅保证计时覆盖 **Host Dispatch**，GPU 异步算子可能还在执行；不能将其称作 GPU Kernel Latency。

需要粗略测量带 GPU 工作完成时间时，可以显式传入 `MiniRuntime(..., sync_cuda_timing=True)`，或运行：

```bash
python -m experiments.record_runtime --device cuda --policy recompute \
  --blocks 2 --sync-cuda-timing \
  --output-json artifacts/observability/cuda_synced.json
```

它在轮首、Decode 后、Prefill 后和轮末进行同步，**同步本身会改变执行时序及性能**。这是同步运行时阶段耗时，不是纯 Kernel Time；需要准确分析 Kernel，应使用 CUDA Events / Nsight Systems / Nsight Compute，并在性能实验时采用统一的计时方法。

## 已知限制

1. 默认每次 `step()` 追加一条记录，长期运行会占用内存；之后可以添加 bounded ring buffer / sampling。
2. 重复 KV Token 统计依赖请求 `seq_id` 的唯一性。现有 `Scheduler.add_request` 尚未统一拒绝重复 ID，请勿用重复 ID 做本实验。
3. 所有请求一次性到达，无真实时间上的动态到达、流式输出与排队 SLA。
4. 采样输出与 Block 使用量均为轮级记录，不能替代 GPU 物理显存采样。
5. 统计过程存在 Python 开销；不要把此前不带运行时统计的绝对耗时与本版本直接对比。
