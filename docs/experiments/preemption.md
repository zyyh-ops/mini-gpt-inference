# Scheduler / Preemption 策略 A/B 实验报告

## 实验目的与范围

比较两种请求准入策略在有限 Paged KV Cache 下的取舍：

1. **Conservative Admission**：对已经准入的请求预留最大可能的 KV Block 数量。
2. **Recompute Policy**：按当前物理 KV 占用乐观准入，执行时若缺少空间则抢占其他请求，并通过重计算恢复 KV。

必须区分**乐观准入带来的收益**和**真正发生 Preemption 的重计算开销**。本实验只是小型同步 MiniRuntime 的 CPU Smoke，不能代表生产环境 vLLM/SGLang 的吞吐与尾延迟。

## 统一实验环境

- CPU、PyTorch 2.12.1+cpu（原始用户实验 JSON 记录）与 FP32，Tiny MiniGPT `dim=16, layers=1, heads=2`。
- Paged KV `block_size=16`；`max_running=4`；`token_budget=64`；greedy decode。
- 四个请求同时提交；两种策略使用同一模型与确定性输入；每次配对验证输出 Token 一致。
- 每轮 `MiniRuntime.step()` 后记录时间；GPU 路径额外同步。计时循环仍含状态检查、Python 调度与同步开销，因此属于端到端小规模 smoke 时间。
- P95 仅为每轮四个请求结束时间的描述性分位数，不是线上统计显著的尾延迟指标。

## 实验 A：Baseline，KV Pool = 2 / 4 / 8 Blocks

负载：A=(Prompt16, Output8)，B=(16,8)，C=(12,6)，D=(16,5)，合计 27 输出 Token。每组 3 次 CPU 重复。

| KV Blocks | Policy | Iterations | Preemptions | Replayed KV Tokens | Mean total wall (ms) | Mean avg E2E (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Conservative | 27 | 0 | 0 | 23.045 | 15.815 |
| 2 | Recompute | 27 | 3 | 44 | 28.727 | 20.089 |
| 4 | Conservative | 14 | 0 | 0 | 18.700 | 14.002 |
| 4 | Recompute | 14 | 2 | 28 | 20.978 | 17.301 |
| 8 | Conservative | 8 | 0 | 0 | 14.964 | 13.921 |
| 8 | Recompute | 8 | 0 | 0 | 15.059 | 13.872 |

**Trace 归因**：

- 2 Blocks：第 2、10、22 轮分别抢占 B/C/D，恢复时重算 16+12+16=44 KV Tokens；Recompute 先提前生成首 Token，后由纯重计算轮抵消，最终仍是 27 轮。
- 4 Blocks：第 2 轮为 A、B 扩展第二个 KV Block，抢占 C、D；第 9 轮恢复 C、D 共 28 Tokens。本轮没有新输出 Token，因此 Conservative 和 Recompute 均 14 轮结束。
- 8 Blocks：两种策略都没有抢占；几乎无额外 KV 重算工作。

**结论**：更激进的准入不保证更多有效 Batched Decode；若请求刚好跨越 Block 边界，又没有足够容量供后续 Decode，额外 Prefill/recompute 可能大于并发收益。

## 实验 B：Mixed-Short，KV Pool = 4 Blocks

负载：A=(Prompt4, Output32)，B/C/D=(4,2)，合计 38 输出 Token。3 次 CPU 重复。

| Policy | Iterations | Preemptions | Replayed KV | Mean total wall | Mean avg TTFT | Mean avg E2E |
|---|---:|---:|---:|---:|---:|---:|
| Conservative | 32 | 0 | 0 | 21.470 ms | 3.044 ms | 8.357 ms |
| Recompute | 32 | 0 | 0 | 21.572 ms | 3.411 ms | 8.410 ms |

Recompute 同时准入四个请求；B/C/D 在第 2 轮结束。Conservative 分批准入，B/C/D 分别在第 2、4、6 轮结束。短请求 D 更早完成，但 B 因较大的首轮工作量可能更慢。两者总轮数均由 A 的 32 输出 Token 决定。

## 实验 C：Mixed-Short，KV Pool = 3 Blocks

与实验 B 相同工作负载，减少 KV Pool 至 3 Blocks；5 次 CPU 重复。

| Policy | Iterations | Preemptions | Replayed KV | Mean total wall | Mean avg TTFT | Mean avg E2E |
|---|---:|---:|---:|---:|---:|---:|
| Conservative | 34 | 0 | 0 | 20.770 ms | 15.450 ms | 20.128 ms |
| Recompute | 32 | 0 | 0 | 20.780 ms | 3.134 ms | 8.065 ms |

| Request | Conservative mean E2E | Recompute mean E2E |
|---|---:|---:|
| A (long) | 18.265 ms | 20.755 ms |
| B (short) | 20.749 ms | 3.239 ms |
| C (short) | 20.749 ms | 3.239 ms |
| D (short) | 20.749 ms | 5.027 ms |

Conservative 初期只准入 A（最大 KV 需求 3 Blocks，虽然当前只用 1 个），B/C/D 直到 A 第 32 轮完成后才运行，共 34 轮。Recompute 第 1 轮准入 A/B/C，第 2 轮 B/C 完成，第 3 轮 D Prefill，第 4 轮 D 完成，第 32 轮 A 完成。

平均 TTFT 下降约 79.7%，平均请求 E2E 下降约 59.9%，但总 wall time 几乎相同（20.770 vs 20.780 ms）。**特别强调：这一负载根本没有发生抢占；延迟变化应归因于乐观准入，而非 Recompute 算法本身。** 长请求 A 的平均 E2E 则上升了。

## 正确性与统计限制

- 每组相同工作负载下，两种策略全部输出 Token 序列一致；所有请求均 FINISHED，最终物理 KV Blocks 已回收。
- CPU 小模型与 Python 控制流的成本占比不同于 GPU；不能据此推导实际 LLM 推理服务性能。
- 在 `run_once()` 的每一轮都执行状态不变量检查，因此包含测量干扰；正式 GPU Benchmark 应移出计时热路径。
- 目前仅测试同一时间到达的四个请求，不具备动态到达、排队负载、长尾规模的代表性。
- 多次重复可初步观察波动，但仍不足以支持关于普遍性能优势的统计结论。

## 复现命令

```bash
python -m experiments.bench_preemption \
  --device cpu --workload baseline --num-blocks 2,4,8 \
  --repeats 3 --output-json artifacts/preemption/cpu_smoke.json

python -m experiments.bench_preemption \
  --device cpu --workload mixed_short --num-blocks 3 \
  --repeats 5 --output-json artifacts/preemption/cpu_mixed_short_blocks3.json

python -m experiments.trace_preemption \
  --workload mixed_short --blocks 3 \
  --output-json artifacts/preemption/trace_mixed_short_blocks3.json
```

建议后续增加 GPU 计时（单独 warmup、同步边界、Kernel/Python 时间分解）、请求动态到达、不同负载分布以及多个 victim selection 策略。
