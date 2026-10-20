# Mini-GPT Inference Runtime

一个面向 **LLM 推理系统原理与性能实验** 的教学型 Mini-GPT Runtime，围绕请求调度、Paged KV Cache、Continuous Batching、Chunked Prefill、Batched Decode 和 **Recompute-based Preemption** 实现端到端推理流程。

> **项目定位**：这是为理解推理引擎底层机制而构建的 educational runtime，**不是**生产级 vLLM/SGLang 替代品。现有 CPU 测试验证了部分核心正确性与调度行为；尚未提供可与生产引擎直接比较的 GPU Serving Benchmark。

## 核心实现

- **Request lifecycle**：`WAITING → PREFILLING → RUNNING → FINISHED`；请求完成后回收其 KV Blocks。
- **KV Block Manager**：按需分配物理 Block，维护每个请求的 Block Table 与所有权映射，支持释放及复用。
- **Chunked Prefill**：受每轮 Token Budget 与 KV 空间约束，将 Prefill 切成多个调度 Chunk。
- **Batched Decode**：批量构造 `positions`、`context_lengths`、`block_tables`、`slot_mapping`；执行批量 QKV、KV Write 与向量化 PagedAttention（仍包含 Python 层逐逻辑 Block 遍历）。
- **Recompute-based Preemption**：KV 空间不足时抢占其他未保护的活跃请求，丢弃物理 KV，保留历史 Token，之后通过 Chunked Prefill 重建 KV，再继续 Decode。完成重计算时不重复采样。
- **Scheduler A/B**：比较 `conservative`（最大 KV 容量承诺）与 `recompute`（乐观准入 + 按需抢占）策略，并保留逐轮 Trace 与 CPU Benchmark。
- **Runtime Observability**：每轮自动生成 `IterationRecord`，记录 Token、抢占、重复 KV 计算、Block 快照及主机/可选 CUDA 同步计时。

## 代码架构

```text
mini-gpt-inference/
├── minivllm/
│   ├── mini_runtime.py          # Runtime 入口与单轮执行循环
│   ├── scheduler.py             # Token/KV Budget、准入与自动抢占
│   ├── sequence.py              # 请求生命周期、历史 Token、状态
│   ├── block_manager.py         # Paged KV Block 分配 / 释放 / 不变量
│   ├── kv_block_pool.py         # KV Tensor 的物理存储
│   ├── model_runner.py          # Prefill / Decode 输入元数据构造
│   ├── model.py                 # KV-aware MiniGPT
│   └── paged_attention_batch.py # 批量 PagedAttention reference + online v1
├── experiments/
│   ├── bench_preemption.py      # 配对 CPU/CUDA 实验
│   ├── trace_preemption.py      # 按 iteration 记录 Scheduler 状态
│   └── record_runtime.py        # Runtime 原生指标 JSON 导出
├── scripts/                     # 统一 CPU 发布冒烟 / Git 文件审计
├── tests/                       # CPU Correctness / Preemption tests
└── minigpt/                     # 早期 MiniGPT 学习实现
```

完整模块责任与关键不变量见 [docs/architecture.md](docs/architecture.md)，旧导入方式升级说明见 [Phase 2 迁移指南](docs/phase2_migration.md)。

## 运行与验证

```bash
python -m pip install -e .
```

主要运行路径依赖 **PyTorch**。建议创建独立虚拟环境，并从仓库根目录执行 `python -m pip install -e .` 安装可编辑包；如果已经单独安装了与 CUDA 匹配的 PyTorch，可使用 `python -m pip install -e . --no-deps`。`requirements.txt` 已精简为运行时的最小依赖。

以下命令从仓库根目录执行，适用于 Linux/macOS：

```bash
# 所有单元测试（原 29 项 + Runtime 观测测试 4 项，合计 33 项）
python -m unittest discover -s tests -p 'test_*.py' -v

# CPU 快速对比，两个策略分别执行相同的四请求工作负载
python -m experiments.bench_preemption \
  --workload mixed_short --device cpu --num-blocks 3 \
  --repeats 5 --output-json artifacts/preemption/cpu_mixed_short_blocks3.json

# Scheduler 逐轮 Trace；这是行为追踪，不用于计时
python -m experiments.trace_preemption \
  --workload mixed_short --blocks 3 \
  --output-json artifacts/preemption/trace_mixed_short_blocks3.json

# 导出 Runtime 原生逐轮指标（不依赖外部 CountingRunner）
python -m experiments.record_runtime --policy recompute --blocks 2 \
  --output-json artifacts/observability/recompute_blocks2.json
```

> **包运行方式**：`minivllm` 使用包内相对导入，`experiments` 使用 `python -m ...` 作为入口。仓库根目录直接运行这些命令不必设置 `PYTHONPATH`；从其他目录运行前，应先执行 `python -m pip install -e .`。

## Scheduler 策略实验（CPU Smoke）

所有实验均使用 Tiny MiniGPT（1 层、hidden size 16、2 heads、FP32、CPU），所有请求同一时刻进入系统，greedy decode；不同容量配置下，两种策略输出 Token 序列一致。**下面的数字仅描述这个简化同步 Runtime 的 CPU 行为，不是生产级在线性能。**

| Workload | KV Blocks | 策略 | Iterations | Preemptions | Repeated KV Tokens | 平均 E2E (ms) |
|---|---:|---|---:|---:|---:|---:|
| Baseline | 2 | Conservative | 27 | 0 | 0 | 15.82 |
| Baseline | 2 | Recompute | 27 | 3 | 44 | 20.09 |
| Baseline | 4 | Conservative | 14 | 0 | 0 | 14.00 |
| Baseline | 4 | Recompute | 14 | 2 | 28 | 17.30 |
| Mixed-Short | 3 | Conservative | 34 | 0 | 0 | 20.13 |
| Mixed-Short | 3 | Recompute | 32 | 0 | 0 | 8.07 |

**实验解读**：Baseline 负载中激进准入导致额外 KV 重算、性能反而变差；Mixed-Short 负载中三个短请求可以在长请求扩展 KV 之前完成，乐观准入显著降低平均请求延迟，但并未在这组 CPU 测试中明确提高整体吞吐。后一个结果由**乐观准入**而非实际发生的 Preemption 带来（抢占次数为零）。

参见 [Preemption 实验分析](docs/experiments/preemption.md) 了解完整配置、统计口径、逐轮解释与局限。

逐轮 `IterationRecord` 的指标含义、导出命令、CUDA 异步计时限制见 [Runtime Observability](docs/runtime_observability.md)。

## 正确性与已知局限

Phase 2 工程化副本在 Python 3.13 / PyTorch 2.10 CPU 环境下，现已 **36 项 CPU 测试 PASS**（原 33 项 + 3 项发布契约测试）。覆盖 KV Block 分配/复用、跨 Block Attention、无效 KV Padding（NaN）、生命周期、Scheduler 资源竞争、手动与自动抢占及恢复。**不能据此推导出 GPU/CUDA kernel 已验证。**

目前尚未完成：动态到达工作负载、GPU Profiler / 吞吐实验、线上级公平性/活性分析。请参见 [公开前检查清单](docs/release_checklist.md)。

## License

本项目采用 **Apache License 2.0** 开源许可证，具体条款参见 [LICENSE](LICENSE)。

本许可证适用于仓库中作者有权授权的原创代码；如包含第三方来源的代码或素材，请保留其各自的授权与署名信息。
