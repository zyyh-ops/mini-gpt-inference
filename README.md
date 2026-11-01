# Mini-GPT Inference Runtime

一个面向 **LLM Inference Runtime 原理、状态管理与性能优化实验**
的教学型推理运行时。项目使用 PyTorch 实现 Mini-GPT
推理路径，重点研究请求调度、Paged KV Cache、Chunked Prefill、Batched
Decode、Recompute-based Preemption，以及 Prefill / Decode Attention
后端优化。

> **项目定位**：Educational Inference Runtime，而不是生产级 vLLM /
> SGLang 替代品。项目使用小型 Mini-GPT
> 与封闭式工作负载验证系统机制，不提供网络
> Serving、真实模型权重管理或线上 SLO 能力。

> **Release Closure**：项目最终 Release Closure
> 已完成，最终检查均已确认通过。

## 核心能力

-   **Request lifecycle**：`WAITING → PREFILLING → RUNNING → FINISHED`。
-   **Scheduler**：按 Decode、已有
    Prefill、新请求准入的顺序构造每轮执行计划；受
    `max_running`、`max_num_batched_tokens` 和 KV Block 容量约束。
-   **Paged KV Cache**：逻辑 Token 位置通过 Block Table 映射到非连续物理
    Block；BlockManager 管理分配、所有权、释放和复用，KVBlockPool
    保存实际 K/V Tensor。
-   **Chunked Prefill**：把长 Prompt 拆成多个 Prefill Chunk，支持 Token
    Budget 与 KV 容量约束。
-   **Batched Decode**：将多个请求的输入 Token、Position、Context
    Length、Block Table 和 Slot Mapping 组织为批量 Tensor。
-   **Recompute-based Preemption**：KV
    容量不足时抢占未受保护的活跃请求，释放物理 KV，保留语义 Token
    历史，之后重新 Prefill 构建 KV，不重复采样已经生成的 pending Token。
-   **Prefill Attention optimization**：Reference、SDPA v1（首 Chunk）和
    Prefix-aware SDPA v2（包含非零起点 Chunk）。
-   **Decode Attention optimization**：PyTorch batched online-softmax
    Reference，以及可显式启用的 Triton fused Paged Decode Attention v0。
-   **Runtime observability**：逐轮记录 Prefill/Decode Token
    数、队列状态、KV Block 使用情况、抢占次数、重复 KV
    计算量和阶段耗时。
-   **Release engineering**：单元测试、CPU Release Smoke、Git
    tracked-file audit 和 Apache License 2.0。

## 架构概览

``` text
Request(Token IDs)
       |
       v
MiniRuntime.add_request
       |
       v
SequenceState <---- semantic history / pending token / status
       |
       v
Scheduler.schedule
       |                         |
       | Decode batch            | Prefill work items
       v                         v
ModelRunner.decode_batch    ModelRunner.prefill_chunk
       |                         |
       v                         v
MiniGPT.decode_batch        MiniGPT.prefill_chunk
       |                         |
       v                         v
KV Write -> Paged Attention  KV Write -> Reference / SDPA v1 / SDPA v2
       |                         |
       +------------+------------+
                    |
                    v
                 Logits
                    |
                    v
              Greedy Argmax
                    |
                    v
       SequenceState / Scheduler.postprocess
                    |
                    v
            Finish and free KV Blocks
```

更完整的组件职责、状态所有权和关键不变量见：

-   [架构设计](docs/architecture.md)
-   [Paged KV Cache 与调度](docs/kv-cache-and-scheduling.md)
-   [Attention 后端与优化](docs/attention-backends.md)
-   [Benchmark 与 Profiling](docs/benchmark-and-profiling.md)
-   [实验结果](docs/benchmark-results.md)
-   [测试、Release 与限制](docs/testing-and-release.md)
-   [已知限制与非目标](docs/known-limitations.md)

## 代码结构

``` text
mini-gpt-inference/
├── minivllm/
│   ├── mini_runtime.py             # Runtime 入口、单轮执行和 IterationRecord
│   ├── scheduler.py                # Decode/Prefill 调度、准入和抢占
│   ├── sequence.py                 # SequenceState、请求状态和 Token 历史
│   ├── block_manager.py            # 物理 Block 分配、映射、释放和不变量
│   ├── kv_block_pool.py            # 实际 K/V Tensor 存储
│   ├── model_runner.py              # Runtime state -> GPU input/metadata
│   ├── decode_batch_metadata.py    # Batched Decode metadata
│   ├── kv_runtime.py               # Prefill/Decode 单次执行上下文
│   ├── model.py                    # KV-aware MiniGPT
│   ├── attention.py                # Prefill/Decode Attention 路径
│   ├── paged_attention_batch.py    # Batched PyTorch online-softmax Reference
│   └── triton_paged_decode.py      # Triton fused Decode Attention v0
├── minigpt/                        # 较早期 Mini-GPT 学习实现
├── experiments/                    # CPU/GPU correctness、benchmark 和 profiling
├── scripts/
│   ├── release_smoke.py             # 一键 CPU Release Smoke
│   └── tracked_file_audit.py        # Git tracked path/size 只读审计
├── tests/                           # 单元测试和后端集成测试
├── docs/                            # 项目技术文档
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Quick Start

建议使用独立 Python 环境。项目运行时依赖 PyTorch；CUDA
环境应先安装与本机驱动/CUDA 兼容的 PyTorch。

``` bash
python -m pip install -e .
```

如果已经安装了正确的 PyTorch，不希望安装过程替换 PyTorch：

``` bash
python -m pip install -e . --no-deps
```

运行单元测试：

``` bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

运行完整 CPU Release Smoke：

``` bash
python -m scripts.release_smoke
```

运行只读 Git tracked-file audit：

``` bash
python -m scripts.tracked_file_audit
```

## 常用实验入口

CPU 调度策略对照：

``` bash
python -m experiments.bench_preemption \
  --device cpu \
  --workload baseline \
  --num-blocks 2 \
  --repeats 1 \
  --output-json artifacts/preemption/cpu_baseline.json
```

导出逐轮 Runtime Telemetry：

``` bash
python -m experiments.record_runtime \
  --device cpu \
  --policy recompute \
  --blocks 2 \
  --output-json artifacts/observability/recompute_blocks2.json
```

GPU Prefill SDPA v2 correctness：

``` bash
python -m experiments.gpu_prefill_fastpath_v2_ab \
  --device cuda \
  --mode correctness \
  --prompts 128,256 \
  --token-budget 64 \
  --dim 256 --layers 4 --heads 8 \
  --output-json artifacts/gpu/prefill_sdpa_v2_correctness.json
```

GPU Triton Decode Runtime A/B：

``` bash
python -m experiments.gpu_day6_runtime_decode_ab \
  --device cuda \
  --mode e2e \
  --prompts 128,256 \
  --batches 1,4 \
  --token-budget 64 \
  --output-len 8 \
  --dim 256 --heads 8 --layers 4 --block-size 16 \
  --warmups 2 --repeats 7 \
  --output-json artifacts/gpu/day6_step5_e2e_ab.json
```

真实 Recompute Preemption correctness / E2E 命令见 [Benchmark 与
Profiling](docs/benchmark-and-profiling.md)。

## 已验证结果摘要

在 NVIDIA RTX 5090、PyTorch `2.6.0a0+ecf3bae40a.nv25.01`、FP32、TF32
关闭的指定实验配置下：

-   Prefix-aware SDPA v2 相比 Prefill
    Reference，在单请求和四请求闭合批次中取得约 **4.69×--10.57×** E2E
    加速。
-   Triton fused Decode Attention v0 的独立 Attention-only
    benchmark，在测试的 Batch/Context 组合中取得约 **12.18×--81.68×**
    加速。
-   Triton Decode 接入 Runtime 后，在无抢占闭合批次中取得约
    **3.80×--6.34×** E2E 加速。
-   在强制发生 Recompute Preemption 的工作负载中，Reference 与 Triton
    后端均发生 1 次抢占、重放 16 个 KV Token，输出 Token
    与生命周期计数一致；Triton 后端取得约 **4.31×--5.44×** E2E 加速。

这些是特定小模型、FP32、闭合批次和指定 GPU
环境下的实验结果，不代表生产模型或在线 Serving
的普遍性能。完整实验条件与数据见
[benchmark-results.md](docs/benchmark-results.md)。

## 项目边界

当前项目是教学型 Runtime：

-   使用 Mini-GPT 小模型和 Token ID 输入，不包含完整网络 Serving 服务。
-   Decode 采用 Greedy Argmax，不提供完整的 Sampling API。
-   当前 Runtime 没有实现 Prefix Cache、Tensor Parallel、Pipeline
    Parallel 或 CUDA Graph。
-   Triton v0 仅覆盖指定 CUDA/FP32/MHA Decode Attention 场景。

更多限制见 [known-limitations.md](docs/known-limitations.md)。

## License

Apache License 2.0，详见 [LICENSE](LICENSE)。
