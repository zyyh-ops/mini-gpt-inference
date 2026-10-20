# 复现指南：CPU Correctness、Trace 与 A/B Benchmark

本指南针对教育型 Mini-GPT Inference Runtime；复现结果是小模型、同步步骤及全部请求同时到达条件下的系统行为，**不是线上 Serving / GPU 性能声明**。

## 环境与安装

已执行验证的独立审计环境：Python 3.13、PyTorch 2.10.0+cpu（具体以个人机器为准）。项目元数据要求 Python ≥3.10、PyTorch ≥2.1。

```bash
python -m pip install -e .
python -c "import sys, torch, minivllm, minigpt; print(sys.version); print(torch.__version__)"
```

对已有 CUDA 环境，可使用 `python -m pip install -e . --no-deps`，避免改变预装的 PyTorch。

## 发布前一键 CPU 冒烟

```bash
python -m scripts.release_smoke
```

自动执行单元测试、Baseline 2 Blocks 策略对比、Mixed-Short 3 Blocks 策略对比及 Native IterationRecord 导出校验。输出 JSON 使用临时目录，正常退出后清理。这里只比较调度轮数、抢占次数、重复 KV Token、输出序列一致性，**不比较绝对耗时**。

预期行为如下（若这些值因调度算法修改而变化，应重新审核而非简单修改预期）：

| Workload | Pool | Conservative（Iterations / Preemptions / Repeated KV） | Recompute（Iterations / Preemptions / Repeated KV） |
|---|---:|---|---|
| Baseline | 2 Blocks | 27 / 0 / 0 | 27 / 3 / 44 |
| Mixed-Short | 3 Blocks | 34 / 0 / 0 | 32 / 0 / 0 |

## 单独导出实验

```bash
python -m experiments.trace_preemption --workload baseline --blocks 2 \
  --output-json artifacts/preemption/trace_blocks2.json
python -m experiments.record_runtime --policy recompute --blocks 2 \
  --output-json artifacts/observability/records.json
python -m experiments.bench_preemption --device cpu --workload mixed_short \
  --num-blocks 3 --repeats 5 \
  --output-json artifacts/preemption/mixed_short_cpu.json
```

Benchmark 包含每轮同步及 correctness 检查，CPU 耗时易受机器负载影响；小样本 P95 不代表在线尾延迟。GPU 未验证，不能直接推导 GPU 吞吐或 Kernel 性能。

## Git 发布检查

```bash
python -m scripts.tracked_file_audit
# 手动审查实际 Git 历史和 staged files
# git status --short
# git ls-files
# git log --stat --oneline -n 10
```

`.gitignore` 只影响未来新发现的未跟踪文件；已被 Git 跟踪的 Profiler 输出/数据不会被自动取消跟踪。审查确认后，可使用 `git rm --cached <path>` 从跟踪索引移除，但要注意它仍可能留在旧的公开提交历史里。请自行审查许可文件、素材来源、模型和测试数据的可发布性。
