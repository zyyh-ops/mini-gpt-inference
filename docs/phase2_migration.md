# Phase 2：Python Package 迁移说明

本阶段只整理模块边界、可安装性、测试与实验入口，不更改 Scheduler 的实际调度算法、Preemption 策略和模型计算。

## 安装

从项目根目录执行：

```bash
python -m pip install -e .
```

如已经单独安装与你的 CUDA 驱动兼容的 PyTorch，可以执行 `python -m pip install -e . --no-deps`，避免 pip 调整现有 PyTorch。

## 新命令

```bash
# CPU 测试：原 26 项 + 3 项导入检查
python -m unittest discover -s tests -p 'test_*.py' -v

# CPU Benchmark
python -m experiments.bench_preemption --workload mixed_short --device cpu \
  --num-blocks 3 --repeats 3 --output-json artifacts/preemption/cpu_phase2.json

# 调度轨迹
python -m experiments.trace_preemption --workload mixed_short --blocks 3 \
  --output-json artifacts/preemption/trace_phase2.json
```

无需设置 `PYTHONPATH`；上面的 `python -m ...` 推荐从仓库根目录运行。`experiments/*.py` 不再直接用 `python experiments/bench_preemption.py` 等方式启动。

## import 兼容性变化

旧代码：

```python
from scheduler import Scheduler
from block_manager import BlockManager
from model import MiniGPT
```

现在改为：

```python
from minivllm.scheduler import Scheduler
from minivllm.block_manager import BlockManager
from minivllm.model import MiniGPT
```

`minivllm/` 内部模块使用相对导入，如 `from .scheduler import Scheduler`；`tests/` 和 `experiments/` 使用绝对包路径。

删除了未使用且失效的 `Scheduler.admit()`、`BlockManager.allocate()` 和 `BlockManager.can_allocate()`；现行 API 为 `Scheduler.schedule()` 和 `BlockManager.reserve_slots()`。

## 核验与限制

此次审计副本在 Python 3.13 / PyTorch 2.10.0+cpu 环境中，29 项测试通过，并确认 CPU Benchmark、Trace 及安装后从其他目录导入包成功。**这不代表 GPU 性能或异步在线 Serving 已经验证**。

原仓库中的大型 `proflie_*.json` 历史 Profiler 文件没有在补丁中删除，仅将其加入 `.gitignore`，而已被 Git 跟踪的文件不会自动被忽略。需要时使用 `git ls-files 'proflie_*.json'` 检查，再自行决定是否从 Git 索引中移除。README/文档中的旧实验绝对耗时来自先前 CPU 环境，请勿跨环境直接比较。
