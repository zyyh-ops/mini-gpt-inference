# 公开发布前检查清单

> 本清单依据 2026-10-20 最新 ZIP 的静态审计与 CPU 运行结果。将修复项分层，不要求一次性重写全部历史实验代码。

## P0 · 发布前建议完成

- [x] 将 `test_block_manager.py`、`test_scheduler_contention.py`、`test_paged_attention_padding.py` 合入仓库；26 项 CPU 测试已通过。
- [x] 修正包初始化文件并规范相对导入；统一改用 `python -m ...` 运行测试与实验。
- [x] 删除失效的 `Scheduler.admit()`；实际调度仅使用 `schedule()`。
- [x] 精简 `requirements.txt`，将 PyTorch 作为核心依赖，并新增 `pyproject.toml`。
- [ ] 增加 `LICENSE`（根据自己意愿决定授权方式）与 README/文档。
- [ ] 确认提交历史没有 Token/密钥、大型缓存、敏感数据；`git status` 和 `git ls-files` 单独核查。

## P1 · 工程质量

- [x] 删除未使用的 `BlockManager.can_allocate()/allocate()` 旧预分配接口。
- [ ] 统一命名和类型（例如 `kv_cache_correnctness.py`、`proflie_*.json`、`SequenceState.last_token` 可标注 Optional）。
- [x] 将 `IterationRecord` append 至 `self.records`，支持每轮抢占、重复 KV 位置、KV Block 快照及 JSON 导出；文档明确 GPU 异步计时口径。
- [ ] 将根目录 6MB～7MB 的原始 Profiler Trace 移入非默认发布目录；保留轻量摘要/采样结果。`.gitignore` 不会自动让已被 Git 跟踪的文件消失。
- [x] 测试/实验改用 `python -m ...` 入口；未引入额外命令行框架。
- [ ] 增加输入边界检测（空请求、重复 seq_id、非法预算、不同 Request 状态）；将断言与异常分类整理。

## P2 · 性能 / 正确性扩展

- [ ] 在 GPU 上重新跑 PagedAttention、Batched Decode、Preemption Test；不能把 CPU PASS 视为 CUDA 已验证。
- [ ] CPU 计时路径中移出频繁 `check_invariants()` 与调试打印，分离 Correctness 模式和 Performance 模式。
- [ ] 跟踪调度循环的 Kernel 计算、host metadata 建立、CPU/GPU 同步、重计算额外 Forward 时间。
- [ ] 扩展动态到达与重尾长短混合负载；分析优先级、饥饿与公平性。
- [ ] 明确实际的 `paged_attention_batch_online_v1` 仍按 Python Logical Block 循环，不宣传为融合 GPU Kernel；如后续实现 Triton/CUDA Kernel，单列对照实验。

## 本轮已验证的基线

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
# Phase 2 Step 2 审计副本：Ran 33 tests ... OK（请在本地复验）
```

审计环境 Python 3.13.5 / torch 2.10.0+cpu；用户之前的实验 JSON 环境 torch 2.12.1+cpu。两者不同；不要跨环境比较绝对耗时。
