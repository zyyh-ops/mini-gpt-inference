# 早期学习实验与当前 Runtime 的边界

仓库保留根目录早期教学实验，目的是展示从连续 KV、Paged KV reference 到完整 Runtime 的演化过程。**这些脚本不属于当前维护的 Serving Runtime 公开 API，也不计入统一 CPU correctness baseline。**

| 学习专题 | 现存脚本 |
|---|---|
| KV Cache 与显存 | `kv_cache_correnctness.py`（历史文件名拼写如此）、`kv_cache_memory.py` |
| 多请求与 Batch | `continuous_batching_simulator.py`、`multi_request_kv_reference.py` |
| Paged KV / Attention | `paged_kv_storage_reference.py`、`paged_attention_reference.py`、`paged_attention_v1_reference.py` |
| Prefill / Decode 性能 | `prefill_decode_workload.py`、`prefill_decode_benchmark.py`、`prefill_decode_profile.py` |

**为什么没有直接移动？** 部分原型脚本直接通过文件名相互导入，例如 `paged_attention_reference.py` 使用 `from paged_kv_storage_reference import ...`。贸然移入子目录会破坏原来从仓库根目录执行的路径。等到单独整理 reference 教程时再进行原子迁移，并增加 smoke test。

当前主线的可复现入口是：

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
python -m scripts.release_smoke
python -m experiments.bench_preemption --help
```

不把早期脚本中的独立速度数据或未复验 GPU 结果当作当前 runtime 的性能证明。
