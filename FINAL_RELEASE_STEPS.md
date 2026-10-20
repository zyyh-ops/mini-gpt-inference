# Mini-GPT Inference Runtime — Apache 2.0 发布收尾

## 包含的改动

- 新增标准 Apache License, Version 2.0 全文 `LICENSE`。
- README 增加 Apache-2.0 声明与 LICENSE 链接。
- 更新 `docs/release_checklist.md` 记录发布检查状态。
- 不修改 Scheduler、PagedAttention、BlockManager 或 Runtime 计算逻辑。

## PowerShell：仓库根目录执行

```powershell
git status --short
git apply --check .\apache2_finalization.patch
git apply .\apache2_finalization.patch

# 仅停止跟踪，保留本地文件
git rm --cached -- proflie_cached_decode_t256.json proflie_naive_t256.json

# 确认本地文件存在，并确认已有的 ignore 规则生效
Test-Path .\proflie_cached_decode_t256.json
Test-Path .\proflie_naive_t256.json
git check-ignore proflie_cached_decode_t256.json proflie_naive_t256.json

python -m unittest discover -s tests -p 'test_*.py' -v
python -m scripts.release_smoke
python -m scripts.tracked_file_audit

git diff --check
git status --short
```

## 检查完成后

```powershell
git add README.md LICENSE docs/release_checklist.md
git diff --cached --stat
git commit -m "chore: finalize Apache-2.0 licensing and release hygiene"
git push
```

注意：`git rm --cached` 不删除本地文件，但并不重写历史提交。历史中已有的 Profiler 数据仍存在于历史版本；作者已确认历史没有不宜公开的信息。若他人已克隆仓库，之后执行 pull 可能会在其工作区删除这些已取消跟踪的文件，若需保留应单独备份。

首次发布后确认 GitHub Actions 工作流在 GitHub 上实际 PASS。GPU correctness、动态到达工作负载、正式 GPU Serving Benchmark 仍未完成，应在 README 中保留限制说明。
