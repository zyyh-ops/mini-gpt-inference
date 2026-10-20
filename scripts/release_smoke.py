"""One-command CPU release smoke: tests, policy A/B, and Runtime telemetry.

Run after editable installation from the repository root:
    python -m scripts.release_smoke

This verifies deterministic *behavior*, never absolute wall time or GPU speed.
Temporary JSON outputs are discarded after successful validation.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _execute(*args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): python -m {' '.join(args)}\n"
            f"{result.stdout[-10000:]}"
        )
    print(f"PASS: python -m {' '.join(args[:1])}")


def verify_benchmark(report: dict, expected: dict[str, tuple[int, int, int]]) -> None:
    """Verify strategy behavior and exact token equality, not noisy performance."""
    grouped: dict[str, dict] = {}
    for run in report["runs"]:
        if run["policy"] in grouped:
            raise AssertionError(f"duplicate policy run: {run['policy']}")
        grouped[run["policy"]] = run
    if set(grouped) != set(expected):
        raise AssertionError(f"unexpected policies: {set(grouped)}")
    first = grouped["conservative"]["per_request_output"]
    assert first == grouped["recompute"]["per_request_output"], "output mismatch"
    for policy, (iterations, preemptions, repeated_tokens) in expected.items():
        run = grouped[policy]
        actual = (run["iterations"], run["preemptions"], run["repeated_kv_tokens"])
        assert actual == (iterations, preemptions, repeated_tokens), (policy, actual)
        assert run["output_tokens"] == sum(len(xs) for xs in run["per_request_output"].values())


def verify_records(report: dict) -> None:
    """Verify that native metrics correspond to the exported iteration records."""
    summary = report["summary"]
    records = report["records"]
    assert len(records) == summary["iterations"] == 27
    assert [r["iteration"] for r in records] == list(range(1, 28))
    assert summary["preemptions"] == sum(r["preemptions"] for r in records) == 3
    assert summary["repeated_kv_tokens"] == sum(r["repeated_kv_tokens"] for r in records) == 44
    for key in ("prefill_tokens", "decode_tokens", "scheduled_tokens", "output_tokens"):
        assert summary[key] == sum(r[key] for r in records), key
    assert summary["scheduled_tokens"] == summary["prefill_tokens"] + summary["decode_tokens"]
    assert all(r["used_blocks"] + r["free_blocks"] == 2 for r in records)
    assert len(report["metadata"]["request_outputs"]) == 4


def main() -> None:
    print("[1/4] All unit tests")
    _execute("unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q")
    with tempfile.TemporaryDirectory(prefix="minivllm-release-") as temp:
        path = Path(temp)
        baseline = path / "baseline.json"
        mixed = path / "mixed.json"
        telemetry = path / "telemetry.json"

        print("[2/4] Baseline / 2 KV Blocks")
        _execute("experiments.bench_preemption", "--device", "cpu", "--workload", "baseline",
                 "--num-blocks", "2", "--repeats", "1", "--warmups", "0",
                 "--output-json", str(baseline))
        verify_benchmark(json.loads(baseline.read_text(encoding="utf-8")), {
            "conservative": (27, 0, 0), "recompute": (27, 3, 44),
        })

        print("[3/4] Mixed-Short / 3 KV Blocks")
        _execute("experiments.bench_preemption", "--device", "cpu", "--workload", "mixed_short",
                 "--num-blocks", "3", "--repeats", "1", "--warmups", "0",
                 "--output-json", str(mixed))
        verify_benchmark(json.loads(mixed.read_text(encoding="utf-8")), {
            "conservative": (34, 0, 0), "recompute": (32, 0, 0),
        })

        print("[4/4] Native telemetry / 2 KV Blocks")
        _execute("experiments.record_runtime", "--policy", "recompute", "--blocks", "2",
                 "--output-json", str(telemetry))
        verify_records(json.loads(telemetry.read_text(encoding="utf-8")))

    print("RELEASE SMOKE PASS — CPU correctness, two workloads, telemetry accounted for")


if __name__ == "__main__":
    main()
