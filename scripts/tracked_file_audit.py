"""Read-only check of Git-tracked file names and sizes before public release.

Run inside your *actual* Git working tree: python -m scripts.tracked_file_audit
Never removes or modifies tracked files; Git history is not inspected here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE = 2 * 1024 * 1024
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", "credentials.json", "service-account.json"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".safetensors", ".ncu-rep", ".nsys-rep"}


def main() -> int:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        print("Not a Git checkout: run this command in your real local repository.")
        return 2
    risky = []
    tracked = [p.decode("utf-8", errors="replace") for p in result.stdout.split(b"\0") if p]
    if not tracked:
        print("No tracked files in this checkout; cannot certify publication hygiene.")
        return 2
    for relative in tracked:
        file = ROOT / relative
        if not file.is_file():
            continue
        reasons = []
        if file.stat().st_size > MAX_SIZE:
            reasons.append(f">2 MiB ({file.stat().st_size} bytes)")
        if file.name.lower() in FORBIDDEN_NAMES or any(file.name.endswith(s) for s in FORBIDDEN_SUFFIXES):
            reasons.append("potential secret/model/profiler artifact")
        if "/__pycache__/" in f"/{relative}" or relative.endswith(".pyc"):
            reasons.append("generated Python cache")
        if reasons:
            risky.append((relative, ", ".join(reasons)))
    print(f"Inspected {len(tracked)} tracked paths")
    if risky:
        print("REVIEW THESE TRACKED FILES (nothing was removed):")
        for path, reason in risky:
            print(f"  {path}: {reason}")
        print("Use git rm --cached <path> only after checking what it contains.")
        return 1
    print("PASS: no obvious large or sensitive artifacts by filename/size")
    print("Still review git log/history and scan for secrets separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
