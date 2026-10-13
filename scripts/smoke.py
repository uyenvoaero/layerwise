"""CPU smoke run: copy the committed fixture under a temporary DATA_ROOT, ingest it and train
outer fold 0 with the `local` profile.

Usage:
    uv run python scripts/smoke.py        # or: make smoke

The temporary DATA_ROOT is printed so later commands (baseline, report, predict, benchmark) can
reuse it: set DATA_ROOT to the printed path.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "aalto"
ENTRY = "from layerwise.cli import app; app()"


def run_cli(env: dict[str, str], *args: str) -> None:
    command = [sys.executable, "-c", ENTRY, *args]
    print(f"$ layerwise {' '.join(args)}", flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def main() -> int:
    start = time.perf_counter()
    data_root = Path(tempfile.mkdtemp(prefix="layerwise_smoke_"))
    shutil.copytree(FIXTURE, data_root / "aalto")
    env = {**os.environ, "DATA_ROOT": str(data_root)}
    run_cli(env, "ingest", "--profile", "fixture")
    run_cli(env, "train", "--profile", "local", "--fold", "0")
    minutes = (time.perf_counter() - start) / 60
    print(f"\nsmoke run finished in {minutes:.1f} minutes")
    print(f"DATA_ROOT={data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
