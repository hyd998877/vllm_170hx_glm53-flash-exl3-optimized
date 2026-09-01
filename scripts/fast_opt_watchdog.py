#!/usr/bin/env python3
"""Supervise an optimization runner and restore formal service after death.

This process is intentionally separate from the runner.  It only restores the
frozen formal command when the runner exits unexpectedly and port 30002 is
vacant; any unknown listener or residual GPU process causes ABORT_REVIEW.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fast_opt_runner as runner  # noqa: E402


def healthy() -> bool:
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:30002/health", timeout=3
        ) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def restore(
    manifest_path: Path, out_dir: Path, protected_snapshot: dict[int, set[int]]
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    campaign = runner.Campaign(manifest_path, execute=False)
    campaign.run_dir = out_dir
    campaign.protected_gpu_snapshot = protected_snapshot
    if runner.port_bindings(30002):
        (out_dir / "ABORT_REVIEW").write_text("watchdog found unknown listener\n")
        return 2
    try:
        campaign.verify_target_gpus_empty()
        campaign.restore_formal()
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ABORT_REVIEW").write_text(f"watchdog restore failed: {exc}\n")
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("WATCHDOG DRY-RUN OK")
        return 0
    manifest = json.loads(args.manifest.read_text())
    out_dir = Path(manifest["output_dir"]).resolve()
    initial_gpus = runner.gpu_compute_processes()
    protected_snapshot = {
        gpu: set(initial_gpus.get(gpu, set())) for gpu in range(4, 8)
    }
    cmd = [
        manifest["base_env"]["PYTHON_BIN"],
        str(ROOT / "scripts" / "fast_opt_runner.py"),
        "--manifest",
        str(args.manifest),
        "--execute",
    ]
    child = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
    rc = child.wait()
    if rc == 0 and healthy():
        return 0
    # A normal runner exit should have restored formal.  On abnormal death,
    # only proceed when there is no listener and no protected GPU drift.
    if runner.port_bindings(30002):
        (out_dir / "ABORT_REVIEW").write_text(
            f"watchdog runner rc={rc}; unknown listener remains\n"
        )
        return 2
    return restore(args.manifest, out_dir, protected_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
