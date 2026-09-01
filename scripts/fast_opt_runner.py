#!/usr/bin/env python3
"""Fail-closed, low-intervention GLM-5.3 optimization campaign runner.

The runner intentionally owns only one vLLM process at a time.  A frozen JSON
manifest supplies all candidates and thresholds; no candidate is invented from
logs during a run.  It is safe-by-default: ``--dry-run`` performs validation,
while ``--execute`` requires an exact identity match for the service currently
holding the configured port before sending it SIGTERM.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runtime" / "fast-opt-manifest.json"
DEFAULT_OUTPUT = Path("/mnt/nvme0/keys-vllm-glm53/runtime")


class AbortReview(RuntimeError):
    """An unsafe state that must stop the campaign before cleanup."""


@dataclasses.dataclass(frozen=True)
class ProcIdentity:
    pid: int
    starttime: str
    cmdline_sha256: str
    cwd: str
    env_sha256: str
    pgid: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def proc_path(pid: int, name: str) -> Path:
    return Path("/proc") / str(pid) / name


def read_proc_identity(pid: int) -> ProcIdentity:
    base = Path("/proc") / str(pid)
    if not base.exists():
        raise AbortReview(f"process {pid} disappeared")
    raw_cmd = (base / "cmdline").read_bytes()
    raw_env = (base / "environ").read_bytes()
    stat_fields = (base / "stat").read_text().split()
    # Linux /proc/<pid>/stat field 22 is process starttime; field 5 is pgrp.
    starttime = stat_fields[21]
    pgid = int(stat_fields[4])
    return ProcIdentity(
        pid=pid,
        starttime=starttime,
        cmdline_sha256=sha256_bytes(raw_cmd),
        cwd=os.readlink(base / "cwd"),
        env_sha256=sha256_bytes(raw_env),
        pgid=pgid,
    )


def proc_cmdline(pid: int) -> str:
    return (
        proc_path(pid, "cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode("utf-8", "replace")
        .strip()
    )


def proc_env(pid: int) -> dict[str, str]:
    values = proc_path(pid, "environ").read_bytes().split(b"\0")
    result: dict[str, str] = {}
    for value in values:
        if b"=" not in value:
            continue
        key, item = value.split(b"=", 1)
        result[key.decode("utf-8", "replace")] = item.decode("utf-8", "replace")
    return result


def port_pid(port: int) -> int | None:
    result = subprocess.run(
        ["ss", "-ltnp"], capture_output=True, text=True, check=False
    )
    pattern = re.compile(rf"0\.0\.0\.0:{port}\b|\*:{port}\b|\[::\]:{port}\b")
    for line in result.stdout.splitlines():
        if not pattern.search(line):
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc)


def metric_value(metrics: str, name: str) -> float | None:
    # vLLM labels vary by version.  The first value is sufficient for the
    # single-engine service used by this campaign.
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return None


def metric_sum(metrics: str, name: str, label: str | None = None) -> float | None:
    """Sum a labeled Prometheus gauge, optionally selecting one label value."""
    values: list[float] = []
    pattern = re.compile(rf"^{re.escape(name)}\{{([^}}]*)\}}\s+([0-9.eE+-]+)$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        labels, raw_value = match.groups()
        if label is not None and label not in labels:
            continue
        values.append(float(raw_value))
    return sum(values) if values else None


def metrics_snapshot(base: str) -> dict[str, float | None]:
    status, body = http_get(base.rstrip("/") + "/metrics", timeout=5)
    if status != 200:
        return {"prefix_queries": None, "prefix_hits": None}
    return {
        "prefix_queries": metric_value(body, "vllm:prefix_cache_queries_total"),
        "prefix_hits": metric_value(body, "vllm:prefix_cache_hits_total"),
        "waiting": metric_value(body, "vllm:num_requests_waiting"),
        "deferred": metric_sum(
            body, "vllm:num_requests_waiting_by_reason", 'reason="deferred"'
        ),
    }


def parse_bench(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = (
        "aggregate_decode_tps_from_last_ttft",
        "median_per_stream_decode_tps",
        "min_per_stream_decode_tps",
        "wall_s",
        "results",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"benchmark result missing {missing}: {path}")
    if not data["results"] or any(
        item.get("completion_tokens") is None for item in data["results"]
    ):
        raise ValueError(f"incomplete benchmark result: {path}")
    return data


def mad(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def bootstrap_ci(
    deltas: list[float], seed: int = 20260901, rounds: int = 4000
) -> tuple[float, float]:
    # Deterministic, dependency-free percentile bootstrap.  The benchmark
    # samples are paired by prompt seed and candidate order.
    if len(deltas) < 3:
        return (float("nan"), float("nan"))
    state = seed & 0xFFFFFFFF
    samples: list[float] = []
    for _ in range(rounds):
        picked = []
        for _ in deltas:
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            picked.append(deltas[state % len(deltas)])
        samples.append(statistics.mean(picked))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def candidate_env(
    base_env: dict[str, str], candidate: dict[str, Any]
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(base_env)
    for key, value in candidate.get("env", {}).items():
        env[key] = str(value)
    return env


class Campaign:
    def __init__(self, manifest_path: Path, execute: bool) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.execute = execute
        self.started = time.time()
        self.run_dir = Path(self.manifest["output_dir"]).resolve()
        self.status_path = self.run_dir / "STATUS.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.formal_identity: ProcIdentity | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self.child_identity: ProcIdentity | None = None

    def event(self, kind: str, **fields: Any) -> None:
        record = {"time": time.time(), "kind": kind, **fields}
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        atomic_json(
            self.status_path,
            {
                "manifest_sha256": sha256_file(self.manifest_path),
                "last_event": record,
                "elapsed_s": time.time() - self.started,
            },
        )

    def validate(self) -> None:
        required = ("formal", "candidates", "output_dir", "baseline")
        missing = [key for key in required if key not in self.manifest]
        if missing:
            raise ValueError(f"manifest missing {missing}")
        formal = self.manifest["formal"]
        if int(formal["port"]) != 30002:
            raise ValueError("this runner is restricted to port 30002")
        if formal["gpus"] != [0, 1, 2, 3]:
            raise ValueError("this runner is restricted to GPU 0-3")
        if not self.manifest["candidates"]:
            raise ValueError("manifest has no candidates")
        for candidate in self.manifest["candidates"]:
            if "id" not in candidate or "env" not in candidate:
                raise ValueError(f"candidate missing id/env: {candidate}")
            if candidate.get("port", 30002) != 30002:
                raise ValueError("candidate port must be 30002")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.run_dir / "manifest.snapshot.json", self.manifest)
        (self.run_dir / "manifest.sha256").write_text(
            sha256_file(self.manifest_path) + "\n"
        )
        self.event("precheck_ok", candidate_count=len(self.manifest["candidates"]))

    def verify_formal(self) -> ProcIdentity:
        formal = self.manifest["formal"]
        pid = port_pid(int(formal["port"]))
        if pid is None:
            raise AbortReview("formal service is not listening on port 30002")
        identity = read_proc_identity(pid)
        cmd = proc_cmdline(pid)
        env = proc_env(pid)
        expected = formal["expected"]
        checks = {
            "model": expected["model"] in cmd,
            "port": f"--port {formal['port']}" in cmd,
            "serve": "vllm.entrypoints.cli.main serve" in cmd,
            "cwd": identity.cwd == expected["cwd"],
            "cuda": env.get("CUDA_VISIBLE_DEVICES") == expected["cuda_visible_devices"],
        }
        if not all(checks.values()):
            raise AbortReview(
                f"formal identity mismatch pid={pid}: {checks}, cmd={cmd}"
            )
        self.formal_identity = identity
        self.event("formal_verified", pid=pid, identity=dataclasses.asdict(identity))
        return identity

    def verify_identity(self, identity: ProcIdentity) -> None:
        current = read_proc_identity(identity.pid)
        if current != identity:
            raise AbortReview(
                f"process identity changed pid={identity.pid}; refusing cleanup"
            )

    @staticmethod
    def wait_exit(pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not Path("/proc").joinpath(str(pid)).exists():
                return True
            time.sleep(1)
        return False

    @staticmethod
    def group_has_members(pgid: int) -> bool:
        result = subprocess.run(
            ["ps", "-eo", "pgid="], capture_output=True, text=True, check=False
        )
        return any(line.strip() == str(pgid) for line in result.stdout.splitlines())

    def wait_group_exit(self, pgid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.group_has_members(pgid):
                return True
            time.sleep(1)
        return False

    def stop_formal(self) -> None:
        if self.formal_identity is None:
            raise AbortReview("formal identity was not captured")
        self.verify_identity(self.formal_identity)
        os.killpg(self.formal_identity.pgid, signal.SIGTERM)
        if not self.wait_exit(self.formal_identity.pid, 90) or not self.wait_group_exit(
            self.formal_identity.pgid, 30
        ):
            raise AbortReview("formal process did not exit after SIGTERM")
        self.event("formal_stopped", pid=self.formal_identity.pid)

    def stop_child(self) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        if self.child_identity is None:
            raise AbortReview("candidate identity was not captured")
        self.verify_identity(self.child_identity)
        os.killpg(self.child_identity.pgid, signal.SIGTERM)
        if not self.wait_exit(self.child_identity.pid, 90) or not self.wait_group_exit(
            self.child_identity.pgid, 30
        ):
            raise AbortReview("candidate did not exit after SIGTERM")
        self.event("candidate_stopped", pid=self.child_identity.pid)

    def start_candidate(self, candidate: dict[str, Any]) -> Path:
        candidate_dir = self.run_dir / candidate["id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        log_path = candidate_dir / "server.log"
        env = candidate_env(self.manifest["base_env"], candidate)
        env["PYTHONUNBUFFERED"] = "1"
        command = [str(ROOT / "scripts" / "serve_glm53_sm80.sh")]
        log = log_path.open("wb")
        self.child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.child_identity = read_proc_identity(self.child.pid)
        if port_pid(int(candidate.get("port", 30002))) not in (None, self.child.pid):
            raise AbortReview("candidate port was already occupied after spawn")
        self.event(
            "candidate_started",
            candidate=candidate["id"],
            pid=self.child.pid,
            identity=dataclasses.asdict(self.child_identity),
        )
        return candidate_dir

    def wait_healthy(self, candidate: dict[str, Any]) -> bool:
        base = f"http://127.0.0.1:{candidate.get('port', 30002)}"
        deadline = time.monotonic() + float(candidate.get("health_timeout_s", 420))
        while time.monotonic() < deadline:
            if self.child is None:
                return False
            if self.child.poll() is not None:
                return False
            status, _ = http_get(base + "/health", timeout=3)
            if status == 200:
                model_status, body = http_get(base + "/v1/models", timeout=5)
                if (
                    model_status == 200
                    and self.manifest["formal"]["expected"]["served_model"] in body
                ):
                    # The launcher starts as bash and execs Python. Refresh
                    # identity after exec so cleanup checks the long-lived
                    # process instead of its transient shell command.
                    if self.child is not None:
                        self.child_identity = read_proc_identity(self.child.pid)
                    self.event("candidate_healthy", candidate=candidate["id"])
                    return True
            time.sleep(2)
        return False

    def run_bench(self, candidate_dir: Path, test: dict[str, Any]) -> dict[str, Any]:
        output = candidate_dir / test["name"]
        command = [
            self.manifest["base_env"]["PYTHON_BIN"],
            str(ROOT / "scripts" / "bench_glm53_concurrent.py"),
            "--base",
            f"http://127.0.0.1:{test.get('port', 30002)}",
            "--model",
            self.manifest["formal"]["expected"]["served_model"],
            "--model-path",
            self.manifest["base_env"]["MODEL"],
            "--chat-template",
            str(ROOT / "chat_templates" / "glm53-enable-thinking-switch.jinja"),
            "--concurrency",
            str(test["concurrency"]),
            "--prompt-tokens",
            str(test["prompt_tokens"]),
            "--max-tokens",
            str(test["max_tokens"]),
            "--prompt-seed",
            str(test.get("prompt_seed", 0)),
            "--out",
            str(output),
        ]
        timeout = float(test.get("timeout_s", 7200))
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=candidate_env(self.manifest["base_env"], {}),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        (candidate_dir / f"{test['name']}.stdout").write_text(completed.stdout)
        (candidate_dir / f"{test['name']}.stderr").write_text(completed.stderr)
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(
                f"benchmark failed rc={completed.returncode}: {test['name']}"
            )
        data = parse_bench(output)
        data["elapsed_runner_s"] = time.monotonic() - started
        return data

    def scan_log(self, candidate_dir: Path) -> list[str]:
        logs = list(candidate_dir.glob("*.log"))
        if not logs:
            return []
        body = "\n".join(path.read_text(errors="replace") for path in logs)
        patterns = (
            r"OutOfMemoryError",
            r"CUDA out of memory",
            r"engine dead",
            r"Traceback \(most recent call last\):",
            r"CUBLAS_STATUS",
            r"illegal memory access",
        )
        return [pattern for pattern in patterns if re.search(pattern, body, re.I)]

    def evaluate_tests(
        self,
        candidate: dict[str, Any],
        candidate_dir: Path,
        tests: list[dict[str, Any]],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        l3_checked = False
        for test in tests:
            before = metrics_snapshot(f"http://127.0.0.1:{test.get('port', 30002)}")
            result = self.run_bench(candidate_dir, test)
            after = metrics_snapshot(f"http://127.0.0.1:{test.get('port', 30002)}")
            result["test_name"] = test["name"]
            result["prefix_queries_delta"] = (
                after["prefix_queries"] - before["prefix_queries"]
                if after["prefix_queries"] is not None
                and before["prefix_queries"] is not None
                else None
            )
            result["prefix_hits_delta"] = (
                after["prefix_hits"] - before["prefix_hits"]
                if after["prefix_hits"] is not None
                and before["prefix_hits"] is not None
                else None
            )
            if (
                test.get("cold", True)
                and result["prefix_hits_delta"] is not None
                and result["prefix_hits_delta"] != 0
            ):
                raise RuntimeError(
                    f"cold test reused {result['prefix_hits_delta']} prefix tokens"
                )
            if after.get("waiting") not in (None, 0.0) or after.get("deferred") not in (
                None,
                0.0,
            ):
                raise RuntimeError(
                    f"requests remained queued: waiting={after.get('waiting')} "
                    f"deferred={after.get('deferred')}"
                )
            results.append(result)
            self.event("test_done", candidate=candidate["id"], test=test["name"])
            errors = self.scan_log(candidate_dir)
            if errors:
                raise RuntimeError(f"log safety errors after {test['name']}: {errors}")

            # Fail fast after the complete L3 triplet.  Expensive context
            # probes are only run for candidates that are already promising.
            if test["name"].startswith("l3-"):
                l3_count = sum(item["test_name"].startswith("l3-") for item in results)
                if l3_count == 3 and not l3_checked:
                    l3_checked = True
                    preliminary = self._l3_decision(results, baseline)
                    if not preliminary["promoted"]:
                        atomic_json(candidate_dir / "decision.json", preliminary)
                        return preliminary

        decision = self._l3_decision(results, baseline)
        atomic_json(candidate_dir / "decision.json", decision)
        return decision

    @staticmethod
    def _l3_decision(
        results: list[dict[str, Any]], baseline: dict[str, Any]
    ) -> dict[str, Any]:
        speed_values = [
            float(item["aggregate_decode_tps_from_last_ttft"])
            for item in results
            if item["test_name"].startswith("l3-")
        ]
        baseline_values = [
            float(item["aggregate_decode_tps_from_last_ttft"])
            for item in baseline["l3"]
        ]
        deltas = [
            candidate_value / baseline_value - 1
            for candidate_value, baseline_value in zip(speed_values, baseline_values)
            if baseline_value > 0
        ]
        ci = bootstrap_ci(deltas)
        median_delta = statistics.median(deltas) if deltas else float("nan")
        decision = {
            "tests": results,
            "l3_relative_deltas": deltas,
            "l3_median_delta": median_delta,
            "l3_bootstrap_ci95": ci,
            "mad": mad(speed_values) if speed_values else float("nan"),
            "promoted": bool(
                deltas
                and median_delta >= 0.05
                and ci[0] >= 0.03
                and all(
                    item["min_per_stream_decode_tps"]
                    >= baseline["min_per_stream_decode_tps"] * 0.97
                    for item in results
                    if item["test_name"].startswith("l3-")
                )
            ),
        }
        return decision

    def restore_formal(self) -> None:
        formal = self.manifest["formal"]
        env = dict(self.manifest["base_env"])
        env.update(formal["env"])
        log_path = self.run_dir / "formal-restored.log"
        log = log_path.open("wb")
        self.child = subprocess.Popen(
            [str(ROOT / "scripts" / "serve_glm53_sm80.sh")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.child_identity = read_proc_identity(self.child.pid)
        if not self.wait_healthy(formal):
            raise AbortReview("formal service failed to restore")
        status, body = http_get("http://127.0.0.1:30002/v1/models", timeout=5)
        expected = formal["expected"]
        if status != 200 or expected["served_model"] not in body:
            raise AbortReview("restored model identity mismatch")
        self.event("formal_restored", pid=self.child.pid)

    def run(self) -> int:
        self.validate()
        if not self.execute:
            self.event("dry_run_complete")
            print(f"DRY-RUN OK: {len(self.manifest['candidates'])} candidates")
            return 0

        self.verify_formal()
        self.stop_formal()
        baseline = self.manifest["baseline"]
        try:
            for candidate in self.manifest["candidates"]:
                candidate_dir = self.start_candidate(candidate)
                if not self.wait_healthy(candidate):
                    self.event(
                        "candidate_rejected", candidate=candidate["id"], reason="health"
                    )
                    self.stop_child()
                    continue
                try:
                    decision = self.evaluate_tests(
                        candidate, candidate_dir, candidate["tests"], baseline
                    )
                    self.event(
                        "candidate_decision",
                        candidate=candidate["id"],
                        promoted=decision["promoted"],
                    )
                except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                    atomic_json(candidate_dir / "decision.json", {"failed": str(exc)})
                    self.event(
                        "candidate_rejected", candidate=candidate["id"], reason=str(exc)
                    )
                finally:
                    self.stop_child()
            self.restore_formal()
            (self.run_dir / "DONE").write_text("campaign completed\n")
            self.event("campaign_done")
            return 0
        except AbortReview as exc:
            self.event("ABORT_REVIEW", reason=str(exc))
            try:
                self.stop_child()
            finally:
                # If restore itself fails, leave the marker and never pretend
                # the formal service is healthy.
                try:
                    if port_pid(30002) is None:
                        self.restore_formal()
                except Exception as restore_exc:  # noqa: BLE001
                    (self.run_dir / "ABORT_REVIEW").write_text(
                        f"{exc}\nrestore failed: {restore_exc}\n"
                    )
                    return 2
            (self.run_dir / "ABORT_REVIEW").write_text(str(exc) + "\n")
            return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        return Campaign(args.manifest, args.execute).run()
    except (AbortReview, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ABORT_REVIEW: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
