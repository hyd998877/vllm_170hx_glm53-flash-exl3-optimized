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
import base64
import dataclasses
import fcntl
import hashlib
import json
import math
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


class RunnerSignal(BaseException):
    """Convert catchable termination signals into the recovery path."""


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        raise RunnerSignal(f"runner received signal {signum}")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGHUP, handle)


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
    bindings = port_bindings(port)
    return bindings[0][1] if bindings else None


def port_bindings(port: int) -> list[tuple[str, int]]:
    """Return (listen address, pid) pairs reported by ss for a TCP port."""
    result = subprocess.run(
        ["ss", "-ltnp"], capture_output=True, text=True, check=False
    )
    pattern = re.compile(rf"(?P<address>0\.0\.0\.0|127\.0\.0\.1|\*|\[::\]):{port}\b")
    bindings: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        address_match = pattern.search(line)
        if not address_match:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            bindings.append((address_match.group("address"), int(match.group(1))))
    return bindings


def port_socket_fingerprint(port: int) -> list[tuple[str, int, int]]:
    """Return listen address, owner PID and kernel socket inode."""
    result = subprocess.run(
        ["ss", "-ltnpe"], capture_output=True, text=True, check=False
    )
    pattern = re.compile(
        rf"(?P<address>0\.0\.0\.0|127\.0\.0\.1|\*|\[::\]):{port}\b"
    )
    fingerprints: list[tuple[str, int, int]] = []
    for line in result.stdout.splitlines():
        address_match = pattern.search(line)
        pid_match = re.search(r"pid=(\d+)", line)
        inode_match = re.search(r"ino:(\d+)", line)
        if address_match and pid_match and inode_match:
            fingerprints.append(
                (
                    address_match.group("address"),
                    int(pid_match.group(1)),
                    int(inode_match.group(1)),
                )
            )
    return fingerprints


def gpu_compute_processes() -> dict[int, set[int]]:
    """Map physical GPU index to compute PIDs, failing closed on query errors."""
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    app_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_result.returncode != 0 or app_result.returncode != 0:
        raise AbortReview(
            "nvidia-smi failed while checking GPU ownership: "
            f"gpu_rc={gpu_result.returncode} app_rc={app_result.returncode}"
        )
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            uuid_to_index[fields[1]] = int(fields[0])
        except ValueError as exc:
            raise AbortReview(f"invalid nvidia-smi GPU row: {line!r}") from exc
    processes: dict[int, set[int]] = {}
    for line in app_result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        gpu_index = uuid_to_index.get(fields[1])
        if gpu_index is None:
            raise AbortReview(f"unknown GPU UUID in nvidia-smi process row: {line!r}")
        processes.setdefault(gpu_index, set()).add(int(fields[0]))
    return processes


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
        return {
            "prefix_queries": None,
            "prefix_hits": None,
            "running": None,
            "waiting": None,
            "deferred": None,
        }
    return {
        "prefix_queries": metric_value(body, "vllm:prefix_cache_queries_total"),
        "prefix_hits": metric_value(body, "vllm:prefix_cache_hits_total"),
        "running": metric_value(body, "vllm:num_requests_running"),
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
        "decode_window_s",
        "ttft_spread_s",
        "median_ttft_s",
        "max_ttft_s",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"benchmark result missing {missing}: {path}")
    if not data["results"]:
        raise ValueError(f"incomplete benchmark result: {path}")
    numeric = (
        "prompt_tokens",
        "completion_tokens",
        "ttft_s",
        "decode_s",
        "decode_tps",
    )
    for item in data["results"]:
        if any(item.get(key) is None for key in numeric):
            raise ValueError(f"incomplete stream metrics: {path}")
        if any(not isinstance(item[key], (int, float)) or not math.isfinite(item[key])
               for key in numeric):
            raise ValueError(f"non-finite stream metrics: {path}")
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
        self.child_role: str | None = None
        self.protected_gpu_snapshot: dict[int, set[int]] | None = None
        self.lock_file = None
        self.socket_fingerprint: list[tuple[str, int, int]] | None = None
        self.last_event: dict[str, Any] | None = None
        self.last_heartbeat = 0.0

    def acquire_lock(self) -> None:
        lock_path = self.run_dir.parent / ".fast-opt-runner.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = lock_path.open("a+")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AbortReview(f"another optimization runner owns {lock_path}") from exc

    def event(self, kind: str, **fields: Any) -> None:
        record = {"time": time.time(), "kind": kind, **fields}
        self.last_event = record
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.heartbeat(force=True)

    def heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_heartbeat < 10:
            return
        max_campaign_s = float(self.manifest.get("max_campaign_s", 21600))
        if now - self.started > max_campaign_s:
            raise AbortReview(f"campaign exceeded hard limit {max_campaign_s}s")
        atomic_json(
            self.status_path,
            {
                "manifest_sha256": sha256_file(self.manifest_path),
                "last_event": self.last_event,
                "heartbeat_time": now,
                "elapsed_s": now - self.started,
            },
        )
        self.last_heartbeat = now

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
        if self.manifest.get("base_env", {}).get("HOST") != "127.0.0.1":
            raise ValueError("candidate base HOST must be 127.0.0.1")
        if formal.get("env", {}).get("HOST") != "0.0.0.0":
            raise ValueError("formal restore HOST must be 0.0.0.0")
        for candidate in self.manifest["candidates"]:
            if "id" not in candidate or "env" not in candidate:
                raise ValueError(f"candidate missing id/env: {candidate}")
            if candidate.get("port", 30002) != 30002:
                raise ValueError("candidate port must be 30002")
            if candidate_env(self.manifest["base_env"], candidate).get("HOST") != (
                "127.0.0.1"
            ):
                raise ValueError(f"candidate must use loopback: {candidate['id']}")
        if self.execute and self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise AbortReview(
                "execute requires a fresh output directory, found existing data: "
                f"{self.run_dir}"
            )
        if self.execute:
            self.run_dir.mkdir(parents=True, exist_ok=False)
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
            "cmdline_sha256": identity.cmdline_sha256 == expected["cmdline_sha256"],
            "serve_script_sha256": sha256_file(
                ROOT / "scripts" / "serve_glm53_sm80.sh"
            )
            == expected["serve_script_sha256"],
            "critical_env": all(
                env.get(key) == value for key, value in expected["env"].items()
            ),
        }
        if not all(checks.values()):
            raise AbortReview(
                f"formal identity mismatch pid={pid}: {checks}, cmd={cmd}"
            )
        bindings = port_bindings(int(formal["port"]))
        if bindings != [("0.0.0.0", pid)]:
            raise AbortReview(f"unexpected formal listen bindings: {bindings}")
        self.socket_fingerprint = port_socket_fingerprint(int(formal["port"]))
        if (
            len(self.socket_fingerprint) != 1
            or self.socket_fingerprint[0][:2] != ("0.0.0.0", pid)
        ):
            raise AbortReview(
                f"unexpected formal socket fingerprint: {self.socket_fingerprint}"
            )
        gpu_processes = gpu_compute_processes()
        self.protected_gpu_snapshot = {
            gpu: set(gpu_processes.get(gpu, set())) for gpu in range(4, 8)
        }
        self.verify_gpu_ownership(identity, gpu_processes)
        self.formal_identity = identity
        self.event("formal_verified", pid=pid, identity=dataclasses.asdict(identity))
        return identity

    def verify_identity(self, identity: ProcIdentity) -> None:
        current = read_proc_identity(identity.pid)
        if current != identity:
            raise AbortReview(
                f"process identity changed pid={identity.pid}; refusing cleanup"
            )

    def verify_protected_gpus(
        self, gpu_processes: dict[int, set[int]] | None = None
    ) -> None:
        if self.protected_gpu_snapshot is None:
            raise AbortReview("protected GPU snapshot was not captured")
        current = gpu_processes or gpu_compute_processes()
        current_protected = {gpu: set(current.get(gpu, set())) for gpu in range(4, 8)}
        if current_protected != self.protected_gpu_snapshot:
            raise AbortReview(
                "GPU4-7 process state changed during campaign: "
                f"expected={self.protected_gpu_snapshot} current={current_protected}"
            )

    def verify_gpu_ownership(
        self,
        identity: ProcIdentity,
        gpu_processes: dict[int, set[int]] | None = None,
    ) -> None:
        current = gpu_processes or gpu_compute_processes()
        for gpu in self.manifest["formal"]["gpus"]:
            pids = current.get(gpu, set())
            if not pids:
                raise AbortReview(f"GPU {gpu} has no compute process")
            for pid in pids:
                try:
                    pgid = int(proc_path(pid, "stat").read_text().split()[4])
                except (FileNotFoundError, IndexError, ValueError) as exc:
                    raise AbortReview(
                        f"cannot identify GPU {gpu} process {pid}"
                    ) from exc
                if pgid != identity.pgid:
                    raise AbortReview(
                        f"unknown process {pid} on GPU {gpu}: "
                        f"pgid={pgid}, expected={identity.pgid}"
                    )
        self.verify_protected_gpus(current)

    def verify_target_gpus_empty(self) -> None:
        current = gpu_compute_processes()
        occupied = {
            gpu: current.get(gpu, set())
            for gpu in self.manifest["formal"]["gpus"]
            if current.get(gpu)
        }
        if occupied:
            raise AbortReview(f"GPU0-3 still have compute processes: {occupied}")
        self.verify_protected_gpus(current)

    @staticmethod
    def wait_exit(pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status_path = Path("/proc").joinpath(str(pid), "stat")
            if not status_path.exists():
                return True
            try:
                state = status_path.read_text().split()[2]
            except (FileNotFoundError, IndexError):
                return True
            if state == "Z":
                return True
            time.sleep(1)
        return False

    @staticmethod
    def group_has_members(pgid: int) -> bool:
        result = subprocess.run(
            ["ps", "-eo", "pgid=,stat="], capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            fields = line.split()
            if (
                len(fields) >= 2
                and fields[0] == str(pgid)
                and not fields[1].startswith("Z")
            ):
                return True
        return False

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
        self.wait_formal_drained()
        os.killpg(self.formal_identity.pgid, signal.SIGTERM)
        if not self.wait_exit(self.formal_identity.pid, 90) or not self.wait_group_exit(
            self.formal_identity.pgid, 30
        ):
            raise AbortReview("formal process did not exit after SIGTERM")
        self.verify_target_gpus_empty()
        self.event("formal_stopped", pid=self.formal_identity.pid)

    def wait_formal_drained(
        self, stable_s: float = 5.0, timeout: float = 120.0
    ) -> None:
        """Require a quiet maintenance window before stopping formal traffic."""
        deadline = time.monotonic() + timeout
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            snapshot = metrics_snapshot("http://127.0.0.1:30002")
            if any(
                snapshot[key] is None for key in ("running", "waiting", "deferred")
            ):
                raise AbortReview("formal drain metrics are unavailable")
            if all(snapshot[key] == 0.0 for key in ("running", "waiting", "deferred")):
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= stable_s:
                    self.event("formal_drained", stable_s=stable_s)
                    return
            else:
                quiet_since = None
            time.sleep(1)
        raise AbortReview("formal service did not reach a stable drained window")

    def stop_child(self) -> None:
        if self.child is None:
            return
        if self.child_identity is None:
            raise AbortReview("candidate identity was not captured")
        if self.child.poll() is None:
            self.verify_identity(self.child_identity)
            os.killpg(self.child_identity.pgid, signal.SIGTERM)
            try:
                self.child.wait(timeout=90)
            except subprocess.TimeoutExpired as exc:
                raise AbortReview(
                    "candidate parent did not exit after SIGTERM"
                ) from exc
        elif self.group_has_members(self.child_identity.pgid):
            # The API parent may have died while workers remain.  We cannot
            # prove a vanished PID's identity, so fail closed instead of
            # sending a signal to a possibly recycled process group.
            raise AbortReview(
                f"candidate parent exited but process group remains: "
                f"pgid={self.child_identity.pgid}"
            )
        else:
            return
        if not self.wait_group_exit(self.child_identity.pgid, 30):
            raise AbortReview("candidate did not exit after SIGTERM")
        self.verify_target_gpus_empty()
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
        self.child_role = "candidate"
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
            self.heartbeat()
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
                    if candidate.get("isolation_required", True):
                        bindings = port_bindings(int(candidate.get("port", 30002)))
                        if bindings != [("127.0.0.1", self.child.pid)]:
                            raise AbortReview(
                                f"candidate has unexpected listen bindings: {bindings}"
                            )
                        self.socket_fingerprint = port_socket_fingerprint(
                            int(candidate.get("port", 30002))
                        )
                        if (
                            len(self.socket_fingerprint) != 1
                            or self.socket_fingerprint[0][:2]
                            != ("127.0.0.1", self.child.pid)
                        ):
                            raise AbortReview(
                                "candidate socket fingerprint does not match child"
                            )
                        if self.child is not None:
                            command = proc_cmdline(self.child.pid)
                            if "--host 127.0.0.1" not in command:
                                raise AbortReview(
                                    f"candidate is not isolated to loopback: {command}"
                                )
                    # The launcher starts as bash and execs Python. Refresh
                    # identity after exec so cleanup checks the long-lived
                    # process instead of its transient shell command.
                    if self.child is not None:
                        self.child_identity = read_proc_identity(self.child.pid)
                        self.verify_gpu_ownership(self.child_identity)
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
        completed = subprocess.Popen(
            command,
            cwd=ROOT,
            env=candidate_env(self.manifest["base_env"], {}),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        peak_waiting = 0.0
        peak_deferred = 0.0
        metric_samples = 0
        try:
            while completed.poll() is None:
                self.heartbeat()
                snapshot = metrics_snapshot(
                    f"http://127.0.0.1:{test.get('port', 30002)}"
                )
                if any(
                    snapshot[key] is None
                    for key in (
                        "prefix_queries",
                        "prefix_hits",
                        "running",
                        "waiting",
                        "deferred",
                    )
                ):
                    raise RuntimeError("required Prometheus metric is unavailable")
                peak_waiting = max(peak_waiting, float(snapshot["waiting"]))
                peak_deferred = max(peak_deferred, float(snapshot["deferred"]))
                metric_samples += 1
                if time.monotonic() - started > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(1)
        except BaseException:
            if completed.poll() is None:
                os.killpg(completed.pid, signal.SIGTERM)
            try:
                completed.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(completed.pid, signal.SIGKILL)
                completed.wait(timeout=30)
            raise
        stdout, stderr = completed.communicate()
        (candidate_dir / f"{test['name']}.stdout").write_text(stdout or "")
        (candidate_dir / f"{test['name']}.stderr").write_text(stderr or "")
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(
                f"benchmark failed rc={completed.returncode}: {test['name']}"
            )
        data = parse_bench(output)
        expected_concurrency = int(test["concurrency"])
        if data.get("concurrency") != expected_concurrency:
            raise ValueError(f"benchmark concurrency mismatch: {test['name']}")
        if len(data["results"]) != expected_concurrency:
            raise ValueError(f"benchmark stream count mismatch: {test['name']}")
        if any(
            item["prompt_tokens"] != int(test["prompt_tokens"])
            or item["completion_tokens"] != int(test["max_tokens"])
            for item in data["results"]
        ):
            raise ValueError(f"benchmark token count mismatch: {test['name']}")
        data["peak_waiting"] = peak_waiting
        data["peak_deferred"] = peak_deferred
        data["metric_samples"] = metric_samples
        data["elapsed_runner_s"] = time.monotonic() - started
        return data

    def run_vision_smoke(self, candidate_dir: Path) -> None:
        image_path = ROOT / "tests" / "multimodal" / "assets" / "image1.png"
        image_data = base64.b64encode(image_path.read_bytes()).decode()
        body = {
            "model": self.manifest["formal"]["expected"]["served_model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                        {"type": "text", "text": "Briefly describe this image."},
                    ],
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:30002/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"multimodal smoke failed: {exc}") from exc
        choices = result.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not message or not (message.get("content") or message.get("reasoning")):
            raise RuntimeError("multimodal smoke returned no content")
        atomic_json(candidate_dir / "vision-smoke.json", result)
        self.event("vision_smoke_done", candidate=candidate_dir.name)

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

    @staticmethod
    def kv_tokens(candidate_dir: Path) -> int:
        log_path = candidate_dir / "server.log"
        body = log_path.read_text(errors="replace") if log_path.exists() else ""
        matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", body)
        if not matches:
            raise RuntimeError("GPU KV cache size is missing from server log")
        return int(matches[-1].replace(",", ""))

    def evaluate_baseline(
        self, candidate: dict[str, Any], candidate_dir: Path
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        if candidate.get("vision_smoke"):
            self.run_vision_smoke(candidate_dir)
        for test in candidate["tests"]:
            result = self.run_bench(candidate_dir, test)
            result["test_name"] = test["name"]
            results.append(result)
            self.event("baseline_test_done", test=test["name"])
        l3 = [item for item in results if item["test_name"].startswith("l3-")]
        if len(l3) != 3:
            raise AbortReview("baseline must contain exactly three measured L3 runs")
        l4 = {
            item["test_name"].split(".json", 1)[0]: item
            for item in results
            if item["test_name"].startswith("l4-")
        }
        baseline = {
            "l3": l3,
            "l4": l4,
            "kv_tokens": self.kv_tokens(candidate_dir),
            "min_per_stream_decode_tps": min(
                item["min_per_stream_decode_tps"] for item in l3
            ),
        }
        atomic_json(candidate_dir / "baseline.json", baseline)
        return baseline

    def evaluate_tests(
        self,
        candidate: dict[str, Any],
        candidate_dir: Path,
        tests: list[dict[str, Any]],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        l3_checked = False
        if candidate.get("vision_smoke"):
            self.run_vision_smoke(candidate_dir)
        for test in tests:
            before = metrics_snapshot(f"http://127.0.0.1:{test.get('port', 30002)}")
            if any(
                before[key] is None
                for key in (
                    "prefix_queries",
                    "prefix_hits",
                    "running",
                    "waiting",
                    "deferred",
                )
            ):
                raise RuntimeError("required Prometheus metric is unavailable")
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
            if any(
                after[key] is None
                for key in ("running", "waiting", "deferred")
            ):
                raise RuntimeError("required completion metrics are unavailable")
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
                    preliminary = self._l3_decision(
                        results, baseline, self.kv_tokens(candidate_dir)
                    )
                    if not preliminary["promoted"]:
                        atomic_json(candidate_dir / "decision.json", preliminary)
                        return preliminary

        l3_decision = self._l3_decision(
            results, baseline, self.kv_tokens(candidate_dir)
        )
        decision = self._l4_decision(results, baseline, l3_decision)
        atomic_json(candidate_dir / "decision.json", decision)
        return decision

    @staticmethod
    def _l3_decision(
        results: list[dict[str, Any]],
        baseline: dict[str, Any],
        candidate_kv_tokens: int,
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
        min_stream_ok = all(
            item["min_per_stream_decode_tps"]
            >= baseline["min_per_stream_decode_tps"] * 0.97
            for item in results
            if item["test_name"].startswith("l3-")
        )
        deferred_ok = all(
            item.get("peak_deferred") == 0.0
            for item in results
            if item["test_name"].startswith("l3-")
        )
        speed_pass = bool(
            deltas
            and median_delta >= 0.05
            and ci[0] >= 0.03
            and min_stream_ok
            and deferred_ok
        )
        capacity_ratio = candidate_kv_tokens / baseline["kv_tokens"]
        capacity_pass = bool(
            capacity_ratio >= 1.15 and median_delta >= -0.03 and min_stream_ok
        )
        decision = {
            "tests": results,
            "l3_relative_deltas": deltas,
            "l3_median_delta": median_delta,
            "l3_bootstrap_ci95": ci,
            "mad": mad(speed_values) if speed_values else float("nan"),
            "candidate_kv_tokens": candidate_kv_tokens,
            "baseline_kv_tokens": baseline["kv_tokens"],
            "capacity_ratio": capacity_ratio,
            "speed_pass": speed_pass,
            "capacity_pass": capacity_pass,
            "peak_waiting": [
                item.get("peak_waiting")
                for item in results
                if item["test_name"].startswith("l3-")
            ],
            "promoted": speed_pass or capacity_pass,
        }
        return decision

    @staticmethod
    def _l4_decision(
        results: list[dict[str, Any]], baseline: dict[str, Any], l3: dict[str, Any]
    ) -> dict[str, Any]:
        """Require long-context probes to stay within latency/capacity bounds."""
        checks: list[dict[str, Any]] = []
        baseline_l4 = baseline.get("l4", {})
        for item in results:
            if not item["test_name"].startswith("l4-"):
                continue
            key = item["test_name"].split(".json", 1)[0]
            ref = baseline_l4.get(key)
            if ref is None:
                return {"promoted": False, "reason": f"missing baseline for {key}"}
            checks.append(
                {
                    "test": key,
                    "decode_ratio": item["aggregate_decode_tps_from_last_ttft"]
                    / ref["aggregate_decode_tps_from_last_ttft"],
                    "ttft_ratio": item["median_ttft_s"] / ref["median_ttft_s"],
                    "peak_waiting": item.get("peak_waiting"),
                    "peak_deferred": item.get("peak_deferred"),
                    "baseline_peak_waiting": ref.get("peak_waiting", float("inf")),
                    "baseline_peak_deferred": ref.get("peak_deferred", float("inf")),
                }
            )
        promoted = bool(
            l3.get("promoted")
            and checks
            and all(
                check["decode_ratio"] >= 0.97
                and check["ttft_ratio"] <= 1.03
                    and check["peak_waiting"] <= check["baseline_peak_waiting"]
                    and check["peak_deferred"] <= check["baseline_peak_deferred"]
                for check in checks
            )
        )
        return {"promoted": promoted, "l3": l3, "l4_checks": checks}

    def restore_formal(self) -> None:
        formal = self.manifest["formal"]
        env = candidate_env(self.manifest["base_env"], formal)
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
        self.child_role = "formal"
        self.child_identity = read_proc_identity(self.child.pid)
        if not self.wait_healthy(formal):
            raise AbortReview("formal service failed to restore")
        status, body = http_get("http://127.0.0.1:30002/v1/models", timeout=5)
        expected = formal["expected"]
        if status != 200 or expected["served_model"] not in body:
            raise AbortReview("restored model identity mismatch")
        self.verify_formal()
        self.event("formal_restored", pid=self.child.pid)

    def run(self) -> int:
        self.acquire_lock()
        self.validate()
        if not self.execute:
            print(f"DRY-RUN OK: {len(self.manifest['candidates'])} candidates")
            return 0

        self.verify_formal()
        formal_stopping = False
        try:
            # Mark before sending SIGTERM: if termination partially succeeds,
            # the exception path must still attempt recovery.
            formal_stopping = True
            self.stop_formal()
            baseline_spec = self.manifest["baseline"]
            baseline_dir = self.start_candidate(baseline_spec)
            if not self.wait_healthy(baseline_spec):
                raise AbortReview("paired baseline service failed health check")
            baseline = self.evaluate_baseline(baseline_spec, baseline_dir)
            self.stop_child()
            decisions: list[dict[str, Any]] = []
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
                    decisions.append({"candidate": candidate["id"], **decision})
                except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                    atomic_json(candidate_dir / "decision.json", {"failed": str(exc)})
                    self.event(
                        "candidate_rejected", candidate=candidate["id"], reason=str(exc)
                    )
                finally:
                    self.stop_child()
            promoted = [item for item in decisions if item.get("promoted")]
            champion = max(
                promoted,
                key=lambda item: item.get("l3", item).get("l3_median_delta", -1.0),
                default=None,
            )
            atomic_json(
                self.run_dir / "SUMMARY.json",
                {
                    "baseline": baseline,
                    "decisions": decisions,
                    "champion": champion["candidate"] if champion else None,
                },
            )
            self.restore_formal()
            formal_stopping = False
            (self.run_dir / "DONE").write_text("campaign completed\n")
            self.event("campaign_done")
            return 0
        except BaseException as exc:  # noqa: BLE001
            self.event("ABORT_REVIEW", reason=str(exc))
            try:
                bindings = port_bindings(30002)
                child_is_loopback_owner = bool(
                    self.child
                    and bindings == [("127.0.0.1", self.child.pid)]
                )
                if child_is_loopback_owner and self.child_role == "candidate":
                    self.stop_child()
                if (
                    port_pid(30002) is None
                    and formal_stopping
                    and self.child_role != "formal"
                ):
                    self.restore_formal()
                    formal_stopping = False
            except BaseException as restore_exc:  # noqa: BLE001
                (self.run_dir / "ABORT_REVIEW").write_text(
                    f"{exc}\nrestore failed: {restore_exc}\n"
                )
                return 2
            (self.run_dir / "ABORT_REVIEW").write_text(str(exc) + "\n")
            return 130 if isinstance(exc, KeyboardInterrupt) else 2


def main() -> int:
    install_signal_handlers()
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
