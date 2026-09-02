from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).parents[2] / "scripts" / "fast_opt_runner.py"
SPEC = importlib.util.spec_from_file_location("fast_opt_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules["fast_opt_runner"] = runner
SPEC.loader.exec_module(runner)


def test_metric_value_accepts_prometheus_labels() -> None:
    metrics = (
        "# HELP vllm:num_requests_waiting Number waiting\n"
        'vllm:num_requests_waiting{engine="0",model_name="glm"} 0.0\n'
    )
    assert runner.metric_value(metrics, "vllm:num_requests_waiting") == 0.0


def test_bootstrap_ci_is_deterministic() -> None:
    deltas = [0.08, 0.1, 0.12]
    first = runner.bootstrap_ci(deltas)
    second = runner.bootstrap_ci(deltas)
    assert first == second
    assert first[0] >= 0.08
    assert first[1] <= 0.12


def test_candidate_env_preserves_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_SENTINEL", "present")
    env = runner.candidate_env({"BASE": "1"}, {"env": {"CANDIDATE": 2}})
    assert env["RUNNER_SENTINEL"] == "present"
    assert env["BASE"] == "1"
    assert env["CANDIDATE"] == "2"


def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "run"),
                "max_campaign_s": 60,
                "base_env": {
                    "HOST": "127.0.0.1",
                    "PYTHON_BIN": sys.executable,
                    "MODEL": "/tmp/model",
                },
                "formal": {
                    "port": 30002,
                    "gpus": [0, 1, 2, 3],
                    "env": {"HOST": "0.0.0.0"},
                    "expected": {
                        "served_model": "model",
                        "model": "model",
                        "cwd": "/tmp",
                        "cuda_visible_devices": "0,1,2,3",
                    },
                },
                "baseline": {"id": "baseline", "env": {}, "tests": []},
                "candidates": [{"id": "candidate", "env": {}, "tests": []}],
            }
        )
    )
    return path


def identity(pid: int = 123, pgid: int = 123) -> runner.ProcIdentity:
    return runner.ProcIdentity(pid, "1", "cmd", "/tmp", "env", pgid)


def test_runner_lock_is_singleton(tmp_path: Path) -> None:
    manifest = manifest_file(tmp_path)
    first = runner.Campaign(manifest, execute=False)
    second = runner.Campaign(manifest, execute=False)
    first.acquire_lock()
    with pytest.raises(runner.AbortReview, match="another optimization runner"):
        second.acquire_lock()


def test_startup_failure_restores_formal(tmp_path: Path, monkeypatch) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=True)
    restored = []
    monkeypatch.setattr(campaign, "verify_formal", lambda: identity())
    monkeypatch.setattr(campaign, "stop_formal", lambda: None)
    monkeypatch.setattr(
        campaign,
        "start_candidate",
        lambda _candidate: (_ for _ in ()).throw(OSError("boom")),
    )
    monkeypatch.setattr(campaign, "restore_formal", lambda: restored.append(True))
    monkeypatch.setattr(runner, "port_bindings", lambda _port: [])
    monkeypatch.setattr(runner, "port_pid", lambda _port: None)
    assert campaign.run() == 2
    assert restored == [True]


def test_benchmark_timeout_terminates_client_group(tmp_path: Path, monkeypatch) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=False)
    killed = []

    class Process:
        pid = 456
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -signal.SIGTERM
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(
        runner,
        "metrics_snapshot",
        lambda _base: {
            "prefix_queries": 0.0,
            "prefix_hits": 0.0,
            "running": 0.0,
            "waiting": 0.0,
            "deferred": 0.0,
        },
    )
    test = {
        "name": "timeout.json",
        "concurrency": 1,
        "prompt_tokens": 1,
        "max_tokens": 1,
        "timeout_s": -1,
    }
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_bench(tmp_path, test)
    assert killed == [(456, signal.SIGTERM)]


def test_sigterm_enters_recovery_exception() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    try:
        runner.install_signal_handlers()
        with pytest.raises(runner.RunnerSignal, match="signal"):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_residual_worker_fails_closed(tmp_path: Path, monkeypatch) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=False)

    class Exited:
        def poll(self):
            return 1

    campaign.child = Exited()
    campaign.child_identity = identity()
    monkeypatch.setattr(campaign, "group_has_members", lambda _pgid: True)
    with pytest.raises(runner.AbortReview, match="process group remains"):
        campaign.stop_child()


def test_candidate_port_conflict_aborts(tmp_path: Path, monkeypatch) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=False)

    class Process:
        pid = 123

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(runner, "read_proc_identity", lambda _pid: identity())
    monkeypatch.setattr(runner, "port_pid", lambda _port: 999)
    with pytest.raises(runner.AbortReview, match="already occupied"):
        campaign.start_candidate({"id": "candidate", "env": {}})


def test_formal_health_does_not_require_candidate_id(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=False)

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    campaign.child = Process()
    campaign.child_role = "formal"
    campaign.child_identity = identity()
    events = []
    monkeypatch.setattr(
        runner,
        "http_get",
        lambda url, timeout: (
            (200, "model") if url.endswith("/v1/models") else (200, "")
        ),
    )
    monkeypatch.setattr(runner, "read_proc_identity", lambda _pid: identity())
    monkeypatch.setattr(campaign, "verify_gpu_ownership", lambda *_args: None)
    monkeypatch.setattr(
        campaign,
        "event",
        lambda kind, **fields: events.append((kind, fields)),
    )

    formal = campaign.manifest["formal"]
    formal["isolation_required"] = False
    assert campaign.wait_healthy(formal)
    assert events == [("formal_healthy", {"pid": 123})]


def test_reused_baseline_is_validated_and_loaded(tmp_path: Path) -> None:
    campaign = runner.Campaign(manifest_file(tmp_path), execute=False)
    source = tmp_path / "baseline.json"
    source.write_text(
        json.dumps(
            {
                "kv_tokens": 100,
                "l3": [
                    {
                        "aggregate_decode_tps_from_last_ttft": 10,
                        "min_per_stream_decode_tps": 1,
                    }
                ]
                * 3,
            }
        )
    )
    events = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            campaign, "event", lambda kind, **fields: events.append(kind)
        )
        baseline = campaign.load_reused_baseline(source)
    finally:
        monkeypatch.undo()
    assert baseline["kv_tokens"] == 100
    assert events == ["baseline_reused"]
