from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
