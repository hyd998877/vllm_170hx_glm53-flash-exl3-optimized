from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).parents[2] / "scripts" / "run_autoround_gate.py"
SPEC = importlib.util.spec_from_file_location("run_autoround_gate", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules["run_autoround_gate"] = runner
SPEC.loader.exec_module(runner)


def test_candidate_env_is_pinned_and_drops_deployment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_MOE_BACKEND", "unexpected")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    env = runner.candidate_env("GPU-a,GPU-b,GPU-c,GPU-d")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert env["MODEL"] == str(runner.MODEL)
    assert env["RUNTIME_DIR"] == str(runner.CANDIDATE_RUNTIME)
    assert env["UTIL"] == env["UTIL_CAP"] == "0.970"
    assert env["EXL3_MARLIN"] == "0"
    assert "CUDA_DEVICE_ORDER" not in env
    assert "VLLM_USE_V2_MODEL_RUNNER" not in env
    assert "VLLM_MOE_BACKEND" not in env
    assert "PYTHONPATH" not in env


def test_gate_lock_is_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "GATE_LOCK", tmp_path / "gate.lock")
    first = runner.acquire_gate_lock()
    try:
        with pytest.raises(runner.GateError, match="another AutoRound gate"):
            runner.acquire_gate_lock()
    finally:
        first.close()


def test_parse_result_requires_every_requested_token(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    rows = [
        {
            "prompt_tokens": 1024,
            "completion_tokens": 512,
            "streamed_tokens": 512,
            "decode_tps": 30.0,
            "ttft_s": 1.0,
        }
        for _ in range(6)
    ]
    path.write_text(json.dumps({"concurrency": 6, "results": rows}))
    assert runner.parse_result(path, 1024, 512)["results"] == rows
    rows[-1]["completion_tokens"] = 511
    path.write_text(json.dumps({"concurrency": 6, "results": rows}))
    with pytest.raises(runner.GateError, match="incomplete or invalid"):
        runner.parse_result(path, 1024, 512)


def test_dispatch_gate_requires_positive_inc_and_marlin_logs(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "quantization=inc, quantization_config=INCConfig(...)\n"
        "Using 'MARLIN' WNA16 MoE backend.\n"
        "Using MarlinLinearKernel for AutoGPTQLinearMethod\n"
    )
    runner.assert_candidate_dispatch(log)
    log.write_text(
        "quantization=inc\n"
        "Marlin unsupported, falling back to another backend\n"
    )
    with pytest.raises(runner.GateError, match="dispatch proof missing"):
        runner.assert_candidate_dispatch(log)


def test_stop_without_identity_never_calls_unsafe_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("launcher stop must not be called")
        ),
    )
    runner.stop_owned_group(None, ["unsafe-stop"], {})


def test_reused_leader_never_runs_stop_or_signals_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ("100", "old-leader", "managed leader")
    child = ("101", "old-child", "managed child")
    reused = ("999", "new-leader", "unrelated process")
    group = (41, 41, "100", {41: original, 42: child})
    monkeypatch.setattr(runner, "group_leader_owned", lambda *_args: False)
    monkeypatch.setattr(
        runner, "process_group_members", lambda _pgid: {41: reused, 42: child}
    )
    monkeypatch.setattr(
        runner,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("launcher stop must not run for a reused leader")
        ),
    )
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("reused PGID must never be signalled")
        ),
    )
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        runner.os, "kill", lambda pid, sig: killed.append((pid, sig))
    )

    with pytest.raises(runner.GateError, match="unverified processes"):
        runner.stop_owned_group(group, ["unsafe-stop"], {})
    assert killed == [(42, signal.SIGTERM)]


def test_failed_launcher_stop_falls_back_to_verified_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = ("100", "digest", "managed leader")
    group = (41, 41, "100", {41: identity})
    monkeypatch.setattr(runner, "group_leader_owned", lambda *_args: True)
    calls = 0

    def members(_pgid: int):
        nonlocal calls
        calls += 1
        return {41: identity} if calls <= 2 else {}

    monkeypatch.setattr(runner, "process_group_members", members)
    monkeypatch.setattr(
        runner,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["managed-stop"])
        ),
    )
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        runner.os, "killpg", lambda pgid, sig: killed.append((pgid, sig))
    )

    runner.stop_owned_group(group, ["managed-stop"], {})
    assert killed == [(41, signal.SIGTERM)]


def test_candidate_start_failure_still_restores_formal_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = (101, 101, "1001", {})
    restored = (202, 202, "2002", {})
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str | Path], **_kwargs: object) -> object:
        rendered = tuple(str(item) for item in command)
        calls.append(rendered)
        if rendered == (str(runner.CANDIDATE_SERVER), "start"):
            raise subprocess.CalledProcessError(1, rendered)
        return object()

    def fake_read(pid_file: Path, *_args: object, **_kwargs: object):
        if pid_file == runner.CANDIDATE_RUNTIME / "server.pid":
            raise runner.GateError("candidate never created a PID file")
        formal_reads = sum(
            call == (str(runner.FORMAL_START),) for call in calls
        )
        return restored if formal_reads else formal

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "snapshot_protected", lambda: {"stable": True})
    monkeypatch.setattr(runner, "candidate_env", lambda: {})
    monkeypatch.setattr(runner, "wait_healthy", lambda *_args, **_kwargs: 101)
    monkeypatch.setattr(
        runner,
        "process_identity",
        lambda _pid: ("1001", "digest", f"{runner.EXL3_MODEL} --port 3000"),
    )
    monkeypatch.setattr(runner, "read_managed_group", fake_read)
    monkeypatch.setattr(runner, "assert_target_group_owned", lambda *_args: None)
    monkeypatch.setattr(runner, "assert_protected", lambda *_args: None)
    monkeypatch.setattr(runner, "api_smokes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "bench",
        lambda *_args, **_kwargs: {"aggregate_decode_tps_from_last_ttft": 1.0},
    )
    monkeypatch.setattr(runner, "wait_drained", lambda: None)
    monkeypatch.setattr(runner, "wait_target_empty", lambda: None)
    def fake_port_pid(_port: int) -> int | None:
        if (str(runner.FORMAL_START),) in calls:
            return 202
        if (str(runner.CANDIDATE_SERVER), "start") in calls:
            return None
        return 101

    monkeypatch.setattr(runner, "port_pid", fake_port_pid)
    monkeypatch.setattr(runner, "ensure_candidate_slot_clear", lambda _env: None)
    monkeypatch.setattr(runner, "stop_owned_group", lambda *_args: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)

    with pytest.raises(runner.GateError, match="candidate never created"):
        runner.run_gate(tmp_path)
    assert (str(runner.FORMAL_START),) in calls


def test_protected_guard_rejects_identity_and_gpu_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "pid": 301,
        "start": "100",
        "digest": "expected",
        "gpus": {"1": [301], "3": [302], "5": [303], "7": [304]},
    }
    monkeypatch.setattr(runner, "port_pid", lambda _port: 301)
    monkeypatch.setattr(
        runner, "process_identity", lambda _pid: ("101", "reused", "DeepSeek")
    )
    monkeypatch.setattr(
        runner,
        "gpu_processes",
        lambda: {1: {301}, 3: {302}, 5: {303}, 7: {304}},
    )
    with pytest.raises(runner.GateError, match="identity changed"):
        runner.assert_protected(snapshot)

    monkeypatch.setattr(
        runner, "process_identity", lambda _pid: ("100", "expected", "DeepSeek")
    )
    monkeypatch.setattr(
        runner,
        "gpu_processes",
        lambda: {1: {301}, 3: {302}, 5: {303}, 7: {999}},
    )
    with pytest.raises(runner.GateError, match="GPU processes changed"):
        runner.assert_protected(snapshot)


def test_restored_pp4_must_own_all_four_target_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = (401, 401, "4001", {})
    identity = ("4001", "digest", "managed PP4")
    monkeypatch.setattr(runner, "group_leader_owned", lambda *_args: True)
    monkeypatch.setattr(
        runner, "process_group_members", lambda _pgid: {401: identity}
    )
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: 401)
    monkeypatch.setattr(
        runner, "gpu_processes", lambda: {0: {401}, 2: {401}, 4: {401}}
    )

    with pytest.raises(runner.GateError, match=r"missing.*\[6\]"):
        runner.assert_target_group_owned(group)


def test_api_smoke_asserts_structured_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "OK"}}]},
            {"choices": [{"message": {"content": "Hello AI World"}}]},
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    }
                                }
                            ]
                        },
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(runner, "http_post", lambda *_args, **_kwargs: next(responses))
    runner.api_smokes("model", tmp_path)
    saved = json.loads((tmp_path / "api-smokes.json").read_text())
    assert saved["tool"]["choices"][0]["message"]["tool_calls"]
