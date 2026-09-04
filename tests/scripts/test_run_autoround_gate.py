from __future__ import annotations

import importlib.util
import json
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
