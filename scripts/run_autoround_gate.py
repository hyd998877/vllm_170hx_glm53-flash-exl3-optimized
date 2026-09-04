#!/usr/bin/env python3
"""Fail-closed AutoRound W4A16 gate for the local GLM service.

The runner only owns port 3000 and physical GPUs 0,2,4,6.  It snapshots the
DeepSeek service on port 3001 and GPUs 1,3,5,7, verifies the downloaded model,
measures a fresh paired EXL3 baseline, tests AutoRound from cheap to expensive,
and always restores the managed EXL3 service before exiting.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/nvme0/keys-vllm-glm53/.venv/bin/python")
MODEL = Path("/mnt/nvme0/models/GLM-5.3-Flash-W4A16-AutoRound")
EXL3_MODEL = Path("/mnt/nvme0/models/GLM-5.3-Flash-tr3-4bpw")
TEMPLATE = ROOT / "chat_templates/glm53-enable-thinking-switch.jinja"
VERIFY = ROOT / "scripts/verify_model_snapshot.py"
BENCH = ROOT / "scripts/bench_glm53_concurrent.py"
NEEDLE = ROOT / "scripts/needle_smoke.py"
CANDIDATE_SERVER = ROOT / "scripts/serve_glm53_autoround_sm80.sh"
FORMAL_START = Path("/mnt/nvme0/start_glm53_3000.sh")
FORMAL_STOP = Path("/mnt/nvme0/stop_glm53_3000.sh")
CANDIDATE_RUNTIME = Path("/mnt/nvme0/keys-vllm-glm53/runtime/autoround-3000")
TARGET_GPUS = (0, 2, 4, 6)
PROTECTED_GPUS = (1, 3, 5, 7)
L3_SEEDS = (20260921, 20260922, 20260923)
L4_CASES = ((8192, 64, 20260931), (32768, 64, 20260932))


class GateError(RuntimeError):
    pass


class GateSignal(BaseException):
    pass


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        raise GateSignal(f"gate runner received signal {signum}")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGHUP, handle)


def run(command: list[str | Path], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command], text=True, check=True, **kwargs
    )


def http_get(path: str, timeout: float = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:3000{path}", timeout=timeout
        ) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError):
        return 0, ""


def http_post(path: str, body: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:3000{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def port_pid(port: int) -> int | None:
    result = run(["ss", "-ltnp"], capture_output=True)
    for line in result.stdout.splitlines():
        if re.search(rf"(?:0\.0\.0\.0|127\.0\.0\.1|\*|\[::\]):{port}\b", line):
            match = re.search(r"pid=(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def process_identity(pid: int) -> tuple[str, str, str]:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().split()
        command = (proc / "cmdline").read_bytes()
    except FileNotFoundError as error:
        raise GateError(f"process disappeared: {pid}") from error
    return stat[21], hashlib.sha256(command).hexdigest(), command.replace(
        b"\0", b" "
    ).decode("utf-8", "replace")


def gpu_processes() -> dict[int, set[int]]:
    gpus = run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
    )
    apps = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    uuid_index = {}
    for line in gpus.stdout.splitlines():
        index, uuid = (item.strip() for item in line.split(",", 1))
        uuid_index[uuid] = int(index)
    result: dict[int, set[int]] = {}
    for line in apps.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 1)]
        if len(values) == 2 and values[0].isdigit() and values[1] in uuid_index:
            result.setdefault(uuid_index[values[1]], set()).add(int(values[0]))
    return result


def snapshot_protected() -> dict[str, Any]:
    pid = port_pid(3001)
    if pid is None:
        raise GateError("protected DeepSeek service is not listening on 3001")
    start, digest, command = process_identity(pid)
    if "DeepSeek" not in command or "--port 3001" not in command:
        raise GateError(f"unexpected protected process: {command}")
    apps = gpu_processes()
    return {
        "pid": pid,
        "start": start,
        "digest": digest,
        "gpus": {str(gpu): sorted(apps.get(gpu, set())) for gpu in PROTECTED_GPUS},
    }


def assert_protected(snapshot: dict[str, Any]) -> None:
    pid = port_pid(3001)
    if pid != snapshot["pid"]:
        raise GateError(f"protected port 3001 changed: {snapshot['pid']} -> {pid}")
    start, digest, _ = process_identity(pid)
    apps = gpu_processes()
    current = {str(gpu): sorted(apps.get(gpu, set())) for gpu in PROTECTED_GPUS}
    if start != snapshot["start"] or digest != snapshot["digest"]:
        raise GateError("protected DeepSeek process identity changed")
    if current != snapshot["gpus"]:
        raise GateError(f"protected GPU processes changed: {current}")


def metric(metrics: str, name: str, label: str | None = None) -> float:
    values = []
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{([^}}]*)\}})?\s+([0-9.eE+-]+)$")
    for line in metrics.splitlines():
        match = pattern.match(line.strip())
        if not match or (label is not None and label not in (match.group(1) or "")):
            continue
        values.append(float(match.group(2)))
    if not values:
        raise GateError(f"missing metric: {name} {label or ''}")
    return sum(values)


def metrics() -> dict[str, float]:
    status, body = http_get("/metrics")
    if status != 200:
        raise GateError("metrics endpoint unavailable")
    return {
        "running": metric(body, "vllm:num_requests_running"),
        "waiting": metric(body, "vllm:num_requests_waiting"),
        "deferred": metric(
            body, "vllm:num_requests_waiting_by_reason", 'reason="deferred"'
        ),
        "preemptions": metric(body, "vllm:num_preemptions_total"),
    }


def wait_healthy(model: str, timeout: float = 900) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = port_pid(3000)
        status, body = http_get("/v1/models")
        if pid and status == 200 and model in body:
            return pid
        time.sleep(2)
    raise GateError(f"service did not become healthy as {model}")


def wait_drained(timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    quiet_since: float | None = None
    while time.monotonic() < deadline:
        state = metrics()
        if state["running"] == state["waiting"] == state["deferred"] == 0:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= 5:
                return
        else:
            quiet_since = None
        time.sleep(1)
    raise GateError("port 3000 did not drain")


def target_gpus_empty() -> bool:
    apps = gpu_processes()
    return all(not apps.get(gpu) for gpu in TARGET_GPUS)


def wait_target_empty(timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if target_gpus_empty():
            return
        time.sleep(1)
    raise GateError("target GPUs 0,2,4,6 did not become empty")


def parse_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    rows = result.get("results") or []
    expected = int(result["concurrency"])
    if len(rows) != expected or any(
        row.get("streamed_tokens") != row.get("completion_tokens") for row in rows
    ):
        raise GateError(f"incomplete token stream: {path}")
    return result


def bench(
    out: Path, model: str, model_path: Path, prompt: int, output: int, seed: int
) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    before = metrics()
    process = subprocess.Popen(
        [
            str(PYTHON),
            str(BENCH),
            "--base",
            "http://127.0.0.1:3000",
            "--model",
            model,
            "--model-path",
            str(model_path),
            "--chat-template",
            str(TEMPLATE),
            "--concurrency",
            "6",
            "--prompt-tokens",
            str(prompt),
            "--max-tokens",
            str(output),
            "--workload",
            "prose",
            "--prompt-seed",
            str(seed),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        start_new_session=True,
    )
    peak_waiting = peak_deferred = 0.0
    try:
        while process.poll() is None:
            state = metrics()
            peak_waiting = max(peak_waiting, state["waiting"])
            peak_deferred = max(peak_deferred, state["deferred"])
            time.sleep(1)
        if process.returncode != 0:
            raise GateError(f"benchmark failed with rc={process.returncode}: {out}")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise
    after = metrics()
    result = parse_result(out)
    result["peak_waiting"] = peak_waiting
    result["peak_deferred"] = peak_deferred
    result["preemptions_delta"] = after["preemptions"] - before["preemptions"]
    if peak_waiting or peak_deferred or result["preemptions_delta"]:
        raise GateError(f"scheduler capacity gate failed: {out}: {result}")
    return result


def response_text(result: dict[str, Any]) -> str:
    message = (result.get("choices") or [{}])[0].get("message") or {}
    return str(
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )


def api_smokes(model: str, out_dir: Path) -> None:
    text = http_post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK"}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    if not response_text(text):
        raise GateError("text API smoke returned no content")

    image = ROOT / "tests/multimodal/assets/image1.png"
    image_data = base64.b64encode(image.read_bytes()).decode()
    vision = http_post(
        "/v1/chat/completions",
        {
            "model": model,
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
                        {"type": "text", "text": "Transcribe the image exactly."},
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    normalized = re.sub(r"[^a-z]+", "", response_text(vision).lower())
    if "helloaiworld" not in normalized:
        raise GateError(f"OCR smoke mismatch: {response_text(vision)!r}")
    (out_dir / "api-smokes.json").write_text(
        json.dumps({"text": text, "vision": vision}, ensure_ascii=False, indent=2)
        + "\n"
    )


def kv_tokens(log_path: Path) -> int:
    body = log_path.read_text(errors="replace")
    matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", body)
    if not matches:
        raise GateError("candidate KV capacity missing from log")
    return int(matches[-1].replace(",", ""))


def run_gate(out_dir: Path) -> dict[str, Any]:
    run(
        [
            PYTHON,
            VERIFY,
            MODEL,
            "--model-id",
            "Intel/GLM-5.3-Flash-W4A16-AutoRound",
        ]
    )
    protected = snapshot_protected()
    formal_pid = wait_healthy("GLM-5.3-Flash-tr3-4bpw", 30)
    _, _, formal_command = process_identity(formal_pid)
    if str(EXL3_MODEL) not in formal_command:
        raise GateError(f"unexpected formal service: {formal_command}")

    baseline_dir = out_dir / "paired-exl3"
    baseline = [
        bench(
            baseline_dir / f"l3-{seed}.json",
            "GLM-5.3-Flash-tr3-4bpw",
            EXL3_MODEL,
            1024,
            512,
            seed,
        )
        for seed in L3_SEEDS
    ]
    baseline_l4 = {
        prompt: bench(
            baseline_dir / f"l4-{prompt}-{seed}.json",
            "GLM-5.3-Flash-tr3-4bpw",
            EXL3_MODEL,
            prompt,
            output,
            seed,
        )
        for prompt, output, seed in L4_CASES
    }
    candidate_started = False
    formal_stopped = False
    summary: dict[str, Any] = {"promoted": False}
    try:
        assert_protected(protected)
        wait_drained()
        # Set the recovery flag before asking the managed launcher to stop. If
        # the stop only partly succeeds, the finally block still restores.
        formal_stopped = True
        run([FORMAL_STOP])
        wait_target_empty()
        assert_protected(protected)
        env = os.environ.copy()
        env.update({"HOST": "127.0.0.1", "PORT": "3000", "UTIL": "0.970"})
        run([CANDIDATE_SERVER, "start"], cwd=ROOT, env=env)
        candidate_started = True
        wait_healthy("GLM-5.3-Flash-W4A16-AutoRound")
        assert_protected(protected)
        log_path = CANDIDATE_RUNTIME / "server-dflash2.log"
        log_offset = log_path.stat().st_size
        body = log_path.read_text(errors="replace")
        if "quantization=inc" not in body or "Marlin" not in body:
            raise GateError("candidate did not prove INC/Marlin dispatch")
        capacity = kv_tokens(log_path)
        api_smokes("GLM-5.3-Flash-W4A16-AutoRound", out_dir)
        bench(
            out_dir / "autoround/warmup-1k-16.json",
            "GLM-5.3-Flash-W4A16-AutoRound",
            MODEL,
            1024,
            16,
            20260920,
        )
        candidate = [
            bench(
                out_dir / f"autoround/l3-{seed}.json",
                "GLM-5.3-Flash-W4A16-AutoRound",
                MODEL,
                1024,
                512,
                seed,
            )
            for seed in L3_SEEDS
        ]
        baseline_speed = statistics.median(
            item["aggregate_decode_tps_from_last_ttft"] for item in baseline
        )
        candidate_speed = statistics.median(
            item["aggregate_decode_tps_from_last_ttft"] for item in candidate
        )
        summary.update(
            {
                "kv_tokens": capacity,
                "baseline_l3_median": baseline_speed,
                "candidate_l3_median": candidate_speed,
                "l3_ratio": candidate_speed / baseline_speed,
            }
        )
        if candidate_speed < baseline_speed * 1.05:
            summary["reason"] = "L3 median improvement below 5%"
            return summary

        candidate_l4 = {
            prompt: bench(
                out_dir / f"autoround/l4-{prompt}-{seed}.json",
                "GLM-5.3-Flash-W4A16-AutoRound",
                MODEL,
                prompt,
                output,
                seed,
            )
            for prompt, output, seed in L4_CASES
        }
        summary["l4_ratios"] = {
            str(prompt): candidate_l4[prompt]["aggregate_decode_tps_from_last_ttft"]
            / baseline_l4[prompt]["aggregate_decode_tps_from_last_ttft"]
            for prompt, _, _ in L4_CASES
        }
        if any(value < 0.97 for value in summary["l4_ratios"].values()):
            summary["reason"] = "8K/32K gate regressed by more than 3%"
            return summary
        if capacity < 1_150_000:
            summary["reason"] = "KV capacity below qualified 6x128K floor"
            return summary

        long_result = bench(
            out_dir / "autoround/c6-128k-512.json",
            "GLM-5.3-Flash-W4A16-AutoRound",
            MODEL,
            131072,
            512,
            20260916,
        )
        summary["long_128k_tps"] = long_result[
            "aggregate_decode_tps_from_last_ttft"
        ]
        if summary["long_128k_tps"] < 230.23927143484553 * 0.97:
            summary["reason"] = "128K common-window throughput regressed >3%"
            return summary

        run(
            [
                PYTHON,
                NEEDLE,
                "--base",
                "http://127.0.0.1:3000",
                "--model",
                "GLM-5.3-Flash-W4A16-AutoRound",
                "--model-path",
                MODEL,
                "--chat-template",
                TEMPLATE,
                "--prompt-tokens",
                "500000",
                "--seed",
                "20260940",
                "--out",
                out_dir / "autoround/needle-500k.json",
            ],
            cwd=ROOT,
        )
        runtime_log = log_path.read_bytes()[log_offset:].decode("utf-8", "replace")
        if re.search(
            r"JIT compilation during inference|Triton kernel JIT", runtime_log
        ):
            summary["reason"] = "runtime JIT observed after startup warmup"
            return summary
        summary.update({"promoted": True, "reason": "all gates passed"})
        return summary
    finally:
        if candidate_started:
            env = os.environ.copy()
            env.update({"HOST": "127.0.0.1", "PORT": "3000"})
            with contextlib.suppress(subprocess.CalledProcessError):
                run([CANDIDATE_SERVER, "stop"], cwd=ROOT, env=env)
        if formal_stopped:
            wait_target_empty()
            assert_protected(protected)
            run([FORMAL_START])
            wait_healthy("GLM-5.3-Flash-tr3-4bpw")
            assert_protected(protected)


def main() -> int:
    install_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/nvme0/keys-vllm-glm53/runtime/autoround-ab-final"),
    )
    args = parser.parse_args()
    if not args.execute:
        print("dry-run only: pass --execute after snapshot verification")
        return 0
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_gate(args.out_dir)
    except BaseException as error:
        summary = {"promoted": False, "reason": f"{type(error).__name__}: {error}"}
        (args.out_dir / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        raise
    (args.out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["promoted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
