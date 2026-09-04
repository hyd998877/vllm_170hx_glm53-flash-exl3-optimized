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
from collections.abc import Callable
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
CANDIDATE_LAUNCHER = Path(
    "/mnt/nvme0/keys-vllm-glm53/launcher/launch_cmp170hx.sh"
)
FORMAL_START = Path("/mnt/nvme0/start_glm53_3000.sh")
FORMAL_STOP = Path("/mnt/nvme0/stop_glm53_3000.sh")
CANDIDATE_RUNTIME = Path("/mnt/nvme0/keys-vllm-glm53/runtime/autoround-3000")
FORMAL_RUNTIME = Path("/mnt/nvme0/keys-vllm-glm53/runtime/production-3000")
GATE_LOCK = Path("/mnt/nvme0/keys-vllm-glm53/runtime/.autoround-gate.lock")
TARGET_GPUS = (0, 2, 4, 6)
PROTECTED_GPUS = (1, 3, 5, 7)
L3_SEEDS = (20260921, 20260922, 20260923)
L4_CASES = ((8192, 64, 20260931), (32768, 64, 20260932))


class GateError(RuntimeError):
    pass


class GateSignal(BaseException):
    pass


ProcessIdentity = tuple[str, str, str]
ManagedGroup = tuple[int, int, str, dict[int, ProcessIdentity]]


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        raise GateSignal(f"gate runner received signal {signum}")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGHUP, handle)
    signal.signal(signal.SIGINT, handle)


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


def process_identity(pid: int) -> ProcessIdentity:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().split()
        command = (proc / "cmdline").read_bytes()
    except FileNotFoundError as error:
        raise GateError(f"process disappeared: {pid}") from error
    return stat[21], hashlib.sha256(command).hexdigest(), command.replace(
        b"\0", b" "
    ).decode("utf-8", "replace")


def same_process_start(pid: int, expected_start: str) -> bool:
    try:
        return process_identity(pid)[0] == expected_start
    except GateError:
        return False


def group_leader_owned(pid: int, pgid: int, expected_start: str) -> bool:
    try:
        return (
            same_process_start(pid, expected_start)
            and os.getpgid(pid) == pgid
            and os.getsid(pid) == pid
        )
    except (ProcessLookupError, PermissionError):
        return False


def process_group_members(pgid: int) -> dict[int, ProcessIdentity]:
    members = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            if os.getpgid(pid) == pgid:
                members[pid] = process_identity(pid)
        except (ProcessLookupError, PermissionError, GateError):
            continue
    return members


def read_managed_group(
    pid_file: Path, model: Path, launcher: Path, timeout: float = 10
) -> ManagedGroup:
    deadline = time.monotonic() + timeout
    last_error = f"managed process did not appear in {pid_file}"
    while time.monotonic() < deadline:
        try:
            raw_pid = pid_file.read_text().strip()
            if not raw_pid.isdigit():
                last_error = f"invalid managed PID file: {pid_file}"
                time.sleep(0.1)
                continue
            pid = int(raw_pid)
            identity = process_identity(pid)
            command = identity[2]
            is_server = str(model) in command and "--port 3000" in command
            is_launcher = str(launcher) in command and "foreground" in command
            if not (is_server or is_launcher):
                # nohup/setsid/bash can expose an empty or transitional
                # cmdline for a few scheduler ticks before exec completes.
                last_error = f"unexpected managed process: {command}"
                time.sleep(0.1)
                continue
            pgid = os.getpgid(pid)
            if pgid != pid or os.getsid(pid) != pid:
                last_error = (
                    f"managed process {pid} does not own a private session/group"
                )
                time.sleep(0.1)
                continue
            return pid, pgid, identity[0], process_group_members(pgid)
        except FileNotFoundError:
            pass
        except GateError as error:
            last_error = str(error)
        except (ProcessLookupError, PermissionError) as error:
            last_error = f"managed process transition: {error}"
        time.sleep(0.1)
    raise GateError(last_error)


def live_pid_file(pid_file: Path) -> int | None:
    try:
        raw = pid_file.read_text().strip()
    except FileNotFoundError:
        return None
    if not raw.isdigit():
        return None
    pid = int(raw)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def stop_owned_group(
    group: ManagedGroup | None, stop_command: list[str | Path], env: dict[str, str]
) -> None:
    """Stop a managed server and clean only its verified process-group residue."""
    if group is None:
        # The generic launcher trusts its PID file. Without our independent
        # identity snapshot it is unsafe to invoke stop at all.
        return
    pid, pgid, root_start, snapshot = group
    leader_owned = group_leader_owned(pid, pgid, root_start)
    if leader_owned:
        snapshot.update(process_group_members(pgid))
        with contextlib.suppress(subprocess.CalledProcessError):
            run(stop_command, cwd=ROOT, env=env)

    for sig, timeout in ((signal.SIGTERM, 15.0), (signal.SIGKILL, 5.0)):
        current = process_group_members(pgid)
        if not current:
            return
        leader_owned = pid in current and group_leader_owned(pid, pgid, root_start)
        if leader_owned:
            snapshot.update(current)
            os.killpg(pgid, sig)
        else:
            unknown = {
                member: identity[2]
                for member, identity in current.items()
                if member not in snapshot
                or identity[0] != snapshot[member][0]
            }
            if unknown:
                # Clean any still-identical ledger members before failing
                # closed on the unknown processes.
                for member, identity in current.items():
                    if (
                        member in snapshot
                        and identity[0] == snapshot[member][0]
                    ):
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(member, sig)
                raise GateError(
                    f"managed leader {pid} disappeared and PGID {pgid} contains "
                    f"unverified processes: {unknown}"
                )
            # The leader has gone, so never signal its bare PGID. Signal only
            # member PIDs whose starttime+cmdline identities were recorded.
            for member in current:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(member, sig)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and process_group_members(pgid):
            time.sleep(0.2)
    if process_group_members(pgid):
        raise GateError(f"verified process group {pgid} survived SIGKILL")


def acquire_gate_lock():
    GATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = GATE_LOCK.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise GateError(f"another AutoRound gate owns {GATE_LOCK}") from error
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} start={process_identity(os.getpid())[0]}\n")
    lock.flush()
    return lock


def target_cuda_devices() -> str:
    result = run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
    )
    mapping = {}
    for line in result.stdout.splitlines():
        index, uuid = (item.strip() for item in line.split(",", 1))
        mapping[int(index)] = uuid
    missing = [gpu for gpu in TARGET_GPUS if gpu not in mapping]
    if missing:
        raise GateError(f"target physical GPU indices are missing: {missing}")
    return ",".join(mapping[gpu] for gpu in TARGET_GPUS)


def candidate_env(cuda_devices: str | None = None) -> dict[str, str]:
    """Return a fully pinned profile; do not inherit deployment parameters."""
    # Keep only process-launch basics. In particular, never inherit CUDA/NCCL,
    # VLLM, PYTHONPATH, quantization, or scheduler knobs from the caller.
    env = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "PATH")
        if key in os.environ
    }
    env.update(
        {
            "LAUNCHER": str(CANDIDATE_LAUNCHER),
            "PYTHON_BIN": str(PYTHON),
            "MODEL": str(MODEL),
            "SERVED_MODEL": "GLM-5.3-Flash-W4A16-AutoRound",
            "CHAT_TEMPLATE": str(TEMPLATE),
            "CUDA_VISIBLE_DEVICES": cuda_devices or target_cuda_devices(),
            "HOST": "127.0.0.1",
            "PORT": "3000",
            "RUNTIME_DIR": str(CANDIDATE_RUNTIME),
            "PIPELINE_PARALLEL_SIZE": "4",
            "VLLM_PP_LAYER_PARTITION": "13,12,11,9",
            "MAX_MODEL_LEN": "524288",
            "MAX_NUM_SEQS": "6",
            "MAX_NUM_BATCHED_TOKENS": "2050",
            "LONG_PREFILL_TOKEN_THRESHOLD": "256",
            "KV_CACHE_DTYPE": "auto",
            "UTIL": "0.970",
            "UTIL_CAP": "0.970",
            "SPEC": "dflash2",
            "DFLASH_MODEL": "/mnt/nvme0/models/GLM-5.3-Flash-DFlash2",
            "DFLASH_K": "2",
            "DFLASH_DRAFT_SAMPLE_METHOD": "probabilistic",
            "DFLASH_REJECTION_SAMPLE_METHOD": "standard",
            "DFLASH_KV_CACHE_DTYPE": "auto",
            "DFLASH_QUANTIZATION": "",
            "DFLASH_QUANTIZATION_CONFIG": "",
            "EXL3_MARLIN": "0",
            "EXL3_MARLIN_DIR": "",
            "EXL3_MARLIN_LAYERS": "",
            "ASYNC_SCHEDULING": "1",
            "PP_DECODE_PHASE_POLICY": "pairpack",
            "PP_FIXED_DECODE_COMM": "0",
            "PP_DIRECT_RECV_BUFFER": "0",
            "PP_PREFILL_COHORT_BARRIER": "0",
            "PP_PREFILL_COHORT_SIZE": "0",
            "PP_PREFILL_COHORT_MIN_TOKENS": "0",
            "PP_ADAPTIVE_PREFILL": "1",
            "PP_ADAPTIVE_PREFILL_MAX_TOKENS": "2048",
            "PP_ADAPTIVE_PREFILL_BUSY_TOKENS": "1550",
            "LANGUAGE_MODEL_ONLY": "0",
            "SKIP_MM_PROFILING": "0",
            "EAGER": "0",
            "CUDAGRAPH_MODE": "FULL_DECODE_ONLY",
            "CUDAGRAPH_CAPTURE_SIZES": "3,6,9,12,15,18",
            "JIT_MONITOR_MODE": "warn",
            "JIT_MONITOR_VERBOSE": "1",
        }
    )
    return env


def assert_target_group_owned(
    group: ManagedGroup, *, require_all_gpus: bool = True
) -> None:
    leader, pgid, start, ledger = group
    if not group_leader_owned(leader, pgid, start):
        raise GateError(f"managed process-group leader {leader} changed or exited")
    ledger.update(process_group_members(pgid))
    apps = gpu_processes()
    unexpected = {}
    missing = []
    for gpu in TARGET_GPUS:
        gpu_apps = apps.get(gpu, set())
        managed_on_gpu = False
        for pid in gpu_apps:
            try:
                owner = os.getpgid(pid)
            except (ProcessLookupError, PermissionError):
                continue
            if owner != pgid:
                unexpected.setdefault(gpu, []).append(pid)
            else:
                managed_on_gpu = True
        if require_all_gpus and not managed_on_gpu:
            missing.append(gpu)
    if unexpected:
        raise GateError(
            f"target GPUs contain processes outside managed group: {unexpected}"
        )
    if missing:
        raise GateError(f"managed PP4 process is missing from target GPUs: {missing}")


def ensure_candidate_slot_clear(env: dict[str, str]) -> None:
    """Reject or clean only a stale candidate process before swapping port 3000."""
    pid_file = CANDIDATE_RUNTIME / "server.pid"
    if not pid_file.exists():
        return
    raw = pid_file.read_text().strip()
    if not raw.isdigit() or not Path("/proc").joinpath(raw).exists():
        pid_file.unlink(missing_ok=True)
        return
    identity = process_identity(int(raw))
    command = identity[2]
    if str(MODEL) not in command or "--port 3000" not in command:
        raise GateError(f"candidate PID file points at an unrelated process: {command}")
    group = read_managed_group(pid_file, MODEL, CANDIDATE_LAUNCHER)
    stop_owned_group(group, [CANDIDATE_SERVER, "stop"], env)
    wait_target_empty()


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
        "completed": metric(body, "vllm:request_success_total"),
    }


def wait_healthy(
    model: str,
    timeout: float = 900,
    group: ManagedGroup | None = None,
    protected: dict[str, Any] | None = None,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if group is not None:
            leader, _, start, _ = group
            if not same_process_start(leader, start):
                raise GateError(f"candidate leader {leader} exited during startup")
            assert_target_group_owned(group, require_all_gpus=False)
        if protected is not None:
            assert_protected(protected)
        pid = port_pid(3000)
        status, body = http_get("/v1/models")
        if pid and status == 200 and model in body:
            if group is not None and pid != group[0]:
                raise GateError(
                    f"port 3000 belongs to PID {pid}, expected candidate {group[0]}"
                )
            if group is not None:
                assert_target_group_owned(group)
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


def parse_result(
    path: Path,
    expected_prompt: int,
    expected_output: int,
    expected_concurrency: int = 6,
) -> dict[str, Any]:
    result = json.loads(path.read_text())
    rows = result.get("results") or []
    expected = int(result.get("concurrency", -1))
    if expected != expected_concurrency or len(rows) != expected_concurrency:
        raise GateError(f"incomplete token stream: {path}")
    for row in rows:
        if (
            row.get("prompt_tokens") != expected_prompt
            or row.get("completion_tokens") != expected_output
            or row.get("streamed_tokens") != expected_output
            or not math.isfinite(float(row.get("decode_tps", math.nan)))
            or not math.isfinite(float(row.get("ttft_s", math.nan)))
        ):
            raise GateError(f"incomplete or invalid stream metrics: {path}: {row}")
    return result


def bench(
    out: Path,
    model: str,
    model_path: Path,
    prompt: int,
    output: int,
    seed: int,
    guard: Callable[[], None] | None = None,
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
    peak_running = peak_waiting = peak_deferred = 0.0
    try:
        while process.poll() is None:
            if guard is not None:
                guard()
            state = metrics()
            peak_running = max(peak_running, state["running"])
            peak_waiting = max(peak_waiting, state["waiting"])
            peak_deferred = max(peak_deferred, state["deferred"])
            time.sleep(1)
        if process.returncode != 0:
            raise GateError(f"benchmark failed with rc={process.returncode}: {out}")
    except BaseException:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        raise
    after = metrics()
    result = parse_result(out, prompt, output)
    result["peak_waiting"] = peak_waiting
    result["peak_deferred"] = peak_deferred
    result["peak_running"] = peak_running
    result["preemptions_delta"] = after["preemptions"] - before["preemptions"]
    result["completed_delta"] = after["completed"] - before["completed"]
    if (
        peak_running > 6
        or peak_waiting
        or peak_deferred
        or result["preemptions_delta"]
        or result["completed_delta"] != 6
    ):
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


def api_smokes(
    model: str, out_dir: Path, guard: Callable[[], None] | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    if response_text(text).strip() != "OK":
        raise GateError(f"text API smoke mismatch: {response_text(text)!r}")
    if guard is not None:
        guard()

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
    if guard is not None:
        guard()

    tool = http_post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for one city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choice = (tool.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    # This GLM parser currently reports the tool-stop token as finish_reason
    # "stop" (the production EXL3 service was probed before the swap). Treat
    # both that deployed behavior and OpenAI's canonical "tool_calls" as valid;
    # the structured call content below remains strict.
    if choice.get("finish_reason") not in ("stop", "tool_calls") or len(calls) != 1:
        raise GateError(f"tool-call smoke did not force one call: {tool!r}")
    function = calls[0].get("function") or {}
    if function.get("name") != "get_weather":
        raise GateError(f"tool-call parser returned wrong function: {tool!r}")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError) as error:
        raise GateError(f"tool-call arguments are not valid JSON: {tool!r}") from error
    if arguments.get("city") != "Paris":
        raise GateError(f"tool-call arguments mismatch: {tool!r}")
    if guard is not None:
        guard()
    (out_dir / "api-smokes.json").write_text(
        json.dumps(
            {"text": text, "vision": vision, "tool": tool},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def kv_tokens(log_path: Path) -> int:
    body = log_path.read_text(errors="replace")
    matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", body)
    if not matches:
        raise GateError("candidate KV capacity missing from log")
    return int(matches[-1].replace(",", ""))


def assert_candidate_dispatch(log_path: Path) -> None:
    body = log_path.read_text(errors="replace")
    required = {
        "INC quantization": r"quantization=inc\b",
        "Marlin WNA16 MoE": (
            r"Using '(?:BATCHED_)?MARLIN' WNA16 MoE backend\."
        ),
    }
    missing = [
        name
        for name, pattern in required.items()
        if not re.search(pattern, body, flags=re.IGNORECASE)
    ]
    if missing:
        raise GateError(f"candidate dispatch proof missing: {missing}")
    linear_logs = re.findall(r"Using (\w+LinearKernel) for AutoGPTQLinearMethod", body)
    if any(kernel != "MarlinLinearKernel" for kernel in linear_logs):
        raise GateError(
            f"candidate AutoGPTQ linear backend is not Marlin: {linear_logs}"
        )


def assert_no_runtime_jit(log_path: Path, offset: int) -> None:
    runtime_log = log_path.read_bytes()[offset:].decode("utf-8", "replace")
    if re.search(r"JIT compilation during inference|Triton kernel JIT", runtime_log):
        raise GateError("runtime JIT observed after explicit warmup")


def run_monitored(
    command: list[str | Path], guard: Callable[[], None], **kwargs: Any
) -> None:
    process = subprocess.Popen(
        [str(item) for item in command], start_new_session=True, **kwargs
    )
    try:
        while process.poll() is None:
            guard()
            time.sleep(1)
        if process.returncode != 0:
            raise GateError(
                f"monitored command failed with rc={process.returncode}: {command}"
            )
    except BaseException:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        raise


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
    candidate_profile = candidate_env()
    # Perform every model/profile check while production is still online.
    run([CANDIDATE_SERVER, "validate"], cwd=ROOT, env=candidate_profile)
    formal_pid = wait_healthy("GLM-5.3-Flash-tr3-4bpw", 30)
    _, _, formal_command = process_identity(formal_pid)
    if str(EXL3_MODEL) not in formal_command:
        raise GateError(f"unexpected formal service: {formal_command}")
    formal_group = read_managed_group(
        FORMAL_RUNTIME / "server.pid", EXL3_MODEL, CANDIDATE_LAUNCHER
    )
    if formal_pid != formal_group[0]:
        raise GateError(
            f"formal listener PID {formal_pid} differs from managed PID "
            f"{formal_group[0]}"
        )
    assert_target_group_owned(formal_group)

    baseline_dir = out_dir / "paired-exl3"

    def formal_guard() -> None:
        assert_protected(protected)
        if port_pid(3000) != formal_group[0]:
            raise GateError("formal port 3000 listener changed during baseline")
        assert_target_group_owned(formal_group)

    api_smokes("GLM-5.3-Flash-tr3-4bpw", baseline_dir, formal_guard)
    formal_guard()

    baseline = [
        bench(
            baseline_dir / f"l3-{seed}.json",
            "GLM-5.3-Flash-tr3-4bpw",
            EXL3_MODEL,
            1024,
            512,
            seed,
            formal_guard,
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
            formal_guard,
        )
        for prompt, output, seed in L4_CASES
    }
    formal_guard()
    candidate_attempted = False
    candidate_group: ManagedGroup | None = None
    formal_stopped = False
    summary: dict[str, Any] = {"promoted": False}
    try:
        assert_protected(protected)
        wait_drained()
        # Set the recovery flag before asking the managed launcher to stop. If
        # the stop only partly succeeds, the finally block still restores.
        formal_stopped = True
        stop_owned_group(formal_group, [FORMAL_STOP], os.environ.copy())
        wait_target_empty()
        assert_protected(protected)
        ensure_candidate_slot_clear(candidate_profile)
        env = candidate_profile
        candidate_attempted = True
        run([CANDIDATE_SERVER, "start"], cwd=ROOT, env=env)
        candidate_group = read_managed_group(
            CANDIDATE_RUNTIME / "server.pid", MODEL, CANDIDATE_LAUNCHER
        )
        wait_healthy(
            "GLM-5.3-Flash-W4A16-AutoRound",
            group=candidate_group,
            protected=protected,
        )
        assert_protected(protected)

        def guard() -> None:
            assert_protected(protected)
            assert_target_group_owned(candidate_group)

        log_path = CANDIDATE_RUNTIME / "server-dflash2.log"
        assert_candidate_dispatch(log_path)
        capacity = kv_tokens(log_path)
        api_smokes(
            "GLM-5.3-Flash-W4A16-AutoRound",
            out_dir / "autoround",
            guard,
        )
        guard()
        bench(
            out_dir / "autoround/warmup-1k-16.json",
            "GLM-5.3-Flash-W4A16-AutoRound",
            MODEL,
            1024,
            16,
            20260920,
            guard,
        )
        # Startup/API warmup JIT is expected. Everything after this byte must
        # use only prewarmed kernels.
        log_offset = log_path.stat().st_size
        candidate = [
            bench(
                out_dir / f"autoround/l3-{seed}.json",
                "GLM-5.3-Flash-W4A16-AutoRound",
                MODEL,
                1024,
                512,
                seed,
                guard,
            )
            for seed in L3_SEEDS
        ]
        baseline_speed = statistics.median(
            item["aggregate_decode_tps_from_last_ttft"] for item in baseline
        )
        candidate_speed = statistics.median(
            item["aggregate_decode_tps_from_last_ttft"] for item in candidate
        )
        assert_no_runtime_jit(log_path, log_offset)
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
                guard,
            )
            for prompt, output, seed in L4_CASES
        }
        summary["l4_ratios"] = {
            str(prompt): candidate_l4[prompt]["aggregate_decode_tps_from_last_ttft"]
            / baseline_l4[prompt]["aggregate_decode_tps_from_last_ttft"]
            for prompt, _, _ in L4_CASES
        }
        assert_no_runtime_jit(log_path, log_offset)
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
            guard,
        )
        summary["long_128k_tps"] = long_result[
            "aggregate_decode_tps_from_last_ttft"
        ]
        assert_no_runtime_jit(log_path, log_offset)
        if summary["long_128k_tps"] < 230.23927143484553 * 0.97:
            summary["reason"] = "128K common-window throughput regressed >3%"
            return summary

        run_monitored(
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
            guard,
            cwd=ROOT,
        )
        assert_no_runtime_jit(log_path, log_offset)
        summary.update({"promoted": True, "reason": "all gates passed"})
        return summary
    finally:
        primary_error = sys.exc_info()[1]
        # Recovery must not be interrupted halfway by a second terminal signal.
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        recovery_errors = []
        if candidate_attempted:
            if candidate_group is None:
                try:
                    candidate_group = read_managed_group(
                        CANDIDATE_RUNTIME / "server.pid",
                        MODEL,
                        CANDIDATE_LAUNCHER,
                        timeout=10,
                    )
                except BaseException as error:
                    recovery_errors.append(f"candidate identity: {error}")
            try:
                stop_owned_group(
                    candidate_group,
                    [CANDIDATE_SERVER, "stop"],
                    candidate_profile,
                )
            except BaseException as error:
                recovery_errors.append(f"candidate stop: {error}")
        if formal_stopped:
            target_clear = False
            try:
                wait_target_empty()
                live_candidate = live_pid_file(
                    CANDIDATE_RUNTIME / "server.pid"
                )
                target_clear = port_pid(3000) is None and live_candidate is None
                if not target_clear:
                    recovery_errors.append(
                        "candidate slot remained occupied: "
                        f"port_pid={port_pid(3000)}, candidate_pid={live_candidate}"
                    )
            except BaseException as error:
                recovery_errors.append(f"target drain: {error}")
            try:
                assert_protected(protected)
            except BaseException as error:
                recovery_errors.append(f"protected service: {error}")
            if target_clear:
                try:
                    run([FORMAL_START])
                    restored_group = read_managed_group(
                        FORMAL_RUNTIME / "server.pid",
                        EXL3_MODEL,
                        CANDIDATE_LAUNCHER,
                    )
                    wait_healthy(
                        "GLM-5.3-Flash-tr3-4bpw",
                        group=restored_group,
                        protected=protected,
                    )
                    assert_target_group_owned(restored_group)
                    assert_protected(protected)
                except BaseException as error:
                    recovery_errors.append(f"formal restore: {error}")
            else:
                recovery_errors.append("formal restore skipped: target slot not clear")
        if recovery_errors:
            raise GateError(
                f"primary={primary_error!r}; recovery=" + "; ".join(recovery_errors)
            )


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
    gate_lock = acquire_gate_lock()
    try:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise SystemExit(f"output directory is not empty: {args.out_dir}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = run_gate(args.out_dir)
        except BaseException as error:
            summary = {
                "promoted": False,
                "reason": f"{type(error).__name__}: {error}",
            }
            (args.out_dir / "SUMMARY.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            raise
        (args.out_dir / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["promoted"] else 3
    finally:
        gate_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
