#!/usr/bin/env python3
"""Validate bidirectional MooncakeStore prefix sharing between two GLM APIs."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
EXTERNAL_QUERIES = "vllm:external_prefix_cache_queries_total"
METRIC_LINE = re.compile(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([^\s]+)")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def request_json(
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 30,
) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc


def request_text(url: str, *, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def metric_snapshot(base_url: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in request_text(f"{base_url}/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, labels_text, raw_value = match.groups()
        if name not in {
            EXTERNAL_HITS,
            EXTERNAL_QUERIES,
            "vllm:mooncake_store_operation_total",
            "vllm:mooncake_store_operation_keys_total",
            "vllm:mooncake_store_operation_bytes_total",
            "vllm:mooncake_store_operation_failed_keys_total",
        }:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = dict(LABEL.findall(labels_text or ""))
        operation = labels.get("operation")
        status = labels.get("status")
        key = name
        if operation:
            key += f":{operation}"
        if status:
            key += f":{status}"
        values[key] = values.get(key, 0.0) + value
    return values


def delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    keys = set(after) | set(before)
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(keys)
        if after.get(key, 0.0) != before.get(key, 0.0)
    }


def wait_for_store_quiescence(
    base_url: str,
    before: dict[str, float],
    *,
    expected_save_exists_calls: int,
    timeout: float = 60,
    stable_polls: int = 3,
) -> dict[str, float]:
    """Wait until asynchronous save counters stop changing.

    A single successful PUT only proves that the first PP rank/job completed.
    Starting the peer load at that point can observe a shorter prefix while the
    remaining ranks are still draining. First require the exact lower bound
    implied by the number of complete transfer chunks and PP workers, then
    require several stable snapshots. The second condition still catches any
    extra partial-tail work without starting a peer load in the middle of it.
    """
    deadline = time.monotonic() + timeout
    previous: tuple[float, float, float] | None = None
    stable = 0
    latest = metric_snapshot(base_url)
    while time.monotonic() < deadline:
        latest = metric_snapshot(base_url)
        changed = delta(latest, before)
        current = (
            changed.get("vllm:mooncake_store_operation_total:save_exists:ok", 0),
            changed.get("vllm:mooncake_store_operation_total:save_put:ok", 0),
            changed.get(
                "vllm:mooncake_store_operation_total:save_put:partial_failure", 0
            ),
        )
        error_calls = sum(
            value
            for key, value in changed.items()
            if key.startswith("vllm:mooncake_store_operation_total:save_")
            and (key.endswith(":error") or key.endswith(":partial_failure"))
        )
        failed_keys = sum(
            value
            for key, value in changed.items()
            if key.startswith("vllm:mooncake_store_operation_failed_keys_total:save_")
        )
        if error_calls or failed_keys:
            raise RuntimeError(
                f"Mooncake save failed for {base_url}: "
                f"error_calls={error_calls}, failed_keys={failed_keys}"
            )
        reached_expected = (
            current[0] >= expected_save_exists_calls
            and current[1] >= expected_save_exists_calls
        )
        if current == previous and reached_expected:
            stable += 1
            if stable >= stable_polls:
                return latest
        else:
            stable = 0
            previous = current
        time.sleep(1)
    raise TimeoutError(f"Mooncake saves did not quiesce for {base_url}")


def make_prompt_ids(base_url: str, model: str, seed: str, target: int) -> list[int]:
    tokenized = request_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "prompt": (
                f"Mooncake validation {seed}. The following immutable corpus tests "
                "cross-instance prefix reuse, ordering, and deterministic output. "
            ),
            "add_special_tokens": False,
        },
    )
    source = list(tokenized["tokens"])
    if not source:
        raise RuntimeError("tokenizer returned no tokens")
    return (source * math.ceil(target / len(source)))[:target]


def completion(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 20260905,
        "logprobs": 0,
    }
    started = time.perf_counter()
    response = request_json(
        f"{base_url}/v1/completions",
        body,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    logprobs = choice.get("logprobs") or {}
    generated_tokens = logprobs.get("tokens") or []
    return {
        "elapsed_seconds": round(elapsed, 6),
        "prompt_tokens": usage.get("prompt_tokens", len(prompt_ids)),
        "completion_tokens": usage.get("completion_tokens"),
        "text": choice.get("text", ""),
        "first_token": generated_tokens[0] if generated_tokens else None,
        "finish_reason": choice.get("finish_reason"),
    }


def model_info(base_url: str) -> dict[str, Any]:
    models = request_json(f"{base_url}/v1/models")
    first = models["data"][0]
    return {
        "id": first["id"],
        "max_model_len": first.get("max_model_len"),
    }


def run_direction(
    producer: str,
    consumer: str,
    model: str,
    prompt_ids: list[int],
    output_tokens: int,
    timeout: float,
    expected_save_exists_calls: int,
) -> dict[str, Any]:
    producer_before = metric_snapshot(producer)
    consumer_before = metric_snapshot(consumer)
    cold = completion(producer, model, prompt_ids, output_tokens, timeout)
    producer_after = metric_snapshot(producer)

    producer_after = wait_for_store_quiescence(
        producer,
        producer_before,
        expected_save_exists_calls=expected_save_exists_calls,
        timeout=timeout,
    )

    shared = completion(consumer, model, prompt_ids, output_tokens, timeout)
    consumer_after = metric_snapshot(consumer)
    consumer_delta = delta(consumer_after, consumer_before)
    return {
        "producer": producer,
        "consumer": consumer,
        "cold": cold,
        "shared": shared,
        "speedup": round(
            cold["elapsed_seconds"] / max(shared["elapsed_seconds"], 1e-9), 4
        ),
        "output_match": cold["text"] == shared["text"],
        "first_token_match": (
            cold["first_token"] is not None
            and cold["first_token"] == shared["first_token"]
        ),
        "producer_metrics_delta": delta(producer_after, producer_before),
        "consumer_metrics_delta": consumer_delta,
        "external_hit_tokens": consumer_delta.get(EXTERNAL_HITS, 0),
        "external_query_tokens": consumer_delta.get(EXTERNAL_QUERIES, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="http://127.0.0.1:3000")
    parser.add_argument("--b", default="http://127.0.0.1:3001")
    parser.add_argument("--model", default="GLM-5.3-Flash-tr3-4bpw")
    parser.add_argument("--tokens", type=int, default=8705)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--transfer-block-size", type=int, default=4352)
    parser.add_argument("--run-id", default=f"run-{uuid.uuid4().hex}")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/mnt/nvme0/keys-vllm-glm53/runtime/mooncake-dual-pp4/sharing-result.json"
        ),
    )
    args = parser.parse_args()

    if args.pp_size < 1:
        parser.error("--pp-size must be positive")
    if args.transfer_block_size < 1:
        parser.error("--transfer-block-size must be positive")
    if args.tokens < 2 * args.transfer_block_size + 1:
        parser.error("--tokens must cover two transfer blocks plus a suffix")
    if args.tokens % args.transfer_block_size == 0:
        parser.error("--tokens must include a non-empty recomputed suffix")
    for base_url in (args.a, args.b):
        request_text(f"{base_url}/health")

    models = {args.a: model_info(args.a), args.b: model_info(args.b)}
    for base_url, info in models.items():
        if info["id"] != args.model:
            raise RuntimeError(
                f"model mismatch at {base_url}: {info['id']!r} != {args.model!r}"
            )
        max_model_len = info.get("max_model_len")
        if not isinstance(max_model_len, int) or max_model_len < (
            args.tokens + args.output_tokens
        ):
            raise RuntimeError(
                f"context limit at {base_url} is {max_model_len!r}, below "
                f"required {args.tokens + args.output_tokens}"
            )
    prompt_ab = make_prompt_ids(
        args.a, args.model, f"{args.run_id}-a-to-b", args.tokens
    )
    prompt_ba = make_prompt_ids(
        args.b, args.model, f"{args.run_id}-b-to-a", args.tokens
    )
    expected_save_exists_calls = args.tokens // args.transfer_block_size * args.pp_size
    result = {
        "schema": "glm53_mooncake_cross_instance_v1",
        "run_id": args.run_id,
        "requested_prompt_tokens": args.tokens,
        "output_tokens": args.output_tokens,
        "expected_save_exists_calls_per_direction": expected_save_exists_calls,
        "models": models,
        "directions": [
            run_direction(
                args.a,
                args.b,
                args.model,
                prompt_ab,
                args.output_tokens,
                args.timeout,
                expected_save_exists_calls,
            ),
            run_direction(
                args.b,
                args.a,
                args.model,
                prompt_ba,
                args.output_tokens,
                args.timeout,
                expected_save_exists_calls,
            ),
        ],
    }

    # Production GLM groups use 4352-token Mooncake chunks. Keep one token
    # beyond two full chunks so the engine recomputes a suffix while the
    # expected shared prefix remains exactly representable in the store.
    transfer_block_size = args.transfer_block_size
    aligned_prefix = (args.tokens - 1) // transfer_block_size * transfer_block_size
    # DFlash/EAGLE regenerates its private draft KV by replaying one target
    # block, so only the preceding target prefix is externally reusable.
    minimum_hit = max(0, aligned_prefix - transfer_block_size)
    failures = []
    for direction in result["directions"]:
        route = f"{direction['producer']} -> {direction['consumer']}"
        if not direction["output_match"]:
            failures.append(f"output mismatch {route}")
        if not direction["first_token_match"]:
            failures.append(f"first token mismatch or unavailable {route}")
        for phase in ("cold", "shared"):
            response = direction[phase]
            if response["prompt_tokens"] != args.tokens:
                failures.append(
                    f"prompt token count {route} ({phase}) was "
                    f"{response['prompt_tokens']}, expected {args.tokens}"
                )
            if response["completion_tokens"] != args.output_tokens:
                failures.append(
                    f"completion token count {route} ({phase}) was "
                    f"{response['completion_tokens']}, expected {args.output_tokens}"
                )
            if response["finish_reason"] != "length":
                failures.append(
                    f"finish reason {route} ({phase}) was "
                    f"{response['finish_reason']!r}, expected 'length'"
                )
        if direction["external_hit_tokens"] < minimum_hit:
            failures.append(
                f"external hit {route} "
                f"was {direction['external_hit_tokens']}, expected >= {minimum_hit}"
            )
        if direction["external_query_tokens"] != args.tokens:
            failures.append(
                f"external query count {route} was "
                f"{direction['external_query_tokens']}, expected {args.tokens}; "
                "ensure the instance is otherwise idle"
            )
        for side in ("producer_metrics_delta", "consumer_metrics_delta"):
            metrics = direction[side]
            failed_keys = sum(
                value
                for key, value in metrics.items()
                if key.startswith("vllm:mooncake_store_operation_failed_keys_total:")
            )
            failed_calls = sum(
                value
                for key, value in metrics.items()
                if key.startswith("vllm:mooncake_store_operation_total:")
                and (key.endswith(":error") or key.endswith(":partial_failure"))
            )
            if failed_keys or failed_calls:
                failures.append(
                    f"Mooncake transfer failure {route} ({side}): "
                    f"failed_calls={failed_calls}, failed_keys={failed_keys}"
                )
    result["minimum_expected_external_hit_tokens"] = minimum_hit
    result["passed"] = not failures
    result["failures"] = failures

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
