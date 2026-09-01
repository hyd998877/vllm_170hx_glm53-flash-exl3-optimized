#!/usr/bin/env python3
"""Strict concurrent long-context decode benchmark for GLM-5.3.

Each request differs at the first user-content token, prompts are calibrated
with the checkpoint tokenizer, EOS is ignored so every stream produces the
requested number of output tokens, and TTFT is kept separate from decode TPS.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

FILLERS = (
    (
        "The northern observatory recorded stable telemetry while engineers reviewed "
        "the archive. "
    ),
    (
        "Copper markers identified each independent record and prevented accidental "
        "prefix reuse. "
    ),
    (
        "A quiet control room tracked latency, throughput, memory pressure, and "
        "scheduling fairness. "
    ),
    (
        "Every paragraph belongs only to this request and carries its own "
        "deterministic "
        "sequence. "
    ),
)


def make_prompt(
    tokenizer, template: str, stream_id: int, target: int, seed: int = 0
) -> str:
    unique = (
        f"SEED-{seed}-STREAM-{stream_id}-UNIQUE-"
        f"{9100003 + stream_id * 104729 + seed * 1000003}. "
        f"This is independent workload number {stream_id}; its prefix must not be "
        "shared. "
    )
    suffix = (
        "\nAfter reading the document, write a detailed continuous technical "
        "discussion "
        "of reliable long-context inference. Continue until the output limit."
    )
    unit = FILLERS[stream_id % len(FILLERS)]

    def count(text: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            chat_template=template,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return len(encoded)

    base_tokens = count(unique + suffix)
    unit_tokens = max(1, count(unique + unit + suffix) - base_tokens)
    # Tokenize O(log N) candidate strings instead of appending one token at a
    # time (which becomes quadratic at 128K).
    lo, hi = 0, max(1, (target - base_tokens) // unit_tokens + 8)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(unique + unit * mid + suffix) <= target:
            lo = mid
        else:
            hi = mid - 1
    prompt = unique + unit * lo + suffix
    current = count(prompt)
    # Usually this is within a handful of tokens; a short pad closes the
    # remaining gap without a large retokenization loop.
    if current < target:
        prompt += " x" * (target - current)
    return prompt


def stream_one(base: str, model: str, prompt: str, max_tokens: int, index: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": stream_one.temperature,
        "top_p": stream_one.top_p,
        "top_k": stream_one.top_k,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            item = json.loads(payload)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", ()):
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if text:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    end = time.perf_counter()
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    decode_seconds = last - first if first is not None and last is not None else None
    return {
        "i": index,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": first - start if first is not None else None,
        "decode_s": decode_seconds,
        "decode_tps": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens and decode_seconds and decode_seconds > 0
            else None
        ),
        "wall_s": end - start,
        "chunks": chunks,
        "first_at": first,
        "last_at": last,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30001")
    parser.add_argument("--model", default="GLM-5.3-Flash-tr3-4bpw")
    parser.add_argument(
        "--model-path", default="/mnt/nvme0/models/GLM-5.3-Flash-tr3-4bpw"
    )
    parser.add_argument(
        "--chat-template",
        default="/mnt/nvme0/keys-vllm-glm53/chat_template/chat_template.enable-thinking-switch.jinja",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--prompt-tokens", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=0,
        help="Change the first user token so repeated runs cannot reuse a prefix.",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    stream_one.temperature = args.temperature
    stream_one.top_p = args.top_p
    stream_one.top_k = args.top_k

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    template = Path(args.chat_template).read_text()
    prompts = [
        make_prompt(tokenizer, template, i, args.prompt_tokens, args.prompt_seed)
        for i in range(args.concurrency)
    ]

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
        futures = [
            executor.submit(
                stream_one, args.base, args.model, prompt, args.max_tokens, i
            )
            for i, prompt in enumerate(prompts)
        ]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - wall_start

    decode_rates = [r["decode_tps"] for r in results if r["decode_tps"]]
    first_times = [r["first_at"] for r in results if r["first_at"] is not None]
    last_times = [r["last_at"] for r in results if r["last_at"] is not None]
    for result in results:
        result.pop("first_at", None)
        result.pop("last_at", None)
    total_completion = sum(r["completion_tokens"] or 0 for r in results)
    steady_window = max(last_times) - max(first_times)
    output = {
        "concurrency": args.concurrency,
        "requested_prompt_tokens_each": args.prompt_tokens,
        "requested_completion_tokens_each": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "all_distinct_from_first_user_token": True,
        "wall_s": wall,
        "aggregate_end_to_end_tps": total_completion / wall,
        "aggregate_decode_tps_from_last_ttft": (
            sum(max(0, (r["completion_tokens"] or 0) - 1) for r in results)
            / steady_window
        ),
        "mean_per_stream_decode_tps": statistics.mean(decode_rates),
        "median_per_stream_decode_tps": statistics.median(decode_rates),
        "min_per_stream_decode_tps": min(decode_rates),
        "results": sorted(results, key=lambda r: r["i"]),
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2)
    print(encoded, flush=True)
    if args.out:
        Path(args.out).write_text(encoded + "\n")


if __name__ == "__main__":
    main()
