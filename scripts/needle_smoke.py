#!/usr/bin/env python3
"""Single-request 512K-context needle correctness smoke for GLM-5.3."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def token_count(tokenizer, template: str, text: str) -> int:
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


def build_prompt(tokenizer, template: str, target_tokens: int, seed: int) -> str:
    marker = f"NEEDLE-{seed}-GLM53-EXACT-7f31c9"
    unit = (
        "This archival record is independent, deterministic, and must remain "
        "available for long-context retrieval. "
    )
    prefix = "Read the document and answer the marker question exactly.\n\n"
    needle = f"The unique retrieval marker is {marker}. Remember this exact marker.\n\n"
    suffix = (
        "\n\nQuestion: what is the unique retrieval marker? Reply with only the "
        "marker and no additional words."
    )

    def count(repetitions: int) -> int:
        return token_count(
            tokenizer, template, prefix + unit * repetitions + needle + suffix
        )

    lo, hi = 0, max(1, target_tokens // 8)
    while count(hi) < target_tokens:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    left = prefix + unit * (lo // 2)
    right = unit * (lo - lo // 2)
    prompt = left + needle + right + suffix
    # Keep the needle near the middle while filling the requested length.
    current = token_count(tokenizer, template, prompt)
    if current < target_tokens:
        lo_pad, hi_pad = 0, target_tokens - current

        def count_pad(repetitions: int) -> int:
            return token_count(
                tokenizer,
                template,
                left + needle + right + "x " * repetitions + suffix,
            )

        while lo_pad < hi_pad:
            mid = (lo_pad + hi_pad + 1) // 2
            if count_pad(mid) <= target_tokens:
                lo_pad = mid
            else:
                hi_pad = mid - 1
        prompt = left + needle + right + "x " * lo_pad + suffix
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default="GLM-5.3-Flash-tr3-4bpw")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=8801)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    template = Path(args.chat_template).read_text()
    prompt = build_prompt(tokenizer, template, args.prompt_tokens, args.seed)
    actual_tokens = token_count(tokenizer, template, prompt)
    marker = f"NEEDLE-{args.seed}-GLM53-EXACT-7f31c9"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started
    message = (result.get("choices") or [{}])[0].get("message", {})
    answer = (
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )
    output = {
        "requested_prompt_tokens": args.prompt_tokens,
        "actual_prompt_tokens": actual_tokens,
        "needle": marker,
        "answer": answer,
        "needle_found": marker in answer,
        "elapsed_s": elapsed,
        "usage": result.get("usage"),
    }
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False))
    if not output["needle_found"]:
        raise SystemExit("needle mismatch")


if __name__ == "__main__":
    main()
