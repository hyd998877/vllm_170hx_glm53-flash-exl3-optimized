#!/usr/bin/env python3
"""Audit OpenAI request bodies for reusable token-prefix stability.

This is intentionally an offline tool: it renders requests with the production
tokenizer/template and never sends their content to a server.  Give it two or
more JSON request bodies from consecutive turns or workers; the report locates
the first token mismatch and common-prefix length and flags common sources of
accidental cache misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VOLATILE_KEY_RE = re.compile(
    r"(^|_)(timestamp|time|date|uuid|nonce|random|trace_id|request_id)(_|$)", re.I
)
VOLATILE_VALUE_RES = (
    re.compile(r"\b20\d\d-\d\d-\d\d[T ][0-2]\d:[0-5]\d"),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.I,
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def find_volatile_fields(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if VOLATILE_KEY_RE.search(str(key)):
                findings.append(f"{child_path}: volatile key")
            findings.extend(find_volatile_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_volatile_fields(child, f"{path}[{index}]"))
    elif isinstance(value, str) and any(rx.search(value) for rx in VOLATILE_VALUE_RES):
        findings.append(f"{path}: timestamp/UUID-like value")
    return findings


def common_prefix_length(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    limit = min(map(len, sequences))
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    return limit


def render_request(tokenizer, template: str, body: dict[str, Any]) -> list[int]:
    if "messages" in body:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs["add_generation_prompt"] = True
        kwargs["tokenize"] = True
        if body.get("tools") is not None:
            kwargs["tools"] = body["tools"]
        encoded = tokenizer.apply_chat_template(
            body["messages"], chat_template=template, **kwargs
        )
    elif isinstance(body.get("prompt"), str):
        encoded = tokenizer(body["prompt"], add_special_tokens=True)["input_ids"]
    else:
        raise ValueError("request needs messages or a string prompt")
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return list(encoded)


def audit(tokenizer, template: str, paths: list[Path]) -> dict[str, Any]:
    records = []
    token_lists: list[list[int]] = []
    for path in paths:
        body = json.loads(path.read_text())
        tokens = render_request(tokenizer, template, body)
        token_lists.append(tokens)
        records.append(
            {
                "path": str(path),
                "tokens": len(tokens),
                "rendered_token_sha256": stable_hash(tokens),
                "messages_sha256": stable_hash(body.get("messages")),
                "tools_sha256": stable_hash(body.get("tools")),
                "chat_template_kwargs_sha256": stable_hash(
                    body.get("chat_template_kwargs")
                ),
                "volatile_fields": find_volatile_fields(
                    {
                        "messages": body.get("messages"),
                        "tools": body.get("tools"),
                        "chat_template_kwargs": body.get("chat_template_kwargs"),
                    }
                ),
            }
        )

    prefix = common_prefix_length(token_lists)
    shortest = min(map(len, token_lists))
    pairs = []
    for index in range(1, len(token_lists)):
        pair_prefix = common_prefix_length([token_lists[0], token_lists[index]])
        pairs.append(
            {
                "left": str(paths[0]),
                "right": str(paths[index]),
                "common_prefix_tokens": pair_prefix,
                "first_mismatch_token_index": (
                    pair_prefix
                    if pair_prefix < min(len(token_lists[0]), len(token_lists[index]))
                    else None
                ),
            }
        )
    return {
        "common_prefix_tokens": prefix,
        "common_prefix_percent_of_shortest": round(100 * prefix / shortest, 3)
        if shortest
        else 0,
        "requests": records,
        "pairs_against_first": pairs,
        "recommendations": [
            (
                "Keep the system prompt, tool list/order, JSON schema and "
                "chat-template version byte-stable."
            ),
            (
                "Put timestamps, request IDs and per-turn metadata after the "
                "reusable conversation prefix."
            ),
            (
                "Use sticky routing only when more than one serving instance "
                "exists; it cannot repair changed tokens."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="+", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.requests) < 2:
        parser.error("at least two request JSON files are required")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    report = audit(tokenizer, args.chat_template.read_text(), args.requests)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(encoded, end="")
    if args.out:
        args.out.write_text(encoded)


if __name__ == "__main__":
    main()
