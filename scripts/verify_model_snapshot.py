#!/usr/bin/env python3
"""Verify a ModelScope snapshot's file sizes and SHA-256 digests.

The verifier is intentionally read-only. It compares every local file listed
by the ModelScope repository API and exits non-zero on missing, partial, or
corrupt files. Unlisted local notes/backups are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.parse
import urllib.request


def _metadata(model_id: str, revision: str) -> dict[str, dict[str, object]]:
    query = urllib.parse.urlencode({"Revision": revision, "Root": ""})
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    files = payload.get("Data", {}).get("Files", [])
    return {str(item["Path"]): item for item in files if item.get("Type") == "blob"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=pathlib.Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()

    expected = _metadata(args.model_id, args.revision)
    if not expected:
        raise SystemExit("ModelScope API returned no files")

    root = args.snapshot.resolve()
    failures: list[str] = []
    for name, item in sorted(expected.items()):
        path = root / name
        size = int(item.get("Size", -1))
        digest = str(item.get("Sha256", "")).lower()
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        actual_size = path.stat().st_size
        if actual_size != size:
            failures.append(f"size: {name}: expected {size}, got {actual_size}")
            continue
        if digest and _sha256(path) != digest:
            failures.append(f"sha256: {name}")

    if failures:
        print("snapshot verification failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"verified {len(expected)} files in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
