#!/usr/bin/env python3
"""Resume a ModelScope snapshot with bounded parallel HTTP range requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import threading
import time
import urllib.parse
import urllib.request


def metadata(model_id: str, revision: str) -> list[tuple[str, int]]:
    query = urllib.parse.urlencode({"Revision": revision, "Root": ""})
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    return sorted(
        (str(item["Path"]), int(item["Size"]))
        for item in payload.get("Data", {}).get("Files", [])
        if item.get("Type") == "blob"
    )


def download_one(
    model_id: str,
    revision: str,
    root: pathlib.Path,
    name: str,
    expected_size: int,
    retries: int,
    print_lock: threading.Lock,
) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    size = path.stat().st_size if path.exists() else 0
    if size == expected_size:
        return f"complete {name}"
    if size > expected_size:
        raise RuntimeError(
            f"refusing to overwrite oversized file {name}: {size}>{expected_size}"
        )

    quoted_model = "/".join(urllib.parse.quote(part) for part in model_id.split("/"))
    quoted_name = "/".join(urllib.parse.quote(part) for part in name.split("/"))
    url = (
        f"https://modelscope.cn/models/{quoted_model}/resolve/"
        f"{urllib.parse.quote(revision)}/{quoted_name}"
    )
    for attempt in range(1, retries + 1):
        size = path.stat().st_size if path.exists() else 0
        request = urllib.request.Request(url)
        if size:
            request.add_header("Range", f"bytes={size}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                if size and (
                    status != 206
                    or not content_range.startswith(f"bytes {size}-")
                ):
                    raise RuntimeError(
                        f"server did not honor resume for {name}: "
                        f"status={status}, content-range={content_range!r}"
                    )
                mode = "ab" if size else "wb"
                with path.open(mode) as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
            actual = path.stat().st_size
            if actual != expected_size:
                raise RuntimeError(f"short file {name}: {actual}/{expected_size}")
            with print_lock:
                print(f"downloaded {name} ({expected_size} bytes)", flush=True)
            return f"downloaded {name}"
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(f"failed {name} after {retries} attempts") from error
            delay = min(30, 2 ** (attempt - 1))
            with print_lock:
                print(
                    f"retry {attempt}/{retries} {name} at {size}/{expected_size}: "
                    f"{error}; sleeping {delay}s",
                    flush=True,
                )
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be positive")

    files = metadata(args.model_id, args.revision)
    if not files:
        raise SystemExit("ModelScope API returned no files")
    args.destination.mkdir(parents=True, exist_ok=True)
    expected = sum(size for _, size in files)
    present = sum(
        min((args.destination / name).stat().st_size, size)
        for name, size in files
        if (args.destination / name).is_file()
    )
    print(
        f"snapshot {args.model_id}: {len(files)} files, "
        f"resume {present}/{expected} bytes with {args.workers} workers",
        flush=True,
    )

    lock = threading.Lock()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        future_names = {
            executor.submit(
                download_one,
                args.model_id,
                args.revision,
                args.destination,
                name,
                size,
                args.retries,
                lock,
            ): name
            for name, size in files
        }
        for future, name in future_names.items():
            try:
                future.result()
            except Exception as error:
                failures.append(f"{name}: {error}")
    if failures:
        print("download failures:\n" + "\n".join(failures), flush=True)
        return 1
    print("snapshot download complete; run verify_model_snapshot.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
