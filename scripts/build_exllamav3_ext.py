#!/usr/bin/env python3
"""Build the ExLlamaV3 native extension for the local CUDA architecture."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib

from torch.utils.cpp_extension import load


def package_source() -> pathlib.Path:
    spec = importlib.util.find_spec("exllamav3")
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise SystemExit("exllamav3 is not installed")
    return pathlib.Path(next(iter(locations))) / "exllamav3_ext"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("stock", "shared-had"),
        default="stock",
        help="Build stock ExLlamaV3 or this fork's experimental overrides",
    )
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (
        package_source()
        if args.source == "stock"
        else root / "third_party" / "exllamav3_ext_shared_had"
    )
    if not source.is_dir():
        raise SystemExit(f"missing extension source: {source}")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        str(path) for path in source.rglob("*") if path.suffix in {".c", ".cpp", ".cu"}
    )
    if not sources:
        raise SystemExit(f"no extension sources found under {source}")

    extra_cuda_flags = [
        "-lineinfo",
        "-O3",
        "--use_fast_math",
        "-Xcudafe",
        "--diag_suppress=177",
        "-Xcudafe",
        "--diag_suppress=20012",
    ]
    if cuda_host_cxx := os.environ.get("CUDAHOSTCXX"):
        extra_cuda_flags += ["-ccbin", cuda_host_cxx]

    module = load(
        name="exllamav3_ext",
        sources=sources,
        extra_include_paths=[str(source)],
        extra_cuda_cflags=extra_cuda_flags,
        extra_cflags=["-Ofast"],
        build_directory=str(args.build_dir),
        verbose=args.verbose,
    )
    print(f"built {module.__file__}")


if __name__ == "__main__":
    main()
