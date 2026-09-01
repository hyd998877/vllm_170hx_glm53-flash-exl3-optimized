#!/usr/bin/env python3
"""Convert one GLM-5 EXL3 MoE layer to Marlin GPTQ-INT4 tensors.

The EXL3 checkpoint is already a 4-bit weight-only representation, but its
SM80 decode kernel is slower than Marlin.  This converter reconstructs one
expert at a time, requantizes to symmetric GPTQ INT4, and writes only the
runtime Marlin tensors.  Layer-at-a-time operation keeps peak GPU memory low;
multiple processes can safely convert disjoint layers in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm.model_executor.layers.quantization.exl3 import _exl3_module  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    marlin_quantize,
)
from vllm.scalar_type import scalar_types  # noqa: E402

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("trellis", "suh", "svh", "mcg")


def checkpoint_map(model: pathlib.Path) -> dict[str, str]:
    with (model / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def load_layer(
    model: pathlib.Path, layer: int, device: torch.device
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    mapping = checkpoint_map(model)
    root = f"model.language_model.layers.{layer}.mlp.experts"
    names = [
        f"{root}.{expert}.{projection}.{suffix}"
        for expert in range(288)
        for projection in PROJECTIONS
        for suffix in SUFFIXES
    ]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(mapping[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(model / shard, framework="pt", device="cpu") as f:
            for name in shard_names:
                tensors[name] = f.get_tensor(name).to(device)
    result: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for expert in range(288):
        result[expert] = {}
        for projection in PROJECTIONS:
            result[expert][projection] = {
                suffix: tensors[f"{root}.{expert}.{projection}.{suffix}"]
                for suffix in SUFFIXES
            }
    return result


def reconstruct(
    ext: Any,
    packed: dict[str, torch.Tensor],
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    linear = ext.LinearEXL3(
        config=None,
        in_features=in_features,
        out_features=out_features,
        trellis=packed["trellis"],
        suh=packed["suh"],
        svh=packed["svh"],
        mcg=packed["mcg"],
        out_dtype=torch.float16,
        transformers_fix=True,
    )
    return linear.get_weight_tensor()


def convert_layer(
    model: pathlib.Path,
    out_dir: pathlib.Path,
    layer: int,
    device: torch.device,
    group_size: int,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"layer-{layer:02d}.safetensors"
    if output.exists():
        print(f"exists {output}", flush=True)
        return output
    ext = _exl3_module()
    packed = load_layer(model, layer, device)
    w13_q: list[torch.Tensor] = []
    w13_s: list[torch.Tensor] = []
    w2_q: list[torch.Tensor] = []
    w2_s: list[torch.Tensor] = []
    for expert in range(288):
        gate = reconstruct(ext, packed[expert]["gate_proj"], 4096, 2048)
        up = reconstruct(ext, packed[expert]["up_proj"], 4096, 2048)
        down = reconstruct(ext, packed[expert]["down_proj"], 2048, 4096)
        # Quantize gate/up together so the Marlin tile permutation sees the
        # exact merged [2 * intermediate, hidden] geometry.
        _, q13, s13, _, _, _ = marlin_quantize(
            torch.cat((gate, up), dim=1),
            scalar_types.uint4b8,
            group_size,
            False,
        )
        _, q2, s2, _, _, _ = marlin_quantize(
            down, scalar_types.uint4b8, group_size, False
        )
        w13_q.append(q13.cpu())
        w13_s.append(s13.cpu())
        w2_q.append(q2.cpu())
        w2_s.append(s2.cpu())
        del gate, up, down, q13, s13, q2, s2
        if expert % 16 == 0:
            torch.cuda.synchronize(device)
            print(
                f"layer={layer} expert={expert}/288 "
                f"alloc={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB",
                flush=True,
            )
    tensors = {
        "w13_qweight": torch.stack(w13_q),
        "w13_scales": torch.stack(w13_s),
        "w2_qweight": torch.stack(w2_q),
        "w2_scales": torch.stack(w2_s),
    }
    metadata = {
        "format": "glm53-marlin-int4",
        "group_size": str(group_size),
        "num_experts": "288",
        "hidden_size": "4096",
        "intermediate_size": "2048",
    }
    # safetensors writes directly to the requested path.  Write beside the
    # final file and publish it atomically so an interrupted conversion can
    # never leave a truncated layer that a later run mistakes for complete.
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {output} size={output.stat().st_size / 2**30:.2f}GiB", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    convert_layer(
        args.model,
        args.out_dir,
        args.layer,
        torch.device("cuda", args.device),
        args.group_size,
    )


if __name__ == "__main__":
    main()
