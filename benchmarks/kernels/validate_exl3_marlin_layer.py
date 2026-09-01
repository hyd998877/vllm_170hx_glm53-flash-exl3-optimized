#!/usr/bin/env python3
"""Compare a converted Marlin sidecar layer against its EXL3 source."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.kernels.benchmark_exl3_decode import load_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    ApplyMoEActivationConfig,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (  # noqa: E402
    fused_marlin_moe,
)
from vllm.scalar_type import scalar_types  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=pathlib.Path, required=True)
    p.add_argument("--extension", type=pathlib.Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--sidecar", type=pathlib.Path, required=True)
    p.add_argument("--samples", type=int, default=32)
    args = p.parse_args()
    sys.path.insert(0, str(args.extension))
    device = torch.device("cuda", 0)
    projections, keepalive = load_experts(
        args.model,
        args.layer,
        8,
        device,
    )
    ext = __import__("exllamav3_ext")
    sidecar = {k: v.to(device) for k, v in load_file(args.sidecar).items()}
    ids = torch.arange(8, dtype=torch.long, device=device).view(1, 8)
    weights16 = torch.softmax(torch.randn(1, 8, device=device), -1).to(torch.float16)
    weights32 = weights16.to(torch.float32)
    had = torch.empty((8, 1, 4096), dtype=torch.float16, device=device)
    gate = torch.empty((8, 1, 2048), dtype=torch.float16, device=device)
    up = torch.empty_like(gate)
    act = torch.empty_like(gate)
    out = torch.empty((8, 1, 4096), dtype=torch.float32, device=device)

    errors = []
    cosines = []
    for _ in range(args.samples):
        x = torch.randn((1, 1, 4096), dtype=torch.float16, device=device)
        for proj, target in (("gate_proj", gate), ("up_proj", up)):
            trellis, suh, svh = projections[proj]
            ext.exl3_mgemm(
                x,
                trellis,
                target,
                suh,
                had,
                svh,
                ids,
                None,
                4,
                -1,
                1,
                0,
                -1,
                -1,
                0,
            )
        ext.silu_mul(gate, up, act, 10.0)
        trellis, suh, svh = projections["down_proj"]
        ext.exl3_mgemm(
            act,
            trellis,
            out,
            suh,
            act,
            svh,
            ids,
            weights16,
            4,
            -1,
            1,
            0,
            -1,
            -1,
            0,
        )
        reference = out[0, 0].clone()
        converted = fused_marlin_moe(
            x.view(1, 4096),
            sidecar["w13_qweight"],
            sidecar["w2_qweight"],
            None,
            None,
            sidecar["w13_scales"],
            sidecar["w2_scales"],
            weights32,
            ids,
            quant_type_id=scalar_types.uint4b8.id,
            global_num_experts=288,
            activation=MoEActivation.SILU,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
        )[0]
        errors.append(((converted - reference).norm() / reference.norm()).item())
        cosines.append(
            torch.nn.functional.cosine_similarity(converted, reference, dim=0).item()
        )
    print(
        {
            "samples": args.samples,
            "relative_l2_mean": sum(errors) / len(errors),
            "relative_l2_max": max(errors),
            "cosine_mean": sum(cosines) / len(cosines),
            "cosine_min": min(cosines),
        }
    )


if __name__ == "__main__":
    main()
