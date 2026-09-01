#!/usr/bin/env python3
"""Benchmark the existing Marlin MoE backend on GLM-5 decode shapes."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tests.kernels.moe.test_moe import MarlinMoEWeightData
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.scalar_type import scalar_types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant", choices=("fp8", "int4", "awq4", "mxfp4"), default="fp8"
    )
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    m, k, n, e, topk = 1, 4096, 2048, 288, 8
    dtype = torch.float16
    # Small random values keep the activation numerics well-conditioned while
    # retaining the exact kernel dimensions of GLM-5 routed experts.
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
    quant_type = {
        "fp8": scalar_types.float8_e4m3fn,
        "int4": scalar_types.uint4b8,
        "awq4": scalar_types.uint4,
        "mxfp4": scalar_types.float4_e2m1f,
    }[args.quant]
    d1 = MarlinMoEWeightData.make(w1, quant_type, args.group_size)
    d2 = MarlinMoEWeightData.make(w2, quant_type, args.group_size)
    score = torch.randn((m, e), device=device, dtype=dtype)
    tw, ids, _ = fused_topk(a, score, topk, False)

    def run():
        return fused_marlin_moe(
            a,
            d1.qweight,
            d2.qweight,
            None,
            None,
            d1.scales,
            d2.scales,
            tw,
            ids,
            quant_type_id=quant_type.id,
            global_num_experts=e,
            global_scale1=d1.global_scale,
            global_scale2=d2.global_scale,
            g_idx1=d1.g_idx,
            g_idx2=d2.g_idx,
            sort_indices1=d1.sort_indices,
            sort_indices2=d2.sort_indices,
            w1_zeros=d1.zeros,
            w2_zeros=d2.zeros,
            input_dtype=None,
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(20):
        g.replay()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(1000):
        g.replay()
    end.record()
    end.synchronize()
    latency = start.elapsed_time(end) / 1000
    print(
        json.dumps(
            {
                "quant": args.quant,
                "group_size": args.group_size,
                "latency_ms": latency,
                "moe_per_second": 1000 / latency,
                "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    )


if __name__ == "__main__":
    main()
