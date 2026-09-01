#!/usr/bin/env python3
"""Microbenchmark one GLM-5.3 EXL3 routed MLP decode operation.

This intentionally loads only eight experts from one layer, which is enough to
exercise the same top-8 ``exl3_mgemm`` shapes as single-token serving without
loading the complete checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("trellis", "suh", "svh")


def load_experts(model: Path, layer: int, experts: int, device: torch.device):
    with (model / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    names = [
        f"{prefix}.{expert}.{projection}.{suffix}"
        for expert in range(experts)
        for projection in PROJECTIONS
        for suffix in SUFFIXES
    ]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    tensors = {}
    for shard, shard_names in by_shard.items():
        with safe_open(model / shard, framework="pt", device="cpu") as f:
            for name in shard_names:
                tensors[name] = f.get_tensor(name).to(device)

    projections = {}
    for projection in PROJECTIONS:
        rows = [
            {
                suffix: tensors[f"{prefix}.{expert}.{projection}.{suffix}"]
                for suffix in SUFFIXES
            }
            for expert in range(experts)
        ]
        projections[projection] = (
            torch.tensor(
                [row["trellis"].data_ptr() for row in rows],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [row["suh"].data_ptr() for row in rows],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [row["svh"].data_ptr() for row in rows],
                dtype=torch.long,
                device=device,
            ),
        )
    return projections, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--extension",
        default=os.path.join(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                os.path.join(Path.home(), ".cache", "torch_extensions"),
            ),
            "exllamav3_ext",
        ),
    )
    parser.add_argument(
        "--shared-input-had",
        action="store_true",
        help="Use the optimized extension's force_num_sms=-1 gate/up mode",
    )
    parser.add_argument(
        "--force-num-sms",
        type=int,
        default=0,
        help="Override cooperative EXL3 grid SMS count (0=autotune)",
    )
    parser.add_argument(
        "--pretransform-input",
        action="store_true",
        help="Transform the shared gate/up input before EXL3 GEMM",
    )
    args = parser.parse_args()

    extension = Path(args.extension)
    sys.path.insert(0, str(extension))
    import exllamav3_ext as ext

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    projections, keepalive = load_experts(
        Path(args.model), args.layer, args.experts, device
    )

    hidden = torch.randn((1, 1, 4096), dtype=torch.float16, device=device)
    ids = torch.arange(args.experts, dtype=torch.long, device=device).view(1, -1)
    weights = torch.full(
        (1, args.experts), 1.0 / args.experts, dtype=torch.float16, device=device
    )
    had_hidden = torch.empty(
        (args.experts, 1, 4096), dtype=torch.float16, device=device
    )
    gate_out = torch.empty((args.experts, 1, 2048), dtype=torch.float16, device=device)
    up_out = torch.empty_like(gate_out)
    activation = torch.empty_like(gate_out)

    def projection(
        x: torch.Tensor,
        output: torch.Tensor,
        had: torch.Tensor,
        name: str,
        shape: int,
        output_weights: torch.Tensor | None = None,
    ) -> None:
        trellis, suh, svh = projections[name]
        force_num_sms = (
            -1 if args.shared_input_had and name != "down_proj" else args.force_num_sms
        )
        ext.exl3_mgemm(
            x,
            trellis,
            output,
            suh,
            had,
            svh,
            ids,
            output_weights,
            4,
            shape,
            1,
            0,
            -1,
            -1,
            force_num_sms,
        )

    for output_dtype in (torch.float16, torch.float32):
        output = torch.empty((args.experts, 1, 4096), dtype=output_dtype, device=device)
        for shape in (-1, 1, 2, 3, 4):
            # Autotuning must happen outside capture. A positive shape bypasses
            # autotuning and directly selects that kernel geometry.
            for _ in range(3):
                projection(hidden, gate_out, had_hidden, "gate_proj", shape)
                projection(hidden, up_out, had_hidden, "up_proj", shape)
                ext.silu_mul(gate_out, up_out, activation, 10.0)
                projection(
                    activation,
                    output,
                    activation,
                    "down_proj",
                    shape,
                    weights,
                )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                projection(hidden, gate_out, had_hidden, "gate_proj", shape)
                projection(hidden, up_out, had_hidden, "up_proj", shape)
                ext.silu_mul(gate_out, up_out, activation, 10.0)
                projection(
                    activation,
                    output,
                    activation,
                    "down_proj",
                    shape,
                    weights,
                )
            for _ in range(20):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) / args.iterations
            print(
                json.dumps(
                    {
                        "shape": shape,
                        "down_dtype": str(output_dtype),
                        "latency_ms": latency,
                        "moe_per_second": 1000.0 / latency,
                    }
                ),
                flush=True,
            )

    # Gate and up use identical geometry and the same source activation. Their
    # pointer tables can therefore be concatenated into one 16-way mgemm,
    # removing one graph node without repacking the multi-gigabyte weights.
    combined = tuple(
        torch.cat((projections["gate_proj"][i], projections["up_proj"][i]))
        for i in range(3)
    )
    combined_ids = torch.cat((ids, ids + args.experts), dim=1).contiguous()
    combined_had = torch.empty(
        (args.experts * 2, 1, 4096), dtype=torch.float16, device=device
    )
    combined_out = torch.empty(
        (args.experts * 2, 1, 2048), dtype=torch.float16, device=device
    )
    transformed_hidden = torch.empty_like(hidden)
    gate0_suh = keepalive[
        f"model.language_model.layers.{args.layer}.mlp.experts.0.gate_proj.suh"
    ]
    down_output = torch.empty(
        (args.experts, 1, 4096), dtype=torch.float32, device=device
    )

    def fused_gate_up(shape: int) -> None:
        gate_up_input = hidden
        gate_up_had = combined_had
        gate_up_num_sms = -1 if args.shared_input_had else args.force_num_sms
        if args.pretransform_input:
            ext.had_r_128(
                hidden.view(1, 4096),
                transformed_hidden.view(1, 4096),
                gate0_suh,
                None,
                1.0,
            )
            gate_up_input = transformed_hidden
            gate_up_had = transformed_hidden
            gate_up_num_sms = -2
        ext.exl3_mgemm(
            gate_up_input,
            combined[0],
            combined_out,
            combined[1],
            gate_up_had,
            combined[2],
            combined_ids,
            None,
            4,
            shape,
            1,
            0,
            -1,
            -1,
            gate_up_num_sms,
        )
        ext.silu_mul(
            combined_out[: args.experts],
            combined_out[args.experts :],
            activation,
            10.0,
        )
        projection(
            activation,
            down_output,
            activation,
            "down_proj",
            shape,
            weights,
        )

    for shape in (-1, 1, 2, 3, 4):
        for _ in range(3):
            fused_gate_up(shape)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_gate_up(shape)
        for _ in range(20):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        latency = start.elapsed_time(end) / args.iterations
        print(
            json.dumps(
                {
                    "shape": shape,
                    "variant": "fused_gate_up",
                    "pretransform_input": args.pretransform_input,
                    "down_dtype": "torch.float32",
                    "latency_ms": latency,
                    "moe_per_second": 1000.0 / latency,
                }
            ),
            flush=True,
        )

    if hasattr(ext, "exl3_mgemm_raw") and hasattr(ext, "exl3_guad"):
        guad_activation = torch.empty_like(activation)

        def fused_guad(shape: int) -> None:
            gate_up_input = hidden
            gate_up_had = combined_had
            gate_up_num_sms = -1
            if args.pretransform_input:
                ext.had_r_128(
                    hidden.view(1, 4096),
                    transformed_hidden.view(1, 4096),
                    gate0_suh,
                    None,
                    1.0,
                )
                gate_up_input = transformed_hidden
                gate_up_had = transformed_hidden
                gate_up_num_sms = -2
            ext.exl3_mgemm_raw(
                gate_up_input,
                combined[0],
                combined_out,
                combined[1],
                gate_up_had,
                combined[2],
                combined_ids,
                4,
                shape,
                1,
                0,
                -1,
                -1,
                gate_up_num_sms,
            )
            ext.exl3_guad(
                combined_out[: args.experts],
                combined_out[args.experts :],
                guad_activation,
                projections["gate_proj"][2],
                projections["up_proj"][2],
                projections["down_proj"][1],
                ids,
                10.0,
            )
            ext.exl3_mgemm(
                guad_activation,
                projections["down_proj"][0],
                down_output,
                projections["down_proj"][1],
                guad_activation,
                projections["down_proj"][2],
                ids,
                weights,
                4,
                shape,
                1,
                0,
                -1,
                -1,
                -2,
            )

        fused_gate_up(-1)
        reference_output = down_output.clone()
        fused_guad(-1)
        torch.cuda.synchronize()
        absolute_error = (down_output - reference_output).abs()
        cosine = torch.nn.functional.cosine_similarity(
            down_output.flatten(), reference_output.flatten(), dim=0
        )
        print(
            json.dumps(
                {
                    "variant": "fused_guad_correctness",
                    "max_abs_error": absolute_error.max().item(),
                    "mean_abs_error": absolute_error.mean().item(),
                    "cosine_similarity": cosine.item(),
                }
            ),
            flush=True,
        )

        def raw_gate_up_only() -> None:
            ext.exl3_mgemm_raw(
                hidden,
                combined[0],
                combined_out,
                combined[1],
                combined_had,
                combined[2],
                combined_ids,
                4,
                -1,
                1,
                0,
                -1,
                -1,
                -1,
            )

        def guad_only() -> None:
            ext.exl3_guad(
                combined_out[: args.experts],
                combined_out[args.experts :],
                guad_activation,
                projections["gate_proj"][2],
                projections["up_proj"][2],
                projections["down_proj"][1],
                ids,
                10.0,
            )

        def transformed_down_only() -> None:
            ext.exl3_mgemm(
                guad_activation,
                projections["down_proj"][0],
                down_output,
                projections["down_proj"][1],
                guad_activation,
                projections["down_proj"][2],
                ids,
                weights,
                4,
                -1,
                1,
                0,
                -1,
                -1,
                -2,
            )

        for name, operation in (
            ("raw_gate_up", raw_gate_up_only),
            ("guad_only", guad_only),
            ("transformed_down", transformed_down_only),
        ):
            operation()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                operation()
            for _ in range(20):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) / args.iterations
            print(
                json.dumps(
                    {
                        "variant": name,
                        "latency_ms": latency,
                    }
                ),
                flush=True,
            )

        for shape in (-1, 1, 2, 3, 4):
            for _ in range(3):
                fused_guad(shape)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_guad(shape)
            for _ in range(20):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) / args.iterations
            print(
                json.dumps(
                    {
                        "shape": shape,
                        "variant": "fused_guad",
                        "down_dtype": "torch.float32",
                        "latency_ms": latency,
                        "moe_per_second": 1000.0 / latency,
                    }
                ),
                flush=True,
            )

    # Repack gate/up as one N=4096 matrix per expert.  Unlike the pointer-table
    # fusion above, this halves the multi-matrix batch from 16 to 8 while
    # preserving the exact packed EXL3 coefficients and output scales.
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts"
    packed_trellis = [
        torch.cat(
            (
                keepalive[f"{prefix}.{expert}.gate_proj.trellis"],
                keepalive[f"{prefix}.{expert}.up_proj.trellis"],
            ),
            dim=1,
        ).contiguous()
        for expert in range(args.experts)
    ]
    packed_svh = [
        torch.cat(
            (
                keepalive[f"{prefix}.{expert}.gate_proj.svh"],
                keepalive[f"{prefix}.{expert}.up_proj.svh"],
            )
        ).contiguous()
        for expert in range(args.experts)
    ]
    packed_gate_up = (
        torch.tensor(
            [tensor.data_ptr() for tensor in packed_trellis],
            dtype=torch.long,
            device=device,
        ),
        projections["gate_proj"][1],
        torch.tensor(
            [tensor.data_ptr() for tensor in packed_svh],
            dtype=torch.long,
            device=device,
        ),
    )
    packed_had = torch.empty(
        (args.experts, 1, 4096), dtype=torch.float16, device=device
    )
    packed_out = torch.empty_like(packed_had)

    def packed_gate_up_moe(shape: int) -> None:
        ext.exl3_mgemm(
            hidden,
            packed_gate_up[0],
            packed_out,
            packed_gate_up[1],
            packed_had,
            packed_gate_up[2],
            ids,
            None,
            4,
            shape,
            1,
            0,
            -1,
            -1,
            -1 if args.shared_input_had else args.force_num_sms,
        )
        ext.silu_mul(
            packed_out[..., :2048],
            packed_out[..., 2048:],
            activation,
            10.0,
        )
        projection(
            activation,
            down_output,
            activation,
            "down_proj",
            shape,
            weights,
        )

    for shape in (-1, 1, 2, 3, 4):
        for _ in range(3):
            packed_gate_up_moe(shape)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed_gate_up_moe(shape)
        for _ in range(20):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        latency = start.elapsed_time(end) / args.iterations
        print(
            json.dumps(
                {
                    "shape": shape,
                    "variant": "packed_gate_up",
                    "down_dtype": "torch.float32",
                    "latency_ms": latency,
                    "moe_per_second": 1000.0 / latency,
                }
            ),
            flush=True,
        )

    # ExLlamaV3 also ships a monolithic grouped-MoE kernel.  Its normal
    # frontend switches to it only at batch size >= 4, but measuring bsz=1 is
    # useful when deciding whether launch fusion can beat the three-stage
    # decode path above.  The microbenchmark loads exactly ``args.experts``
    # experts, all of which are selected by the single source token.
    concurrency = int(ext.exl3_moe_max_concurrency(args.device))
    max_rows = 128
    fused_buffers = (
        torch.empty((concurrency, max_rows, 4096), dtype=torch.float16, device=device),
        torch.empty((concurrency, max_rows, 4096), dtype=torch.float16, device=device),
        torch.empty((concurrency, max_rows, 2048), dtype=torch.float16, device=device),
        torch.empty((concurrency, max_rows, 2048), dtype=torch.float16, device=device),
    )
    expert_count = torch.ones(args.experts + 1, dtype=torch.long, device=device)
    expert_count[-1] = 0
    token_sorted = torch.zeros(args.experts, dtype=torch.long, device=device)
    weight_sorted = weights.reshape(-1).contiguous()
    fused_output = torch.zeros((1, 4096), dtype=torch.float32, device=device)

    def fused_moe() -> None:
        fused_output.zero_()
        ext.exl3_moe(
            hidden.view(1, 4096),
            fused_output,
            expert_count,
            token_sorted,
            weight_sorted,
            *fused_buffers,
            0,
            4,
            4,
            4,
            projections["gate_proj"][0],
            projections["gate_proj"][1],
            projections["gate_proj"][2],
            projections["up_proj"][0],
            projections["up_proj"][1],
            projections["up_proj"][2],
            projections["down_proj"][0],
            projections["down_proj"][1],
            projections["down_proj"][2],
            True,
            False,
            True,
            False,
            True,
            False,
            10.0,
        )

    for _ in range(3):
        fused_moe()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_moe()
    for _ in range(20):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        graph.replay()
    end.record()
    end.synchronize()
    latency = start.elapsed_time(end) / args.iterations
    print(
        json.dumps(
            {
                "variant": "monolithic_moe_bsz1",
                "experts": args.experts,
                "latency_ms": latency,
                "moe_per_second": 1000.0 / latency,
            }
        ),
        flush=True,
    )

    # The extension also contains a small-M EXL3 GEMV kernel that is not used
    # by LinearEXL3.  Measure it here before investing in a pointer-table
    # top-k wrapper.  Gate/up emit the inner (Hadamard-domain) activations;
    # exl3_guad applies their output transforms, the activation, and the down
    # input transform exactly as the regular path does.  Keep the proven
    # multi-matrix down projection so this isolates the potential win from a
    # decode-specialized gate/up kernel.
    if hasattr(ext, "exl3_gemv") and hasattr(ext, "exl3_guad"):
        gemv_hidden = torch.empty_like(hidden.view(1, 4096))
        gemv_gate_inner = torch.empty_like(gate_out.view(args.experts, 2048))
        gemv_up_inner = torch.empty_like(up_out.view(args.experts, 2048))
        gemv_activation = torch.empty_like(activation)

        def gemv_gate_up_moe() -> None:
            ext.had_r_128(
                hidden.view(1, 4096),
                gemv_hidden,
                keepalive[f"{prefix}.0.gate_proj.suh"],
                None,
                1.0,
            )
            for expert in range(args.experts):
                for proj_name, target in (
                    ("gate_proj", gemv_gate_inner),
                    ("up_proj", gemv_up_inner),
                ):
                    ext.exl3_gemv(
                        gemv_hidden,
                        keepalive[f"{prefix}.{expert}.{proj_name}.trellis"],
                        target[expert : expert + 1],
                        None,
                        None,
                        None,
                        True,
                        False,
                    )
            ext.exl3_guad(
                gemv_gate_inner.view(args.experts, 1, 2048),
                gemv_up_inner.view(args.experts, 1, 2048),
                gemv_activation,
                projections["gate_proj"][2],
                projections["up_proj"][2],
                projections["down_proj"][1],
                ids,
                10.0,
            )
            projection(
                gemv_activation,
                down_output,
                gemv_activation,
                "down_proj",
                -1,
                weights,
            )

        for _ in range(3):
            gemv_gate_up_moe()
        torch.cuda.synchronize()
        gemv_reference = down_output.clone()
        fused_guad(-1)
        torch.cuda.synchronize()
        gemv_error = (gemv_reference - down_output).abs()
        print(
            json.dumps(
                {
                    "variant": "gemv_gate_up_correctness",
                    "max_abs_error": gemv_error.max().item(),
                    "mean_abs_error": gemv_error.mean().item(),
                }
            ),
            flush=True,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gemv_gate_up_moe()
        for _ in range(20):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        latency = start.elapsed_time(end) / args.iterations
        print(
            json.dumps(
                {
                    "variant": "gemv_gate_up",
                    "latency_ms": latency,
                    "moe_per_second": 1000.0 / latency,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
