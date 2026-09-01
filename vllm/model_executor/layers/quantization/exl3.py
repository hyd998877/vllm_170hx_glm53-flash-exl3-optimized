# SPDX-License-Identifier: Apache-2.0
"""Selective EXL3 reader for the GLM-5.3 routed-expert checkpoint.

The checkpoint stores only routed MoE projections in ExLlamaV3's packed EXL3
format.  The rest of the model remains native BF16 and is intentionally left
to the normal vLLM loaders.  This module is a small vLLM adapter around the
reviewed ExLlamaV3 LinearEXL3 kernel; it does not reconstruct a dense expert
weight at load time.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import re
import sys
from collections.abc import Iterable
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

_PACKED_NAME = re.compile(
    r"(?:^|\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)$"
)
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SUFFIXES = ("trellis", "suh", "svh", "mcg")
_EXLLAMA_EXT = os.environ.get(
    "VLLM_EXL3_EXTENSION_DIR",
    os.path.join(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            os.path.join(pathlib.Path.home(), ".cache", "torch_extensions"),
        ),
        "exllamav3_ext",
    ),
)
# The upstream BC_BlockSparseMLP object has a fixed 32-row graph table.  The
# raw three-kernel route below does not use that table and can safely replay a
# larger, statically-unrolled batch inside vLLM's outer CUDA graph.  DFlash
# verification for six concurrent requests is normally 48 rows (6 * 8), while
# vLLM also captures a 96-row profile size, so keep enough scratch for both.
_TEMP_ROWS_GRAPH = 32
_TEMP_ROWS_RAW_GRAPH = 96
_TEMP_ROWS_FUSED = 128
# ``determine_available_memory`` profiles the model with
# ``max_num_batched_tokens`` rows.  The Marlin sidecar layers no longer retain
# their EXL3 tensors, so they must stay on Marlin for that profiling pass as
# well as normal decode.  The long-context serving profile uses a 2048-token
# scheduler budget to avoid splitting each 128K prefill into 256 tiny chunks.
_TEMP_ROWS_MARLIN = 2048
_FUSED_BUFFER_CACHE: dict[torch.device, tuple[torch.Tensor, ...]] = {}
_FUSED_MOE_BUFFER_CACHE: dict[torch.device, tuple[torch.Tensor, ...]] = {}
_MARLIN_CACHE: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def _marlin_sidecar_path(layer: RoutedExperts) -> str | None:
    root = os.environ.get("VLLM_EXL3_MARLIN_DIR", "")
    if not root:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
    if match is None:
        return None
    layer_index = int(match.group(1))
    selected = os.environ.get("VLLM_EXL3_MARLIN_LAYERS", "")
    if selected:
        allowed: set[int] = set()
        for item in selected.split(","):
            bounds = item.split("-", 1)
            if len(bounds) == 1:
                allowed.add(int(bounds[0]))
            else:
                allowed.update(range(int(bounds[0]), int(bounds[1]) + 1))
        if layer_index not in allowed:
            return None
    path = os.path.join(root, f"layer-{layer_index:02d}.safetensors")
    return path if os.path.isfile(path) else None


def _marlin_scratch(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Stable decode scratch for the sidecar Marlin path.

    The tensors are intentionally oversized for the normal decode graph and
    are sliced by ``fused_marlin_moe`` for smaller requests.  Reusing them is
    important: allocating the intermediate buffers on every token otherwise
    erases most of Marlin's kernel advantage.
    """
    device = torch.device(device)
    cached = _MARLIN_CACHE.get(device)
    if cached is None:
        max_rows = _TEMP_ROWS_MARLIN
        topk = 8
        cached = (
            torch.empty((max_rows * topk * 4096,), dtype=torch.float16, device=device),
            torch.empty((max_rows * topk, 2048), dtype=torch.float16, device=device),
        )
        _MARLIN_CACHE[device] = cached
    return cached


def _use_online_fp8_for_linear(prefix: str) -> bool:
    """Return whether a BF16 linear follows the official FP8 scope.

    The released GLM-5.3 FP8 checkpoint quantizes shared experts and the
    first three dense MLP layers.  Attention, mHC, router, embeddings, and
    the LM head remain BF16.  ``all`` retains the original broad experiment;
    ``official`` matches the released checkpoint's scope and avoids putting
    Marlin on attention/LM-head matrices where it is slower on SM80.
    """
    if os.environ.get("VLLM_EXL3_NON_ROUTED_FP8", "0") != "1":
        return False
    scope = os.environ.get("VLLM_EXL3_NON_ROUTED_FP8_SCOPE", "all")
    if scope == "all":
        return True
    if scope != "official":
        raise ValueError(
            f"VLLM_EXL3_NON_ROUTED_FP8_SCOPE must be 'all' or 'official', got {scope!r}"
        )
    if ".mlp.shared_experts." in prefix:
        return True
    match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.", prefix)
    return match is not None and int(match.group(1)) < 3


def _shared_input_had_enabled() -> bool:
    """Return whether the separately-built shared-Hadamard extension is live."""
    return os.environ.get(
        "VLLM_EXL3_SHARED_INPUT_HAD", "0"
    ) == "1" and "shared_had" in os.path.basename(_EXLLAMA_EXT)


def _exl3_module():
    """Import the already-built ExLlamaV3 module lazily at model load time."""
    # ExLlamaV3's package initializer imports optional FlashAttention and
    # Formatron modules.  The serving adapter needs only ``modules.quant``;
    # install namespace parents to keep those optional dependencies out of the
    # vLLM environment.  The reviewed extension is precompiled for SM80.
    package_root = os.environ.get("VLLM_EXL3_PACKAGE_ROOT")
    if package_root:
        root = pathlib.Path(package_root)
    else:
        spec = importlib.util.find_spec("exllamav3")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise ImportError(
                "exllamav3 is not installed; install exllamav3==0.0.43 or "
                "set VLLM_EXL3_PACKAGE_ROOT"
            )
        root = pathlib.Path(next(iter(locations)))
    if "exllamav3" not in sys.modules:
        for name, path in {
            "exllamav3": root,
            "exllamav3.modules": root / "modules",
            "exllamav3.modules.quant": root / "modules" / "quant",
            "exllamav3.model": root / "model",
        }.items():
            module = type(sys)(name)
            module.__file__ = str(path / "__init__.py")
            module.__package__ = name
            module.__path__ = [str(path)]
            sys.modules[name] = module
        config = type(sys)("exllamav3.model.config")
        config.__file__ = str(root / "model" / "config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})
        sys.modules[config.__name__] = config
    if _EXLLAMA_EXT not in sys.path:
        sys.path.insert(0, _EXLLAMA_EXT)
    return importlib.import_module("exllamav3.modules.quant.exl3")


def _fused_buffers(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return scratch buffers shared by sequential MoE layers on one device."""
    device = torch.device(device)
    cached = _FUSED_BUFFER_CACHE.get(device)
    if cached is None:
        hidden_size = 4096
        intermediate_size = 2048
        raw_rows = max(_TEMP_ROWS_GRAPH, _TEMP_ROWS_RAW_GRAPH)
        temp_hidden = torch.empty(
            (raw_rows * 2, hidden_size),
            dtype=torch.float16,
            device=device,
        )
        temp_intermediate = torch.empty(
            (raw_rows * 2, intermediate_size),
            dtype=torch.float16,
            device=device,
        )
        temp_activation = torch.empty(
            (raw_rows, intermediate_size),
            dtype=torch.float16,
            device=device,
        )
        temp_output = torch.empty(
            (raw_rows, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        # The bound class needs these for its large-row fallback. All layers on
        # a PP rank execute sequentially on the same stream, so they can share.
        dq_up = torch.empty(
            (hidden_size, intermediate_size),
            dtype=torch.float16,
            device=device,
        )
        dq_down = dq_up.view(intermediate_size, hidden_size)
        cached = (
            temp_hidden,
            temp_intermediate,
            temp_activation,
            temp_output,
            dq_up,
            dq_down,
        )
        _FUSED_BUFFER_CACHE[device] = cached
    return cached


def _fused_moe_buffers(ext, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return scratch for ExLlamaV3's single-kernel grouped MoE path."""
    device = torch.device(device)
    cached = _FUSED_MOE_BUFFER_CACHE.get(device)
    if cached is None:
        concurrency = int(ext.exl3_moe_max_concurrency(device.index))
        cached = (
            torch.empty(
                (concurrency, _TEMP_ROWS_FUSED, 4096),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, _TEMP_ROWS_FUSED, 4096),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, _TEMP_ROWS_FUSED, 2048),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, _TEMP_ROWS_FUSED, 2048),
                dtype=torch.float16,
                device=device,
            ),
        )
        _FUSED_MOE_BUFFER_CACHE[device] = cached
    return cached


class Exl3Config(QuantizationConfig):
    """Configuration for the selective GLM-5.3 EXL3-MCG format."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        if config.get("quant_method") != "exl3":
            raise ValueError("EXL3 config requires quant_method='exl3'")
        if config.get("bits") != 4 or config.get("codebook") != "mcg":
            raise ValueError("vLLM EXL3 adapter currently supports only K4 MCG")
        if config.get("scope") != "glm53_routed_experts_only":
            raise ValueError("EXL3 checkpoint is not the selective GLM-5.3 format")
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Any | None:
        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(self, layer.moe_config)
        # The quantization config is global, but only RoutedExperts are packed.
        # On the normal path native BF16 linear layers retain their ordinary
        # implementation.  The opt-in online FP8 path is useful for this
        # checkpoint because its non-routed tensors are deliberately stored in
        # BF16; on SM80 those matrices are substantially faster in Marlin's
        # weight-only FP8 kernel and use less memory.  Keep this behind an
        # environment flag so the production reader remains byte-for-byte
        # unchanged unless explicitly requested.
        if isinstance(layer, LinearBase):
            if _use_online_fp8_for_linear(prefix):
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineLinearMethod,
                )
                from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                    get_marlin_input_dtype,
                )

                method = Fp8PerTensorOnlineLinearMethod()
                method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return method
            return UnquantizedLinearMethod()
        return None


class Exl3MoEMethod(FusedMoEMethodBase):
    """Eager routed dispatch backed by one EXL3 kernel object per projection."""

    def __init__(self, quant_config: Exl3Config, moe):
        super().__init__(moe)
        self.quant_config = quant_config
        self._inners: list[dict[str, Any]] | None = None
        self._fused_bsz1 = None
        self._raw_bsz1 = None
        self._fused_moe = None
        self._marlin = None
        self._fused_output: torch.Tensor | None = None
        self._fused_batch_output: torch.Tensor | None = None
        # FusedMoEConfig has already resolved the effective TP width (including
        # sequence/DP flattening) before the quant method is constructed.
        self._tp_size = int(moe.tp_size)
        if self._tp_size != 1:
            raise ValueError(
                "GLM-5.3 EXL3 vLLM reader requires TP=1; use PP=4 across four GPUs"
            )

    @property
    def is_monolithic(self) -> bool:
        return False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            num_experts != 288
            or hidden_size != 4096
            or intermediate_size_per_partition != 2048
        ):
            raise ValueError("unexpected GLM-5.3 routed expert geometry")
        layer.quant_config = self.quant_config
        layer._exl3_num_experts = num_experts
        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_size = intermediate_size_per_partition
        packed_shapes = {
            "w13": {
                "trellis": (
                    num_experts,
                    2,
                    hidden_size // 16,
                    intermediate_size_per_partition // 16,
                    64,
                ),
                "suh": (num_experts, 2, hidden_size),
                "svh": (num_experts, 2, intermediate_size_per_partition),
                "mcg": (num_experts, 2, 1),
            },
            "w2": {
                "trellis": (
                    num_experts,
                    intermediate_size_per_partition // 16,
                    hidden_size // 16,
                    64,
                ),
                "suh": (num_experts, intermediate_size_per_partition),
                "svh": (num_experts, hidden_size),
                "mcg": (num_experts, 1),
            },
        }
        weight_attrs = dict(extra_weight_attrs)
        weight_attrs["weight_loader"] = self.weight_loader
        for packed_name, shapes in packed_shapes.items():
            for suffix, shape in shapes.items():
                parameter = nn.Parameter(
                    torch.empty(
                        shape,
                        dtype={
                            "trellis": torch.int16,
                            "suh": torch.float16,
                            "svh": torch.float16,
                            "mcg": torch.int32,
                        }[suffix],
                    ),
                    requires_grad=False,
                )
                layer.register_parameter(f"{packed_name}_{suffix}", parameter)
                set_weight_attrs(parameter, weight_attrs)

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        value = loaded_weight.to(device=param.device, dtype=param.dtype)
        target = param.data[expert_id]
        if shard_id in ("w1", "w3"):
            target = target[0 if shard_id == "w1" else 1]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"EXL3 {weight_name} shape {tuple(value.shape)} "
                f"!= {tuple(target.shape)}"
            )
        target.copy_(value)
        return True if return_success else None

    def load_weights(
        self, layer: RoutedExperts, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        """Load individual ``experts.N.{projection}.{suffix}`` packed tensors."""

        loaded: set[str] = set()
        parameters = dict(layer.named_parameters(recurse=False))
        for expert_name, loaded_weight in weights:
            match = _PACKED_NAME.search("experts." + expert_name)
            if match is None:
                raise ValueError(f"unexpected EXL3 routed weight: {expert_name}")
            expert_id, projection, suffix = match.groups()
            expert_id = int(expert_id)
            shard_id = {
                "gate_proj": "w1",
                "up_proj": "w3",
                "down_proj": "w2",
            }[projection]
            parameter_name = f"{'w13' if shard_id != 'w2' else 'w2'}_{suffix}"
            parameter = parameters[parameter_name]
            self.weight_loader(
                parameter,
                loaded_weight,
                expert_name,
                shard_id,
                expert_id,
            )
            loaded.add(parameter_name)
        return loaded

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        sidecar = _marlin_sidecar_path(layer)
        if sidecar is not None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_make_workspace_new,
            )

            tensors: dict[str, torch.Tensor] = {}
            with safe_open(sidecar, framework="pt", device="cpu") as f:
                for name in ("w13_qweight", "w13_scales", "w2_qweight", "w2_scales"):
                    tensors[name] = f.get_tensor(name).to(layer.w13_trellis.device)
            self._marlin = (
                tensors["w13_qweight"],
                tensors["w2_qweight"],
                tensors["w13_scales"],
                tensors["w2_scales"],
                marlin_make_workspace_new(layer.w13_trellis.device, 4),
                _marlin_scratch(layer.w13_trellis.device),
            )
            # Release the EXL3 storage for this layer after the sidecar is live.
            # The parameters remain registered with empty tensors so model
            # loading/accounting code that inspects their names still works.
            for name in tuple(layer._parameters):
                if name.startswith(("w13_", "w2_")):
                    layer._parameters[name] = nn.Parameter(
                        torch.empty(
                            0, dtype=torch.uint8, device=layer.w13_trellis.device
                        ),
                        requires_grad=False,
                    )
            return
        # Construct BC_LinearEXL3 objects only after all four packed tensors are
        # resident on their final device.  Each wrapper keeps the compressed
        # storage and never materializes a dense [4096, 2048] expert matrix.
        exl3 = _exl3_module()
        linear_cls = exl3.LinearEXL3
        self._inners = []
        for expert_id in range(layer._exl3_num_experts):
            row: dict[str, Any] = {}
            for projection in _PROJECTIONS:
                is_w13 = projection != "down_proj"
                prefix = "w13_" if is_w13 else "w2_"
                projection_index = 0 if projection == "gate_proj" else 1

                def packed(
                    suffix: str,
                    *,
                    prefix: str = prefix,
                    expert_id: int = expert_id,
                    projection_index: int = projection_index,
                    is_w13: bool = is_w13,
                ):
                    value = getattr(layer, prefix + suffix)[expert_id]
                    return value[projection_index] if is_w13 else value

                row[projection] = linear_cls(
                    config=None,
                    in_features=(4096 if projection != "down_proj" else 2048),
                    out_features=(2048 if projection != "down_proj" else 4096),
                    trellis=packed("trellis"),
                    suh=packed("suh"),
                    svh=packed("svh"),
                    mcg=packed("mcg"),
                    out_dtype=torch.float16,
                    transformers_fix=True,
                )
            self._inners.append(row)
        self._make_fused_bsz1(exl3.ext, layer)

    def _make_fused_bsz1(self, ext, layer: RoutedExperts) -> None:
        """Create ExLlamaV3's bound top-k MoE kernel for decode-sized batches."""
        assert self._inners is not None
        device = layer.w13_trellis.device
        topk = int(layer.moe_config.experts_per_token)
        if topk > _TEMP_ROWS_GRAPH:
            raise ValueError(f"EXL3 top-k {topk} exceeds {_TEMP_ROWS_GRAPH}")

        def projection_ptrs(projection: str):
            linears = [row[projection] for row in self._inners]

            def pointers(name: str):
                return torch.tensor(
                    [getattr(linear, name).data_ptr() for linear in linears],
                    dtype=torch.long,
                    device=device,
                )

            first = linears[0]
            if any(linear.K != first.K for linear in linears[1:]):
                raise ValueError(f"mixed EXL3 K values in {projection}")
            if any(linear.mcg != first.mcg for linear in linears[1:]):
                raise ValueError(f"mixed EXL3 MCG flags in {projection}")
            if any(linear.mul1 != first.mul1 for linear in linears[1:]):
                raise ValueError(f"mixed EXL3 mul1 flags in {projection}")
            return (
                linears,
                pointers("trellis"),
                pointers("suh"),
                pointers("svh"),
                first.K,
                first.mcg,
                first.mul1,
            )

        gate = projection_ptrs("gate_proj")
        up = projection_ptrs("up_proj")
        down = projection_ptrs("down_proj")
        temp_hidden, temp_intermediate, temp_activation, temp_output, dq_up, dq_down = (
            _fused_buffers(device)
        )
        hidden_size = layer._exl3_hidden_size
        intermediate_size = layer._exl3_intermediate_size
        yh = temp_hidden[:topk].view(topk, 1, hidden_size)
        intermediate_gate = temp_intermediate[:topk].view(topk, 1, intermediate_size)
        intermediate_up = temp_intermediate[topk : topk * 2].view(
            topk, 1, intermediate_size
        )
        intermediate_activation = temp_activation[:topk].view(
            topk, 1, intermediate_size
        )
        output = temp_output[:topk].view(topk, 1, hidden_size)
        gu_trellis = torch.stack((gate[1], up[1]), dim=1).contiguous()
        gu_suh = torch.stack((gate[2], up[2]), dim=1).contiguous()
        gu_svh = torch.stack((gate[3], up[3]), dim=1).contiguous()
        # The raw decode path addresses the concatenated table with
        # ``[gate_ids, up_ids + num_experts]``.  A single 16-way mgemm then
        # replaces two otherwise identical 8-way cooperative launches.
        raw_gu = (
            torch.cat((gate[1], up[1])).contiguous(),
            torch.cat((gate[2], up[2])).contiguous(),
            torch.cat((gate[3], up[3])).contiguous(),
            gate[4],
            gate[5],
            gate[6],
        )
        raw_gu_had = temp_hidden[: topk * 2].view(topk * 2, 1, hidden_size)
        raw_gu_output = temp_intermediate[: topk * 2].view(
            topk * 2, 1, intermediate_size
        )
        raw_gu_ids = torch.empty((1, topk * 2), dtype=torch.long, device=device)
        self._fused_bsz1 = ext.BC_BlockSparseMLP(
            temp_hidden,
            yh,
            temp_intermediate,
            intermediate_gate,
            intermediate_up,
            intermediate_activation,
            temp_activation,
            output,
            temp_output,
            None,
            None,
            dq_up,
            dq_down,
            -1,
            -1,
            gate[1],
            gate[2],
            gate[3],
            gate[4],
            gate[5],
            gate[6],
            up[1],
            up[2],
            up[3],
            up[4],
            up[5],
            up[6],
            down[1],
            down[2],
            down[3],
            down[4],
            down[5],
            down[6],
            True,
            False,
            None,
            None,
            10.0,
            [linear.bc for linear in gate[0]],
            [linear.bc for linear in up[0]],
            [linear.bc for linear in down[0]],
            gu_trellis,
            gu_suh,
            gu_svh,
        )
        self._fused_output = output[0].reshape(-1)
        self._fused_batch_output = temp_output
        # Same three-kernel route as BC_BlockSparseMLP.run_bsz1(), exposed
        # without ExLlamaV3's private CUDA Graph wrapper.  This lets vLLM own
        # the outer full-decode graph instead of trying to nest graph replay.
        self._raw_bsz1 = (
            ext,
            yh,
            intermediate_gate,
            intermediate_up,
            intermediate_activation,
            output,
            gate[1:7],
            up[1:7],
            down[1:7],
            raw_gu_had,
            raw_gu_output,
            raw_gu,
            layer._exl3_num_experts,
            raw_gu_ids,
        )
        self._fused_moe = (
            ext,
            _fused_moe_buffers(ext, device),
            gate[1:7],
            up[1:7],
            down[1:7],
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        # No vLLM fused-MoE kernel consumes EXL3 tensors.  Returning None keeps
        # the runner on its modular dispatch path and calls apply() below.
        return None

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        # ExLlamaV3's bound MoE class reads 64-bit expert indices. Passing
        # vLLM's common int32 representation is an out-of-bounds ABI mismatch.
        return torch.int64

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        if self._inners is None and self._marlin is None:
            self.process_weights_after_loading(layer)
        rows = x.reshape(-1, x.shape[-1]).to(torch.float16)
        if self._marlin is not None and rows.shape[0] <= _TEMP_ROWS_MARLIN:
            from vllm.model_executor.layers.fused_moe.activation import (
                ApplyMoEActivationConfig,
            )
            from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
                fused_marlin_moe,
            )
            from vllm.scalar_type import scalar_types

            w13, w2, s13, s2, workspace, scratch = self._marlin
            cache13, cache2 = scratch
            result = torch.empty_like(rows)
            fused_marlin_moe(
                rows,
                w13,
                w2,
                None,
                None,
                s13,
                s2,
                topk_weights.to(torch.float32),
                topk_ids,
                quant_type_id=scalar_types.uint4b8.id,
                global_num_experts=layer._exl3_num_experts,
                workspace=workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=result,
                activation=layer.moe_config.activation,
                activation_config=ApplyMoEActivationConfig(
                    clamp_limit=layer.swiglu_limit,
                    alpha=1.0 if layer.swiglu_alpha is None else layer.swiglu_alpha,
                    beta=0.0 if layer.swiglu_beta is None else layer.swiglu_beta,
                ),
            )
            return result.to(x.dtype).reshape_as(x)
        assert self._inners is not None
        if (
            rows.shape[0] <= _TEMP_ROWS_RAW_GRAPH
            and self._raw_bsz1 is not None
            and os.environ.get("VLLM_EXL3_OUTER_GRAPH", "0") == "1"
        ):
            ids = topk_ids.to(dtype=torch.long).contiguous()
            weights = topk_weights.to(dtype=torch.float16).contiguous()
            if len(self._raw_bsz1) == 9:
                # Compatibility with callers/tests that construct the raw
                # tuple directly (before the optional fused gate/up fields).
                (
                    ext,
                    yh,
                    gate_out,
                    up_out,
                    activation,
                    output,
                    gate,
                    up,
                    down,
                ) = self._raw_bsz1
                gu_had = gu_out = gu = gu_ids = None
                num_experts = 0
            else:
                (
                    ext,
                    yh,
                    gate_out,
                    up_out,
                    activation,
                    output,
                    gate,
                    up,
                    down,
                    gu_had,
                    gu_out,
                    gu,
                    num_experts,
                    gu_ids,
                ) = self._raw_bsz1
            # The optimized extension interprets force_num_sms=-1 as an
            # explicit promise that every selected gate/up matrix has the
            # same input Hadamard scale.  That property is checked when the
            # checkpoint is loaded; the stock extension remains the default.
            shared_input_had = -1 if _shared_input_had_enabled() else 0
            result = (
                None
                if rows.shape[0] == 1
                else torch.empty(
                    (rows.shape[0], layer._exl3_hidden_size),
                    device=x.device,
                    dtype=torch.float32,
                )
            )
            # ``exl3_mgemm`` represents one source token as a batch of its
            # top-k expert assignments.  Decode verification (and MTP) can
            # present a few source tokens at once, so replay the same raw
            # three-kernel sequence per row.  This remains capture-safe under
            # vLLM's outer graph and avoids BC_BlockSparseMLP trying to create
            # its own nested CUDA graph for rows > 1.
            for row_idx in range(rows.shape[0]):
                row = rows[row_idx : row_idx + 1].unsqueeze(0)
                row_ids = ids[row_idx : row_idx + 1]
                row_weights = weights[row_idx : row_idx + 1]
                if (
                    gu is not None
                    and os.environ.get("VLLM_EXL3_FUSED_GATE_UP", "1") != "0"
                ):
                    gu_ids[:, : row_ids.shape[1]].copy_(row_ids)
                    gu_ids[:, row_ids.shape[1] :].copy_(row_ids + num_experts)
                    ext.exl3_mgemm(
                        row,
                        gu[0],
                        gu_out,
                        gu[1],
                        gu_had,
                        gu[2],
                        gu_ids,
                        None,
                        gu[3],
                        -1,
                        gu[4],
                        gu[5],
                        -1,
                        -1,
                        shared_input_had,
                    )
                else:
                    ext.exl3_mgemm(
                        row,
                        gate[0],
                        gate_out,
                        gate[1],
                        yh,
                        gate[2],
                        row_ids,
                        None,
                        gate[3],
                        -1,
                        gate[4],
                        gate[5],
                        -1,
                        -1,
                        shared_input_had,
                    )
                    ext.exl3_mgemm(
                        row,
                        up[0],
                        up_out,
                        up[1],
                        yh,
                        up[2],
                        row_ids,
                        None,
                        up[3],
                        -1,
                        up[4],
                        up[5],
                        -1,
                        -1,
                        shared_input_had,
                    )
                ext.silu_mul(gate_out, up_out, activation, 10.0)
                ext.exl3_mgemm(
                    activation,
                    down[0],
                    output,
                    down[1],
                    activation,
                    down[2],
                    row_ids,
                    row_weights,
                    down[3],
                    -1,
                    down[4],
                    down[5],
                    -1,
                    -1,
                    0,
                )
                if result is not None:
                    result[row_idx].copy_(output[0, 0])
            if result is None:
                result = output[0]
            return result.to(x.dtype).reshape_as(x)

        if (
            self._fused_moe is not None
            and rows.shape[0] >= 4
            and rows.shape[0] * topk_ids.shape[1] <= _TEMP_ROWS_FUSED
            and os.environ.get("VLLM_EXL3_FUSED_MOE", "1") != "0"
        ):
            ids = topk_ids.to(dtype=torch.long).contiguous()
            weights = topk_weights.to(dtype=torch.float16).contiguous()
            flat_ids = ids.reshape(-1)
            flat_weights = weights.reshape(-1)
            order = torch.argsort(flat_ids)
            # Assignments were flattened token-major, so the original flat
            # position divided by top-k is the source token row.
            token_sorted = torch.div(
                order, ids.shape[1], rounding_mode="floor"
            ).contiguous()
            weight_sorted = flat_weights.index_select(0, order).contiguous()
            expert_count = torch.bincount(
                flat_ids, minlength=layer._exl3_num_experts + 1
            )
            result = torch.zeros(
                (rows.shape[0], 4096), device=x.device, dtype=torch.float32
            )
            ext, buffers, gate, up, down = self._fused_moe
            ext.exl3_moe(
                rows,
                result,
                expert_count,
                token_sorted,
                weight_sorted,
                *buffers,
                0,  # SiLU
                gate[3],
                up[3],
                down[3],
                gate[0],
                gate[1],
                gate[2],
                up[0],
                up[1],
                up[2],
                down[0],
                down[1],
                down[2],
                gate[4],
                gate[5],
                up[4],
                up[5],
                down[4],
                down[5],
                10.0,
            )
            return result.to(x.dtype).reshape_as(x)

        if self._fused_bsz1 is not None and rows.shape[0] <= _TEMP_ROWS_GRAPH:
            ids = topk_ids.to(dtype=torch.long).contiguous()
            weights = topk_weights.to(dtype=torch.float16).contiguous()
            result = torch.zeros(
                (rows.shape[0], 4096), device=x.device, dtype=torch.float32
            )
            assert self._fused_output is not None
            if rows.shape[0] == 1:
                self._fused_bsz1.run_bsz1(rows, ids, weights)
                result.copy_(self._fused_output)
                return result.to(x.dtype).reshape_as(x)

            if torch.cuda.is_current_stream_capturing():
                # ``unique().tolist()`` synchronizes the device and cannot be
                # captured. Decode graph sizes are static, so an unrolled
                # per-row top-k call is deterministic and capture-safe.
                for row_idx in range(rows.shape[0]):
                    self._fused_bsz1.run_bsz1(
                        rows[row_idx : row_idx + 1],
                        ids[row_idx : row_idx + 1],
                        weights[row_idx : row_idx + 1],
                    )
                    result[row_idx].copy_(self._fused_output)
                return result.to(x.dtype).reshape_as(x)

            # For a small decode batch, group assignments by expert and use
            # the bound three-projection kernel once per expert. This avoids
            # the 3 * (tokens * top-k) Python/kernel launches of the fallback.
            assert self._fused_batch_output is not None
            can_fuse = True
            for expert_id in torch.unique(ids).tolist():
                positions = (ids == expert_id).nonzero(as_tuple=False)
                token_ids = positions[:, 0]
                choice_ids = positions[:, 1]
                current = rows.index_select(0, token_ids)
                count = current.shape[0]
                if count > _TEMP_ROWS_GRAPH:
                    # A large prefill assignment is uncommon in this branch;
                    # leave it to the proven per-expert LinearEXL3 fallback.
                    can_fuse = False
                    break
                self._fused_bsz1.run_single_expert(current, int(expert_id))
                current = self._fused_batch_output[:count]
                current.mul_(
                    weights[token_ids, choice_ids].to(torch.float32).unsqueeze(1)
                )
                result.index_add_(0, token_ids, current)
            # If any expert exceeded the bound scratch size, recompute the
            # whole batch through the general path below rather than returning
            # a partial result.
            if can_fuse:
                return result.to(x.dtype).reshape_as(x)

        result = torch.zeros(
            (rows.shape[0], 4096), device=x.device, dtype=torch.float32
        )
        # Group token assignments by expert to amortize kernel launch overhead.
        for expert_id in torch.unique(topk_ids).tolist():
            positions = (topk_ids == expert_id).nonzero(as_tuple=False)
            token_ids = positions[:, 0]
            choice_ids = positions[:, 1]
            current = rows.index_select(0, token_ids)
            expert = self._inners[int(expert_id)]
            gate = expert["gate_proj"].forward(current, {}, out_dtype=torch.float16)
            up = expert["up_proj"].forward(current, {}, out_dtype=torch.float16)
            current = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
                -10.0, 10.0
            )
            current = expert["down_proj"].forward(current, {}, out_dtype=torch.float32)
            current = current * topk_weights[token_ids, choice_ids, None].to(
                current.dtype
            )
            result.index_add_(0, token_ids, current)
        return result.to(x.dtype).reshape_as(x)
