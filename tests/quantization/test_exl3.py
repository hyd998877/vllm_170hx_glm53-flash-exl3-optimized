# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3MoEMethod,
    _marlin_sidecar_path,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)


def test_exl3_config_accepts_only_selective_glm53_format() -> None:
    config = Exl3Config.from_config(
        {
            "quant_method": "exl3",
            "bits": 4,
            "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
        }
    )
    assert config.get_name() == "exl3"

    for override in (
        {"quant_method": "gptq"},
        {"bits": 3},
        {"codebook": "other"},
        {"scope": "whole_model"},
    ):
        invalid = {
            "quant_method": "exl3",
            "bits": 4,
            "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
            **override,
        }
        with pytest.raises(ValueError):
            Exl3Config.from_config(invalid)


def test_exl3_requires_tp1_and_requests_int64_expert_ids() -> None:
    method = Exl3MoEMethod(Exl3Config(), SimpleNamespace(tp_size=1))
    assert method.topk_indices_dtype == torch.int64

    with pytest.raises(ValueError, match="requires TP=1"):
        Exl3MoEMethod(Exl3Config(), SimpleNamespace(tp_size=2))


def test_exl3_can_online_quantize_non_routed_linears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXL3_NON_ROUTED_FP8", "1")
    layer = object.__new__(LinearBase)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with set_current_vllm_config(config):
        method = Exl3Config().get_quant_method(layer, "model.layers.3.self_attn")

    assert isinstance(method, Fp8PerTensorOnlineLinearMethod)
    assert method.uses_meta_device


@pytest.mark.parametrize(
    ("prefix", "expected_fp8"),
    [
        ("model.layers.0.mlp.gate_proj", True),
        ("model.layers.2.mlp.down_proj", True),
        ("model.layers.3.mlp.shared_experts.up_proj", True),
        ("model.layers.44.mlp.shared_experts.down_proj", True),
        ("model.layers.3.self_attn.o_proj", False),
        ("model.layers.3.mlp.gate", False),
        ("lm_head", False),
    ],
)
def test_exl3_official_non_routed_fp8_scope(
    monkeypatch: pytest.MonkeyPatch, prefix: str, expected_fp8: bool
) -> None:
    monkeypatch.setenv("VLLM_EXL3_NON_ROUTED_FP8", "1")
    monkeypatch.setenv("VLLM_EXL3_NON_ROUTED_FP8_SCOPE", "official")
    from vllm.model_executor.layers.quantization.exl3 import (
        _use_online_fp8_for_linear,
    )

    assert _use_online_fp8_for_linear(prefix) is expected_fp8


def test_exl3_custom_loader_maps_split_projections() -> None:
    layer = nn.Module()
    layer.register_parameter(
        "w13_mcg", nn.Parameter(torch.zeros(2, 2, 1), requires_grad=False)
    )
    layer.register_parameter(
        "w2_mcg", nn.Parameter(torch.zeros(2, 1), requires_grad=False)
    )
    method = Exl3MoEMethod(Exl3Config(), SimpleNamespace(tp_size=1))

    loaded = method.load_weights(
        layer,
        [
            ("0.gate_proj.mcg", torch.tensor([11], dtype=torch.int32)),
            ("0.up_proj.mcg", torch.tensor([12], dtype=torch.int32)),
            ("1.down_proj.mcg", torch.tensor([13], dtype=torch.int32)),
        ],
    )

    assert loaded == {"w13_mcg", "w2_mcg"}
    torch.testing.assert_close(layer.w13_mcg[0, :, 0], torch.tensor([11.0, 12.0]))
    torch.testing.assert_close(layer.w2_mcg[1], torch.tensor([13.0]))


def test_exl3_marlin_layer_range_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("VLLM_EXL3_MARLIN_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_EXL3_MARLIN_LAYERS", "3-5,8")
    for index in (3, 4, 5, 8, 9):
        (tmp_path / f"layer-{index:02d}.safetensors").touch()

    assert _marlin_sidecar_path(SimpleNamespace(layer_name="model.layers.3.mlp"))
    assert _marlin_sidecar_path(SimpleNamespace(layer_name="model.layers.8.mlp"))
    assert (
        _marlin_sidecar_path(SimpleNamespace(layer_name="model.layers.9.mlp")) is None
    )


def test_exl3_outer_graph_path_handles_six_request_dflash_verify_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExt:
        def __init__(self) -> None:
            self.calls = 0
            self.down_calls = 0

        def exl3_mgemm(self, *args) -> None:
            self.calls += 1
            output = args[2]
            weights = args[7]
            if weights is not None:
                self.down_calls += 1
                output[0].fill_(self.down_calls)

        @staticmethod
        def silu_mul(*args) -> None:
            pass

    monkeypatch.setenv("VLLM_EXL3_OUTER_GRAPH", "1")
    method = Exl3MoEMethod(Exl3Config(), SimpleNamespace(tp_size=1))
    method._inners = []
    ext = FakeExt()
    topk = 8
    yh = torch.empty(topk, 1, 4096, dtype=torch.float16)
    gate_out = torch.empty(topk, 1, 2048, dtype=torch.float16)
    up_out = torch.empty_like(gate_out)
    activation = torch.empty_like(gate_out)
    output = torch.empty(topk, 1, 4096, dtype=torch.float32)
    projection = (None, None, None, 4, False, False)
    method._raw_bsz1 = (
        ext,
        yh,
        gate_out,
        up_out,
        activation,
        output,
        projection,
        projection,
        projection,
    )
    layer = SimpleNamespace(_exl3_hidden_size=4096)
    # Six requests times eight target verification tokens.  This deliberately
    # exceeds BC_BlockSparseMLP's 32-row internal graph-table limit.
    rows = 6 * 8
    x = torch.zeros(rows, 4096, dtype=torch.float16)
    ids = torch.zeros(rows, topk, dtype=torch.int64)
    weights = torch.ones(rows, topk, dtype=torch.float16)

    result = method.apply(layer, x, weights, ids)

    assert ext.calls == rows * 3
    torch.testing.assert_close(result[0], torch.ones(4096, dtype=torch.float16))
    torch.testing.assert_close(
        result[-1], torch.full((4096,), rows, dtype=torch.float16)
    )
