# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


def test_mamba_group_info_refreshes_after_pp_projection_changes() -> None:
    state = object.__new__(MambaHybridModelState)
    state._mamba_ctx = None
    state._mamba_group_ids = []
    state._mamba_spec = None
    state._mamba_group_signature = ()
    spec = MambaSpec(
        block_size=256,
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
    )

    def make_config(num_groups: int) -> KVCacheConfig:
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"layer.{i}"], spec) for i in range(num_groups)
            ],
        )

    group_ids, _ = state._get_mamba_group_info(make_config(5))
    assert group_ids == [0, 1, 2, 3, 4]

    state._mamba_ctx = Mock()
    group_ids, _ = state._get_mamba_group_info(make_config(4))

    assert group_ids == [0, 1, 2, 3]
    assert state._mamba_ctx is None


def test_prepare_attn_forwards_positions_to_sparse_metadata(monkeypatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = False
    state.recoverssm = None
    state.max_model_len = 512
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)

    positions = torch.arange(8, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=8,
        num_tokens_after_padding=8,
        query_start_loc_np=torch.tensor([0, 8], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([8], dtype=torch.int32).numpy(),
        seq_lens_cpu_upper_bound=torch.tensor([8], dtype=torch.int32),
        is_prefilling_np=torch.tensor([True]).numpy(),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        dcp_local_seq_lens=None,
        prompt_lens=None,
        positions=positions,
    )
    build = Mock(return_value={})
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata", build
    )

    state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty((0, 8), dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )

    assert build.call_args.kwargs["positions"] is positions


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
