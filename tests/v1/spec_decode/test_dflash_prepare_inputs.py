# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    _prepare_dflash_inputs_kernel,
    prepare_dflash_inputs,
)


def test_request_metadata_alignment_does_not_specialize() -> None:
    expected = {
        "out_input_ids_ptr",
        "out_query_positions_ptr",
        "out_query_start_loc_ptr",
        "out_seq_lens_ptr",
        "out_query_slot_mapping_ptr",
        "out_context_positions_ptr",
        "out_context_slot_mapping_ptr",
        "out_sample_indices_ptr",
        "out_sample_pos_ptr",
        "out_sample_idx_mapping_ptr",
        "out_temperature_ptr",
        "out_seeds_ptr",
        "target_positions_ptr",
        "target_query_start_loc_ptr",
        "idx_mapping_ptr",
        "last_sampled_ptr",
        "next_prefill_tokens_ptr",
        "num_sampled_ptr",
        "num_rejected_ptr",
        "temperature_ptr",
        "seeds_ptr",
        "block_table_ptr",
    }
    assert set(_prepare_dflash_inputs_kernel.do_not_specialize_on_alignment) == expected


def test_warmup_input_preparation_covers_power_of_two_classes(monkeypatch) -> None:
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        dflash,
        "prepare_dflash_inputs",
        lambda *args: calls.append(
            (int(args[9].num_scheduled_tokens.max()), args[22])
        ),
    )

    max_num_tokens = 256
    input_buffers = InputBuffers(6, max_num_tokens, torch.device("cpu"))
    speculator = object.__new__(DFlashSpeculator)
    speculator.device = torch.device("cpu")
    speculator.target_input_buffers = input_buffers
    speculator.input_buffers = input_buffers
    speculator.draft_kv_cache_group_ids = [0]
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros((1, max_num_tokens), dtype=torch.int64),
        input_block_tables=[torch.zeros((6, 4), dtype=torch.int32)],
        kernel_block_sizes=[16],
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    speculator.context_positions = torch.zeros(max_num_tokens, dtype=torch.int64)
    speculator._context_slot_mappings = torch.zeros(
        (1, max_num_tokens), dtype=torch.int64
    )
    speculator.sample_indices = torch.zeros(12, dtype=torch.int64)
    speculator.sample_pos = torch.zeros(12, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.zeros(12, dtype=torch.int32)
    speculator.temperature = torch.zeros(6)
    speculator.seeds = torch.zeros(6, dtype=torch.int64)
    speculator.parallel_drafting_token_id = 0
    speculator.num_query_per_req = 3
    speculator.num_speculative_steps = 2
    speculator.max_num_reqs = 6
    speculator.max_num_tokens = max_num_tokens
    speculator.max_model_len = 524288
    speculator.sample_from_anchor = False

    speculator.warmup_input_preparation(
        torch.zeros(6, dtype=torch.int64),
        torch.zeros(6, dtype=torch.int32),
        torch.zeros(6),
        torch.zeros(6, dtype=torch.int64),
    )

    assert calls == [(126, 3), (62, 3), (30, 3), (14, 3), (6, 3), (2, 3), (1, 3)]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
):
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3

    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    sample_pos = torch.full_like(sample_indices, -1)
    sample_idx_mapping = torch.full(
        sample_indices.shape, -1, dtype=torch.int32, device=device
    )
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    input_temperature = torch.tensor(
        [0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device
    )
    input_seeds = torch.tensor([0, 0, 17, 0], dtype=torch.int64, device=device)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


def test_prepare_dflash_inputs_excludes_rejected_context_suffix():
    # Positions 10/11 use physical block 7. Rejected positions 12/13 would use
    # block 8, but must be PAD context rather than contaminating draft KV.
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]

    # The replacement query starts immediately after the two valid rows and
    # advances from the last accepted position (11).
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.positions[:3].cpu().tolist() == [12, 13, 14]
    assert out.query_slot_mapping[:3].tolist() == [32, 33, 34]
    assert out.sample_indices[:3].tolist() == [0, 1, 2]
    assert out.sample_pos[:3].tolist() == [13, 14, 15]
    assert out.sample_idx_mapping[:3].tolist() == [2, 2, 2]
    assert out.temperature[2].item() == 1.0
    assert out.seeds[2].item() == 17


def test_prepare_dflash_inputs_excludes_rejected_context_suffix_with_dcp():
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 7, 8, 9],
        cp_rank=1,
        cp_size=2,
        cp_interleave=2,
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slot_mapping[:4].tolist() == [28, 29, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out.query_slot_mapping[:3].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID, 30]


def test_prepare_dflash_inputs_never_writes_the_null_block():
    # The valid context uses logical block 0 and the replacement query uses
    # logical block 1. Both map to the null block and must remain unwritable.
    out = _run_prepare(
        target_positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_slot_mapping[:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out.query_slot_mapping[:3].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
