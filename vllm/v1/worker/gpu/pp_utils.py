# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    draft_tokens: torch.Tensor | None  # [num_reqs, num_speculative_steps]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Hidden-state P2P uses adjacent two-rank groups, so the full PP
    `device_group` is reserved for this sampled-token broadcast.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.packed_width = self.max_sample_len + 2 + num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Keep the sampled-feedback collective on the PP communicator.  The
        # hidden-state path uses adjacent two-rank groups in parallel_state;
        # using a single communicator here keeps send/recv ordering identical
        # on all ranks while the side stream overlaps the next forward.
        self.broadcast_group = get_pp_group().device_group

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            draft_tokens=slot.draft_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if input_batch.num_reqs == 0:
            return False
        if need_sampled_mask is None:
            # All PP ranks must enter the same broadcasts even when all
            # requests finish on this step. Keep a false mask so the values
            # received below are ignored by the deferred postprocess.
            need_sampled_mask = np.zeros(input_batch.num_reqs, dtype=bool)

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # All payloads have a fixed int64 row layout so the sender and
            # receivers issue exactly one NCCL operation per pipeline step:
            # [sampled(max_sample_len), num_sampled, num_rejected,
            #  draft(num_speculative_steps)].  The counts are widened to
            # int64 only on the wire and converted back to int32 views below.
            msl = self.max_sample_len
            packed = torch.empty(
                num_reqs, self.packed_width, dtype=torch.int64, device=self.device
            )
            torch.distributed.broadcast(
                packed, src=self.last_rank, group=self.broadcast_group
            )
            sampled_tokens = packed[:, :msl].contiguous()
            num_sampled = packed[:, msl].to(torch.int32)
            num_rejected = packed[:, msl + 1].to(torch.int32)
            draft_tokens = (
                packed[:, msl + 2 :] if self.num_speculative_steps else None
            )
            event = self.broadcast_stream.record_event()
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            packed.record_stream(self.main_stream)
            sampled_tokens.record_stream(self.main_stream)
            num_sampled.record_stream(self.main_stream)
            num_rejected.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            draft_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if input_batch.num_reqs == 0:
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        # The sampler and request-state tensors are persistent buffers. Snapshot
        # every payload into one temporary row before handing it to the side
        # stream; record_stream() alone cannot prevent a subsequent sampler
        # step from mutating a persistent source buffer.
        msl = self.max_sample_len
        num_reqs = sampled_token_ids.shape[0]
        width = sampled_token_ids.shape[1]
        assert width <= msl
        packed = torch.full(
            (num_reqs, self.packed_width), -1, dtype=torch.int64, device=self.device
        )
        packed[:, :width].copy_(sampled_token_ids)
        packed[:, msl] = num_sampled.to(torch.int64)
        packed[:, msl + 1] = num_rejected.to(torch.int64)
        if self.num_speculative_steps:
            assert draft_token_ids is not None
            assert draft_token_ids.shape == (num_reqs, self.num_speculative_steps)
            packed[:, msl + 2 :].copy_(draft_token_ids.to(torch.int64))

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                packed, src=self.last_rank, group=self.broadcast_group
            )
            packed.record_stream(self.broadcast_stream)
