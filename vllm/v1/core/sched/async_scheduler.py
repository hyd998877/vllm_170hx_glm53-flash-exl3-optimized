# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size
        self._pp_decode_phase_policy = envs.VLLM_PP_DECODE_PHASE_POLICY
        if self._pp_decode_phase_policy not in ("native", "pairpack", "pack"):
            raise ValueError(
                "VLLM_PP_DECODE_PHASE_POLICY must be native, pairpack, or pack"
            )
        self._pp_decode_phases: dict[str, int] = {}
        if self._pp_decode_phase_policy != "native":
            logger.info(
                "Async PP decode phase policy enabled: %s",
                self._pp_decode_phase_policy,
            )
        self._pp_prefill_cohort_barrier = envs.VLLM_PP_PREFILL_COHORT_BARRIER
        self._pp_prefill_cohort_size = envs.VLLM_PP_PREFILL_COHORT_SIZE
        self._pp_prefill_cohort_min_tokens = (
            envs.VLLM_PP_PREFILL_COHORT_MIN_TOKENS
        )
        self._pp_prefill_cohort_ids: set[str] = set()
        self._pp_prefill_cohort_held: set[str] = set()
        self._pp_prefill_cohort_ready: set[str] = set()
        self._pp_prefill_cohort_final_pending: set[str] = set()
        self._pp_prefill_cohort_finalizing = False
        self._pp_prefill_cohort_released = False
        self._pp_prefill_final_chunk = (
            self.scheduler_config.long_prefill_token_threshold
        )
        self._adaptive_prefill_enabled = envs.VLLM_PP_ADAPTIVE_PREFILL
        speculative_config = self.vllm_config.speculative_config
        draft_slots = (
            speculative_config.max_num_new_slots_for_drafting
            if speculative_config is not None
            else 0
        )
        self._adaptive_prefill_max_tokens = min(
            envs.VLLM_PP_ADAPTIVE_PREFILL_MAX_TOKENS,
            self.scheduler_config.max_num_batched_tokens - draft_slots,
            self.max_num_scheduled_tokens,
        )
        self._adaptive_prefill_busy_tokens = min(
            envs.VLLM_PP_ADAPTIVE_PREFILL_BUSY_TOKENS,
            self.scheduler_config.max_num_batched_tokens,
            self.max_num_scheduled_tokens,
        )
        self._adaptive_prefill_request_id: str | None = None
        self._balanced_prefill_ids: set[str] = set()
        self._min_prefill_tokens_by_priority: dict[int, int] = {}
        if self._adaptive_prefill_enabled:
            if self.scheduler_config.long_prefill_token_threshold <= 0:
                raise ValueError(
                    "--long-prefill-token-threshold must be positive when "
                    "adaptive prefill is enabled"
                )
            if self._adaptive_prefill_max_tokens <= 0:
                raise ValueError("VLLM_PP_ADAPTIVE_PREFILL_MAX_TOKENS must be positive")
            if self._adaptive_prefill_busy_tokens <= draft_slots:
                raise ValueError(
                    "VLLM_PP_ADAPTIVE_PREFILL_BUSY_TOKENS must exceed the "
                    "speculative draft-slot reservation"
                )
            logger.info(
                "Adaptive PP prefill enabled: idle_chunk=%d busy_chunk=%d "
                "busy_budget=%d",
                self._adaptive_prefill_max_tokens,
                self.scheduler_config.long_prefill_token_threshold,
                self._adaptive_prefill_busy_tokens,
            )
        if self._pp_prefill_cohort_barrier:
            if self._pp_prefill_cohort_size < 2:
                raise ValueError("VLLM_PP_PREFILL_COHORT_SIZE must be at least 2")
            if self._pp_prefill_cohort_min_tokens <= 0:
                raise ValueError(
                    "VLLM_PP_PREFILL_COHORT_MIN_TOKENS must be positive"
                )
            if self._pp_prefill_final_chunk <= 0:
                raise ValueError(
                    "--long-prefill-token-threshold must be positive when "
                    "prefill cohort barrier is enabled"
                )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._update_adaptive_prefill_request()
        self._balance_adaptive_prefills()
        if self._pp_prefill_cohort_barrier:
            self._update_prefill_cohort_barrier()
        scheduler_output = super().schedule(throttle_prefills)
        if self._should_release_stalled_prefill_cohort(scheduler_output):
            # The configured cohort cannot coexist in the current KV pool.
            # Fall back to native progress instead of holding resident members
            # forever while another cohort member waits for their blocks.
            self._pp_prefill_cohort_held.clear()
            self._pp_prefill_cohort_ready.clear()
            self._pp_prefill_cohort_released = True
            self._pp_prefill_cohort_finalizing = False
            logger.warning(
                "PP prefill cohort barrier released early due to KV capacity"
            )
            scheduler_output = super().schedule(throttle_prefills)
        return scheduler_output

    def _update_adaptive_prefill_request(self) -> None:
        previous_request_id = self._adaptive_prefill_request_id
        adaptive_request_id = None
        exit_reason = "disabled"
        if self._adaptive_prefill_enabled:
            active_requests = [
                request
                for request in self.requests.values()
                if not request.is_finished()
            ]
            if len(active_requests) == 1:
                request = active_requests[0]
                if (
                    request.num_output_tokens == 0
                    and request.num_computed_tokens < request.num_prompt_tokens
                    and request.status
                    in (
                        RequestStatus.WAITING,
                        RequestStatus.RUNNING,
                        RequestStatus.PREEMPTED,
                    )
                ):
                    adaptive_request_id = request.request_id
                else:
                    exit_reason = "decode-or-non-runnable"
            elif len(active_requests) > 1:
                exit_reason = (
                    "decode"
                    if any(
                        request.num_output_tokens > 0
                        or request.num_computed_tokens >= request.num_prompt_tokens
                        for request in active_requests
                    )
                    else "concurrency"
                )
            else:
                exit_reason = "idle"

        self._adaptive_prefill_request_id = adaptive_request_id
        request_changed = adaptive_request_id != previous_request_id
        if previous_request_id is not None and request_changed:
            logger.info(
                "Adaptive PP prefill exited: request=%s reason=%s chunk=%d",
                previous_request_id,
                exit_reason,
                self.scheduler_config.long_prefill_token_threshold,
            )
        if adaptive_request_id is not None and request_changed:
            logger.info(
                "Adaptive PP prefill entered: request=%s chunk=%d",
                adaptive_request_id,
                self._adaptive_prefill_max_tokens,
            )

    def _get_long_prefill_token_threshold(self, request: Request) -> int:
        if request.request_id == self._adaptive_prefill_request_id:
            return self._adaptive_prefill_max_tokens
        return super()._get_long_prefill_token_threshold(request)

    def _get_token_budget(self) -> int:
        if (
            self._adaptive_prefill_enabled
            and self._adaptive_prefill_request_id is None
        ):
            return self._adaptive_prefill_busy_tokens
        return super()._get_token_budget()

    def _get_input_budget(self) -> int:
        if (
            self._adaptive_prefill_enabled
            and self._adaptive_prefill_request_id is None
        ):
            return self._adaptive_prefill_busy_tokens
        return super()._get_input_budget()

    def _active_adaptive_prefills(self) -> list[Request]:
        if not self._adaptive_prefill_enabled:
            return []
        return [
            request
            for request in self.requests.values()
            if not request.is_finished()
            and request.num_output_tokens == 0
            and request.num_computed_tokens < request.num_prompt_tokens
            and request.status
            in (
                RequestStatus.WAITING,
                RequestStatus.RUNNING,
                RequestStatus.PREEMPTED,
            )
        ]

    def _adaptive_prefill_priority(self, request: Request) -> int:
        return request.priority if self.policy == SchedulingPolicy.PRIORITY else 0

    def _balance_adaptive_prefills(self) -> None:
        """Keep concurrent prefills within one scheduling quantum."""
        self._balanced_prefill_ids.clear()
        self._min_prefill_tokens_by_priority.clear()
        prefills = self._active_adaptive_prefills()
        if len(prefills) < 2:
            return

        priority_counts = Counter(
            self._adaptive_prefill_priority(request) for request in prefills
        )
        self._balanced_prefill_ids.update(
            request.request_id
            for request in prefills
            if priority_counts[self._adaptive_prefill_priority(request)] > 1
        )
        for request in prefills:
            if request.request_id not in self._balanced_prefill_ids:
                continue
            priority = self._adaptive_prefill_priority(request)
            previous = self._min_prefill_tokens_by_priority.get(priority)
            if previous is None or request.num_computed_tokens < previous:
                self._min_prefill_tokens_by_priority[priority] = (
                    request.num_computed_tokens
                )

        active_requests = [
            request
            for request in self.requests.values()
            if not request.is_finished()
        ]
        if len(prefills) != len(active_requests):
            return

        prefill_ids = {request.request_id for request in prefills}
        self.running.sort(
            key=lambda request: (
                self._adaptive_prefill_priority(request),
                0 if request.request_id in prefill_ids else 1,
                request.num_computed_tokens,
                request.arrival_time,
            )
        )

    def _should_yield_adaptive_prefill(self, request: Request) -> bool:
        if request.request_id not in self._balanced_prefill_ids:
            return False
        priority = self._adaptive_prefill_priority(request)
        min_computed_tokens = self._min_prefill_tokens_by_priority[priority]
        return request.num_computed_tokens > min_computed_tokens

    def _should_release_stalled_prefill_cohort(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        if (
            not self._pp_prefill_cohort_barrier
            or self._pp_prefill_cohort_released
            or self._pp_prefill_cohort_finalizing
            or not self._pp_prefill_cohort_held
            or scheduler_output.total_num_scheduled_tokens > 0
        ):
            return False
        return any(
            request.request_id in self._pp_prefill_cohort_ids
            for queue in (self.waiting, self.skipped_waiting)
            for request in queue
        )

    def _update_prefill_cohort_barrier(self) -> None:
        if self._pp_prefill_cohort_ids and not any(
            req_id in self.requests and not self.requests[req_id].is_finished()
            for req_id in self._pp_prefill_cohort_ids
        ):
            # The previous cohort has drained. Reset all generation-local
            # state so the next independent batch can arm its own barrier.
            self._pp_prefill_cohort_ids.clear()
            self._pp_prefill_cohort_held.clear()
            self._pp_prefill_cohort_ready.clear()
            self._pp_prefill_cohort_final_pending.clear()
            self._pp_prefill_cohort_finalizing = False
            self._pp_prefill_cohort_released = False

        candidates = [
            request
            for request in sorted(
                self.requests.values(), key=lambda request: request.arrival_time
            )
            if not request.is_finished()
            and request.num_output_tokens == 0
            and request.num_prompt_tokens >= self._pp_prefill_cohort_min_tokens
        ]
        if (
            not self._pp_prefill_cohort_ids
            and len(candidates) >= self._pp_prefill_cohort_size
        ):
            self._pp_prefill_cohort_ids = {
                request.request_id
                for request in candidates[: self._pp_prefill_cohort_size]
            }
            self._pp_prefill_cohort_released = False
            logger.info(
                "PP prefill cohort barrier armed: cohort=%d",
                len(self._pp_prefill_cohort_ids),
            )

        if not self._pp_prefill_cohort_ids or self._pp_prefill_cohort_released:
            return
        cohort = [
            self.requests[req_id]
            for req_id in self._pp_prefill_cohort_ids
            if req_id in self.requests and not self.requests[req_id].is_finished()
        ]
        if len(cohort) < 2:
            self._pp_prefill_cohort_ids.clear()
            self._pp_prefill_cohort_held.clear()
            self._pp_prefill_cohort_ready.clear()
            self._pp_prefill_cohort_final_pending.clear()
            self._pp_prefill_cohort_finalizing = False
            return

        # A ready request has only reached its last chunk boundary; in async
        # scheduling its num_computed_tokens is advanced before the forward
        # completes. Keep decode requests held until every final prefill chunk
        # has actually been scheduled. Otherwise the first request can enter
        # decode while the remaining requests are still prefilling.
        if self._pp_prefill_cohort_finalizing:
            if not self._pp_prefill_cohort_final_pending and all(
                request.num_computed_tokens >= request.num_prompt_tokens
                for request in cohort
            ):
                self._pp_prefill_cohort_held.clear()
                self._pp_prefill_cohort_ready.clear()
                self._pp_prefill_cohort_released = True
                self._pp_prefill_cohort_finalizing = False
                logger.info("PP prefill cohort barrier released after final chunks")
            return

        for request in cohort:
            remaining = max(
                request.num_prompt_tokens - request.num_computed_tokens, 0
            )
            if remaining <= self._pp_prefill_final_chunk:
                self._mark_prefill_cohort_ready(request.request_id)

    def _mark_prefill_cohort_ready(self, request_id: str) -> bool:
        if self._pp_prefill_cohort_finalizing:
            return True
        self._pp_prefill_cohort_ready.add(request_id)
        self._pp_prefill_cohort_held.add(request_id)
        if self._pp_prefill_cohort_ids <= self._pp_prefill_cohort_ready:
            self._pp_prefill_cohort_finalizing = True
            self._pp_prefill_cohort_held.clear()
            logger.info("PP prefill cohort final chunks unblocked")
            return True
        return False

    def _should_defer_prefill_chunk(self, request: Request) -> bool:
        cohort_deferred = (
            self._pp_prefill_cohort_barrier
            and not self._pp_prefill_cohort_released
            and request.request_id in self._pp_prefill_cohort_held
        )
        return cohort_deferred or self._should_yield_adaptive_prefill(request)

    def _should_defer_decode_request(self, request: Request) -> bool:
        return (
            self._pp_prefill_cohort_barrier
            and not self._pp_prefill_cohort_released
            and request.request_id in self._pp_prefill_cohort_ids
            and not request.is_prefill_chunk
        )

    def _should_defer_waiting_prefill(
        self, request: Request, num_computed_tokens: int
    ) -> bool:
        if (
            self._pp_prefill_cohort_barrier
            and not self._pp_prefill_cohort_ids
            and request.num_output_tokens == 0
            and request.num_prompt_tokens >= self._pp_prefill_cohort_min_tokens
        ):
            # API-side tokenization can enqueue otherwise-concurrent long
            # requests several seconds apart. Do not let early arrivals finish
            # prefill (or start decode) before the configured cohort exists.
            return True
        if (
            not self._pp_prefill_cohort_barrier
            or self._pp_prefill_cohort_released
            or not self._pp_prefill_cohort_ids
            or self._pp_prefill_cohort_finalizing
            or request.request_id not in self._pp_prefill_cohort_ids
            or request.num_output_tokens > 0
        ):
            return False
        remaining = max(request.num_prompt_tokens - num_computed_tokens, 0)
        if remaining > self._pp_prefill_final_chunk:
            return False
        # Transition and schedule the last ready member atomically instead of
        # relying on a later zero-work scheduler pass to release the cohort.
        return not self._mark_prefill_cohort_ready(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        if self._pp_prefill_cohort_finalizing:
            self._pp_prefill_cohort_final_pending.update(
                req_id
                for req_id in scheduler_output.num_scheduled_tokens
                if req_id in self._pp_prefill_cohort_ids
                and self.requests[req_id].num_output_tokens == 0
                and self.requests[req_id].num_computed_tokens
                >= self.requests[req_id].num_prompt_tokens
            )
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        self._prune_decode_phases()
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self._next_decode_step(req_id)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Async scheduling advances num_computed_tokens before the worker
        # completes. Keep the cohort barrier closed until each final-chunk
        # execution has returned, not merely been enqueued.
        final_pending = getattr(self, "_pp_prefill_cohort_final_pending", None)
        if final_pending is not None:
            final_pending.difference_update(scheduler_output.num_scheduled_tokens)
        return super().update_from_output(scheduler_output, model_runner_output)

    def _prune_decode_phases(self) -> None:
        if not self._pp_decode_phases:
            return
        active_req_ids = {
            request.request_id
            for request in self.running
            if request.status == RequestStatus.RUNNING
        }
        self._pp_decode_phases = {
            req_id: phase
            for req_id, phase in self._pp_decode_phases.items()
            if req_id in active_req_ids
        }

    def _next_decode_step(self, req_id: str) -> int:
        earliest = self.current_step + self.pp_size
        if self._pp_decode_phase_policy == "native" or self.pp_size <= 1:
            return earliest

        target_phase = self._pp_decode_phases.get(req_id)
        if target_phase is None:
            counts = Counter(self._pp_decode_phases.values())
            current_phase = self.current_step % self.pp_size
            if self._pp_decode_phase_policy == "pack" and counts:
                target_count = max(counts.values())
                candidates = [
                    phase
                    for phase in range(self.pp_size)
                    if counts.get(phase, 0) == target_count
                ]
                target_phase = min(
                    candidates,
                    key=lambda phase: (phase - current_phase) % self.pp_size,
                )
            else:
                target_phase = None
            open_phases = [
                phase
                for phase in range(self.pp_size)
                if 0 < counts.get(phase, 0) < 2
            ]
            if target_phase is not None:
                pass
            elif open_phases:
                target_phase = min(
                    open_phases,
                    key=lambda phase: (phase - current_phase) % self.pp_size,
                )
            else:
                empty_phases = [
                    phase
                    for phase in range(self.pp_size)
                    if counts.get(phase, 0) == 0
                ]
                candidates = empty_phases or list(range(self.pp_size))
                target_count = min(counts.get(phase, 0) for phase in candidates)
                candidates = [
                    phase
                    for phase in candidates
                    if counts.get(phase, 0) == target_count
                ]
                target_phase = min(
                    candidates,
                    key=lambda phase: (phase - current_phase) % self.pp_size,
                )
            self._pp_decode_phases[req_id] = target_phase

        return earliest + (target_phase - earliest) % self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
