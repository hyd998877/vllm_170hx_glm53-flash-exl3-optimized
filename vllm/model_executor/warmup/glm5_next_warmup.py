# SPDX-License-Identifier: Apache-2.0
"""Warm up GLM-5.3 Flash kernels that are shape-driven at runtime.

The generic JIT registry does not see the lazy Triton/TileLang calls used by
the GLM sparse indexer and mHC path.  Without this small model-specific pass,
the first long prefill or speculative step compiles kernels in the serving
critical path.  All buffers below are short-lived and the calls are guarded by
class/attribute checks so other models are a no-op.
"""

from __future__ import annotations

import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _find_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if (
            module.__class__.__name__ == "Glm5NextDecoderLayer"
            and getattr(module, "mhc", False)
            and all(
                hasattr(module, name)
                for name in ("hc_pre", "hc_post", "hc_attn_fn", "hc_ffn_fn")
            )
        ):
            return module
    return None


def _warmup_mhc(layer: torch.nn.Module, max_tokens: int) -> None:
    """Compile the mHC prenorm GEMM for the decode/prefill token shapes."""
    sizes = [
        size
        for size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048)
        if size <= max_tokens
    ]
    if not sizes:
        return

    device = layer.hc_attn_fn.device
    dtype = torch.bfloat16
    with torch.inference_mode():
        for size in sizes:
            residual = torch.zeros(
                size, layer.n, layer.hidden_size, dtype=dtype, device=device
            )
            post, comb, layer_input = layer.hc_pre(
                residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
            )
            layer.hc_post(layer_input, residual, post, comb)
            del residual, post, comb, layer_input


def _warmup_indexer_kernels(model: torch.nn.Module) -> None:
    """Compile the GLM sparse-indexer kernels used by real serving buffers."""
    # The checkpoint has one indexer per sparse MLA layer.  The constants are
    # shared by all layers, so one representative module is sufficient.
    indexer = None
    for module in model.modules():
        if module.__class__.__name__ == "Indexer":
            indexer = module
            break
    if indexer is None:
        return

    device = indexer.index_kpool_compress_ape.device
    head_dim = int(indexer.head_dim)
    kpool = int(indexer.index_kpool)
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_compress_and_write_cache,
        kpool_seed_tail_cache,
    )

    # Use the bound cache views when available.  The tail/cache block count and
    # physical strides are constexprs in the Triton kernels; a one-block dummy
    # therefore misses the real serving specialization on long contexts.
    bound_tail = getattr(getattr(indexer, "tail_cache", None), "kv_cache", None)
    tail = bound_tail
    if tail is None or tail.numel() == 0:
        tail = torch.empty((1, 2, kpool, head_dim), dtype=torch.bfloat16, device=device)
    key = torch.zeros((1, head_dim), dtype=torch.bfloat16, device=device)
    gate = torch.zeros((1, head_dim), dtype=torch.bfloat16, device=device)
    # -1 takes the early return and avoids modifying the null block while still
    # compiling the exact NUM_TAIL_BLOCKS/stride specialization.
    # Runtime slot mappings are int64.  Triton includes pointer element type in
    # its specialization key, so an int32 dummy compiles a cache entry that can
    # never be reused by serving.
    # Triton distinguishes aligned and unaligned tensor pointers.  Production
    # metadata is commonly a slice into a shared int64 buffer (storage_offset
    # > 0), while the old warmup allocated only an aligned one-element tensor.
    # Prime both variants.  Negative slots take the no-write path, so the
    # model's real tail cache remains untouched.
    for offset in (0, 1):
        tslot_storage = torch.full((2,), -1, dtype=torch.int64, device=device)
        tslot = tslot_storage[offset : offset + 1]
        kpool_seed_tail_cache(tail, key, gate, tslot, kpool, head_dim=head_dim)

    # Prefill uses a separate fused softmax/rotate/quantize/cache-write kernel.
    # Compile it against the bound cache: page size and physical page stride
    # are constexprs and a small synthetic cache would therefore miss the
    # serving specialization.  An all-false mask executes no stores.
    bound_cache = getattr(getattr(indexer, "k_cache", None), "kv_cache", None)
    if bound_cache is not None and bound_cache.numel() > 0:
        slot_k = torch.zeros(
            (1, kpool, head_dim), dtype=torch.bfloat16, device=device
        )
        slot_score = torch.zeros_like(slot_k)
        ape = indexer.index_kpool_compress_ape.detach().float().contiguous()
        write_mask = torch.zeros((1,), dtype=torch.bool, device=device)
        for offset in (0, 1):
            loc_storage = torch.full((2,), -1, dtype=torch.int64, device=device)
            loc = loc_storage[offset : offset + 1]
            kpool_compress_and_write_cache(
                bound_cache,
                slot_k,
                slot_score,
                ape,
                loc,
                pool_size=kpool,
                head_dim=head_dim,
                write_mask=write_mask,
                round_scale=True,
                return_compressed=False,
                write_cache=True,
            )
        del slot_k, slot_score, ape, write_mask, loc, loc_storage
    del tail, key, gate, tslot, tslot_storage

    # The SM80 fallback pads the checkpoint's 16 index heads to 32 before
    # dispatch.  Calling its public warmup helper primes both autotune and
    # Triton compilation without touching model state.
    from vllm.v1.attention.ops.mqa_logits_triton import (
        warmup_fp8_mqa_logits_triton,
        warmup_fp8_paged_mqa_logits_triton,
    )

    warmup_fp8_mqa_logits_triton(32, head_dim, device)

    if bound_cache is not None and bound_cache.numel() > 0:
        from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
            kv_cache_as_quant_view,
        )
        from vllm.v1.attention.ops.mqa_logits_triton import (
            fp8_paged_mqa_logits_triton,
        )

        paged_cache = kv_cache_as_quant_view(bound_cache, head_dim, False)
        block_size = int(paged_cache.shape[1])
        q = torch.empty((1, 1, 32, head_dim), dtype=torch.float8_e4m3fn, device=device)
        weights = torch.zeros((1, 32), dtype=torch.float32, device=device)
        context_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
        block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)
        fp8_paged_mqa_logits_triton(
            q,
            paged_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len=block_size,
        )
        del paged_cache, q, weights, context_lens, block_tables
    else:
        warmup_fp8_paged_mqa_logits_triton(32, head_dim, 64, device)


def _warmup_mamba_acceptance_kernel(device: torch.device) -> None:
    # DFlash's Mamba state postprocess uses this tiny kernel for integer
    # num_sampled values.  It is otherwise invisible to the generic registry.
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import _fill_num_accepted_kernel

    # Runtime idx_mapping is int64; match it exactly so the warmup and serving
    # calls share a Triton specialization.
    index = torch.zeros((1,), dtype=torch.int64, device=device)
    accepted = torch.zeros((1,), dtype=torch.int32, device=device)
    # ``num_sampled`` is a constexpr scalar in Triton.  Speculative decoding
    # can pass 0..(k+1), so prime the small set used by the production profiles.
    for num_sampled in range(0, 9):
        _fill_num_accepted_kernel[(1,)](index, accepted, num_sampled)
    del index, accepted


def glm5_next_warmup(model: torch.nn.Module, *, max_tokens: int) -> None:
    """Best-effort GLM warmup; no-op for non-GLM models."""
    if not any(
        module.__class__.__name__.startswith("Glm5Next")
        for module in model.modules()
    ):
        return

    started = time.perf_counter()
    try:
        layer = _find_mhc_layer(model)
        if layer is not None:
            _warmup_mhc(layer, max_tokens)
        _warmup_indexer_kernels(model)
        _warmup_mamba_acceptance_kernel(next(model.parameters()).device)
        torch.cuda.synchronize()
    except Exception:
        # Keep startup robust on unsupported devices; the JIT monitor will
        # expose any remaining lazy kernel on the first request.
        logger.exception("GLM-5.3 kernel warmup failed; continuing without it")
        return
    logger.info(
        "GLM-5.3 kernel warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
