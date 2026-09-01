# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def _load_pp_dflash_embedding(
    draft_model: nn.Module,
    vllm_config: VllmConfig,
) -> bool:
    """Load the target embedding for a DFlash head hosted on a PP rank.

    DFlash checkpoints intentionally omit ``embed_tokens`` because the draft
    normally aliases the target embedding. With PP the target embedding only
    exists on stage 0, while the DFlash head runs on the last stage; aliasing
    is impossible and the draft would otherwise use an uninitialized table.
    Read only the target embedding shard into the draft's local table.
    """
    draft_inner = getattr(draft_model, "model", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if draft_embed is None or not hasattr(draft_embed, "weight"):
        return False

    model_path = Path(vllm_config.model_config.model)
    index_path = model_path / "model.safetensors.index.json"
    weight_key = "model.language_model.embed_tokens.weight"
    shard_path: Path | None = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            shard_name = index.get("weight_map", {}).get(weight_key)
            if shard_name:
                shard_path = model_path / shard_name
        except (OSError, json.JSONDecodeError):
            shard_path = None
    candidates = [shard_path] if shard_path is not None else sorted(
        model_path.glob("*.safetensors")
    )

    from safetensors import safe_open

    source = None
    source_path = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with safe_open(str(candidate), framework="pt", device="cpu") as f:
            if weight_key in f.keys():
                source = f.get_tensor(weight_key)
                source_path = candidate
                break
    if source is None:
        logger.warning(
            "Could not find target embedding %s under %s; DFlash PP "
            "draft embedding remains uninitialized",
            weight_key,
            model_path,
        )
        return False

    target_weight = draft_embed.weight
    if tuple(source.shape) != tuple(target_weight.shape):
        raise ValueError(
            "Target embedding shape does not match DFlash draft embedding: "
            f"checkpoint={tuple(source.shape)} draft={tuple(target_weight.shape)}"
        )
    target_weight.data.copy_(
        source.to(device=target_weight.device, dtype=target_weight.dtype)
    )
    logger.info(
        "Loaded target embedding into PP DFlash head from %s (%s)",
        source_path,
        tuple(source.shape),
    )
    return True


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
        dflash_target_rope_is_neox_style,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # The drafter must rotate Q/K the way its target does. Take that from the
    # built target before super() constructs the draft.
    is_neox_style = dflash_target_rope_is_neox_style(target_model)
    if is_neox_style is not None:
        draft_model_config.hf_config.is_neox_style = is_neox_style
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        # The DFlash head is loaded on the last target PP worker and executes
        # as one local model rather than as a second pipeline.
        parallel_config=replace(
            vllm_config.parallel_config,
            pipeline_parallel_size=1,
        ),
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    # Under PP the target embedding exists only on stage 0 while the draft is
    # hosted on the last stage. Load the omitted draft embedding separately.
    if get_pp_group().world_size > 1:
        _load_pp_dflash_embedding(dflash_model, vllm_config)
    else:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
