# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse MLA backend for Ampere CUDA GPUs."""

from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseMetadata,
    FlashAttnMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill


class TritonMLASparseSM80Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_SM80"

    @staticmethod
    def get_builder_cls() -> type[FlashAttnMLASparseMetadataBuilder]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return TritonMLASparseSM80Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return "SM80 sparse MLA requires a BF16 KV cache"
        return None


class TritonMLASparseSM80Impl(
    SparseMLACommonImpl[FlashAttnMLASparseMetadata]
):
    """Top-k sparse MLA using a portable Triton BF16 attention kernel."""

    # Ampere has no dense MLA prefill kernel compatible with GLM-5's
    # rope-free 512-wide latent representation.  Keep both prefill and decode
    # on the sparse MQA path below.
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        if self.qk_rope_head_dim != 0 or self.kv_lora_rank != 512:
            raise NotImplementedError(
                "SM80 sparse MLA currently supports rope-free 512-wide GLM MLA"
            )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("SM80 sparse MLA requires a BF16 KV cache")
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "SM80 sparse MLA does not support alibi, sliding window, or softcap"
            )
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q_nope, q_rope = q
            q = q_nope if q_rope.shape[-1] == 0 else torch.cat((q_nope, q_rope), -1)

        num_tokens = q.shape[0]
        assert self.topk_indices_buffer is not None
        # The model-owned buffer may have padded/kpool-tail columns.  Only the
        # first ``topk_tokens`` columns contain the sparse history selected for
        # this attention call.
        logical_indices = self.topk_indices_buffer[
            :num_tokens, : attn_metadata.topk_tokens
        ]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        physical_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            logical_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=logical_indices.shape[1],
            return_valid_counts=True,
        )
        out = torch.empty_like(q, dtype=torch.bfloat16)
        rocm_sparse_attn_prefill(
            q=q,
            kv=kv_rows.unsqueeze(1),
            indices=physical_indices,
            topk_length=valid_counts,
            scale=self.scale,
            head_dim=self.head_size,
            nope_head_dim=self.kv_lora_rank,
            rope_head_dim=self.qk_rope_head_dim,
            attn_sink=None,
            output=out,
        )
        return out, None
