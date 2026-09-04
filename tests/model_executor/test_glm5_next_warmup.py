# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.glm5_next_warmup import _warmup_indexer_kernels


class Indexer:
    def __init__(self) -> None:
        self.head_dim = 128
        self.index_kpool = 16
        self.index_kpool_compress_ape = torch.zeros(16, 128)
        self.k_cache = SimpleNamespace(
            kv_cache=torch.empty(3, 32, 1, 132, dtype=torch.uint8)
        )
        self.tail_cache = SimpleNamespace(
            kv_cache=torch.empty(5, 2, 16, 128, dtype=torch.bfloat16)
        )


class Model:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def modules(self):
        return iter((self, self.indexer))


def test_runtime_lengths_do_not_create_indexer_specializations() -> None:
    import vllm.models.glm5next.nvidia.ops.kpool_compress as kpool_ops
    import vllm.v1.attention.ops.mqa_logits_triton as mqa_ops

    assert "n_tokens" in kpool_ops._kpool_tail_seed_kernel.do_not_specialize
    assert "N" in mqa_ops._fp8_mqa_logits_kernel.fn.do_not_specialize


def test_indexer_warmup_uses_bound_cache_and_pointer_variants(monkeypatch) -> None:
    # Import the model side first, matching production construction order and
    # avoiding the package-level GLM export importing attention while the
    # sparse-indexer module is only partially initialized.
    importlib.import_module("vllm.models.glm5next.nvidia.attention")
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as indexer_ops
    import vllm.models.glm5next.nvidia.ops.kpool_compress as kpool_ops
    import vllm.v1.attention.ops.mqa_logits_triton as mqa_ops

    indexer = Indexer()
    tail_calls = []
    compress_calls = []
    mqa_calls = []
    paged_calls = []

    monkeypatch.setattr(
        kpool_ops,
        "kpool_seed_tail_cache",
        lambda tail, key, gate, slot, *args, **kwargs: tail_calls.append(
            (tail, slot.storage_offset())
        ),
    )
    monkeypatch.setattr(
        kpool_ops,
        "kpool_compress_and_write_cache",
        lambda cache, *args, **kwargs: compress_calls.append(
            (cache, args[3].storage_offset(), kwargs)
        ),
    )
    monkeypatch.setattr(
        mqa_ops,
        "warmup_fp8_mqa_logits_triton",
        lambda *args: mqa_calls.append(args),
    )
    monkeypatch.setattr(indexer_ops, "kv_cache_as_quant_view", lambda cache, *_: cache)
    monkeypatch.setattr(
        mqa_ops,
        "fp8_paged_mqa_logits_triton",
        lambda *args, **kwargs: paged_calls.append((args, kwargs)),
    )

    _warmup_indexer_kernels(Model(indexer))  # type: ignore[arg-type]

    assert [offset for _, offset in tail_calls] == [0, 1]
    assert all(cache is indexer.tail_cache.kv_cache for cache, _ in tail_calls)
    assert [offset for _, offset, _ in compress_calls] == [0, 1]
    assert all(cache is indexer.k_cache.kv_cache for cache, _, _ in compress_calls)
    assert all(call[2]["write_mask"].eq(False).all() for call in compress_calls)
    assert len(mqa_calls) == 1
    assert len(paged_calls) == 1
