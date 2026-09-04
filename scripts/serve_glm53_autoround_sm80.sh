#!/usr/bin/env bash
set -euo pipefail

# Candidate profile for Intel's W4A16 AutoRound checkpoint on 4x SM80.
# This intentionally uses a separate runtime directory from the EXL3 service.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${LAUNCHER:-/mnt/nvme0/keys-vllm-glm53/launcher/launch_cmp170hx.sh}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nvme0/keys-vllm-glm53/.venv/bin/python}"
MODEL="${MODEL:-/mnt/nvme0/models/GLM-5.3-Flash-W4A16-AutoRound}"

[[ -x "$LAUNCHER" ]] || { echo "missing launcher: $LAUNCHER" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "missing model config: $MODEL" >&2; exit 2; }
[[ -f "$MODEL/model.safetensors.index.json" ]] || {
  echo "incomplete AutoRound snapshot: missing model.safetensors.index.json" >&2
  exit 2
}

"$PYTHON_BIN" - "$MODEL" <<'PY'
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
quant = config.get("quantization_config", {})
expected = {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "packing_format": "auto_round:auto_gptq",
}
actual = {key: quant.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"unexpected AutoRound config: {actual!r}")

index = json.loads((model / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in shards if not (model / name).is_file()]
if missing:
    raise SystemExit(f"incomplete AutoRound snapshot: missing {missing[:4]!r}")
print(f"validated AutoRound snapshot index: {len(shards)} shards")
PY

export MODEL
export SERVED_MODEL="${SERVED_MODEL:-GLM-5.3-Flash-W4A16-AutoRound}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,6}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-3000}"
export RUNTIME_DIR="${RUNTIME_DIR:-/mnt/nvme0/keys-vllm-glm53/runtime/autoround-3000}"

export PIPELINE_PARALLEL_SIZE=4
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-13,12,11,9}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2050}"
export LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-256}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export UTIL="${UTIL:-0.96}"
export UTIL_CAP="${UTIL_CAP:-0.97}"

export SPEC="${SPEC:-dflash2}"
export DFLASH_MODEL="${DFLASH_MODEL:-/mnt/nvme0/models/GLM-5.3-Flash-DFlash2}"
export DFLASH_K="${DFLASH_K:-2}"
export DFLASH_DRAFT_SAMPLE_METHOD="${DFLASH_DRAFT_SAMPLE_METHOD:-probabilistic}"
export DFLASH_REJECTION_SAMPLE_METHOD="${DFLASH_REJECTION_SAMPLE_METHOD:-standard}"
export DFLASH_KV_CACHE_DTYPE="${DFLASH_KV_CACHE_DTYPE:-auto}"

# The checkpoint carries its own INC/GPTQ packing. It must not inherit EXL3
# sidecar overrides from a parent shell or the production EXL3 profile.
export EXL3_MARLIN=0
unset VLLM_EXL3_MARLIN_DIR VLLM_EXL3_MARLIN_LAYERS
unset VLLM_EXL3_NON_ROUTED_FP8 VLLM_EXL3_NON_ROUTED_FP8_SCOPE
unset VLLM_EXL3_OUTER_GRAPH

export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
export PP_DECODE_PHASE_POLICY="${PP_DECODE_PHASE_POLICY:-pairpack}"
export PP_FIXED_DECODE_COMM=0
export PP_DIRECT_RECV_BUFFER=0
export PP_PREFILL_COHORT_BARRIER="${PP_PREFILL_COHORT_BARRIER:-0}"
export PP_PREFILL_COHORT_SIZE="${PP_PREFILL_COHORT_SIZE:-0}"
export PP_PREFILL_COHORT_MIN_TOKENS="${PP_PREFILL_COHORT_MIN_TOKENS:-0}"
export PP_ADAPTIVE_PREFILL="${PP_ADAPTIVE_PREFILL:-1}"
export PP_ADAPTIVE_PREFILL_MAX_TOKENS="${PP_ADAPTIVE_PREFILL_MAX_TOKENS:-2048}"
export PP_ADAPTIVE_PREFILL_BUSY_TOKENS="${PP_ADAPTIVE_PREFILL_BUSY_TOKENS:-1550}"

export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
export SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-0}"
export EAGER="${EAGER:-0}"
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-3,6,9,12,15,18}"
export JIT_MONITOR_MODE="${JIT_MONITOR_MODE:-warn}"
export JIT_MONITOR_VERBOSE="${JIT_MONITOR_VERBOSE:-1}"

exec "$LAUNCHER" "${1:-start}"
