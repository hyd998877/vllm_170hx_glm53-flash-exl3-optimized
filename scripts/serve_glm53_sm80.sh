#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:?set MODEL to the GLM-5.3-Flash EXL3 checkpoint directory}"
DFLASH_MODEL="${DFLASH_MODEL:?set DFLASH_MODEL to the DFlash2 checkpoint directory}"
DFLASH_K="${DFLASH_K:-2}"
DFLASH_BATCH_SCHEDULE_JSON="${DFLASH_BATCH_SCHEDULE_JSON:-}"
MARLIN_DIR="${MARLIN_DIR:?set MARLIN_DIR to the converted Marlin sidecar directory}"
PROFILE="${PROFILE:-multimodal}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-4}"
SERVED_MODEL="${SERVED_MODEL:-GLM-5.3-Flash-tr3-4bpw}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT/chat_templates/glm53-enable-thinking-switch.jinja}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
EXL3_EXTENSION_DIR="${EXL3_EXTENSION_DIR:-$TORCH_EXTENSIONS_DIR/exllamav3_ext}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-3,6,9,12,15,18}"
ADAPTIVE_PREFILL="${ADAPTIVE_PREFILL:-1}"
ADAPTIVE_PREFILL_MAX_TOKENS="${ADAPTIVE_PREFILL_MAX_TOKENS:-2048}"
ADAPTIVE_PREFILL_BUSY_TOKENS="${ADAPTIVE_PREFILL_BUSY_TOKENS:-1550}"
PREFIX_CACHING_HASH_ALGO="${PREFIX_CACHING_HASH_ALGO:-sha256}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"
EXTRA_ARGS=()
DFLASH_BATCH_SCHEDULE_FIELD=""

if [[ "$ADAPTIVE_PREFILL" == 1 ]]; then
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((ADAPTIVE_PREFILL_MAX_TOKENS + DFLASH_K))}"
else
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
fi

if [[ -n "$DFLASH_BATCH_SCHEDULE_JSON" ]]; then
  DFLASH_BATCH_SCHEDULE_FIELD=",\"num_speculative_tokens_per_batch_size\":$DFLASH_BATCH_SCHEDULE_JSON"
fi

case "$PROFILE" in
  multimodal)
    PP_PARTITION="${PP_PARTITION:-13,12,11,9}"
    COHORT_BARRIER="${COHORT_BARRIER:-0}"
    ;;
  text)
    PP_PARTITION="${PP_PARTITION:-13,11,11,10}"
    COHORT_BARRIER="${COHORT_BARRIER:-0}"
    EXTRA_ARGS=(--language-model-only --skip-mm-profiling)
    ;;
  text-benchmark)
    PP_PARTITION="${PP_PARTITION:-13,11,11,10}"
    COHORT_BARRIER="${COHORT_BARRIER:-1}"
    EXTRA_ARGS=(--language-model-only --skip-mm-profiling)
    ;;
  *)
    echo "PROFILE must be multimodal, text, or text-benchmark" >&2
    exit 2
    ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "invalid MODEL: $MODEL" >&2; exit 2; }
[[ -f "$DFLASH_MODEL/config.json" ]] || {
  echo "invalid DFLASH_MODEL: $DFLASH_MODEL" >&2
  exit 2
}
[[ -f "$CHAT_TEMPLATE" ]] || { echo "missing template: $CHAT_TEMPLATE" >&2; exit 2; }
if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  "$PYTHON_BIN" - "$KV_TRANSFER_CONFIG" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit("KV_TRANSFER_CONFIG must be a JSON object")
if not value.get("kv_connector") or not value.get("kv_role"):
    raise SystemExit("KV_TRANSFER_CONFIG requires kv_connector and kv_role")
PY
fi
if [[ -n "${MOONCAKE_CONFIG_PATH:-}" ]]; then
  [[ -f "$MOONCAKE_CONFIG_PATH" ]] || {
    echo "missing Mooncake config: $MOONCAKE_CONFIG_PATH" >&2
    exit 2
  }
  "$PYTHON_BIN" - "$MOONCAKE_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    value = json.load(config_file)
if not isinstance(value, dict):
    raise SystemExit("MOONCAKE_CONFIG_PATH must contain a JSON object")
PY
fi
[[ -f "$EXL3_EXTENSION_DIR/exllamav3_ext.so" ]] || {
  echo "missing EXL3 extension: $EXL3_EXTENSION_DIR/exllamav3_ext.so" >&2
  echo "build it with scripts/build_exllamav3_ext.py" >&2
  exit 2
}
for layer in $(seq 3 44); do
  printf -v sidecar '%s/layer-%02d.safetensors' "$MARLIN_DIR" "$layer"
  [[ -s "$sidecar" ]] || { echo "missing Marlin sidecar: $sidecar" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,6}"
export TORCH_EXTENSIONS_DIR
export PYTHONPATH="$ROOT:$EXL3_EXTENSION_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_EXL3_EXTENSION_DIR="$EXL3_EXTENSION_DIR"
export VLLM_EXL3_MARLIN_DIR="$MARLIN_DIR"
export VLLM_EXL3_MARLIN_LAYERS="3-44"
export VLLM_EXL3_NON_ROUTED_FP8=1
export VLLM_EXL3_NON_ROUTED_FP8_SCOPE=official
export VLLM_EXL3_OUTER_GRAPH=1
export VLLM_PP_LAYER_PARTITION="$PP_PARTITION"
export VLLM_PP_DECODE_PHASE_POLICY="${PP_DECODE_PHASE_POLICY:-pairpack}"
export VLLM_PP_FIXED_DECODE_COMM="${PP_FIXED_DECODE_COMM:-0}"
export VLLM_PP_DIRECT_RECV_BUFFER="${PP_DIRECT_RECV_BUFFER:-0}"
export VLLM_PP_PREFILL_COHORT_BARRIER="$COHORT_BARRIER"
export VLLM_PP_PREFILL_COHORT_SIZE="${COHORT_SIZE:-$([[ "$COHORT_BARRIER" == 1 ]] && echo 6 || echo 0)}"
export VLLM_PP_PREFILL_COHORT_MIN_TOKENS="${COHORT_MIN_TOKENS:-$([[ "$COHORT_BARRIER" == 1 ]] && echo 65536 || echo 0)}"
export VLLM_PP_ADAPTIVE_PREFILL="$ADAPTIVE_PREFILL"
export VLLM_PP_ADAPTIVE_PREFILL_MAX_TOKENS="$ADAPTIVE_PREFILL_MAX_TOKENS"
export VLLM_PP_ADAPTIVE_PREFILL_BUSY_TOKENS="$ADAPTIVE_PREFILL_BUSY_TOKENS"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=1
if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "KV connectors require stable CUDA virtual addresses; unset expandable_segments" >&2
    exit 2
  fi
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native}"
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  EXTRA_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi
if [[ -n "$MAMBA_CACHE_MODE" ]]; then
  EXTRA_ARGS+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
fi

exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-256}" \
  --block-size 256 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.970}" \
  --kv-cache-dtype auto \
  --prefix-caching-hash-algo "$PREFIX_CACHING_HASH_ALGO" \
  --trust-remote-code \
  --mm-processor-cache-gb 0 \
  --chat-template "$CHAT_TEMPLATE" \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --linear-backend marlin \
  --async-scheduling \
  --jit-monitor-mode warn \
  --compilation-config \
    "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}" \
  --speculative-config \
    "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$DFLASH_K,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"auto\"$DFLASH_BATCH_SCHEDULE_FIELD}" \
  "${EXTRA_ARGS[@]}"
