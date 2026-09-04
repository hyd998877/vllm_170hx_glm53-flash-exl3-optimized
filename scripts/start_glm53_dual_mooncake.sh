#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=glm53_dual_mooncake_common.sh
source "$SCRIPT_DIR/glm53_dual_mooncake_common.sh"

STARTED_MASTER=0
STARTED_PORTS=()
TARGET_INDICES=(0 1)

case "${1:-}" in
  "") ;;
  --only-3000) TARGET_INDICES=(0) ;;
  --only-3001) TARGET_INDICES=(1) ;;
  *)
    printf '用法: %s [--only-3000|--only-3001]\n' "$0" >&2
    exit 2
    ;;
esac

cleanup_partial_start() {
  local port runtime pid
  mooncake_log "启动未完成，清理本脚本已经启动的 Mooncake/GLM 进程"
  for port in "${STARTED_PORTS[@]}"; do
    if ! mooncake_stop_instance "$port"; then
      # Session validation may itself be why startup failed. Fall back only
      # to the just-recorded PID+starttime+cmdline identity, never a bare PID.
      if mooncake_validate_launched_instance_pid "$port"; then
        pid="$(<"$(mooncake_instance_pid_file "$port")")"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    fi
  done
  if ((STARTED_MASTER == 1)); then
    if mooncake_validate_master_pid || mooncake_validate_launched_master_pid; then
      kill -TERM "$(<"$MOONCAKE_RUNTIME/master.pid")" 2>/dev/null || true
    fi
  fi
}

handle_signal() {
  local exit_code=$1
  exit "$exit_code"
}

trap cleanup_partial_start EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

for command_name in awk curl flock nvidia-smi sed setsid ss tr; do
  command -v "$command_name" >/dev/null 2>&1 || \
    mooncake_die "缺少命令: $command_name"
done
[[ -x "$MOONCAKE_SERVER" ]] || mooncake_die "服务脚本不存在: $MOONCAKE_SERVER"
[[ -x "$MOONCAKE_PYTHON" ]] || mooncake_die "Python 不存在: $MOONCAKE_PYTHON"
[[ -x "$MOONCAKE_MASTER_BIN" ]] || \
  mooncake_die "mooncake_master 不存在: $MOONCAKE_MASTER_BIN"
[[ -f "$MOONCAKE_CONFIG" ]] || mooncake_die "Mooncake 配置不存在: $MOONCAKE_CONFIG"
[[ -f "$MOONCAKE_MODEL/config.json" ]] || mooncake_die "模型不存在: $MOONCAKE_MODEL"

mkdir -p "$MOONCAKE_RUNTIME"
exec 9>"$MOONCAKE_RUNTIME/.control.lock"
flock -n 9 || mooncake_die "另一项 Mooncake 启停操作正在进行"

if mooncake_validate_master_pid && mooncake_port_in_use "$MOONCAKE_MASTER_PORT"; then
  mooncake_log "复用已托管的 Mooncake master"
elif mooncake_port_in_use "$MOONCAKE_MASTER_PORT"; then
  mooncake_die "端口 $MOONCAKE_MASTER_PORT 已被非托管进程占用"
else
  rm -f "$MOONCAKE_RUNTIME/master.pid" "$MOONCAKE_RUNTIME/master.starttime"
  mooncake_log "启动 Mooncake master: 127.0.0.1:$MOONCAKE_MASTER_PORT"
  nohup setsid "$MOONCAKE_MASTER_BIN" \
    --rpc_address=127.0.0.1 \
    --rpc_port="$MOONCAKE_MASTER_PORT" \
    --metrics_host=127.0.0.1 \
    --metrics_port="$MOONCAKE_METRICS_PORT" \
    >"$MOONCAKE_RUNTIME/master.log" 2>&1 </dev/null 9>&- &
  printf '%s\n' "$!" >"$MOONCAKE_RUNTIME/master.pid"
  STARTED_MASTER=1
  mooncake_record_private_session \
    "$!" "$MOONCAKE_RUNTIME/master.starttime" || \
    mooncake_die "Mooncake master 未建立私有 session"
  sleep 1
  for _ in $(seq 1 30); do
    mooncake_port_in_use "$MOONCAKE_MASTER_PORT" && break
    mooncake_validate_master_pid || \
      mooncake_die "Mooncake master 初始化期间退出"
    sleep 1
  done
  mooncake_port_in_use "$MOONCAKE_MASTER_PORT" || \
    mooncake_die "Mooncake master 未在 $MOONCAKE_MASTER_PORT 监听"
fi

start_instance() {
  local index=$1 port gpu_set runtime rpc_path kv_transfer_config
  port="${MOONCAKE_PORTS[$index]}"
  gpu_set="${MOONCAKE_GPU_SETS[$index]}"
  runtime="$(mooncake_instance_runtime "$port")"
  rpc_path="/tmp/vm-$port-$$"

  if mooncake_validate_instance_pid "$port" && mooncake_healthy "$port"; then
    mooncake_log "端口 $port 的托管实例已健康，跳过重复启动"
    return
  fi
  mooncake_port_in_use "$port" && \
    mooncake_die "端口 $port 已被非健康或非托管进程占用"
  mooncake_validate_gpu_free "$gpu_set"

  kv_transfer_config="{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cache_prefix\":\"$MOONCAKE_CACHE_PREFIX\",\"load_async\":true,\"lookup_async\":false,\"lookup_rpc_port\":$port}}"

  mkdir -p "$runtime" "$rpc_path"
  rm -f "$runtime/server.pid" "$runtime/server.starttime"
  printf '%s\n' "$rpc_path" >"$runtime/ipc.path"
  mooncake_log "顺序启动 GLM 实例：端口=$port GPU=$gpu_set"
  nohup setsid env \
    PYTHON_BIN="$MOONCAKE_PYTHON" \
    CUDA_VISIBLE_DEVICES="$gpu_set" \
    HOST="0.0.0.0" \
    PORT="$port" \
    MODEL="$MOONCAKE_MODEL" \
    SERVED_MODEL="GLM-5.3-Flash-tr3-4bpw" \
    PIPELINE_PARALLEL_SIZE=4 \
    PP_PARTITION="13,12,11,9" \
    MAX_MODEL_LEN=524288 \
    MAX_NUM_SEQS=6 \
    MAX_NUM_BATCHED_TOKENS=2050 \
    LONG_PREFILL_TOKEN_THRESHOLD=256 \
    GPU_MEMORY_UTILIZATION=0.970 \
    PROFILE=multimodal \
    DFLASH_MODEL="$MOONCAKE_DFLASH_MODEL" \
    DFLASH_K=2 \
    MARLIN_DIR="$MOONCAKE_MARLIN_DIR" \
    TORCH_EXTENSIONS_DIR="/mnt/nvme0/models/.torch_extensions" \
    PP_DECODE_PHASE_POLICY=pairpack \
    PP_FIXED_DECODE_COMM=0 \
    PP_DIRECT_RECV_BUFFER=0 \
    COHORT_BARRIER=0 \
    COHORT_SIZE=0 \
    COHORT_MIN_TOKENS=0 \
    ADAPTIVE_PREFILL=1 \
    ADAPTIVE_PREFILL_MAX_TOKENS=2048 \
    ADAPTIVE_PREFILL_BUSY_TOKENS=1550 \
    CUDAGRAPH_CAPTURE_SIZES="3,6,9,12,15,18" \
    MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG" \
    KV_TRANSFER_CONFIG="$kv_transfer_config" \
    PREFIX_CACHING_HASH_ALGO=sha256 \
    MAMBA_CACHE_MODE=align \
    VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4352 \
    PYTHONHASHSEED=20260905 \
    VLLM_RPC_BASE_PATH="$rpc_path" \
    "$MOONCAKE_SERVER" >"$runtime/server-dflash2.log" 2>&1 </dev/null 9>&- &
  printf '%s\n' "$!" >"$runtime/server.pid"
  STARTED_PORTS+=("$port")
  mooncake_record_private_session "$!" "$runtime/server.starttime" || \
    mooncake_die "端口 $port 的 GLM 未建立私有 session"
  sleep 1

  for _ in $(seq 1 720); do
    if mooncake_healthy "$port"; then
      mooncake_log "READY: http://0.0.0.0:$port/v1"
      return
    fi
    mooncake_validate_instance_pid "$port" || {
      tail -n 120 "$runtime/server-dflash2.log" >&2 || true
      mooncake_die "端口 $port 的 GLM 实例初始化期间退出"
    }
    sleep 1
  done
  tail -n 120 "$runtime/server-dflash2.log" >&2 || true
  mooncake_die "等待端口 $port 就绪超时"
}

for index in "${TARGET_INDICES[@]}"; do
  start_instance "$index"
done

trap - EXIT INT TERM
if ((${#TARGET_INDICES[@]} == 2)); then
  mooncake_log "双实例已启动：3000/GPU0-3，3001/GPU4-7，共享 cache_prefix=$MOONCAKE_CACHE_PREFIX"
  mooncake_log "运行跨实例门禁：$MOONCAKE_PYTHON $MOONCAKE_REPO/scripts/verify_glm53_mooncake_sharing.py"
else
  mooncake_log "指定的单个实例已启动；跨实例共享门禁需要 3000 和 3001 同时健康"
fi
