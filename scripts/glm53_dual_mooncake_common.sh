#!/usr/bin/env bash

MOONCAKE_REPO="/mnt/nvme0/keys-vllm-glm53/vllm-glm53-pr"
MOONCAKE_ROOT="/mnt/nvme0/keys-vllm-glm53"
MOONCAKE_PYTHON="$MOONCAKE_ROOT/.venv/bin/python"
MOONCAKE_MASTER_BIN="$MOONCAKE_ROOT/.venv/bin/mooncake_master"
MOONCAKE_SERVER="$MOONCAKE_REPO/scripts/serve_glm53_sm80.sh"
MOONCAKE_CONFIG="$MOONCAKE_REPO/configs/mooncake_store_dual_pp4.json"
MOONCAKE_RUNTIME="$MOONCAKE_ROOT/runtime/mooncake-dual-pp4"
MOONCAKE_MODEL="/mnt/nvme0/models/GLM-5.3-Flash-tr3-4bpw"
MOONCAKE_DFLASH_MODEL="/mnt/nvme0/models/GLM-5.3-Flash-DFlash2"
MOONCAKE_MARLIN_DIR="/mnt/nvme0/models/GLM-5.3-Flash-tr3-4bpw-marlin-int4-gs64"
MOONCAKE_MASTER_PORT=50051
MOONCAKE_METRICS_PORT=50052
MOONCAKE_CACHE_PREFIX="glm53-tr3-4bpw-pp4-bs256-kvauto-dflash2-k2-v1"
MOONCAKE_PORTS=(3000 3001)
MOONCAKE_GPU_SETS=("0,1,2,3" "4,5,6,7")

mooncake_log() {
  printf '%s [INFO] %s\n' "$(date '+%F %T')" "$*"
}

mooncake_die() {
  printf '%s [ERROR] %s\n' "$(date '+%F %T')" "$*" >&2
  exit 1
}

mooncake_port_in_use() {
  local port=$1
  ss -ltnH | awk -v suffix=":$port" \
    'substr($4, length($4) - length(suffix) + 1) == suffix {found=1} END {exit !found}'
}

mooncake_healthy() {
  curl -fsS --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

mooncake_pid_running() {
  local pid_file=$1 pid
  [[ -s "$pid_file" ]] || return 1
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

mooncake_instance_runtime() {
  printf '%s/instance-%s' "$MOONCAKE_RUNTIME" "$1"
}

mooncake_instance_pid_file() {
  printf '%s/server.pid' "$(mooncake_instance_runtime "$1")"
}

mooncake_instance_starttime_file() {
  printf '%s/server.starttime' "$(mooncake_instance_runtime "$1")"
}

mooncake_process_identity() {
  local pid=$1 stat tail
  [[ -r "/proc/$pid/stat" ]] || return 1
  stat="$(<"/proc/$pid/stat")"
  tail="${stat##*) }"
  set -- $tail
  # Fields after comm start at proc field 3 (state).
  printf '%s %s %s\n' "$3" "$4" "${20}"
}

mooncake_record_private_session() {
  local pid=$1 starttime_file=$2 identity pgid sid starttime
  for _ in $(seq 1 50); do
    identity="$(mooncake_process_identity "$pid" 2>/dev/null || true)"
    if [[ -n "$identity" ]]; then
      read -r pgid sid starttime <<<"$identity"
      # Persist the identity before checking setsid. If private-session
      # validation fails, the launcher's EXIT trap can still identify and
      # terminate exactly the child it just created without trusting a bare
      # PID that might later be reused.
      if [[ ! -s "$starttime_file" ]]; then
        printf '%s\n' "$starttime" >"$starttime_file"
      fi
      if [[ "$pgid" == "$pid" && "$sid" == "$pid" ]]; then
        return 0
      fi
    fi
    sleep 0.1
  done
  return 1
}

mooncake_validate_launched_instance_pid() {
  local port=$1 pid_file starttime_file pid identity starttime expected
  pid_file="$(mooncake_instance_pid_file "$port")"
  starttime_file="$(mooncake_instance_starttime_file "$port")"
  mooncake_pid_running "$pid_file" || return 1
  [[ -s "$starttime_file" ]] || return 1
  pid="$(<"$pid_file")"
  identity="$(mooncake_process_identity "$pid")" || return 1
  read -r _ _ starttime <<<"$identity"
  expected="$(<"$starttime_file")"
  [[ "$starttime" == "$expected" ]] || return 1
  local cmdline environment
  cmdline="$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline" || true)"
  if [[ "$cmdline" == *"$MOONCAKE_MODEL"* && "$cmdline" == *"--port $port"* ]]; then
    return 0
  fi
  environment="$(tr '\0' '\n' 2>/dev/null <"/proc/$pid/environ" || true)"
  [[ "$cmdline" == *"$MOONCAKE_SERVER"* && \
    "$environment" == *$'MODEL='"$MOONCAKE_MODEL"$'\n'* && \
    "$environment" == *$'PORT='"$port"$'\n'* ]]
}

mooncake_validate_launched_master_pid() {
  local pid_file="$MOONCAKE_RUNTIME/master.pid"
  local starttime_file="$MOONCAKE_RUNTIME/master.starttime"
  local pid identity starttime expected cmdline
  mooncake_pid_running "$pid_file" || return 1
  [[ -s "$starttime_file" ]] || return 1
  pid="$(<"$pid_file")"
  identity="$(mooncake_process_identity "$pid")" || return 1
  read -r _ _ starttime <<<"$identity"
  expected="$(<"$starttime_file")"
  [[ "$starttime" == "$expected" ]] || return 1
  cmdline="$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline" || true)"
  [[ "$cmdline" == *"mooncake_master"* && \
    ( "$cmdline" == *"--rpc_port=$MOONCAKE_MASTER_PORT"* || \
      "$cmdline" == *"--port=$MOONCAKE_MASTER_PORT"* ) ]]
}

mooncake_validate_gpu_free() {
  local gpu_set=$1 busy proc cmdline visible requested_gpu visible_gpu overlap
  busy="$(nvidia-smi -i "$gpu_set" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "$busy" ]]; then
    printf '%s\n' "$busy" >&2
    mooncake_die "GPU $gpu_set 已有计算进程；拒绝终止未知进程或重叠启动"
  fi

  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" && -r "$proc/environ" ]] || continue
    cmdline="$(tr '\0' ' ' <"$proc/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"vllm.entrypoints"* || "$cmdline" == *"vllm serve"* ]] || \
      continue
    visible="$(tr '\0' '\n' <"$proc/environ" 2>/dev/null | \
      sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1)"
    overlap=0
    if [[ -z "$visible" ]]; then
      overlap=1
    else
      IFS=',' read -r -a requested_gpus <<<"$gpu_set"
      IFS=',' read -r -a visible_gpus <<<"$visible"
      for requested_gpu in "${requested_gpus[@]}"; do
        for visible_gpu in "${visible_gpus[@]}"; do
          [[ "$requested_gpu" == "$visible_gpu" ]] && overlap=1
        done
      done
    fi
    if ((overlap == 1)); then
      mooncake_die "检测到尚未占用显存但目标 GPU 重叠的 vLLM PID=${proc##*/}: $cmdline"
    fi
  done
}

mooncake_validate_instance_pid() {
  local port=$1 pid_file starttime_file pid identity pgid sid starttime expected
  pid_file="$(mooncake_instance_pid_file "$port")"
  starttime_file="$(mooncake_instance_starttime_file "$port")"
  mooncake_pid_running "$pid_file" || return 1
  [[ -s "$starttime_file" ]] || return 1
  pid="$(<"$pid_file")"
  identity="$(mooncake_process_identity "$pid")" || return 1
  read -r pgid sid starttime <<<"$identity"
  expected="$(<"$starttime_file")"
  [[ "$pgid" == "$pid" && "$sid" == "$pid" && "$starttime" == "$expected" ]] || \
    return 1
  local cmdline environment
  cmdline="$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline" || true)"
  if [[ "$cmdline" == *"$MOONCAKE_MODEL"* && "$cmdline" == *"--port $port"* ]]; then
    return 0
  fi
  environment="$(tr '\0' '\n' 2>/dev/null <"/proc/$pid/environ" || true)"
  [[ "$cmdline" == *"$MOONCAKE_SERVER"* && \
    "$environment" == *$'MODEL='"$MOONCAKE_MODEL"$'\n'* && \
    "$environment" == *$'PORT='"$port"$'\n'* ]]
}

mooncake_stop_instance() {
  local port=$1 pid_file pid
  pid_file="$(mooncake_instance_pid_file "$port")"
  mooncake_validate_instance_pid "$port" || return 1
  pid="$(<"$pid_file")"
  kill -TERM "$pid"
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || {
      rm -f "$pid_file" "$(mooncake_instance_starttime_file "$port")"
      return 0
    }
    sleep 1
  done
  return 1
}

mooncake_validate_master_pid() {
  local pid_file="$MOONCAKE_RUNTIME/master.pid"
  local starttime_file="$MOONCAKE_RUNTIME/master.starttime"
  local pid identity pgid sid starttime expected cmdline
  mooncake_pid_running "$pid_file" || return 1
  [[ -s "$starttime_file" ]] || return 1
  pid="$(<"$pid_file")"
  identity="$(mooncake_process_identity "$pid")" || return 1
  read -r pgid sid starttime <<<"$identity"
  expected="$(<"$starttime_file")"
  [[ "$pgid" == "$pid" && "$sid" == "$pid" && "$starttime" == "$expected" ]] || \
    return 1
  cmdline="$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline" || true)"
  [[ "$cmdline" == *"mooncake_master"* && \
    ( "$cmdline" == *"--rpc_port=$MOONCAKE_MASTER_PORT"* || \
      "$cmdline" == *"--port=$MOONCAKE_MASTER_PORT"* ) ]]
}
