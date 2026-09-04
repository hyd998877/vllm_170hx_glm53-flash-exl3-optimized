#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=glm53_dual_mooncake_common.sh
source "$SCRIPT_DIR/glm53_dual_mooncake_common.sh"

mkdir -p "$MOONCAKE_RUNTIME"
exec 9>"$MOONCAKE_RUNTIME/.control.lock"
flock -n 9 || mooncake_die "另一项 Mooncake 启停操作正在进行"

for port in "${MOONCAKE_PORTS[@]}"; do
  runtime="$(mooncake_instance_runtime "$port")"
  pid_file="$(mooncake_instance_pid_file "$port")"
  if mooncake_validate_instance_pid "$port"; then
    mooncake_log "停止托管的 GLM 实例：端口=$port PID=$(<"$pid_file")"
    mooncake_stop_instance "$port" || \
      mooncake_die "端口 $port 的托管实例在 60 秒内未退出；未强制终止"
  elif mooncake_port_in_use "$port"; then
    mooncake_die "端口 $port 在监听，但 PID 身份不匹配；拒绝终止未知进程"
  else
    rm -f "$pid_file" "$(mooncake_instance_starttime_file "$port")"
    mooncake_log "端口 $port 已停止"
  fi
  if [[ -s "$runtime/ipc.path" ]]; then
    rpc_path="$(<"$runtime/ipc.path")"
    if [[ "$rpc_path" == /tmp/vm-"$port"-* && -d "$rpc_path" ]]; then
      find "$rpc_path" -mindepth 1 -maxdepth 1 -type s -delete
      rmdir "$rpc_path" 2>/dev/null || true
    fi
    rm -f "$runtime/ipc.path"
  fi
done

pid_file="$MOONCAKE_RUNTIME/master.pid"
if mooncake_validate_master_pid; then
  pid="$(<"$pid_file")"
  mooncake_log "停止托管的 Mooncake master：PID=$pid"
  kill -TERM "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    mooncake_die "Mooncake master 在 30 秒内未退出；保留进程，不强制终止"
  fi
  rm -f "$pid_file" "$MOONCAKE_RUNTIME/master.starttime"
elif mooncake_port_in_use "$MOONCAKE_MASTER_PORT"; then
  mooncake_die "端口 $MOONCAKE_MASTER_PORT 在监听，但 master PID 身份不匹配"
else
  rm -f "$pid_file" "$MOONCAKE_RUNTIME/master.starttime"
  mooncake_log "Mooncake master 已停止"
fi
