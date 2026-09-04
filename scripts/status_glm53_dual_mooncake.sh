#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=glm53_dual_mooncake_common.sh
source "$SCRIPT_DIR/glm53_dual_mooncake_common.sh"

failed=0
if mooncake_validate_master_pid && mooncake_port_in_use "$MOONCAKE_MASTER_PORT"; then
  mooncake_log "Mooncake master: RUNNING PID=$(<"$MOONCAKE_RUNTIME/master.pid") port=$MOONCAKE_MASTER_PORT"
else
  mooncake_log "Mooncake master: STOPPED/UNMANAGED port=$MOONCAKE_MASTER_PORT"
  failed=1
fi

for port in "${MOONCAKE_PORTS[@]}"; do
  if mooncake_validate_instance_pid "$port" && mooncake_healthy "$port"; then
    mooncake_log "GLM $port: HEALTHY PID=$(<"$(mooncake_instance_pid_file "$port")")"
  elif mooncake_validate_instance_pid "$port"; then
    mooncake_log "GLM $port: INITIALIZING/UNHEALTHY PID=$(<"$(mooncake_instance_pid_file "$port")")"
    failed=1
  elif mooncake_port_in_use "$port"; then
    mooncake_log "GLM $port: UNMANAGED LISTENER"
    failed=1
  else
    mooncake_log "GLM $port: STOPPED"
    failed=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
exit "$failed"
