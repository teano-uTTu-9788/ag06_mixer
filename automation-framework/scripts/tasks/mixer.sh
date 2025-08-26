#!/bin/bash
set -Eeuo pipefail
IFS=$'\n\t'

# shellcheck source=../lib/log.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib/log.sh"

LOG_DIR=".mixer_logs"; mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/gunicorn.pid"

task_mixer() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    start)
      log_info "Starting mixer on :8080..."
      if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log_warn "Already running (PID $(cat "$PID_FILE"))."
        exit 0
      fi
      # Free port if needed
      kill -TERM $(lsof -ti tcp:8080) 2>/dev/null || true
      sleep 0.5
      
      # Start with python for now (gunicorn has issues with sounddevice)
      nohup python3 production_mixer.py > "$LOG_DIR/server.log" 2>&1 & echo $! > "$PID_FILE"
      sleep 0.6
      
      # Start the audio engine automatically
      /usr/bin/curl -s -X POST http://127.0.0.1:8080/api/start >/dev/null 2>&1 || true
      
      log_ok "Started PID $(cat "$PID_FILE"). Logs: $LOG_DIR/server.log"
      ;;
    stop)
      if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        log_info "Stopping PID $(cat "$PID_FILE")..."
        # Stop audio engine first
        /usr/bin/curl -s -X POST http://127.0.0.1:8080/api/stop >/dev/null 2>&1 || true
        kill -TERM "$(cat "$PID_FILE")" || true
        sleep 1
        kill -KILL "$(cat "$PID_FILE")" 2>/dev/null || true
      else
        log_warn "No running PID."
      fi
      rm -f "$PID_FILE"
      ;;
    status)
      if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "RUNNING (PID $(cat "$PID_FILE"))"
        # Show audio status
        /usr/bin/curl -s http://127.0.0.1:8080/api/status | python3 -c "
import sys, json
data = json.load(sys.stdin)
m = data['metrics']
c = data['config']
print(f\"  Audio: {'ON' if m['running'] else 'OFF'}\")
print(f\"  Levels: RMS={m['rms_db']:.1f}dB, Peak={m['peak_db']:.1f}dB, LUFS={m['lufs_est']:.1f}\")
print(f\"  Config: AI Mix={c['ai_mix']:.1%}, Target={c['target_lufs']}LUFS\")
print(f\"  Device: In={m['device_in'] or 'default'}, Out={m['device_out'] or 'default'}\")
" 2>/dev/null || echo "  (Could not fetch status)"
      else
        echo "STOPPED"
      fi
      ;;
    logs)
      tail -n 200 -f "$LOG_DIR/server.log"
      ;;
    config)
      # Configure mixer parameters
      local ai_mix="${2:-0.8}"
      local target_lufs="${3:--14}"
      log_info "Setting AI mix=$ai_mix, target LUFS=$target_lufs"
      /usr/bin/curl -s -X POST http://127.0.0.1:8080/api/config \
        -H 'Content-Type: application/json' \
        -d "{\"ai_mix\":$ai_mix,\"target_lufs\":$target_lufs}" | python3 -m json.tool
      ;;
    monitor)
      # Live monitoring
      watch -n 0.5 '/usr/bin/curl -s http://127.0.0.1:8080/api/status | python3 -c "
import sys, json
data = json.load(sys.stdin)
m = data[\"metrics\"]
print(f\"RMS: {m[\"rms_db\"]:.1f}dB  Peak: {m[\"peak_db\"]:.1f}dB  LUFS: {m[\"lufs_est\"]:.1f}  Clip: {m[\"clipping\"]}\")"'
      ;;
    help|*)
      cat <<EOF
dev mixer <command> [args]

Commands:
  start          Start mixer server and audio engine
  stop           Stop mixer server
  status         Show server and audio status
  logs           Tail server logs
  config <mix> <lufs>  Configure AI mix (0-1) and target LUFS
  monitor        Live monitoring of audio levels
  help           Show this help

Examples:
  dev mixer start
  dev mixer config 0.8 -14
  dev mixer monitor
EOF
      ;;
  esac
}

# If sourced, export function; if executed, run it
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  task_mixer "$@"
fi