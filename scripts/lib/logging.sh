#!/usr/bin/env bash
# Logging Library - Following Google SRE structured logging practices

set -euo pipefail

# Configuration
LOG_DIR="${LOG_DIR:-$HOME/.local/log/dev-framework}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dev.log}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FORMAT="${LOG_FORMAT:-text}"  # text or json

# Colors for terminal output
readonly COLOR_RESET='\033[0m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_YELLOW='\033[0;33m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_BLUE='\033[0;34m'

# Ensure log directory exists
[[ ! -d "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"

# Core logging function
_log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")"
    
    # Log to file
    if [[ "$LOG_FORMAT" == "json" ]]; then
        # Structured logging for observability platforms
        printf '{"timestamp":"%s","level":"%s","message":"%s","pid":%d,"hostname":"%s"}\n' \
            "$timestamp" "$level" "$message" "$$" "$(hostname)" >> "$LOG_FILE"
    else
        printf "[%s] %s: %s\n" "$timestamp" "$level" "$message" >> "$LOG_FILE"
    fi
    
    # Console output with colors
    case "$level" in
        ERROR)   echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $message" >&2 ;;
        WARN)    echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $message" ;;
        SUCCESS) echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $message" ;;
        INFO)    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $message" ;;
        DEBUG)   [[ "$LOG_LEVEL" == "DEBUG" ]] && echo "[DEBUG] $message" ;;
    esac
}

# Public logging functions
log_info() { _log "INFO" "$1"; }
log_warn() { _log "WARN" "$1"; }
log_error() { _log "ERROR" "$1"; }
log_success() { _log "SUCCESS" "$1"; }
log_debug() { _log "DEBUG" "$1"; }

# Structured event logging (Google SRE pattern)
log_event() {
    local event_type="$1"
    local event_data="$2"
    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")"
    
    if [[ "$LOG_FORMAT" == "json" ]]; then
        printf '{"timestamp":"%s","event_type":"%s","data":%s}\n' \
            "$timestamp" "$event_type" "$event_data" >> "$LOG_FILE"
    else
        _log "EVENT" "$event_type: $event_data"
    fi
}

# Performance logging (latency tracking)
log_latency() {
    local operation="$1"
    local start_time="$2"
    local end_time="$3"
    local duration=$((end_time - start_time))
    
    log_event "performance" "{\"operation\":\"$operation\",\"duration_ms\":$duration}"
}

export -f log_info log_warn log_error log_success log_debug log_event log_latency