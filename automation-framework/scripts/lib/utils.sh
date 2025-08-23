#!/usr/bin/env bash
# shellcheck shell=bash
# Defensive helpers with strict-mode-friendly patterns

set -Eeo pipefail
shopt -s inherit_errexit 2>/dev/null || true

# Source logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"

trap 's=$?; log_error "ERR at line $LINENO: $BASH_COMMAND (exit=$s)"; exit $s' ERR

required() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"
}

is_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1
}

current_branch() {
  git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
}

realpath_portable() {
  python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$1"
}

# Check if running on macOS
is_macos() {
  [[ "$(uname -s)" == "Darwin" ]]
}

# Check if running in CI
is_ci() {
  [[ -n "${CI:-}" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]]
}

# Get number of CPU cores
get_cpu_cores() {
  if is_macos; then
    sysctl -n hw.ncpu
  else
    nproc 2>/dev/null || echo 1
  fi
}

# Retry with exponential backoff
retry_with_backoff() {
  local max_attempts="${1:-3}"
  local delay="${2:-1}"
  shift 2
  
  local attempt=1
  while [[ $attempt -le $max_attempts ]]; do
    if "$@"; then
      return 0
    fi
    
    log_warn "Attempt $attempt failed, retrying in ${delay}s..."
    sleep "$delay"
    delay=$((delay * 2))
    ((attempt++))
  done
  
  log_error "All $max_attempts attempts failed"
  return 1
}