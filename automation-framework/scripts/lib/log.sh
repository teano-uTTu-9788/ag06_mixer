#!/usr/bin/env bash
# shellcheck shell=bash
# Logging with levels + CI-friendly, Google-style guidance
# Ref: Google Shell Style Guide
# https://google.github.io/styleguide/shellguide.html

LOG_LEVEL="${LOG_LEVEL:-info}"  # debug|info|warn|error
NO_COLOR="${NO_COLOR:-}"

_ts() { 
  date +"%Y-%m-%dT%H:%M:%S%z"
}

_color() { 
  [[ -n "$NO_COLOR" ]] && return 0
  case "$1" in
    green) printf '\033[32m';;
    yellow) printf '\033[33m';;
    red) printf '\033[31m';;
    blue) printf '\033[34m';;
    *) :;;
  esac
}

_reset() { 
  [[ -n "$NO_COLOR" ]] || printf '\033[0m'
}

_log() { 
  local lvl="$1"
  shift
  printf "%s [%s] %s\n" "$(_ts)" "$lvl" "$*" >&2
}

log_debug() { 
  [[ "$LOG_LEVEL" == "debug" ]] && _log DEBUG "$@"
}

log_info() {
  _log INFO "$@"
}

log_warn() {
  _log WARN "$@"
}

log_error() {
  _log ERROR "$@"
}

log_ok() {
  _color green
  _log OK "$@"
  _reset
}

die() {
  log_error "$*"
  exit 1
}