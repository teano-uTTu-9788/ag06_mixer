#!/usr/bin/env bash
# Core Library Functions - Google/Meta Inspired Patterns (Fixed)
# Provides fundamental utilities for all automation scripts

set -euo pipefail  # Google Shell Style: Fail fast with proper error handling

# Color definitions for better UX (Netflix pattern)
if [[ -z "${RED:-}" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
fi

# Logging levels (Google Cloud pattern)
if [[ -z "${LOG_LEVEL_DEBUG:-}" ]]; then
    LOG_LEVEL_DEBUG=0
    LOG_LEVEL_INFO=1
    LOG_LEVEL_WARN=2
    LOG_LEVEL_ERROR=3
    LOG_LEVEL_FATAL=4
fi

# Default log level
: ${LOG_LEVEL:=$LOG_LEVEL_INFO}

# ============================================================================
# Logging Functions (Google Cloud Logging Pattern)
# ============================================================================

log::timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log::debug() {
    [[ $LOG_LEVEL -le $LOG_LEVEL_DEBUG ]] && echo -e "${CYAN}[DEBUG $(log::timestamp)]${NC} $*" >&2
}

log::info() {
    [[ $LOG_LEVEL -le $LOG_LEVEL_INFO ]] && echo -e "${GREEN}[INFO $(log::timestamp)]${NC} $*" >&2
}

log::warn() {
    [[ $LOG_LEVEL -le $LOG_LEVEL_WARN ]] && echo -e "${YELLOW}[WARN $(log::timestamp)]${NC} $*" >&2
}

log::error() {
    [[ $LOG_LEVEL -le $LOG_LEVEL_ERROR ]] && echo -e "${RED}[ERROR $(log::timestamp)]${NC} $*" >&2
}

log::fatal() {
    echo -e "${RED}[FATAL $(log::timestamp)]${NC} $*" >&2
    exit 1
}

# ============================================================================
# Platform Detection (Uber's Multi-Platform Pattern)
# ============================================================================

platform::detect() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

platform::is_macos() {
    [[ "$(platform::detect)" == "macos" ]]
}

platform::is_linux() {
    [[ "$(platform::detect)" == "linux" ]]
}

platform::arch() {
    uname -m
}

# ============================================================================
# Validation Functions (Netflix Quality Pattern)
# ============================================================================

validate::command_exists() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log::error "Command '$cmd' not found. Please install it first."
        return 1
    fi
}

validate::file_exists() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        log::error "File '$file' does not exist"
        return 1
    fi
}

validate::directory_exists() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log::error "Directory '$dir' does not exist"
        return 1
    fi
}

# ============================================================================
# Simple Dependency Management
# ============================================================================

deps::require() {
    local lib="$1"
    local lib_path="${AUTOMATION_LIB_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
    
    if [[ -f "$lib_path/${lib}_fixed.sh" ]]; then
        source "$lib_path/${lib}_fixed.sh"
    elif [[ -f "$lib_path/$lib.sh" ]]; then
        source "$lib_path/$lib.sh" || true
    else
        log::warn "Library '$lib' not found, continuing anyway"
    fi
}

# ============================================================================
# Export core functions for use in other scripts
# ============================================================================

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    export -f log::debug log::info log::warn log::error log::fatal 2>/dev/null || true
    export -f platform::detect platform::is_macos platform::is_linux 2>/dev/null || true
    export -f validate::command_exists validate::file_exists validate::directory_exists 2>/dev/null || true
    export -f deps::require 2>/dev/null || true
fi