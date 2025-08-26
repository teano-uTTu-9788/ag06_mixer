#!/usr/bin/env bash
# Core Library Functions - Google/Meta Inspired Patterns
# Provides fundamental utilities for all automation scripts

set -euo pipefail  # Google Shell Style: Fail fast with proper error handling

# Color definitions for better UX (Netflix pattern)
declare -r RED='\033[0;31m' 2>/dev/null || true
declare -r GREEN='\033[0;32m' 2>/dev/null || true
declare -r YELLOW='\033[1;33m' 2>/dev/null || true
declare -r BLUE='\033[0;34m' 2>/dev/null || true
declare -r MAGENTA='\033[0;35m' 2>/dev/null || true
declare -r CYAN='\033[0;36m' 2>/dev/null || true
declare -r NC='\033[0m' 2>/dev/null || true

# Logging levels (Google Cloud pattern)
readonly LOG_LEVEL_DEBUG=0
readonly LOG_LEVEL_INFO=1
readonly LOG_LEVEL_WARN=2
readonly LOG_LEVEL_ERROR=3
readonly LOG_LEVEL_FATAL=4

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
# Error Handling (Meta's Resilience Pattern)
# ============================================================================

# Stack trace on error (Meta pattern)
error::traceback() {
    local frame=0
    echo "Traceback (most recent call last):" >&2
    while caller $frame; do
        ((frame++))
    done | tac >&2
}

# Enhanced error trap
error::trap() {
    local exit_code=$?
    local line_no=$1
    log::error "Command failed with exit code $exit_code at line $line_no"
    error::traceback
    exit $exit_code
}

# Set global error trap
trap 'error::trap $LINENO' ERR

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

validate::env_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        log::error "Environment variable '$var_name' is not set"
        return 1
    fi
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
# Retry Logic (Netflix Resilience Pattern)
# ============================================================================

retry::exponential_backoff() {
    local max_attempts="${1:-3}"
    local delay="${2:-1}"
    local max_delay="${3:-60}"
    local attempt=1
    local exit_code=0

    shift 3
    local command=("$@")

    until [[ $attempt -gt $max_attempts ]]; do
        log::debug "Attempt $attempt/$max_attempts: ${command[*]}"
        
        if "${command[@]}"; then
            return 0
        else
            exit_code=$?
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            log::error "Command failed after $max_attempts attempts"
            return $exit_code
        fi

        log::warn "Command failed, retrying in ${delay}s..."
        sleep "$delay"
        
        # Exponential backoff with max delay
        delay=$((delay * 2))
        [[ $delay -gt $max_delay ]] && delay=$max_delay
        
        ((attempt++))
    done
}

# ============================================================================
# Parallel Execution (Dropbox's Resource Optimization Pattern)
# ============================================================================

parallel::run() {
    local max_jobs="${1:-4}"
    shift
    local commands=("$@")
    
    local pids=()
    local failed=0
    
    for cmd in "${commands[@]}"; do
        while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
            sleep 0.1
        done
        
        log::debug "Starting parallel job: $cmd"
        eval "$cmd" &
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
            log::error "Parallel job $pid failed"
        fi
    done
    
    return $failed
}

# ============================================================================
# Configuration Management (Airbnb Infrastructure as Code Pattern)
# ============================================================================

config::load() {
    local config_file="${1:-}"
    local default_config="${2:-$HOME/.config/automation/config}"
    
    # Try multiple config locations
    local config_locations=(
        "$config_file"
        "$default_config"
        "./config"
        "./.automation/config"
    )
    
    for location in "${config_locations[@]}"; do
        if [[ -f "$location" ]]; then
            log::debug "Loading configuration from: $location"
            # shellcheck source=/dev/null
            source "$location"
            return 0
        fi
    done
    
    log::warn "No configuration file found"
    return 1
}

config::get() {
    local key="$1"
    local default="${2:-}"
    
    echo "${!key:-$default}"
}

# ============================================================================
# Progress Indicators (Developer Experience Pattern)
# ============================================================================

progress::spinner() {
    local pid=$1
    local message="${2:-Processing}"
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    
    while kill -0 "$pid" 2>/dev/null; do
        local temp=${spinstr#?}
        printf " [%c] %s..." "${spinstr}" "${message}"
        spinstr=$temp${spinstr%"$temp"}
        sleep 0.1
        printf "\r"
    done
    
    wait "$pid"
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        printf " ${GREEN}[✓]${NC} %s... Done\n" "${message}"
    else
        printf " ${RED}[✗]${NC} %s... Failed\n" "${message}"
    fi
    
    return $exit_code
}

progress::bar() {
    local current=$1
    local total=$2
    local width=${3:-50}
    
    local percent=$((current * 100 / total))
    local filled=$((width * current / total))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((width - filled))s" | tr ' ' '-'
    printf "] %3d%%" "$percent"
    
    [[ $current -eq $total ]] && echo
}

# ============================================================================
# Dependency Management (Google's Module Pattern)
# ============================================================================

deps::require() {
    local lib="$1"
    local lib_path="${AUTOMATION_LIB_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
    
    if [[ ! -f "$lib_path/$lib.sh" ]]; then
        log::fatal "Required library '$lib' not found in $lib_path"
    fi
    
    # shellcheck source=/dev/null
    source "$lib_path/$lib.sh"
    log::debug "Loaded library: $lib"
}

# ============================================================================
# Cleanup Functions (Meta's Resource Management Pattern)
# ============================================================================

declare -a CLEANUP_FUNCTIONS=()

cleanup::register() {
    CLEANUP_FUNCTIONS+=("$1")
}

cleanup::execute() {
    local exit_code=${1:-$?}
    
    for func in "${CLEANUP_FUNCTIONS[@]}"; do
        log::debug "Running cleanup: $func"
        $func || true
    done
    
    exit "$exit_code"
}

# Register cleanup on exit
trap cleanup::execute EXIT

# ============================================================================
# Export core functions for use in other scripts
# ============================================================================

export -f log::debug log::info log::warn log::error log::fatal
export -f validate::command_exists validate::file_exists validate::directory_exists
export -f platform::detect platform::is_macos platform::is_linux
export -f retry::exponential_backoff
export -f parallel::run
export -f config::load config::get
export -f progress::spinner progress::bar
export -f deps::require
export -f cleanup::register