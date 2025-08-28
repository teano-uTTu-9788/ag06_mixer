#!/bin/bash
# Core library functions following Google Shell Style Guide
# Provides: Logging, error handling, validation utilities
# Used by: All framework scripts

set -euo pipefail  # Google recommended: Exit on error, undefined vars, pipe failures

# Color constants for output formatting (Meta/Google practice)
# Use framework-specific variable names to avoid conflicts
if [[ -z "${FRAMEWORK_RED:-}" ]]; then
  readonly FRAMEWORK_RED='\033[0;31m'
  readonly FRAMEWORK_GREEN='\033[0;32m'
  readonly FRAMEWORK_YELLOW='\033[1;33m'
  readonly FRAMEWORK_BLUE='\033[0;34m'
  readonly FRAMEWORK_PURPLE='\033[0;35m'
  readonly FRAMEWORK_CYAN='\033[0;36m'
  readonly FRAMEWORK_WHITE='\033[1;37m'
  readonly FRAMEWORK_NC='\033[0m' # No Color
  
  # Aliases for convenience
  readonly RED="${FRAMEWORK_RED}"
  readonly GREEN="${FRAMEWORK_GREEN}"
  readonly YELLOW="${FRAMEWORK_YELLOW}"
  readonly BLUE="${FRAMEWORK_BLUE}"
  readonly PURPLE="${FRAMEWORK_PURPLE}"
  readonly CYAN="${FRAMEWORK_CYAN}"
  readonly WHITE="${FRAMEWORK_WHITE}"
  readonly NC="${FRAMEWORK_NC}"
fi

# Global configuration (protected against multiple sourcing)
if [[ -z "${FRAMEWORK_SCRIPT_DIR:-}" ]]; then
  readonly FRAMEWORK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  readonly FRAMEWORK_PROJECT_ROOT="$(cd "${FRAMEWORK_SCRIPT_DIR}/../.." && pwd)"
  readonly FRAMEWORK_LOG_LEVEL="${LOG_LEVEL:-INFO}"
  readonly FRAMEWORK_LOG_FILE="${LOG_FILE:-${FRAMEWORK_PROJECT_ROOT}/automation.log}"
  
  # Aliases for convenience (non-readonly to avoid conflicts)
  SCRIPT_DIR="${FRAMEWORK_SCRIPT_DIR}"
  PROJECT_ROOT="${FRAMEWORK_PROJECT_ROOT}"
  LOG_LEVEL="${FRAMEWORK_LOG_LEVEL}"
  LOG_FILE="${FRAMEWORK_LOG_FILE}"
fi

# Logging functions (Meta-inspired structured logging)
log_debug() {
    [[ "${LOG_LEVEL}" == "DEBUG" ]] || return 0
    echo -e "${PURPLE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    echo "[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "${LOG_FILE}"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    echo "[OK] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >> "${LOG_FILE}"
}

log_fatal() {
    log_error "$@"
    exit 1
}

# Error handling (Google SRE practices)
setup_error_handling() {
    local script_name="${1:-${0##*/}}"
    
    # Trap for cleanup on exit
    trap 'cleanup_on_exit $?' EXIT
    
    # Trap for errors
    trap 'handle_error ${LINENO} "${BASH_COMMAND}"' ERR
    
    log_debug "Error handling setup for: ${script_name}"
}

handle_error() {
    local line_number="$1"
    local command="$2"
    local exit_code=$?
    
    log_error "Script failed at line ${line_number}: ${command} (exit code: ${exit_code})"
    
    # Optional: Send to monitoring system (Google-style incident response)
    if command -v send_to_monitoring >/dev/null 2>&1; then
        send_to_monitoring "script_error" "${0##*/}" "${line_number}" "${exit_code}"
    fi
}

cleanup_on_exit() {
    local exit_code="$1"
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_debug "Script completed successfully"
    else
        log_error "Script exited with code: ${exit_code}"
    fi
    
    # Clean up temp files if they exist
    if [[ -n "${TEMP_FILES:-}" ]]; then
        log_debug "Cleaning up temporary files"
        # shellcheck disable=SC2086
        rm -rf ${TEMP_FILES} 2>/dev/null || true
    fi
}

# Validation functions (Meta-style input validation)
validate_command() {
    local cmd="$1"
    local required="${2:-true}"
    
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        if [[ "${required}" == "true" ]]; then
            log_fatal "Required command '${cmd}' not found. Please install it and try again."
        else
            log_warn "Optional command '${cmd}' not found."
            return 1
        fi
    fi
    
    log_debug "Command '${cmd}' is available"
    return 0
}

validate_file() {
    local file="$1"
    local required="${2:-true}"
    
    if [[ ! -f "${file}" ]]; then
        if [[ "${required}" == "true" ]]; then
            log_fatal "Required file '${file}' not found."
        else
            log_warn "Optional file '${file}' not found."
            return 1
        fi
    fi
    
    log_debug "File '${file}' exists"
    return 0
}

validate_directory() {
    local dir="$1"
    local create="${2:-false}"
    
    if [[ ! -d "${dir}" ]]; then
        if [[ "${create}" == "true" ]]; then
            log_info "Creating directory: ${dir}"
            mkdir -p "${dir}"
        else
            log_fatal "Required directory '${dir}' not found."
        fi
    fi
    
    log_debug "Directory '${dir}' exists"
    return 0
}

# System information functions (Google-style environment detection)
get_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux) echo "linux" ;;
        *) echo "unknown" ;;
    esac
}

get_arch() {
    case "$(uname -m)" in
        x86_64) echo "amd64" ;;
        arm64|aarch64) echo "arm64" ;;
        *) echo "unknown" ;;
    esac
}

is_ci() {
    [[ -n "${CI:-}" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]] || [[ -n "${BUILD_ID:-}" ]]
}

is_macos() {
    [[ "$(get_os)" == "macos" ]]
}

# Progress indication (Meta-style user feedback)
show_spinner() {
    local pid="$1"
    local delay=0.1
    local spinstr='|/-\'
    
    log_info "Running process (PID: ${pid})..."
    
    while kill -0 "${pid}" 2>/dev/null; do
        local temp=${spinstr#?}
        printf " [%c]  " "${spinstr}"
        local spinstr=${temp}${spinstr%"${temp}"}
        sleep ${delay}
        printf "\b\b\b\b\b\b"
    done
    
    wait "${pid}"
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]]; then
        printf "    \b\b\b\b"
        log_ok "Process completed successfully"
    else
        printf "    \b\b\b\b"
        log_error "Process failed with exit code: ${exit_code}"
    fi
    
    return ${exit_code}
}

# Retry logic (Google SRE reliability patterns)
retry() {
    local max_attempts="$1"
    local delay="$2"
    shift 2
    
    local attempt=1
    while [[ ${attempt} -le ${max_attempts} ]]; do
        log_debug "Attempt ${attempt}/${max_attempts}: $*"
        
        if "$@"; then
            log_debug "Command succeeded on attempt ${attempt}"
            return 0
        fi
        
        if [[ ${attempt} -eq ${max_attempts} ]]; then
            log_error "Command failed after ${max_attempts} attempts: $*"
            return 1
        fi
        
        log_warn "Attempt ${attempt} failed, retrying in ${delay} seconds..."
        sleep "${delay}"
        ((attempt++))
    done
}

# Performance timing (Meta-style profiling)
timer_start() {
    readonly TIMER_START=$(date +%s.%N)
}

timer_end() {
    if [[ -z "${TIMER_START:-}" ]]; then
        log_warn "Timer not started"
        return 1
    fi
    
    local end_time
    end_time=$(date +%s.%N)
    local duration
    duration=$(echo "${end_time} - ${TIMER_START}" | bc -l 2>/dev/null || echo "0")
    
    log_info "Execution time: ${duration}s"
    
    # Optional: Send to metrics system
    if command -v send_metrics >/dev/null 2>&1; then
        send_metrics "execution_time" "${0##*/}" "${duration}"
    fi
}

# Template functions for common patterns
confirm() {
    local message="${1:-Are you sure?}"
    local default="${2:-n}"
    
    if is_ci; then
        log_info "CI environment detected, auto-confirming: ${message}"
        return 0
    fi
    
    local prompt
    if [[ "${default}" == "y" ]]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    echo -n "${message} ${prompt} "
    read -r response
    
    response="${response:-${default}}"
    case "${response,,}" in
        y|yes) return 0 ;;
        n|no) return 1 ;;
        *) 
            log_warn "Invalid response. Please answer yes or no."
            confirm "${message}" "${default}"
            ;;
    esac
}

# Export all functions for use in other scripts
export -f log_debug log_info log_ok log_warn log_error log_fatal
export -f setup_error_handling handle_error cleanup_on_exit
export -f validate_command validate_file validate_directory
export -f get_os get_arch is_ci is_macos
export -f show_spinner retry timer_start timer_end confirm