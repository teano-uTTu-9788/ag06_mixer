#!/usr/bin/env bash
#
# Logger - Structured logging following industry best practices
# Inspired by Google's glog and Meta's logging standards
#
set -euo pipefail

# Logger namespace
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FORMAT="${LOG_FORMAT:-pretty}"
LOG_TIMESTAMP="${LOG_TIMESTAMP:-true}"
LOG_FILE="${LOG_FILE:-}"

# Log level conversion
log::level_to_num() {
    case "$1" in
        TRACE) echo 0 ;;
        DEBUG) echo 1 ;;
        INFO) echo 2 ;;
        WARN|WARNING) echo 3 ;;
        ERROR) echo 4 ;;
        FATAL) echo 5 ;;
        *) echo 2 ;;  # Default to INFO
    esac
}

# Color codes for pretty printing
log::get_color() {
    case "$1" in
        TRACE) echo '\033[0;37m' ;;    # Light gray
        DEBUG) echo '\033[0;36m' ;;    # Cyan
        INFO) echo '\033[0;32m' ;;     # Green
        WARN|WARNING) echo '\033[1;33m' ;;  # Yellow
        ERROR) echo '\033[0;31m' ;;    # Red
        FATAL) echo '\033[1;31m' ;;    # Bold red
        *) echo '\033[0m' ;;           # Reset
    esac
}

# Get current timestamp in ISO 8601 format
log::timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%S.%3NZ'
}

# Check if log level should be printed
log::should_log() {
    local level="$1"
    local current_level_num="$(log::level_to_num "$LOG_LEVEL")"
    local check_level_num="$(log::level_to_num "$level")"
    
    [[ $check_level_num -ge $current_level_num ]]
}

# Core logging function
log::write() {
    local level="$1"
    local message="$2"
    local caller="${3:-}"
    
    # Check if we should log this level
    if ! log::should_log "$level"; then
        return 0
    fi
    
    local timestamp=""
    if [[ "$LOG_TIMESTAMP" == "true" ]]; then
        timestamp="$(log::timestamp) "
    fi
    
    local formatted_message=""
    case "$LOG_FORMAT" in
        json)
            # JSON structured logging (for machine parsing)
            formatted_message=$(jq -n \
                --arg timestamp "$timestamp" \
                --arg level "$level" \
                --arg message "$message" \
                --arg caller "$caller" \
                --arg pid "$$" \
                --arg user "${FRAMEWORK_USER:-unknown}" \
                --arg host "${FRAMEWORK_HOST:-unknown}" \
                '{
                    timestamp: $timestamp,
                    level: $level,
                    message: $message,
                    caller: $caller,
                    pid: ($pid | tonumber),
                    user: $user,
                    host: $host
                }')
            ;;
        pretty)
            # Human-readable format with colors
            local color="$(log::get_color "$level")"
            local reset="$(log::get_color "RESET")"
            formatted_message="${timestamp}${color}[${level}]${reset} ${message}"
            
            if [[ -n "$caller" ]]; then
                local trace_color="$(log::get_color "TRACE")"
                formatted_message+=" ${trace_color}(${caller})${reset}"
            fi
            ;;
        simple)
            # Simple format without colors
            formatted_message="${timestamp}[${level}] ${message}"
            ;;
    esac
    
    # Output to stderr (following Unix conventions)
    echo -e "$formatted_message" >&2
    
    # Also write to log file if specified
    if [[ -n "$LOG_FILE" ]]; then
        # Always use simple format for file logging
        echo "${timestamp}[${level}] ${message}" >> "$LOG_FILE"
    fi
}

# Get caller information for debugging
log::caller() {
    local frame="${1:-2}"  # Default to 2 frames up
    echo "${BASH_SOURCE[$frame]##*/}:${BASH_LINENO[$((frame-1))]}"
}

# Log level functions
log::trace() {
    log::write "TRACE" "$*" "$(log::caller)"
}

log::debug() {
    log::write "DEBUG" "$*" "$(log::caller)"
}

log::info() {
    log::write "INFO" "$*"
}

log::warn() {
    log::write "WARN" "$*"
}

log::warning() {
    log::warn "$@"
}

log::error() {
    log::write "ERROR" "$*" "$(log::caller)"
}

log::fatal() {
    log::write "FATAL" "$*" "$(log::caller)"
    exit 1
}

log::success() {
    # Use INFO level but with success indicator
    local message="✅ $*"
    log::write "INFO" "$message"
}

# Performance logging
log::perf_start() {
    local operation="$1"
    declare "PERF_START_${operation}=$(date +%s%N)"
}

log::perf_end() {
    local operation="$1"
    local start_var="PERF_START_${operation}"
    local start_time="${!start_var:-}"
    
    if [[ -n "$start_time" ]]; then
        local end_time=$(date +%s%N)
        local duration_ns=$((end_time - start_time))
        local duration_ms=$((duration_ns / 1000000))
        
        log::debug "Performance: $operation took ${duration_ms}ms"
        unset "$start_var"
    else
        log::warn "Performance: No start time found for operation '$operation'"
    fi
}

# Context logging (for structured logging)
log::with_context() {
    local context="$1"
    shift
    local level="$1"
    shift
    local message="$*"
    
    log::write "$level" "[$context] $message"
}

# Audit logging (for compliance and security)
log::audit() {
    local action="$1"
    local resource="${2:-}"
    local user="${FRAMEWORK_USER:-unknown}"
    local timestamp="$(log::timestamp)"
    
    local audit_message="AUDIT: user=$user action=$action"
    if [[ -n "$resource" ]]; then
        audit_message+=" resource=$resource"
    fi
    
    # Always log audit events regardless of log level
    local original_log_level="$LOG_LEVEL"
    LOG_LEVEL="TRACE"
    log::write "INFO" "$audit_message"
    LOG_LEVEL="$original_log_level"
    
    # Also write to dedicated audit log if configured
    if [[ -n "${AUDIT_LOG_FILE:-}" ]]; then
        echo "$timestamp $audit_message" >> "$AUDIT_LOG_FILE"
    fi
}

# Configure logger from environment or config
log::configure() {
    local config_file="${1:-}"
    
    if [[ -n "$config_file" ]] && [[ -f "$config_file" ]]; then
        # Source configuration file
        source "$config_file"
    fi
    
    # Validate log level
    if [[ -z "${LOG_LEVELS[$LOG_LEVEL]:-}" ]]; then
        log::warn "Invalid log level '$LOG_LEVEL', using INFO"
        LOG_LEVEL="INFO"
    fi
    
    # Create log file directory if needed
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
    fi
}

# Export functions for use by other modules
export -f log::trace log::debug log::info log::warn log::warning log::error log::fatal
export -f log::success log::with_context log::audit
export -f log::perf_start log::perf_end