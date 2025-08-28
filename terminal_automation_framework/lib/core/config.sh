#!/usr/bin/env bash
#
# Config - Simplified configuration management for macOS bash compatibility
# Basic configuration using regular variables
#
set -euo pipefail

# Configuration namespace
CONFIG_LOADED=false

# Default configuration values
CONFIG_log_level="INFO"
CONFIG_log_format="pretty"
CONFIG_log_file=""
CONFIG_homebrew_auto_install="true"
CONFIG_github_token=""
CONFIG_ci_enabled="false"
CONFIG_debug_mode="false"
CONFIG_temp_dir="/tmp/terminal-automation"
CONFIG_max_parallel_jobs="4"
CONFIG_timeout_seconds="300"

# Load configuration from environment variables if available
config::load() {
    if [[ "$CONFIG_LOADED" == "true" ]]; then
        return 0
    fi
    
    # Override with environment variables if set
    CONFIG_log_level="${FRAMEWORK_LOG_LEVEL:-$CONFIG_log_level}"
    CONFIG_log_format="${FRAMEWORK_LOG_FORMAT:-$CONFIG_log_format}"
    CONFIG_log_file="${FRAMEWORK_LOG_FILE:-$CONFIG_log_file}"
    CONFIG_homebrew_auto_install="${FRAMEWORK_HOMEBREW_AUTO_INSTALL:-$CONFIG_homebrew_auto_install}"
    CONFIG_github_token="${FRAMEWORK_GITHUB_TOKEN:-$CONFIG_github_token}"
    CONFIG_ci_enabled="${FRAMEWORK_CI_ENABLED:-$CONFIG_ci_enabled}"
    CONFIG_debug_mode="${FRAMEWORK_DEBUG_MODE:-$CONFIG_debug_mode}"
    CONFIG_temp_dir="${FRAMEWORK_TEMP_DIR:-$CONFIG_temp_dir}"
    CONFIG_max_parallel_jobs="${FRAMEWORK_MAX_PARALLEL_JOBS:-$CONFIG_max_parallel_jobs}"
    CONFIG_timeout_seconds="${FRAMEWORK_TIMEOUT_SECONDS:-$CONFIG_timeout_seconds}"
    
    CONFIG_LOADED=true
}

# Get configuration value
config::get() {
    local key="$1"
    local default_value="${2:-}"
    
    # Ensure config is loaded
    config::load
    
    case "$key" in
        log_level) echo "$CONFIG_log_level" ;;
        log_format) echo "$CONFIG_log_format" ;;
        log_file) echo "$CONFIG_log_file" ;;
        homebrew_auto_install) echo "$CONFIG_homebrew_auto_install" ;;
        github_token) echo "$CONFIG_github_token" ;;
        ci_enabled) echo "$CONFIG_ci_enabled" ;;
        debug_mode) echo "$CONFIG_debug_mode" ;;
        temp_dir) echo "$CONFIG_temp_dir" ;;
        max_parallel_jobs) echo "$CONFIG_max_parallel_jobs" ;;
        timeout_seconds) echo "$CONFIG_timeout_seconds" ;;
        *) echo "$default_value" ;;
    esac
}

# Set configuration value
config::set() {
    local key="$1"
    local value="$2"
    
    case "$key" in
        log_level) CONFIG_log_level="$value" ;;
        log_format) CONFIG_log_format="$value" ;;
        log_file) CONFIG_log_file="$value" ;;
        homebrew_auto_install) CONFIG_homebrew_auto_install="$value" ;;
        github_token) CONFIG_github_token="$value" ;;
        ci_enabled) CONFIG_ci_enabled="$value" ;;
        debug_mode) CONFIG_debug_mode="$value" ;;
        temp_dir) CONFIG_temp_dir="$value" ;;
        max_parallel_jobs) CONFIG_max_parallel_jobs="$value" ;;
        timeout_seconds) CONFIG_timeout_seconds="$value" ;;
    esac
}

# Check if configuration key has a value
config::has() {
    local key="$1"
    local value
    value="$(config::get "$key")"
    [[ -n "$value" ]]
}

# Get boolean configuration value
config::get_bool() {
    local key="$1"
    local default_value="${2:-false}"
    local value
    value="$(config::get "$key" "$default_value")"
    
    case "$value" in
        true|True|TRUE|1|yes|Yes|YES|on|On|ON)
            echo "true"
            ;;
        *)
            echo "false"
            ;;
    esac
}

# Get integer configuration value
config::get_int() {
    local key="$1"
    local default_value="${2:-0}"
    local value
    value="$(config::get "$key" "$default_value")"
    
    # Validate it's a number
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value"
    else
        echo "$default_value"
    fi
}

# Export functions for use by other modules
export -f config::load config::get config::set config::has config::get_bool config::get_int