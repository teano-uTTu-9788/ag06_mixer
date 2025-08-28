#!/usr/bin/env bash
#
# Config - Configuration management following 12-factor app principles
# Multi-source configuration: files, environment variables, CLI flags
#
set -euo pipefail

# Configuration namespace (using variables since macOS bash doesn't support associative arrays)
CONFIG_LOADED=false

# Default configuration
config::defaults() {
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
}

# Load configuration from multiple sources (precedence order: CLI flags > env vars > config files > defaults)
config::load() {
    if [[ "$CONFIG_LOADED" == "true" ]]; then
        return 0
    fi
    
    # 1. Load defaults
    config::defaults
    
    # 2. Load from system config file
    config::load_file "/etc/terminal-automation/config"
    
    # 3. Load from user config file
    config::load_file "$HOME/.config/terminal-automation/config"
    
    # 4. Load from project config file
    config::load_file "./config/terminal-automation.conf"
    
    # 5. Load from environment variables (prefixed with TA_)
    config::load_env
    
    # 6. CLI flags will be handled by individual commands
    
    CONFIG_LOADED=true
    log::debug "Configuration loaded successfully"
}

# Load configuration from file
config::load_file() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        log::trace "Config file not found: $config_file"
        return 0
    fi
    
    log::debug "Loading config from: $config_file"
    
    # Read config file line by line
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ "$key" =~ ^[[:space:]]*$ ]] && continue
        
        # Trim whitespace
        key="$(echo "$key" | xargs)"
        value="$(echo "$value" | xargs)"
        
        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        
        # Store in configuration
        FRAMEWORK_CONFIG["$key"]="$value"
        
    done < "$config_file"
}

# Load configuration from environment variables
config::load_env() {
    local env_var
    local config_key
    
    # Check all environment variables starting with TA_
    while IFS= read -r env_var; do
        if [[ "$env_var" =~ ^TA_(.+)=(.*)$ ]]; then
            config_key="${BASH_REMATCH[1],,}"  # Convert to lowercase
            config_key="${config_key//_/.}"     # Replace underscores with dots
            local value="${BASH_REMATCH[2]}"
            
            FRAMEWORK_CONFIG["$config_key"]="$value"
            log::trace "Loaded from env: $config_key=$value"
        fi
    done < <(env | grep '^TA_')
}

# Get configuration value
config::get() {
    local key="$1"
    local default_value="${2:-}"
    
    if [[ ! "$CONFIG_LOADED" == "true" ]]; then
        config::load
    fi
    
    echo "${FRAMEWORK_CONFIG[$key]:-$default_value}"
}

# Set configuration value
config::set() {
    local key="$1"
    local value="$2"
    
    FRAMEWORK_CONFIG["$key"]="$value"
    log::trace "Config set: $key=$value"
}

# Check if configuration key exists
config::has() {
    local key="$1"
    
    if [[ ! "$CONFIG_LOADED" == "true" ]]; then
        config::load
    fi
    
    [[ -n "${FRAMEWORK_CONFIG[$key]:-}" ]]
}

# Get configuration as boolean
config::get_bool() {
    local key="$1"
    local default_value="${2:-false}"
    
    local value
    value="$(config::get "$key" "$default_value")"
    
    case "${value,,}" in
        true|yes|1|on|enabled)
            echo "true"
            ;;
        *)
            echo "false"
            ;;
    esac
}

# Get configuration as integer
config::get_int() {
    local key="$1"
    local default_value="${2:-0}"
    
    local value
    value="$(config::get "$key" "$default_value")"
    
    # Validate it's a number
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value"
    else
        log::warn "Invalid integer value for $key: $value, using default: $default_value"
        echo "$default_value"
    fi
}

# Save current configuration to file
config::save() {
    local config_file="$1"
    
    log::info "Saving configuration to: $config_file"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$config_file")"
    
    # Write configuration
    {
        echo "# Terminal Automation Framework Configuration"
        echo "# Generated on $(date)"
        echo ""
        
        for key in "${!FRAMEWORK_CONFIG[@]}"; do
            echo "$key=\"${FRAMEWORK_CONFIG[$key]}\""
        done
    } > "$config_file"
}

# Validate configuration
config::validate() {
    local errors=0
    
    # Validate log level
    local log_level
    log_level="$(config::get "log_level")"
    if [[ ! "$log_level" =~ ^(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)$ ]]; then
        log::error "Invalid log_level: $log_level"
        ((errors++))
    fi
    
    # Validate log format
    local log_format
    log_format="$(config::get "log_format")"
    if [[ ! "$log_format" =~ ^(json|pretty|simple)$ ]]; then
        log::error "Invalid log_format: $log_format"
        ((errors++))
    fi
    
    # Validate max_parallel_jobs
    local max_jobs
    max_jobs="$(config::get_int "max_parallel_jobs")"
    if [[ $max_jobs -lt 1 ]] || [[ $max_jobs -gt 32 ]]; then
        log::error "max_parallel_jobs must be between 1 and 32: $max_jobs"
        ((errors++))
    fi
    
    # Validate timeout
    local timeout
    timeout="$(config::get_int "timeout_seconds")"
    if [[ $timeout -lt 1 ]]; then
        log::error "timeout_seconds must be positive: $timeout"
        ((errors++))
    fi
    
    if [[ $errors -gt 0 ]]; then
        log::error "Configuration validation failed with $errors errors"
        return 1
    fi
    
    log::debug "Configuration validation passed"
    return 0
}

# Show current configuration
config::show() {
    if [[ ! "$CONFIG_LOADED" == "true" ]]; then
        config::load
    fi
    
    echo "Current Configuration:"
    echo "====================="
    
    for key in $(printf '%s\n' "${!FRAMEWORK_CONFIG[@]}" | sort); do
        local value="${FRAMEWORK_CONFIG[$key]}"
        
        # Mask sensitive values
        if [[ "$key" =~ (token|password|secret|key) ]]; then
            if [[ -n "$value" ]]; then
                value="***masked***"
            else
                value="<not set>"
            fi
        fi
        
        printf "%-20s = %s\n" "$key" "$value"
    done
}

# Initialize configuration on load
config::load

# Export functions for use by other modules
export -f config::get config::set config::has config::get_bool config::get_int
export -f config::show config::save config::validate