#!/usr/bin/env bash
#
# Bootstrap - Core framework initialization
# Following Google SRE and Meta DevInfra best practices
#
set -euo pipefail

# Framework namespace
FRAMEWORK_INITIALIZED=false

# Initialize the framework
framework::init() {
    if [[ "$FRAMEWORK_INITIALIZED" == "true" ]]; then
        return 0
    fi
    
    # Set strict error handling
    set -euo pipefail
    
    # Set framework globals (only if not already set)
    if [[ -z "${FRAMEWORK_START_TIME:-}" ]]; then
        readonly FRAMEWORK_START_TIME=$(date +%s)
    fi
    if [[ -z "${FRAMEWORK_PID:-}" ]]; then
        readonly FRAMEWORK_PID=$$
    fi
    if [[ -z "${FRAMEWORK_USER:-}" ]]; then
        readonly FRAMEWORK_USER="${USER:-unknown}"
    fi
    if [[ -z "${FRAMEWORK_HOST:-}" ]]; then
        readonly FRAMEWORK_HOST="${HOSTNAME:-unknown}"
    fi
    
    # OS Detection (macOS focused) - only if not already set
    if [[ -z "${FRAMEWORK_OS:-}" ]]; then
        case "$(uname -s)" in
            Darwin)
                readonly FRAMEWORK_OS="macos"
                readonly FRAMEWORK_OS_VERSION="$(sw_vers -productVersion)"
                ;;
            Linux)
                readonly FRAMEWORK_OS="linux"
                readonly FRAMEWORK_OS_VERSION="$(uname -r)"
                ;;
            *)
                readonly FRAMEWORK_OS="unknown"
                readonly FRAMEWORK_OS_VERSION="unknown"
                ;;
        esac
    fi
    
    # Homebrew detection and setup
    if [[ "$FRAMEWORK_OS" == "macos" ]] && command -v brew >/dev/null 2>&1; then
        if [[ -z "${HOMEBREW_PREFIX:-}" ]]; then
            readonly HOMEBREW_PREFIX="$(brew --prefix)"
        fi
        if [[ -z "${HOMEBREW_VERSION:-}" ]]; then
            readonly HOMEBREW_VERSION="$(brew --version | head -n1 | cut -d' ' -f2)"
        fi
        
        # Add Homebrew binaries to PATH if not already present
        if [[ ":$PATH:" != *":$HOMEBREW_PREFIX/bin:"* ]]; then
            export PATH="$HOMEBREW_PREFIX/bin:$PATH"
        fi
    fi
    
    # XDG Base Directory Specification - only if not already set
    if [[ -z "${XDG_CONFIG_HOME:-}" ]]; then
        readonly XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
    fi
    if [[ -z "${XDG_DATA_HOME:-}" ]]; then
        readonly XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
    fi
    if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
        readonly XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    fi
    
    # Framework directories - only if not already set
    if [[ -z "${FRAMEWORK_CONFIG_DIR:-}" ]]; then
        readonly FRAMEWORK_CONFIG_DIR="$XDG_CONFIG_HOME/terminal-automation"
    fi
    if [[ -z "${FRAMEWORK_DATA_DIR:-}" ]]; then
        readonly FRAMEWORK_DATA_DIR="$XDG_DATA_HOME/terminal-automation"
    fi
    if [[ -z "${FRAMEWORK_CACHE_DIR:-}" ]]; then
        readonly FRAMEWORK_CACHE_DIR="$XDG_CACHE_HOME/terminal-automation"
    fi
    
    # Create framework directories
    mkdir -p "$FRAMEWORK_CONFIG_DIR" "$FRAMEWORK_DATA_DIR" "$FRAMEWORK_CACHE_DIR"
    
    # Signal that framework is initialized
    FRAMEWORK_INITIALIZED=true
    
    return 0
}

# Cleanup function
framework::cleanup() {
    local exit_code=$?
    
    # Log cleanup if logger is available
    if declare -f log::debug >/dev/null 2>&1; then
        log::debug "Framework cleanup initiated (exit code: $exit_code)"
    fi
    
    # Remove temporary files
    if [[ -n "${FRAMEWORK_TEMP_DIR:-}" ]] && [[ -d "$FRAMEWORK_TEMP_DIR" ]]; then
        rm -rf "$FRAMEWORK_TEMP_DIR"
    fi
    
    # Additional cleanup tasks can be added here
    
    exit $exit_code
}

# Trap cleanup on exit
trap 'framework::cleanup' EXIT

# Auto-initialize framework
framework::init

# Export key functions for use by other modules
export -f framework::init framework::cleanup