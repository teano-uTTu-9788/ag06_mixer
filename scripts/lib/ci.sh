#!/usr/bin/env bash
# CI/CD Library - GitHub Actions and CI helpers

set -euo pipefail

source "${BASH_SOURCE%/*}/logging.sh"

# Detect CI environment
detect_ci() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "github"
    elif [[ -n "${CIRCLECI:-}" ]]; then
        echo "circle"
    elif [[ -n "${GITLAB_CI:-}" ]]; then
        echo "gitlab"
    else
        echo "local"
    fi
}

# GitHub Actions annotations
github_annotation() {
    local type="$1"  # error, warning, notice
    local message="$2"
    local file="${3:-}"
    local line="${4:-}"
    
    if [[ "$(detect_ci)" == "github" ]]; then
        echo "::$type file=$file,line=$line::$message"
    else
        log_info "[$type] $message"
    fi
}

# Set GitHub Actions output
set_output() {
    local name="$1"
    local value="$2"
    
    if [[ "$(detect_ci)" == "github" ]]; then
        echo "$name=$value" >> "$GITHUB_OUTPUT"
    else
        export "$name=$value"
    fi
}

# Create GitHub Actions summary
add_summary() {
    local content="$1"
    
    if [[ "$(detect_ci)" == "github" ]]; then
        echo "$content" >> "$GITHUB_STEP_SUMMARY"
    else
        log_info "Summary: $content"
    fi
}

# Cache operations
cache_restore() {
    local key="$1"
    local path="$2"
    
    if [[ "$(detect_ci)" == "github" ]]; then
        log_info "Restoring cache: $key"
        # GitHub Actions cache is handled via action
    else
        # Local cache
        local cache_dir="$HOME/.cache/dev-framework"
        local cache_file="$cache_dir/$key.tar.gz"
        
        if [[ -f "$cache_file" ]]; then
            tar -xzf "$cache_file" -C "$path"
            log_success "Cache restored from $cache_file"
        fi
    fi
}

cache_save() {
    local key="$1"
    local path="$2"
    
    if [[ "$(detect_ci)" != "github" ]]; then
        local cache_dir="$HOME/.cache/dev-framework"
        mkdir -p "$cache_dir"
        
        tar -czf "$cache_dir/$key.tar.gz" -C "$path" .
        log_success "Cache saved to $cache_dir/$key.tar.gz"
    fi
}

export -f detect_ci github_annotation set_output add_summary cache_restore cache_save
