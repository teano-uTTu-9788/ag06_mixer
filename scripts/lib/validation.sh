#!/usr/bin/env bash
# Validation Library - Input validation and sanitization

set -euo pipefail

source "${BASH_SOURCE%/*}/logging.sh"

# Validate environment variables
validate_env() {
    local -a required_vars=("$@")
    local missing=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("$var")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing[*]}"
        return 1
    fi
    
    return 0
}

# Validate semver version
validate_version() {
    local version="$1"
    local pattern='^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?(\+[a-zA-Z0-9]+)?$'
    
    if [[ ! "$version" =~ $pattern ]]; then
        log_error "Invalid version format: $version"
        return 1
    fi
    
    return 0
}

# Validate URL
validate_url() {
    local url="$1"
    local pattern='^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$'
    
    if [[ ! "$url" =~ $pattern ]]; then
        log_error "Invalid URL format: $url"
        return 1
    fi
    
    return 0
}

# Validate file path (prevent directory traversal)
validate_path() {
    local path="$1"
    local base_dir="${2:-$PWD}"
    
    # Resolve to absolute path
    local abs_path
    abs_path="$(cd "$(dirname "$path")" 2>/dev/null && pwd)/$(basename "$path")"
    
    # Check if path is within allowed directory
    if [[ ! "$abs_path" == "$base_dir"* ]]; then
        log_error "Path traversal detected: $path"
        return 1
    fi
    
    return 0
}

# Sanitize user input
sanitize_input() {
    local input="$1"
    # Remove potentially dangerous characters
    echo "$input" | sed 's/[;&|`$<>]//g'
}

export -f validate_env validate_version validate_url validate_path sanitize_input
