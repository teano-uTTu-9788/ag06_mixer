#!/usr/bin/env bash
# Utilities Library - Common functions following Meta patterns

set -euo pipefail

# Source logging
source "${BASH_SOURCE%/*}/logging.sh"

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Retry mechanism (Netflix resilience pattern)
retry_with_backoff() {
    local max_attempts="${1:-3}"
    local delay="${2:-1}"
    local max_delay="${3:-60}"
    shift 3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if "$@"; then
            return 0
        fi
        
        log_warn "Attempt $attempt/$max_attempts failed. Retrying in ${delay}s..."
        sleep "$delay"
        
        # Exponential backoff with jitter
        delay=$((delay * 2 + RANDOM % 3))
        [[ $delay -gt $max_delay ]] && delay=$max_delay
        
        ((attempt++))
    done
    
    log_error "All $max_attempts attempts failed"
    return 1
}

# Parallel execution (Google pattern)
run_parallel() {
    local -a pids=()
    local failed=0
    
    for cmd in "$@"; do
        eval "$cmd" &
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    
    return $failed
}

# Safe file operations
safe_copy() {
    local src="$1"
    local dst="$2"
    
    if [[ ! -f "$src" ]]; then
        log_error "Source file does not exist: $src"
        return 1
    fi
    
    # Create backup if destination exists
    if [[ -f "$dst" ]]; then
        cp "$dst" "${dst}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    cp "$src" "$dst"
    log_success "Copied $src to $dst"
}

# Template rendering (Meta DotSlash pattern)
render_template() {
    local template="$1"
    local output="$2"
    shift 2
    
    local content
    content=$(<"$template")
    
    # Replace variables
    for var in "$@"; do
        local key="${var%%=*}"
        local value="${var#*=}"
        content="${content//\{\{$key\}\}/$value}"
    done
    
    echo "$content" > "$output"
    log_info "Rendered template $template to $output"
}

# JSON operations (modern shell patterns)
json_get() {
    local json="$1"
    local key="$2"
    
    if command_exists jq; then
        echo "$json" | jq -r "$key"
    else
        # Fallback to Python
        python3 -c "import json; print(json.loads('$json')$key)"
    fi
}

export -f command_exists retry_with_backoff run_parallel safe_copy render_template json_get
