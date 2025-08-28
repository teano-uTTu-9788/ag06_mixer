#!/usr/bin/env bash
#
# Validation - Input validation and sanitization utilities
# Following OWASP security guidelines and defensive programming practices
#
set -euo pipefail

# Validation namespace
readonly VALIDATION_EMAIL_REGEX='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
readonly VALIDATION_URL_REGEX='^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
readonly VALIDATION_SEMVER_REGEX='^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'
readonly VALIDATION_UUID_REGEX='^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'

# Validate required tools are installed
validate::tools() {
    local tools=("$@")
    local missing=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log::error "Missing required tools: ${missing[*]}"
        log::info "Install missing tools with: brew install ${missing[*]}"
        return 1
    fi
    
    return 0
}

# Validate file exists and is readable
validate::file_readable() {
    local file="$1"
    local description="${2:-File}"
    
    if [[ ! -f "$file" ]]; then
        log::error "$description does not exist: $file"
        return 1
    fi
    
    if [[ ! -r "$file" ]]; then
        log::error "$description is not readable: $file"
        return 1
    fi
    
    return 0
}

# Validate directory exists and is writable
validate::dir_writable() {
    local dir="$1"
    local description="${2:-Directory}"
    
    if [[ ! -d "$dir" ]]; then
        log::error "$description does not exist: $dir"
        return 1
    fi
    
    if [[ ! -w "$dir" ]]; then
        log::error "$description is not writable: $dir"
        return 1
    fi
    
    return 0
}

# Validate string is not empty
validate::not_empty() {
    local value="$1"
    local name="${2:-Value}"
    
    if [[ -z "$value" ]]; then
        log::error "$name cannot be empty"
        return 1
    fi
    
    return 0
}

# Validate string matches pattern
validate::pattern() {
    local value="$1"
    local pattern="$2"
    local name="${3:-Value}"
    
    if [[ ! "$value" =~ $pattern ]]; then
        log::error "$name does not match required pattern: $value"
        return 1
    fi
    
    return 0
}

# Validate email address
validate::email() {
    local email="$1"
    
    validate::pattern "$email" "$VALIDATION_EMAIL_REGEX" "Email address"
}

# Validate URL
validate::url() {
    local url="$1"
    
    validate::pattern "$url" "$VALIDATION_URL_REGEX" "URL"
}

# Validate semantic version
validate::semver() {
    local version="$1"
    
    validate::pattern "$version" "$VALIDATION_SEMVER_REGEX" "Semantic version"
}

# Validate UUID
validate::uuid() {
    local uuid="$1"
    
    validate::pattern "$uuid" "$VALIDATION_UUID_REGEX" "UUID"
}

# Validate integer within range
validate::int_range() {
    local value="$1"
    local min="$2"
    local max="$3"
    local name="${4:-Value}"
    
    # Check if it's an integer
    if ! [[ "$value" =~ ^-?[0-9]+$ ]]; then
        log::error "$name must be an integer: $value"
        return 1
    fi
    
    # Check range
    if [[ $value -lt $min ]] || [[ $value -gt $max ]]; then
        log::error "$name must be between $min and $max: $value"
        return 1
    fi
    
    return 0
}

# Validate port number
validate::port() {
    local port="$1"
    
    validate::int_range "$port" 1 65535 "Port number"
}

# Validate IP address (IPv4)
validate::ipv4() {
    local ip="$1"
    
    local ipv4_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
    
    if ! validate::pattern "$ip" "$ipv4_regex" "IPv4 address"; then
        return 1
    fi
    
    # Validate each octet is 0-255
    IFS='.' read -ra octets <<< "$ip"
    for octet in "${octets[@]}"; do
        if [[ $octet -gt 255 ]]; then
            log::error "Invalid IPv4 address octet: $octet"
            return 1
        fi
    done
    
    return 0
}

# Validate hostname
validate::hostname() {
    local hostname="$1"
    
    local hostname_regex='^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    
    validate::pattern "$hostname" "$hostname_regex" "Hostname"
}

# Validate path is safe (no directory traversal)
validate::safe_path() {
    local path="$1"
    
    # Check for directory traversal attempts
    if [[ "$path" == *".."* ]]; then
        log::error "Unsafe path detected (directory traversal): $path"
        return 1
    fi
    
    # Check for absolute paths (depending on context, might be unwanted)
    if [[ "$path" =~ ^/ ]]; then
        log::warn "Absolute path provided: $path"
    fi
    
    return 0
}

# Validate GitHub repository format
validate::github_repo() {
    local repo="$1"
    
    local github_repo_regex='^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
    
    validate::pattern "$repo" "$github_repo_regex" "GitHub repository"
}

# Validate JSON string
validate::json() {
    local json="$1"
    
    if ! echo "$json" | jq . >/dev/null 2>&1; then
        log::error "Invalid JSON format"
        return 1
    fi
    
    return 0
}

# Validate YAML string
validate::yaml() {
    local yaml="$1"
    
    if command -v yq >/dev/null 2>&1; then
        if ! echo "$yaml" | yq . >/dev/null 2>&1; then
            log::error "Invalid YAML format"
            return 1
        fi
    else
        log::warn "yq not available, skipping YAML validation"
    fi
    
    return 0
}

# Sanitize input by removing/escaping dangerous characters
sanitize::filename() {
    local filename="$1"
    
    # Remove dangerous characters, keep alphanumeric, dots, dashes, underscores
    echo "$filename" | sed 's/[^a-zA-Z0-9._-]//g'
}

# Sanitize path by resolving .. and removing dangerous elements
sanitize::path() {
    local path="$1"
    
    # Use realpath to resolve .. and symlinks safely
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "$path" 2>/dev/null || echo "$path"
    else
        # Fallback: basic cleanup
        echo "$path" | sed 's|/\./|/|g; s|//\+|/|g'
    fi
}

# Sanitize shell input to prevent injection
sanitize::shell_input() {
    local input="$1"
    
    # Escape shell metacharacters
    printf '%q' "$input"
}

# Validate and sanitize user input with multiple checks
validate::user_input() {
    local input="$1"
    local type="${2:-string}"  # string, filename, path, email, url, etc.
    local max_length="${3:-1000}"
    
    # Length check
    if [[ ${#input} -gt $max_length ]]; then
        log::error "Input too long (max $max_length characters): ${#input}"
        return 1
    fi
    
    # Type-specific validation
    case "$type" in
        string)
            # Basic string validation
            validate::not_empty "$input" "String input"
            ;;
        filename)
            validate::not_empty "$input" "Filename"
            sanitize::filename "$input"
            ;;
        path)
            validate::safe_path "$input"
            sanitize::path "$input"
            ;;
        email)
            validate::email "$input"
            ;;
        url)
            validate::url "$input"
            ;;
        port)
            validate::port "$input"
            ;;
        ipv4)
            validate::ipv4 "$input"
            ;;
        hostname)
            validate::hostname "$input"
            ;;
        semver)
            validate::semver "$input"
            ;;
        uuid)
            validate::uuid "$input"
            ;;
        json)
            validate::json "$input"
            ;;
        yaml)
            validate::yaml "$input"
            ;;
        *)
            log::warn "Unknown validation type: $type"
            ;;
    esac
}

# Comprehensive system validation
validate::system() {
    local errors=0
    
    log::info "Running system validation..."
    
    # Check OS
    if [[ "$(uname -s)" != "Darwin" ]]; then
        log::warn "This framework is optimized for macOS (Darwin)"
    fi
    
    # Check required tools
    local required_tools=(bash git curl jq)
    if ! validate::tools "${required_tools[@]}"; then
        ((errors++))
    fi
    
    # Check optional but recommended tools
    local optional_tools=(brew shellcheck shfmt bats)
    for tool in "${optional_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log::warn "Recommended tool not found: $tool"
        fi
    done
    
    # Check disk space (at least 1GB free)
    local free_space
    free_space=$(df -h . | awk 'NR==2 {print $4}' | sed 's/[^0-9.]//g')
    if [[ $(echo "$free_space < 1" | bc 2>/dev/null || echo "0") -eq 1 ]]; then
        log::warn "Low disk space: ${free_space}GB available"
    fi
    
    # Check permissions for framework directories
    for dir in "$FRAMEWORK_CONFIG_DIR" "$FRAMEWORK_DATA_DIR" "$FRAMEWORK_CACHE_DIR"; do
        if ! validate::dir_writable "$dir" "Framework directory"; then
            ((errors++))
        fi
    done
    
    if [[ $errors -gt 0 ]]; then
        log::error "System validation failed with $errors errors"
        return 1
    fi
    
    log::success "System validation passed"
    return 0
}

# Export functions for use by other modules
export -f validate::tools validate::file_readable validate::dir_writable
export -f validate::not_empty validate::pattern validate::email validate::url
export -f validate::semver validate::uuid validate::int_range validate::port
export -f validate::ipv4 validate::hostname validate::safe_path validate::github_repo
export -f validate::json validate::yaml validate::user_input validate::system
export -f sanitize::filename sanitize::path sanitize::shell_input