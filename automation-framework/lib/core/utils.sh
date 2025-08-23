#!/usr/bin/env bash
# Common utility functions following Google Shell Style Guide
# Provides reusable functions for command operations, version checking, and system utilities

# Prevent multiple sourcing
[[ -n "${_UTILS_SH_LOADED:-}" ]] && return 0
readonly _UTILS_SH_LOADED=1

# Check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Compare version strings
# Usage: version_ge "1.2.3" "1.2.0" returns 0 (true)
version_ge() {
  local version1="$1"
  local version2="$2"
  
  # Use sort -V for version comparison if available
  if command_exists sort && sort --help 2>&1 | grep -q -- '-V'; then
    [[ "$(printf '%s\n' "$version2" "$version1" | sort -V | head -n1)" == "$version2" ]]
  else
    # Fallback to simple comparison
    [[ "$version1" == "$version2" ]] || [[ "$version1" > "$version2" ]]
  fi
}

# Get OS information
get_os() {
  case "$(uname -s)" in
    Darwin*)  echo "macos" ;;
    Linux*)   echo "linux" ;;
    CYGWIN*)  echo "cygwin" ;;
    MINGW*)   echo "mingw" ;;
    *)        echo "unknown" ;;
  esac
}

# Get macOS version
get_macos_version() {
  if [[ "$(get_os)" == "macos" ]]; then
    sw_vers -productVersion
  else
    echo "not-macos"
  fi
}

# Check if running on Apple Silicon
is_apple_silicon() {
  if [[ "$(get_os)" == "macos" ]]; then
    [[ "$(uname -m)" == "arm64" ]]
  else
    return 1
  fi
}

# Get number of CPU cores
get_cpu_cores() {
  if [[ "$(get_os)" == "macos" ]]; then
    sysctl -n hw.ncpu
  elif [[ "$(get_os)" == "linux" ]]; then
    nproc
  else
    echo 1
  fi
}

# Get available memory in MB
get_available_memory() {
  if [[ "$(get_os)" == "macos" ]]; then
    local pages_free pages_size
    pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    pages_size=$(pagesize)
    echo $(( pages_free * pages_size / 1024 / 1024 ))
  elif [[ "$(get_os)" == "linux" ]]; then
    free -m | awk 'NR==2{print $7}'
  else
    echo 0
  fi
}

# Create directory if it doesn't exist
ensure_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
  fi
}

# Backup file with timestamp
backup_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$file" "$backup"
    echo "$backup"
  fi
}

# Safe file replacement with atomic move
safe_replace() {
  local source="$1"
  local target="$2"
  
  if [[ ! -f "$source" ]]; then
    return 1
  fi
  
  # Backup existing file
  if [[ -f "$target" ]]; then
    backup_file "$target" >/dev/null
  fi
  
  # Atomic move
  mv -f "$source" "$target"
}

# Get file age in seconds
file_age() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "-1"
    return 1
  fi
  
  local current_time modified_time
  current_time=$(date +%s)
  
  if [[ "$(get_os)" == "macos" ]]; then
    modified_time=$(stat -f %m "$file")
  else
    modified_time=$(stat -c %Y "$file")
  fi
  
  echo $(( current_time - modified_time ))
}

# Check if file is older than N seconds
file_older_than() {
  local file="$1"
  local seconds="$2"
  local age
  
  age=$(file_age "$file")
  [[ $age -gt $seconds ]]
}

# URL encode string
url_encode() {
  local string="$1"
  printf '%s' "$string" | jq -sRr @uri
}

# URL decode string
url_decode() {
  local string="$1"
  printf '%b' "${string//%/\\x}"
}

# Generate random string
random_string() {
  local length="${1:-16}"
  LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c "$length"
}

# Get script's real directory (resolving symlinks)
get_script_dir() {
  local source="${BASH_SOURCE[0]}"
  local dir
  
  # Resolve symlinks
  while [[ -h "$source" ]]; do
    dir="$(cd -P "$(dirname "$source")" && pwd)"
    source="$(readlink "$source")"
    [[ $source != /* ]] && source="$dir/$source"
  done
  
  cd -P "$(dirname "$source")" && pwd
}

# Retry command with exponential backoff
retry_with_backoff() {
  local max_attempts="${1:-3}"
  local delay="${2:-1}"
  local max_delay="${3:-60}"
  shift 3
  
  local attempt=1
  local exit_code
  
  while [[ $attempt -le $max_attempts ]]; do
    if "$@"; then
      return 0
    fi
    
    exit_code=$?
    
    if [[ $attempt -eq $max_attempts ]]; then
      return $exit_code
    fi
    
    echo "Attempt $attempt failed. Retrying in ${delay}s..." >&2
    sleep "$delay"
    
    # Exponential backoff
    delay=$(( delay * 2 ))
    [[ $delay -gt $max_delay ]] && delay=$max_delay
    
    (( attempt++ ))
  done
}

# Check if running in CI environment
is_ci() {
  [[ -n "${CI:-}" ]] || \
  [[ -n "${GITHUB_ACTIONS:-}" ]] || \
  [[ -n "${JENKINS_URL:-}" ]] || \
  [[ -n "${CIRCLECI:-}" ]] || \
  [[ -n "${TRAVIS:-}" ]] || \
  [[ -n "${GITLAB_CI:-}" ]]
}

# Check if running interactively
is_interactive() {
  [[ -t 0 ]] && [[ -t 1 ]]
}

# Get terminal width
terminal_width() {
  if command_exists tput; then
    tput cols
  elif [[ -n "${COLUMNS:-}" ]]; then
    echo "${COLUMNS}"
  else
    echo 80
  fi
}

# Truncate string to terminal width
truncate_to_width() {
  local string="$1"
  local width="${2:-$(terminal_width)}"
  local suffix="${3:-...}"
  
  if [[ ${#string} -le $width ]]; then
    echo "$string"
  else
    local truncated_length=$(( width - ${#suffix} ))
    echo "${string:0:$truncated_length}${suffix}"
  fi
}

# Join array elements with delimiter
join_array() {
  local delimiter="$1"
  shift
  local first="$1"
  shift
  printf '%s' "$first" "${@/#/$delimiter}"
}

# Check if array contains element
array_contains() {
  local needle="$1"
  shift
  local element
  
  for element in "$@"; do
    [[ "$element" == "$needle" ]] && return 0
  done
  
  return 1
}

# Remove duplicates from array
array_unique() {
  local -A seen
  local element
  
  for element in "$@"; do
    if [[ -z "${seen[$element]:-}" ]]; then
      echo "$element"
      seen[$element]=1
    fi
  done
}

# Export all utility functions
export -f command_exists
export -f version_ge
export -f get_os
export -f get_macos_version
export -f is_apple_silicon
export -f get_cpu_cores
export -f get_available_memory
export -f ensure_dir
export -f backup_file
export -f safe_replace
export -f file_age
export -f file_older_than
export -f url_encode
export -f url_decode
export -f random_string
export -f get_script_dir
export -f retry_with_backoff
export -f is_ci
export -f is_interactive
export -f terminal_width
export -f truncate_to_width
export -f join_array
export -f array_contains
export -f array_unique