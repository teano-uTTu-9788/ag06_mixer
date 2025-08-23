#!/usr/bin/env bash
# Input validation and sanitization functions
# Following security best practices from Google and OWASP

# Prevent multiple sourcing
[[ -n "${_VALIDATION_SH_LOADED:-}" ]] && return 0
readonly _VALIDATION_SH_LOADED=1

# Source dependencies
source "${BASH_SOURCE%/*}/utils.sh"

# Validate email address format
validate_email() {
  local email="$1"
  local regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
  
  if [[ $email =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate URL format
validate_url() {
  local url="$1"
  local regex="^(https?|ftp)://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$"
  
  if [[ $url =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate IP address (IPv4)
validate_ipv4() {
  local ip="$1"
  local regex="^([0-9]{1,3}\.){3}[0-9]{1,3}$"
  
  if [[ ! $ip =~ $regex ]]; then
    return 1
  fi
  
  # Check each octet is valid (0-255)
  local IFS='.'
  local -a octets=($ip)
  local octet
  
  for octet in "${octets[@]}"; do
    if [[ $octet -lt 0 ]] || [[ $octet -gt 255 ]]; then
      return 1
    fi
  done
  
  return 0
}

# Validate port number
validate_port() {
  local port="$1"
  
  if [[ ! $port =~ ^[0-9]+$ ]]; then
    return 1
  fi
  
  if [[ $port -lt 1 ]] || [[ $port -gt 65535 ]]; then
    return 1
  fi
  
  return 0
}

# Validate semantic version (x.y.z)
validate_semver() {
  local version="$1"
  local regex="^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$"
  
  if [[ $version =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate file path (no directory traversal)
validate_safe_path() {
  local path="$1"
  
  # Check for directory traversal attempts
  if [[ $path == *".."* ]]; then
    return 1
  fi
  
  # Check for absolute paths (optional - depends on requirements)
  if [[ $path == /* ]]; then
    return 1
  fi
  
  # Check for special characters that could be dangerous
  if [[ $path == *";"* ]] || [[ $path == *"|"* ]] || [[ $path == *"&"* ]]; then
    return 1
  fi
  
  return 0
}

# Validate alphanumeric string
validate_alphanumeric() {
  local string="$1"
  local allow_underscore="${2:-false}"
  local allow_dash="${3:-false}"
  
  local regex="^[a-zA-Z0-9"
  [[ $allow_underscore == "true" ]] && regex="${regex}_"
  [[ $allow_dash == "true" ]] && regex="${regex}-"
  regex="${regex}]+$"
  
  if [[ $string =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate integer
validate_integer() {
  local value="$1"
  local min="${2:-}"
  local max="${3:-}"
  
  # Check if it's an integer
  if [[ ! $value =~ ^-?[0-9]+$ ]]; then
    return 1
  fi
  
  # Check min boundary
  if [[ -n $min ]] && [[ $value -lt $min ]]; then
    return 1
  fi
  
  # Check max boundary
  if [[ -n $max ]] && [[ $value -gt $max ]]; then
    return 1
  fi
  
  return 0
}

# Validate float/decimal
validate_float() {
  local value="$1"
  local regex="^-?[0-9]+(\.[0-9]+)?$"
  
  if [[ $value =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate boolean
validate_boolean() {
  local value="$1"
  
  case "${value,,}" in
    true|false|yes|no|1|0|on|off)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Validate date format (YYYY-MM-DD)
validate_date() {
  local date="$1"
  local regex="^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
  
  if [[ ! $date =~ $regex ]]; then
    return 1
  fi
  
  # Validate with date command
  if command_exists date; then
    date -j -f "%Y-%m-%d" "$date" >/dev/null 2>&1
    return $?
  fi
  
  return 0
}

# Validate time format (HH:MM:SS)
validate_time() {
  local time="$1"
  local regex="^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$"
  
  if [[ $time =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate UUID
validate_uuid() {
  local uuid="$1"
  local regex="^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
  
  if [[ $uuid =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate hex color code
validate_hex_color() {
  local color="$1"
  local regex="^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"
  
  if [[ $color =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Sanitize string for safe shell usage
sanitize_for_shell() {
  local string="$1"
  
  # Remove potentially dangerous characters
  string="${string//[;&|<>$\`\\]/}"
  
  # Escape remaining special characters
  printf '%q' "$string"
}

# Sanitize filename
sanitize_filename() {
  local filename="$1"
  local replacement="${2:-_}"
  
  # Remove directory separators and dangerous characters
  filename="${filename//[\/\\:*?\"<>|]/$replacement}"
  
  # Remove leading/trailing dots and spaces
  filename="${filename#.}"
  filename="${filename%.}"
  filename="${filename## }"
  filename="${filename%% }"
  
  # Limit length (255 chars typical filesystem limit)
  if [[ ${#filename} -gt 255 ]]; then
    filename="${filename:0:255}"
  fi
  
  echo "$filename"
}

# Validate environment variable name
validate_env_var_name() {
  local name="$1"
  local regex="^[a-zA-Z_][a-zA-Z0-9_]*$"
  
  if [[ $name =~ $regex ]]; then
    return 0
  else
    return 1
  fi
}

# Validate JSON format
validate_json() {
  local json="$1"
  
  if command_exists jq; then
    echo "$json" | jq empty >/dev/null 2>&1
    return $?
  else
    # Basic check if jq is not available
    [[ $json == "{"* ]] && [[ $json == *"}" ]]
    return $?
  fi
}

# Validate YAML format
validate_yaml() {
  local yaml="$1"
  
  if command_exists yq; then
    echo "$yaml" | yq eval '.' >/dev/null 2>&1
    return $?
  else
    # Very basic check if yq is not available
    [[ $yaml != *"{"* ]] && [[ $yaml != *"["* ]]
    return $?
  fi
}

# Validate required parameters
validate_required() {
  local param_name="$1"
  local param_value="$2"
  
  if [[ -z "$param_value" ]]; then
    echo "Error: Required parameter '$param_name' is missing" >&2
    return 1
  fi
  
  return 0
}

# Validate enum (value in list)
validate_enum() {
  local value="$1"
  shift
  local valid_values=("$@")
  
  if array_contains "$value" "${valid_values[@]}"; then
    return 0
  else
    echo "Error: Invalid value '$value'. Must be one of: ${valid_values[*]}" >&2
    return 1
  fi
}

# Validate file exists
validate_file_exists() {
  local file="$1"
  
  if [[ ! -f "$file" ]]; then
    echo "Error: File '$file' does not exist" >&2
    return 1
  fi
  
  return 0
}

# Validate directory exists
validate_dir_exists() {
  local dir="$1"
  
  if [[ ! -d "$dir" ]]; then
    echo "Error: Directory '$dir' does not exist" >&2
    return 1
  fi
  
  return 0
}

# Validate file is readable
validate_file_readable() {
  local file="$1"
  
  if [[ ! -r "$file" ]]; then
    echo "Error: File '$file' is not readable" >&2
    return 1
  fi
  
  return 0
}

# Validate file is writable
validate_file_writable() {
  local file="$1"
  
  if [[ ! -w "$file" ]]; then
    echo "Error: File '$file' is not writable" >&2
    return 1
  fi
  
  return 0
}

# Export validation functions
export -f validate_email
export -f validate_url
export -f validate_ipv4
export -f validate_port
export -f validate_semver
export -f validate_safe_path
export -f validate_alphanumeric
export -f validate_integer
export -f validate_float
export -f validate_boolean
export -f validate_date
export -f validate_time
export -f validate_uuid
export -f validate_hex_color
export -f sanitize_for_shell
export -f sanitize_filename
export -f validate_env_var_name
export -f validate_json
export -f validate_yaml
export -f validate_required
export -f validate_enum
export -f validate_file_exists
export -f validate_dir_exists
export -f validate_file_readable
export -f validate_file_writable