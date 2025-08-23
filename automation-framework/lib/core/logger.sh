#!/usr/bin/env bash
# Logger library - Structured logging following Google's practices
# Provides consistent logging across all scripts

# Prevent multiple sourcing
[[ -n "${_LOGGER_SH_LOADED:-}" ]] && return 0
readonly _LOGGER_SH_LOADED=1

# Source dependencies
source "${BASH_SOURCE%/*}/colors.sh"

# Log levels
readonly LOG_LEVEL_DEBUG=0
readonly LOG_LEVEL_INFO=1
readonly LOG_LEVEL_WARNING=2
readonly LOG_LEVEL_ERROR=3
readonly LOG_LEVEL_FATAL=4

# Default log level
: "${LOG_LEVEL:=${LOG_LEVEL_INFO}}"
: "${LOG_FILE:=""}"
: "${LOG_DATE_FORMAT:="%Y-%m-%d %H:%M:%S"}"

# Check if output supports colors
if [[ -t 1 ]] && [[ -n "${TERM:-}" ]] && [[ "${TERM}" != "dumb" ]]; then
  readonly COLOR_OUTPUT=1
else
  readonly COLOR_OUTPUT=0
fi

# Get current timestamp
_get_timestamp() {
  date +"${LOG_DATE_FORMAT}"
}

# Get caller information
_get_caller() {
  local frame=${1:-1}
  local file line func
  
  # Get caller info from bash call stack
  if [[ ${#BASH_SOURCE[@]} -gt $((frame + 1)) ]]; then
    file="${BASH_SOURCE[$((frame + 1))]}"
    func="${FUNCNAME[$((frame + 1))]}"
    line="${BASH_LINENO[${frame}]}"
    
    # Make path relative to project root if possible
    if [[ -n "${FRAMEWORK_ROOT:-}" ]]; then
      file="${file#${FRAMEWORK_ROOT}/}"
    else
      file="$(basename "${file}")"
    fi
    
    echo "${file}:${line}:${func}()"
  else
    echo "unknown"
  fi
}

# Core logging function
_log() {
  local level="$1"
  local level_name="$2"
  local level_color="$3"
  local message="$4"
  
  # Check if we should log this level
  if [[ ${level} -lt ${LOG_LEVEL} ]]; then
    return 0
  fi
  
  # Skip if quiet mode is enabled (except for errors)
  if [[ -n "${QUIET:-}" ]] && [[ ${level} -lt ${LOG_LEVEL_ERROR} ]]; then
    return 0
  fi
  
  local timestamp
  timestamp="$(_get_timestamp)"
  
  local caller
  if [[ -n "${DEBUG:-}" ]] || [[ ${level} -ge ${LOG_LEVEL_ERROR} ]]; then
    caller="$(_get_caller 2)"
  else
    caller=""
  fi
  
  # Format message
  local formatted_message
  if [[ ${COLOR_OUTPUT} -eq 1 ]] && [[ -z "${LOG_FILE}" ]]; then
    # Colored output for terminal
    formatted_message="${GRAY}${timestamp}${RESET} "
    formatted_message+="${level_color}[${level_name}]${RESET} "
    [[ -n "${caller}" ]] && formatted_message+="${GRAY}${caller}${RESET} "
    formatted_message+="${message}"
  else
    # Plain text for files or non-terminal output
    formatted_message="${timestamp} [${level_name}] "
    [[ -n "${caller}" ]] && formatted_message+="${caller} "
    formatted_message+="${message}"
  fi
  
  # Output to stderr for warnings and errors
  if [[ ${level} -ge ${LOG_LEVEL_WARNING} ]]; then
    echo -e "${formatted_message}" >&2
  else
    echo -e "${formatted_message}"
  fi
  
  # Also write to log file if specified
  if [[ -n "${LOG_FILE}" ]]; then
    echo -e "${timestamp} [${level_name}] ${caller:+${caller} }${message}" >> "${LOG_FILE}"
  fi
  
  # Exit on fatal errors
  if [[ ${level} -eq ${LOG_LEVEL_FATAL} ]]; then
    exit 1
  fi
}

# Public logging functions
log_debug() {
  _log ${LOG_LEVEL_DEBUG} "DEBUG" "${CYAN}" "$*"
}

log_info() {
  _log ${LOG_LEVEL_INFO} "INFO" "${BLUE}" "$*"
}

log_success() {
  _log ${LOG_LEVEL_INFO} "SUCCESS" "${GREEN}" "$*"
}

log_warning() {
  _log ${LOG_LEVEL_WARNING} "WARNING" "${YELLOW}" "$*"
}

log_error() {
  _log ${LOG_LEVEL_ERROR} "ERROR" "${RED}" "$*"
}

log_fatal() {
  _log ${LOG_LEVEL_FATAL} "FATAL" "${BOLD}${RED}" "$*"
}

# Progress logging
log_progress() {
  local message="$1"
  if [[ ${COLOR_OUTPUT} -eq 1 ]] && [[ -z "${QUIET:-}" ]]; then
    echo -ne "\r${BLUE}⠋${RESET} ${message}..."
  fi
}

log_progress_done() {
  if [[ ${COLOR_OUTPUT} -eq 1 ]] && [[ -z "${QUIET:-}" ]]; then
    echo -e "\r${GREEN}✓${RESET} ${1:-Done}"
  fi
}

# Section headers for better organization
log_section() {
  local title="$1"
  local width=60
  local padding=$(( (width - ${#title} - 2) / 2 ))
  
  if [[ ${COLOR_OUTPUT} -eq 1 ]]; then
    echo
    echo -e "${BOLD}${BLUE}$(printf '=%.0s' {1..60})${RESET}"
    echo -e "${BOLD}${BLUE}=$(printf ' %.0s' $(seq 1 ${padding}))${title}$(printf ' %.0s' $(seq 1 ${padding}))=${RESET}"
    echo -e "${BOLD}${BLUE}$(printf '=%.0s' {1..60})${RESET}"
    echo
  else
    echo
    echo "$(printf '=%.0s' {1..60})"
    echo "=$(printf ' %.0s' $(seq 1 ${padding}))${title}$(printf ' %.0s' $(seq 1 ${padding}))="
    echo "$(printf '=%.0s' {1..60})"
    echo
  fi
}

# Step logging for multi-step processes
_step_counter=0
log_step() {
  ((_step_counter++))
  local message="$1"
  if [[ ${COLOR_OUTPUT} -eq 1 ]]; then
    echo -e "${BOLD}${BLUE}[${_step_counter}]${RESET} ${message}"
  else
    echo "[${_step_counter}] ${message}"
  fi
}

reset_step_counter() {
  _step_counter=0
}

# Command execution with logging
log_exec() {
  local cmd="$*"
  log_debug "Executing: ${cmd}"
  
  if [[ -n "${DRY_RUN:-}" ]]; then
    log_info "[DRY RUN] Would execute: ${cmd}"
    return 0
  fi
  
  if eval "${cmd}"; then
    log_debug "Command succeeded: ${cmd}"
    return 0
  else
    local exit_code=$?
    log_error "Command failed with exit code ${exit_code}: ${cmd}"
    return ${exit_code}
  fi
}

# Indent multiline output
log_indent() {
  local indent="${1:-  }"
  while IFS= read -r line; do
    echo "${indent}${line}"
  done
}

# Export functions
export -f log_debug
export -f log_info
export -f log_success
export -f log_warning
export -f log_error
export -f log_fatal
export -f log_progress
export -f log_progress_done
export -f log_section
export -f log_step
export -f reset_step_counter
export -f log_exec
export -f log_indent