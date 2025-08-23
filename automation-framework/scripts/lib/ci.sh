#!/usr/bin/env bash
# shellcheck shell=bash
# CI/CD utilities for shell and Python

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"
# shellcheck source=./utils.sh
source "${SCRIPT_DIR}/utils.sh"

ci_shell() {
  log_info "Running ShellCheck on all shell scripts"
  
  # Find all shell scripts
  local scripts
  scripts=$(find . -type f -name "*.sh" -o -name "dev" | grep -v ".venv" | grep -v "node_modules")
  
  local failed=0
  for script in $scripts; do
    if ! shellcheck -S warning -x "$script"; then
      ((failed++))
    fi
  done
  
  if [[ $failed -gt 0 ]]; then
    log_error "ShellCheck found issues in $failed file(s)"
    return 1
  fi
  
  log_ok "ShellCheck passed"
  
  # Check formatting with shfmt
  log_info "Checking shell script formatting"
  if command -v shfmt >/dev/null 2>&1; then
    if ! shfmt -d -i 2 -ci scripts dev 2>/dev/null; then
      log_error "Shell scripts need formatting (run: dev fmt)"
      return 1
    fi
    log_ok "Shell formatting check passed"
  else
    log_warn "shfmt not installed, skipping format check"
  fi
}

ci_python() {
  # Source Python utilities
  # shellcheck source=./python.sh
  source "${SCRIPT_DIR}/python.sh"
  
  python_lint
  python_typecheck
  python_run_tests
}

ci_full() {
  log_info "Running full CI suite"
  
  local failed=0
  
  # Shell checks
  if ! ci_shell; then
    ((failed++))
  fi
  
  # Python checks
  if [[ -f "pyproject.toml" ]] || [[ -f "python/pyproject.toml" ]]; then
    if ! ci_python; then
      ((failed++))
    fi
  fi
  
  if [[ $failed -gt 0 ]]; then
    log_error "CI checks failed"
    return 1
  fi
  
  log_ok "All CI checks passed"
}

# GitHub Actions helpers
is_github_actions() {
  [[ -n "${GITHUB_ACTIONS:-}" ]]
}

github_annotation() {
  local level="$1"  # notice, warning, error
  local message="$2"
  local file="${3:-}"
  local line="${4:-}"
  
  if is_github_actions; then
    if [[ -n "$file" ]] && [[ -n "$line" ]]; then
      echo "::${level} file=${file},line=${line}::${message}"
    else
      echo "::${level}::${message}"
    fi
  else
    log_info "[$level] $message"
  fi
}

github_group() {
  local title="$1"
  if is_github_actions; then
    echo "::group::${title}"
  else
    log_info "=== ${title} ==="
  fi
}

github_endgroup() {
  if is_github_actions; then
    echo "::endgroup::"
  fi
}