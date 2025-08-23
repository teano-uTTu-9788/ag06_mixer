#!/usr/bin/env bash
# shellcheck shell=bash
# Python toolchain with uv (fast) and pip fallback

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"

ensure_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    die "python3 missing"
  fi
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    log_info "Installing uv (fast Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

python_bootstrap() {
  ensure_python
  ensure_uv
  
  # Create venv in .venv using uv
  local python_version
  python_version="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
  
  log_info "Creating Python environment with uv (Python ${python_version})"
  uv venv --python "${python_version}"
  
  # Sync dependencies
  if [[ -f "pyproject.toml" ]]; then
    uv sync --frozen || uv sync
    log_ok "Python dependencies ready (uv)"
  else
    log_warn "No pyproject.toml found"
  fi
}

python_run_tests() {
  ensure_uv
  if [[ -f "pyproject.toml" ]]; then
    log_info "Running Python tests"
    uv run pytest -q || log_warn "Tests failed or pytest not configured"
  fi
}

python_lint() {
  ensure_uv
  if command -v ruff >/dev/null 2>&1 || [[ -f "pyproject.toml" ]]; then
    log_info "Running Python linter (ruff)"
    uv run ruff check . || log_warn "Linting issues found"
  fi
}

python_format() {
  ensure_uv
  if command -v ruff >/dev/null 2>&1 || [[ -f "pyproject.toml" ]]; then
    log_info "Formatting Python code (ruff)"
    uv run ruff format .
  fi
}

python_typecheck() {
  ensure_uv
  if [[ -f "pyproject.toml" ]]; then
    log_info "Running type checker (mypy)"
    uv run mypy . || log_warn "Type checking issues found"
  fi
}

# Fallback to pip if uv is not available
python_pip_install() {
  ensure_python
  local package="$1"
  
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$package"
  else
    python3 -m pip install "$package"
  fi
}