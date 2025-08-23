#!/usr/bin/env bash
# shellcheck shell=bash
# Homebrew package management utilities

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"

brew_ensure() {
  if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew not installed. See https://brew.sh/"
  fi
  
  if ! brew list --versions "$1" >/dev/null 2>&1; then
    log_info "brew install $1"
    brew install "$1"
  fi
}

brew_bundle() {
  if [[ -f Brewfile ]]; then
    log_info "Running brew bundle"
    brew bundle --no-lock || true
  else
    log_warn "No Brewfile found"
  fi
}

brew_cleanup() {
  log_info "Cleaning up Homebrew..."
  brew cleanup -s
  brew autoremove
}

brew_doctor_check() {
  if brew doctor 2>&1 | grep -q "Your system is ready to brew"; then
    log_ok "Homebrew is healthy"
    return 0
  else
    log_warn "Homebrew doctor found issues"
    return 1
  fi
}

# Install Homebrew if not present
brew_install() {
  if command -v brew >/dev/null 2>&1; then
    log_info "Homebrew already installed"
    return 0
  fi
  
  log_info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Add to PATH for current session
  if [[ -d "/opt/homebrew" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    eval "$(/usr/local/bin/brew shellenv)"
  fi
  
  log_ok "Homebrew installed"
}