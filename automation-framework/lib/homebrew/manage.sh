#!/usr/bin/env bash
# Homebrew package management automation
# Following Homebrew best practices and Google Shell Style Guide

set -euo pipefail

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FRAMEWORK_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

# Source core libraries
source "${FRAMEWORK_ROOT}/lib/core/colors.sh"
source "${FRAMEWORK_ROOT}/lib/core/logger.sh"
source "${FRAMEWORK_ROOT}/lib/core/utils.sh"
source "${FRAMEWORK_ROOT}/lib/core/validation.sh"

# Homebrew paths for different architectures
readonly HOMEBREW_PREFIX_INTEL="/usr/local"
readonly HOMEBREW_PREFIX_ARM="/opt/homebrew"

# Get Homebrew prefix based on architecture
get_brew_prefix() {
  if is_apple_silicon; then
    echo "${HOMEBREW_PREFIX_ARM}"
  else
    echo "${HOMEBREW_PREFIX_INTEL}"
  fi
}

# Check if Homebrew is installed
check_brew_installed() {
  if ! command_exists brew; then
    log_error "Homebrew is not installed"
    log_info "Install it from: https://brew.sh"
    return 1
  fi
  
  local brew_prefix
  brew_prefix="$(get_brew_prefix)"
  
  if [[ ! -d "${brew_prefix}" ]]; then
    log_error "Homebrew directory not found at ${brew_prefix}"
    return 1
  fi
  
  return 0
}

# Install Homebrew
install_brew() {
  log_info "Installing Homebrew..."
  
  if command_exists brew; then
    log_warning "Homebrew is already installed"
    return 0
  fi
  
  # Download and run official installer
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Add to PATH
  local brew_prefix
  brew_prefix="$(get_brew_prefix)"
  
  if [[ -f "${HOME}/.zshrc" ]]; then
    echo "eval \"\$(${brew_prefix}/bin/brew shellenv)\"" >> "${HOME}/.zshrc"
  fi
  
  if [[ -f "${HOME}/.bash_profile" ]]; then
    echo "eval \"\$(${brew_prefix}/bin/brew shellenv)\"" >> "${HOME}/.bash_profile"
  fi
  
  # Source brew environment for current session
  eval "$(${brew_prefix}/bin/brew shellenv)"
  
  log_success "Homebrew installed successfully"
}

# Update Homebrew
update_brew() {
  log_info "Updating Homebrew..."
  
  if ! check_brew_installed; then
    return 1
  fi
  
  brew update
  log_success "Homebrew updated successfully"
}

# Upgrade all packages
upgrade_packages() {
  log_info "Upgrading Homebrew packages..."
  
  if ! check_brew_installed; then
    return 1
  fi
  
  # Get list of outdated packages
  local outdated
  outdated=$(brew outdated --formula)
  
  if [[ -z "$outdated" ]]; then
    log_info "All packages are up to date"
    return 0
  fi
  
  log_info "Outdated packages:"
  echo "$outdated" | log_indent
  
  # Upgrade packages
  brew upgrade --formula
  
  # Upgrade casks if any are outdated
  if brew outdated --cask | grep -q .; then
    log_info "Upgrading casks..."
    brew upgrade --cask
  fi
  
  log_success "Packages upgraded successfully"
}

# Clean up old versions
cleanup_brew() {
  log_info "Cleaning up Homebrew..."
  
  if ! check_brew_installed; then
    return 1
  fi
  
  # Remove old versions
  brew cleanup -s
  
  # Remove cache
  rm -rf "$(brew --cache)"
  
  # Prune broken symlinks
  brew cleanup --prune=all
  
  log_success "Cleanup complete"
}

# Install package
install_package() {
  local package="$1"
  shift
  local options=("$@")
  
  if ! check_brew_installed; then
    return 1
  fi
  
  # Check if already installed
  if brew list --formula | grep -q "^${package}$"; then
    log_warning "Package '${package}' is already installed"
    return 0
  fi
  
  log_info "Installing ${package}..."
  
  if [[ ${#options[@]} -gt 0 ]]; then
    brew install "${package}" "${options[@]}"
  else
    brew install "${package}"
  fi
  
  log_success "Package '${package}' installed successfully"
}

# Uninstall package
uninstall_package() {
  local package="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  # Check if installed
  if ! brew list --formula | grep -q "^${package}$"; then
    log_warning "Package '${package}' is not installed"
    return 0
  fi
  
  log_info "Uninstalling ${package}..."
  brew uninstall "${package}"
  log_success "Package '${package}' uninstalled successfully"
}

# List installed packages
list_packages() {
  if ! check_brew_installed; then
    return 1
  fi
  
  log_section "Installed Packages"
  
  log_info "Formulae:"
  brew list --formula | log_indent
  
  echo
  log_info "Casks:"
  brew list --cask | log_indent
}

# Search for packages
search_packages() {
  local query="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Searching for '${query}'..."
  
  echo
  log_info "Formulae:"
  brew search --formula "${query}" | log_indent
  
  echo
  log_info "Casks:"
  brew search --cask "${query}" | log_indent
}

# Show package info
package_info() {
  local package="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  # Try formula first
  if brew list --formula | grep -q "^${package}$"; then
    brew info --formula "${package}"
    return 0
  fi
  
  # Try cask
  if brew list --cask | grep -q "^${package}$"; then
    brew info --cask "${package}"
    return 0
  fi
  
  # Not installed, search for it
  brew info "${package}"
}

# Check system health
doctor() {
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Running Homebrew doctor..."
  brew doctor
}

# Bundle operations (Brewfile)
bundle_install() {
  local brewfile="${1:-${FRAMEWORK_ROOT}/config/Brewfile}"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  if [[ ! -f "${brewfile}" ]]; then
    log_error "Brewfile not found: ${brewfile}"
    return 1
  fi
  
  log_info "Installing from Brewfile: ${brewfile}"
  brew bundle --file="${brewfile}"
  log_success "Bundle installation complete"
}

bundle_dump() {
  local brewfile="${1:-${FRAMEWORK_ROOT}/config/Brewfile}"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Dumping current packages to: ${brewfile}"
  brew bundle dump --file="${brewfile}" --force
  log_success "Bundle dump complete"
}

bundle_cleanup() {
  local brewfile="${1:-${FRAMEWORK_ROOT}/config/Brewfile}"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  if [[ ! -f "${brewfile}" ]]; then
    log_error "Brewfile not found: ${brewfile}"
    return 1
  fi
  
  log_info "Removing packages not in Brewfile..."
  brew bundle cleanup --file="${brewfile}"
}

# Services management
service_list() {
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Homebrew services:"
  brew services list
}

service_start() {
  local service="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Starting service: ${service}"
  brew services start "${service}"
  log_success "Service '${service}' started"
}

service_stop() {
  local service="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Stopping service: ${service}"
  brew services stop "${service}"
  log_success "Service '${service}' stopped"
}

service_restart() {
  local service="$1"
  
  if ! check_brew_installed; then
    return 1
  fi
  
  log_info "Restarting service: ${service}"
  brew services restart "${service}"
  log_success "Service '${service}' restarted"
}

# Main command handler
main() {
  local command="${1:-help}"
  shift || true
  
  case "${command}" in
    install-brew)
      install_brew "$@"
      ;;
    update)
      update_brew "$@"
      ;;
    upgrade)
      upgrade_packages "$@"
      ;;
    cleanup)
      cleanup_brew "$@"
      ;;
    install)
      install_package "$@"
      ;;
    uninstall|remove)
      uninstall_package "$@"
      ;;
    list)
      list_packages "$@"
      ;;
    search)
      search_packages "$@"
      ;;
    info)
      package_info "$@"
      ;;
    doctor)
      doctor "$@"
      ;;
    bundle-install)
      bundle_install "$@"
      ;;
    bundle-dump)
      bundle_dump "$@"
      ;;
    bundle-cleanup)
      bundle_cleanup "$@"
      ;;
    service-list)
      service_list "$@"
      ;;
    service-start)
      service_start "$@"
      ;;
    service-stop)
      service_stop "$@"
      ;;
    service-restart)
      service_restart "$@"
      ;;
    help)
      cat <<EOF
${BOLD}Homebrew Management${RESET}

${BOLD}USAGE:${RESET}
    dev brew <command> [options]

${BOLD}COMMANDS:${RESET}
    ${GREEN}install-brew${RESET}      Install Homebrew
    ${GREEN}update${RESET}            Update Homebrew itself
    ${GREEN}upgrade${RESET}           Upgrade all packages
    ${GREEN}cleanup${RESET}           Clean up old versions
    ${GREEN}install${RESET}           Install a package
    ${GREEN}uninstall${RESET}         Uninstall a package
    ${GREEN}list${RESET}              List installed packages
    ${GREEN}search${RESET}            Search for packages
    ${GREEN}info${RESET}              Show package information
    ${GREEN}doctor${RESET}            Check system health
    ${GREEN}bundle-install${RESET}    Install from Brewfile
    ${GREEN}bundle-dump${RESET}       Export to Brewfile
    ${GREEN}bundle-cleanup${RESET}    Remove unlisted packages
    ${GREEN}service-list${RESET}      List services
    ${GREEN}service-start${RESET}     Start a service
    ${GREEN}service-stop${RESET}      Stop a service
    ${GREEN}service-restart${RESET}   Restart a service

${BOLD}EXAMPLES:${RESET}
    dev brew install node
    dev brew search python
    dev brew upgrade
    dev brew service-start postgresql
EOF
      ;;
    *)
      log_error "Unknown command: ${command}"
      echo "Run 'dev brew help' for usage information."
      exit 1
      ;;
  esac
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi