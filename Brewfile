# Brewfile - Homebrew Bundle for AG06 Mixer Development
# Install with: brew bundle

# Taps (Third-party repositories)
tap "homebrew/bundle"
tap "homebrew/services"

# ============================================================================
# Development Tools
# ============================================================================

# Version Control
brew "git"
brew "git-lfs"
brew "gh"  # GitHub CLI

# Shell and Terminal
brew "bash"
brew "zsh"
brew "tmux"
brew "starship"  # Modern prompt

# Text Editors and IDEs
brew "neovim"
brew "ripgrep"  # Fast grep
brew "fd"       # Fast find
brew "bat"      # Better cat
brew "exa"      # Better ls
brew "fzf"      # Fuzzy finder

# ============================================================================
# Programming Languages
# ============================================================================

# Python
brew "python@3.11"
brew "pyenv"
brew "pipx"
brew "poetry"

# Node.js
brew "node"
brew "nvm"
brew "yarn"
brew "pnpm"

# Go
brew "go"

# Rust
brew "rust"

# ============================================================================
# Development Utilities
# ============================================================================

# Build Tools
brew "cmake"
brew "make"
brew "automake"
brew "autoconf"

# Package Managers
brew "pkg-config"

# JSON/YAML Tools
brew "jq"       # JSON processor
brew "yq"       # YAML processor

# HTTP Tools
brew "curl"
brew "wget"
brew "httpie"

# ============================================================================
# Code Quality Tools
# ============================================================================

# Linters and Formatters
brew "shellcheck"   # Shell script linter
brew "shfmt"       # Shell script formatter
brew "hadolint"    # Dockerfile linter
brew "yamllint"    # YAML linter
brew "prettier"    # Code formatter

# Testing
brew "bats-core"   # Bash testing

# Security
brew "gitleaks"    # Secret scanner
brew "trivy"       # Vulnerability scanner

# ============================================================================
# Container and Cloud Tools
# ============================================================================

# Docker
cask "docker"

# Kubernetes
brew "kubectl"
brew "k9s"         # Kubernetes CLI UI
brew "helm"
brew "kind"        # Kubernetes in Docker

# Cloud CLIs
brew "awscli"
brew "azure-cli"
brew "google-cloud-sdk"

# Infrastructure as Code
brew "terraform"
brew "ansible"

# ============================================================================
# Database Tools
# ============================================================================

brew "postgresql@14", restart_service: true
brew "redis", restart_service: true
brew "sqlite"
brew "mysql"

# Database Clients
brew "pgcli"       # PostgreSQL CLI
brew "redis"       # Redis CLI
brew "mycli"       # MySQL CLI

# ============================================================================
# Monitoring and Performance
# ============================================================================

brew "htop"        # Process viewer
brew "ctop"        # Container metrics
brew "hyperfine"   # Benchmarking tool
brew "wrk"         # HTTP benchmarking
brew "vegeta"      # Load testing

# ============================================================================
# Audio/Video Processing (for AG06 Mixer)
# ============================================================================

brew "ffmpeg"
brew "sox"         # Sound processing
brew "opus"        # Audio codec
brew "lame"        # MP3 encoder
brew "flac"        # Lossless audio

# ============================================================================
# Networking Tools
# ============================================================================

brew "nmap"
brew "netcat"
brew "mtr"         # Network diagnostic
brew "iperf3"      # Network performance

# ============================================================================
# macOS Specific Tools
# ============================================================================

cask "iterm2"              # Terminal emulator
cask "visual-studio-code"  # VS Code
cask "postman"             # API testing
cask "wireshark"           # Network analysis

# macOS Utilities
brew "mas"         # Mac App Store CLI
brew "dockutil"    # Dock management

# ============================================================================
# Optional Development Applications
# ============================================================================

# cask "slack"
# cask "zoom"
# cask "notion"
# cask "figma"

# ============================================================================
# Fonts
# ============================================================================

tap "homebrew/cask-fonts"
cask "font-fira-code"
cask "font-jetbrains-mono"
cask "font-cascadia-code"
cask "font-hack-nerd-font"