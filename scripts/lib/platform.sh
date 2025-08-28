#!/usr/bin/env bash
# Platform Library - OS and architecture detection

set -euo pipefail

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)      echo "unknown" ;;
    esac
}

# Detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        armv7l|armv7)  echo "arm" ;;
        i386|i686)     echo "386" ;;
        *)             echo "unknown" ;;
    esac
}

# Get platform string
get_platform() {
    echo "$(detect_os)-$(detect_arch)"
}

# Check for Apple Silicon
is_apple_silicon() {
    [[ "$(detect_os)" == "macos" && "$(detect_arch)" == "arm64" ]]
}

# Check for Homebrew
has_homebrew() {
    command -v brew >/dev/null 2>&1
}

# Get Homebrew prefix
get_brew_prefix() {
    if has_homebrew; then
        brew --prefix
    elif is_apple_silicon; then
        echo "/opt/homebrew"
    else
        echo "/usr/local"
    fi
}

export -f detect_os detect_arch get_platform is_apple_silicon has_homebrew get_brew_prefix
