#!/bin/bash
# Install missing development tools for the Terminal Automation Framework
# Provides alternative installation methods if Homebrew is not available

set -e

echo "🔧 Development Tools Installation Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Architecture detection
ARCH=$(uname -m)
OS=$(uname -s)

echo "System: $OS ($ARCH)"
echo ""

# Function to install via Homebrew
install_with_brew() {
    local tool=$1
    echo -e "${BLUE}Installing $tool via Homebrew...${NC}"
    if brew install "$tool"; then
        echo -e "${GREEN}✅ $tool installed successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Homebrew installation failed for $tool${NC}"
        return 1
    fi
}

# Function to install ShellCheck manually
install_shellcheck_manual() {
    echo -e "${BLUE}Installing ShellCheck manually...${NC}"
    
    local SHELLCHECK_VERSION="v0.10.0"
    local SHELLCHECK_URL=""
    
    if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
        SHELLCHECK_URL="https://github.com/koalaman/shellcheck/releases/download/${SHELLCHECK_VERSION}/shellcheck-${SHELLCHECK_VERSION}.darwin.aarch64.tar.xz"
    else
        SHELLCHECK_URL="https://github.com/koalaman/shellcheck/releases/download/${SHELLCHECK_VERSION}/shellcheck-${SHELLCHECK_VERSION}.darwin.x86_64.tar.xz"
    fi
    
    # Download and install
    curl -LO "$SHELLCHECK_URL"
    tar -xf shellcheck-*.tar.xz
    
    # Try to move to /usr/local/bin (may require sudo)
    if [[ -w "/usr/local/bin" ]]; then
        mv shellcheck-*/shellcheck /usr/local/bin/
        echo -e "${GREEN}✅ ShellCheck installed to /usr/local/bin${NC}"
    else
        mkdir -p "$HOME/bin"
        mv shellcheck-*/shellcheck "$HOME/bin/"
        echo -e "${GREEN}✅ ShellCheck installed to ~/bin${NC}"
        echo -e "${YELLOW}Add ~/bin to your PATH: export PATH=\"\$HOME/bin:\$PATH\"${NC}"
    fi
    
    # Cleanup
    rm -rf shellcheck-*
    
    return 0
}

# Function to install shfmt manually
install_shfmt_manual() {
    echo -e "${BLUE}Installing shfmt manually...${NC}"
    
    local SHFMT_VERSION="v3.7.0"
    local SHFMT_URL=""
    
    if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
        SHFMT_URL="https://github.com/mvdan/sh/releases/download/${SHFMT_VERSION}/shfmt_${SHFMT_VERSION}_darwin_arm64"
    else
        SHFMT_URL="https://github.com/mvdan/sh/releases/download/${SHFMT_VERSION}/shfmt_${SHFMT_VERSION}_darwin_amd64"
    fi
    
    # Download
    curl -LO "$SHFMT_URL"
    chmod +x shfmt_*
    
    # Try to move to /usr/local/bin
    if [[ -w "/usr/local/bin" ]]; then
        mv shfmt_* /usr/local/bin/shfmt
        echo -e "${GREEN}✅ shfmt installed to /usr/local/bin${NC}"
    else
        mkdir -p "$HOME/bin"
        mv shfmt_* "$HOME/bin/shfmt"
        echo -e "${GREEN}✅ shfmt installed to ~/bin${NC}"
        echo -e "${YELLOW}Add ~/bin to your PATH: export PATH=\"\$HOME/bin:\$PATH\"${NC}"
    fi
    
    return 0
}

# Function to install BATS manually
install_bats_manual() {
    echo -e "${BLUE}Installing BATS manually...${NC}"
    
    local BATS_VERSION="v1.11.0"
    local TEMP_DIR=$(mktemp -d)
    
    cd "$TEMP_DIR"
    
    # Clone bats-core
    git clone --depth 1 --branch "$BATS_VERSION" https://github.com/bats-core/bats-core.git
    
    # Install to user directory
    cd bats-core
    ./install.sh "$HOME/.bats"
    
    # Create symlink in ~/bin
    mkdir -p "$HOME/bin"
    ln -sf "$HOME/.bats/bin/bats" "$HOME/bin/bats"
    
    echo -e "${GREEN}✅ BATS installed to ~/.bats${NC}"
    echo -e "${YELLOW}Add ~/bin to your PATH: export PATH=\"\$HOME/bin:\$PATH\"${NC}"
    
    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"
    
    return 0
}

# Main installation process
echo "📋 Tools to install:"
echo "  1. ShellCheck - Shell script linting"
echo "  2. shfmt - Shell script formatting"
echo "  3. BATS - Bash Automated Testing System"
echo ""

# Check if Homebrew is available
if command -v brew >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Homebrew detected${NC}"
    echo "Attempting Homebrew installation first..."
    echo ""
    
    # Try Homebrew installation
    BREW_SUCCESS=true
    install_with_brew "shellcheck" || BREW_SUCCESS=false
    install_with_brew "shfmt" || BREW_SUCCESS=false
    install_with_brew "bats-core" || BREW_SUCCESS=false
    
    if $BREW_SUCCESS; then
        echo ""
        echo -e "${GREEN}🎉 All tools installed successfully via Homebrew!${NC}"
    else
        echo ""
        echo -e "${YELLOW}Some tools failed to install via Homebrew.${NC}"
        echo "Falling back to manual installation..."
    fi
else
    echo -e "${YELLOW}⚠️ Homebrew not available${NC}"
    echo "Using manual installation methods..."
    echo ""
    BREW_SUCCESS=false
fi

# Manual installation for missing tools
if ! command -v shellcheck >/dev/null 2>&1; then
    install_shellcheck_manual
fi

if ! command -v shfmt >/dev/null 2>&1; then
    install_shfmt_manual
fi

if ! command -v bats >/dev/null 2>&1; then
    install_bats_manual
fi

# Verification
echo ""
echo "📊 Installation Summary:"
echo "========================"

# Check ShellCheck
if command -v shellcheck >/dev/null 2>&1; then
    echo -e "${GREEN}✅ ShellCheck:${NC} $(shellcheck --version | head -1)"
else
    echo -e "${RED}❌ ShellCheck: Not installed${NC}"
fi

# Check shfmt
if command -v shfmt >/dev/null 2>&1; then
    echo -e "${GREEN}✅ shfmt:${NC} $(shfmt --version)"
else
    echo -e "${RED}❌ shfmt: Not installed${NC}"
fi

# Check BATS
if command -v bats >/dev/null 2>&1; then
    echo -e "${GREEN}✅ BATS:${NC} $(bats --version)"
else
    echo -e "${RED}❌ BATS: Not installed${NC}"
fi

echo ""
echo "🔍 PATH Configuration:"
echo "======================"
echo "Current PATH: $PATH"
echo ""

# Check if ~/bin is in PATH
if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo -e "${YELLOW}⚠️ ~/bin is not in your PATH${NC}"
    echo ""
    echo "Add this line to your ~/.zshrc or ~/.bashrc:"
    echo -e "${BLUE}export PATH=\"\$HOME/bin:\$PATH\"${NC}"
    echo ""
    echo "Then reload your shell:"
    echo -e "${BLUE}source ~/.zshrc${NC}"
else
    echo -e "${GREEN}✅ ~/bin is in PATH${NC}"
fi

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. If tools were installed to ~/bin, update your PATH"
echo "2. Run './dev doctor' to verify tool installation"
echo "3. Run './dev lint' to test ShellCheck"
echo "4. Run './dev format' to test shfmt"
echo "5. Run './dev test' to test BATS"

echo ""
echo "🎉 Installation script complete!"