#!/bin/bash
# Fix Homebrew /private/tmp permission issue
# This script requires sudo access

set -e

echo "🔧 Homebrew Permission Fix Script"
echo "=================================="
echo ""
echo "This script will fix the /private/tmp permission issue preventing Homebrew from working."
echo "You will need to enter your password when prompted."
echo ""

# Check if running on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

echo "📋 Current status:"
if [[ -d "/private/tmp" ]]; then
    echo "  /private/tmp exists"
    ls -ld /private/tmp
else
    echo "  /private/tmp does not exist (will create)"
fi

echo ""
echo "🔐 Requesting sudo access to fix permissions..."
echo ""

# Create /private/tmp if it doesn't exist
if [[ ! -d "/private/tmp" ]]; then
    echo "Creating /private/tmp directory..."
    sudo mkdir -p /private/tmp
fi

# Fix permissions (1777 = sticky bit + world writable)
echo "Setting correct permissions on /private/tmp..."
sudo chmod 1777 /private/tmp

# Fix ownership
echo "Setting correct ownership on /private/tmp..."
sudo chown root:wheel /private/tmp

# Verify the fix
echo ""
echo "✅ Permissions fixed! New status:"
ls -ld /private/tmp

# Test Homebrew
echo ""
echo "🧪 Testing Homebrew..."
if command -v brew >/dev/null 2>&1; then
    if brew --version >/dev/null 2>&1; then
        echo "✅ Homebrew is now working!"
        echo ""
        echo "Homebrew version:"
        brew --version
    else
        echo "⚠️ Homebrew still has issues. You may need to reinstall it."
    fi
else
    echo "❌ Homebrew is not installed. Run './dev bootstrap' to install it."
fi

echo ""
echo "🎉 Permission fix complete!"
echo ""
echo "Next steps:"
echo "1. Run './dev doctor' to verify system health"
echo "2. Run './dev bootstrap' to install missing tools"
echo "3. Run './dev ci' to test the complete pipeline"