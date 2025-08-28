#!/bin/bash
# AG06 Hardware Validation Script
# Ensures AG06 mixer is properly connected and configured

set -euo pipefail

validate_ag06_connection() {
    echo "🔍 Validating AG06 connection..."
    
    local validation_passed=0
    
    # Check USB connection
    if system_profiler SPUSBDataType 2>/dev/null | grep -q "AG06"; then
        echo "✅ USB connection detected"
        ((validation_passed++))
    else
        echo "❌ USB connection not found"
    fi
    
    # Check audio system recognition
    if system_profiler SPAudioDataType 2>/dev/null | grep -q "AG06"; then
        echo "✅ Audio system recognition confirmed"
        ((validation_passed++))
    else
        echo "❌ Audio system not recognizing AG06"
    fi
    
    # Overall validation result
    if [[ $validation_passed -eq 2 ]]; then
        echo "🎉 AG06 validation successful"
        return 0
    else
        echo "⚠️  AG06 validation failed ($validation_passed/2 checks passed)"
        return 1
    fi
}

# Run validation
validate_ag06_connection
