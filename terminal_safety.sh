#!/bin/bash
# AG06 Mixer App Terminal Safety Configuration
# Prevents context overflow during audio/MIDI development

# ============================================
# CORE SAFETY SETTINGS FOR AG06 DEVELOPMENT
# ============================================

# Disable aggressive auto-continue for mixer development
export AUTO_CONTINUE_LEVEL=manual
export CLAUDE_AUTO_CONTINUE=off
export AG06_DEV_MODE=safe

# Context limits for audio/MIDI debugging
export MAX_TERMINAL_CONTEXT_LINES=150
export MAX_AUDIO_LOG_LINES=100
export MAX_MIDI_EVENT_LINES=50

# AG06-specific paths
export AG06_PROJECT_DIR="/Users/nguythe/ag06_mixer"
export AG06_LOGS_DIR="/Users/nguythe/ag06_mixer/logs"
export AG06_TEST_AUDIO="/System/Library/Sounds/Glass.aiff"

# ============================================
# AG06 MIXER SAFETY FUNCTIONS
# ============================================

# Initialize AG06 safe development environment
ag06_safe() {
    echo "🎛️  AG06 Mixer Safe Mode Activated"
    export AUTO_CONTINUE_LEVEL=manual
    export CLAUDE_AUTO_CONTINUE=off
    
    # Create logs directory if needed
    mkdir -p "$AG06_LOGS_DIR"
    
    echo "Settings:"
    echo "  - Auto-continue: DISABLED"
    echo "  - Audio logs: Limited to $MAX_AUDIO_LOG_LINES lines"
    echo "  - MIDI events: Limited to $MAX_MIDI_EVENT_LINES lines"
    echo "  - Project dir: $AG06_PROJECT_DIR"
}

# Check AG06 audio device status (minimal output)
ag06_status() {
    echo "🎛️  AG06 Status Check"
    echo "===================="
    
    # Check if AG06 is connected (concise)
    echo -n "AG06 USB: "
    if system_profiler SPUSBDataType 2>/dev/null | grep -q "AG06"; then
        echo "✅ Connected"
    else
        echo "❌ Not found"
        return 1
    fi
    
    # Check audio device (limited output)
    echo -n "Audio Device: "
    system_profiler SPAudioDataType 2>/dev/null | grep -A2 "AG06" | grep "Manufacturer" | head -1 | cut -d: -f2 | xargs
    
    # Check MIDI device
    echo -n "MIDI Device: "
    if [ -x /usr/local/bin/sendmidi ]; then
        /usr/local/bin/sendmidi list 2>/dev/null | grep -i "AG06" | head -1 || echo "Not detected"
    else
        echo "sendmidi not installed"
    fi
}

# Test AG06 audio routing (minimal, safe)
ag06_test_audio() {
    echo "🔊 Testing AG06 Audio..."
    
    # Save current output device
    local current_device=$(SwitchAudioSource -c 2>/dev/null)
    
    # Try to switch to AG06
    if SwitchAudioSource -s "AG06" 2>/dev/null; then
        echo "✅ Switched to AG06"
        
        # Play test sound
        afplay "$AG06_TEST_AUDIO" 2>/dev/null && echo "✅ Test sound played"
        
        # Restore previous device
        [ -n "$current_device" ] && SwitchAudioSource -s "$current_device" 2>/dev/null
    else
        echo "⚠️  Could not switch to AG06"
    fi
}

# Monitor AG06 MIDI events (limited output)
ag06_midi_monitor() {
    local duration="${1:-5}"
    echo "🎹 Monitoring MIDI for ${duration} seconds..."
    
    if [ -x /usr/local/bin/receivemidi ]; then
        timeout "$duration" /usr/local/bin/receivemidi 2>&1 | head -n "$MAX_MIDI_EVENT_LINES"
    else
        echo "receivemidi not installed. Install with: brew install gbevin/tools/receivemidi"
    fi
}

# Run AG06 app with output limiting
ag06_run() {
    echo "🚀 Starting AG06 Mixer App (output limited)..."
    
    cd "$AG06_PROJECT_DIR" || return 1
    
    # Run with output limiting and logging
    if [ -f "main.py" ]; then
        python3 main.py 2>&1 | tee "$AG06_LOGS_DIR/app_$(date +%Y%m%d_%H%M%S).log" | tail -n "$MAX_TERMINAL_CONTEXT_LINES"
    elif [ -f "package.json" ]; then
        npm start 2>&1 | tee "$AG06_LOGS_DIR/app_$(date +%Y%m%d_%H%M%S).log" | tail -n "$MAX_TERMINAL_CONTEXT_LINES"
    else
        echo "❌ No main.py or package.json found"
    fi
}

# Test specific AG06 mixer controls
ag06_test_controls() {
    echo "🎛️  Testing AG06 Controls..."
    
    python3 - <<'EOF' 2>&1 | head -n 50
import sys
sys.path.insert(0, '/Users/nguythe/ag06_mixer')

try:
    from ag06_controller import AG06Controller
    
    controller = AG06Controller()
    print("✅ Controller initialized")
    
    # Test basic controls
    print("\nTesting controls:")
    print(f"  Channel 1 gain: {controller.get_channel_gain(1)}")
    print(f"  Monitor level: {controller.get_monitor_level()}")
    print(f"  Effect send: {controller.get_effect_send()}")
    
except ImportError as e:
    print(f"⚠️  Controller not found: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
}

# Clear AG06 logs and reset context
ag06_reset() {
    echo "🔄 Resetting AG06 development context..."
    
    # Clear old logs (keep last 5)
    if [ -d "$AG06_LOGS_DIR" ]; then
        ls -t "$AG06_LOGS_DIR"/*.log 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
        echo "✅ Cleaned old logs"
    fi
    
    # Clear terminal
    clear && printf '\e[3J'
    
    # Reset environment
    ag06_safe
    
    echo "✅ AG06 context reset complete"
}

# Safe command wrapper for AG06 development
ag06_exec() {
    local cmd="$*"
    
    # Check for dangerous audio/system commands
    if [[ "$cmd" =~ (sudo.*kill|killall.*coreaudio|launchctl.*stop.*audio) ]]; then
        echo "⚠️  Blocked potentially dangerous audio command"
        return 1
    fi
    
    echo "📝 Command: $cmd"
    echo -n "Execute? [y/N]: "
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        eval "$cmd" 2>&1 | tail -n "$MAX_TERMINAL_CONTEXT_LINES"
    else
        echo "Command cancelled."
    fi
}

# Log AG06 debug info to file (not terminal)
ag06_debug() {
    local debug_file="$AG06_LOGS_DIR/debug_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "AG06 Debug Information"
        echo "======================"
        echo "Date: $(date)"
        echo ""
        
        echo "USB Devices:"
        system_profiler SPUSBDataType | grep -A10 -B2 "AG06"
        echo ""
        
        echo "Audio Devices:"
        system_profiler SPAudioDataType | grep -A10 -B2 "AG06"
        echo ""
        
        echo "MIDI Devices:"
        /usr/local/bin/sendmidi list 2>/dev/null || echo "sendmidi not available"
        echo ""
        
        echo "Python Environment:"
        python3 --version
        pip3 list | grep -E "(midi|audio|pyaudio|sounddevice)"
        echo ""
        
    } > "$debug_file" 2>&1
    
    echo "✅ Debug info saved to: $debug_file"
    echo "   (Use 'cat $debug_file | head -50' to view safely)"
}

# ============================================
# AG06 DEVELOPMENT ALIASES
# ============================================

alias ag06='ag06_safe'
alias ag06s='ag06_status'
alias ag06t='ag06_test_audio'
alias ag06m='ag06_midi_monitor'
alias ag06r='ag06_run'
alias ag06c='ag06_test_controls'
alias ag06x='ag06_reset'
alias ag06d='ag06_debug'
alias ag06e='ag06_exec'

# ============================================
# AUTO-ACTIVATION
# ============================================

# Activate AG06 safe mode if in mixer directory
if [[ "$PWD" == *"ag06_mixer"* ]]; then
    ag06_safe
fi

echo "🎛️  AG06 Mixer Terminal Safety Loaded"
echo "Commands:"
echo "  ag06  - Activate safe mode"
echo "  ag06s - Check AG06 status"
echo "  ag06t - Test audio routing"
echo "  ag06m - Monitor MIDI events"
echo "  ag06r - Run mixer app (limited output)"
echo "  ag06c - Test mixer controls"
echo "  ag06x - Reset context"
echo "  ag06d - Save debug info to file"
echo "  ag06e - Execute with confirmation"