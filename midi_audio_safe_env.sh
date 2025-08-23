#!/bin/bash
# Safe MIDI/Audio Testing Environment for AG06 Mixer
# Prevents system audio crashes and provides isolated testing

# ============================================
# MIDI/AUDIO SAFETY CONFIGURATION
# ============================================

export AG06_SAFE_MODE=true
export AUDIO_BUFFER_SIZE=512  # Safe buffer size
export MIDI_TIMEOUT_MS=1000   # 1 second timeout for MIDI ops
export AUDIO_TIMEOUT_MS=5000  # 5 second timeout for audio ops

# Backup current audio settings
backup_audio_settings() {
    local backup_file="/Users/nguythe/ag06_mixer/logs/audio_backup_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p "$(dirname "$backup_file")"
    
    {
        echo "Audio Settings Backup - $(date)"
        echo "================================"
        SwitchAudioSource -c  # Current output device
        SwitchAudioSource -t input -c  # Current input device
        osascript -e "output volume of (get volume settings)"  # System volume
    } > "$backup_file" 2>/dev/null
    
    echo "$backup_file"
}

# Safe MIDI test (non-blocking, limited duration)
test_midi_safe() {
    echo "🎹 Safe MIDI Test (5 second limit)..."
    
    # Check MIDI tools
    if ! command -v sendmidi &> /dev/null; then
        echo "⚠️  sendmidi not installed. Install with: brew install gbevin/tools/sendmidi"
        return 1
    fi
    
    # List MIDI devices (safe, read-only)
    echo "MIDI Devices:"
    timeout 2 sendmidi list 2>/dev/null | head -5 || echo "  No devices found"
    
    # Send test note (if AG06 found)
    if sendmidi list 2>/dev/null | grep -q "AG06"; then
        echo "Sending test MIDI note..."
        timeout 1 sendmidi dev "AG06" ch 1 on 60 64 2>/dev/null && \
        sleep 0.5 && \
        timeout 1 sendmidi dev "AG06" ch 1 off 60 2>/dev/null
        echo "✅ MIDI test complete"
    else
        echo "⚠️  AG06 MIDI not detected"
    fi
}

# Safe audio test with volume limits
test_audio_safe() {
    echo "🔊 Safe Audio Test..."
    
    # Save current volume
    local current_vol=$(osascript -e "output volume of (get volume settings)" 2>/dev/null)
    
    # Set safe volume (30%)
    osascript -e "set volume output volume 30" 2>/dev/null
    
    # Check if AG06 is available
    if SwitchAudioSource -a | grep -q "AG06"; then
        # Save current device
        local current_device=$(SwitchAudioSource -c)
        
        # Switch to AG06
        echo "Switching to AG06..."
        SwitchAudioSource -s "AG06" 2>/dev/null
        
        # Play short test sound
        if [ -f "/System/Library/Sounds/Glass.aiff" ]; then
            timeout 2 afplay "/System/Library/Sounds/Glass.aiff" 2>/dev/null
            echo "✅ Audio test complete"
        fi
        
        # Restore previous device
        [ -n "$current_device" ] && SwitchAudioSource -s "$current_device" 2>/dev/null
    else
        echo "⚠️  AG06 audio device not available"
    fi
    
    # Restore original volume
    [ -n "$current_vol" ] && osascript -e "set volume output volume $current_vol" 2>/dev/null
}

# Create isolated Python environment for testing
setup_test_env() {
    echo "🔧 Setting up isolated test environment..."
    
    local test_env="/Users/nguythe/ag06_mixer/.test_env"
    
    # Create virtual environment if needed
    if [ ! -d "$test_env" ]; then
        python3 -m venv "$test_env"
        echo "✅ Created virtual environment"
    fi
    
    # Activate and install minimal requirements
    source "$test_env/bin/activate"
    
    # Install only if needed
    if ! python3 -c "import pyaudio" 2>/dev/null; then
        pip install --quiet pyaudio 2>/dev/null || echo "⚠️  PyAudio installation failed"
    fi
    
    if ! python3 -c "import mido" 2>/dev/null; then
        pip install --quiet mido python-rtmidi 2>/dev/null || echo "⚠️  MIDI libs installation failed"
    fi
    
    echo "✅ Test environment ready"
    echo "   Python: $(which python3)"
    echo "   Packages: pyaudio, mido"
}

# Safe audio routing test
test_routing_safe() {
    echo "🔀 Testing AG06 Routing (safe mode)..."
    
    python3 - <<'EOF' 2>&1 | head -20
import sys
import time

try:
    import pyaudio
    
    # Initialize PyAudio with safe settings
    p = pyaudio.PyAudio()
    
    # Find AG06 device
    ag06_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'AG06' in info.get('name', ''):
            ag06_index = i
            print(f"✅ Found AG06: {info['name']}")
            print(f"   Inputs: {info.get('maxInputChannels', 0)}")
            print(f"   Outputs: {info.get('maxOutputChannels', 0)}")
            break
    
    if ag06_index is None:
        print("⚠️  AG06 not found in PyAudio devices")
    
    # Clean up
    p.terminate()
    
except ImportError:
    print("⚠️  PyAudio not available")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
}

# Monitor audio system health
check_audio_health() {
    echo "🏥 Audio System Health Check..."
    
    # Check Core Audio
    if pgrep -x "coreaudiod" > /dev/null; then
        echo "✅ Core Audio: Running"
    else
        echo "❌ Core Audio: Not running!"
        echo "   To restart: sudo killall coreaudiod"
    fi
    
    # Check for audio crashes
    local crash_count=$(find ~/Library/Logs/DiagnosticReports -name "coreaudiod*.crash" -mtime -1 2>/dev/null | wc -l)
    if [ "$crash_count" -gt 0 ]; then
        echo "⚠️  Recent Core Audio crashes: $crash_count"
    else
        echo "✅ No recent audio crashes"
    fi
    
    # Check AG06 USB power
    system_profiler SPUSBDataType 2>/dev/null | grep -A5 "AG06" | grep "Current Available" | head -1 || echo "   USB Power: Unknown"
}

# Kill runaway audio processes
kill_audio_zombies() {
    echo "🧟 Checking for runaway audio processes..."
    
    # Find high-CPU audio processes
    ps aux | grep -E "(coreaudio|AG06|pyaudio)" | grep -v grep | \
    while read -r line; do
        cpu=$(echo "$line" | awk '{print $3}')
        pid=$(echo "$line" | awk '{print $2}')
        cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
        
        # If CPU > 50%, it might be runaway
        if (( $(echo "$cpu > 50" | bc -l) )); then
            echo "⚠️  High CPU process: $cmd (${cpu}%)"
            echo -n "   Kill process $pid? [y/N]: "
            read -r response
            [[ "$response" =~ ^[Yy]$ ]] && kill -TERM "$pid"
        fi
    done
    
    echo "✅ Process check complete"
}

# Main safety menu
ag06_safe_menu() {
    echo """
╔══════════════════════════════════════════════╗
║      AG06 SAFE MIDI/AUDIO TEST MENU         ║
╚══════════════════════════════════════════════╝

1) Test MIDI (safe, 5 second limit)
2) Test Audio (safe, low volume)
3) Test Routing (PyAudio check)
4) Setup Test Environment
5) Check Audio Health
6) Kill Runaway Processes
7) Backup Audio Settings
8) Run All Safe Tests

q) Quit

Select option: """
    
    read -r option
    
    case $option in
        1) test_midi_safe ;;
        2) test_audio_safe ;;
        3) test_routing_safe ;;
        4) setup_test_env ;;
        5) check_audio_health ;;
        6) kill_audio_zombies ;;
        7) backup_file=$(backup_audio_settings)
           echo "✅ Settings backed up to: $backup_file" ;;
        8) echo "Running all safe tests..."
           test_midi_safe
           echo ""
           test_audio_safe
           echo ""
           test_routing_safe
           echo ""
           check_audio_health ;;
        q) echo "👋 Exiting safe test environment" ;;
        *) echo "Invalid option" ;;
    esac
}

# Export functions for use in other scripts
export -f test_midi_safe
export -f test_audio_safe
export -f test_routing_safe
export -f check_audio_health
export -f kill_audio_zombies

# Show menu if run directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    ag06_safe_menu
fi