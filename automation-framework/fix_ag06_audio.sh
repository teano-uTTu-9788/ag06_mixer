#!/bin/bash

# AG06 Audio Troubleshooting Script
# Fixes common "no audio/mic" issues

echo "🔧 AG06 Audio Fix Script"
echo "========================"
echo ""

# Step 1: Check if AG06 is detected
echo "1️⃣ Checking if AG06 is detected..."
if system_profiler SPAudioDataType | grep -q "AG06\|AG03"; then
    echo "   ✅ AG06 detected by system"
else
    echo "   ❌ AG06 not detected! Check USB connection"
    echo "   Try: Unplug and replug USB cable"
    exit 1
fi

# Step 2: Set AG06 as default audio device
echo ""
echo "2️⃣ Setting AG06 as default audio device..."
# Check current default
if system_profiler SPAudioDataType | grep -A2 "AG06\|AG03" | grep -q "Default Output Device: Yes"; then
    echo "   ✅ AG06 already set as default output"
else
    echo "   ⚠️ AG06 not default - please set in System Settings > Sound"
    echo "   Opening Sound settings..."
    open /System/Library/PreferencePanes/Sound.prefPane
fi

# Step 3: Check audio service
echo ""
echo "3️⃣ Checking Core Audio service..."
if pmset -g | grep -q "coreaudiod"; then
    echo "   ✅ Core Audio is running"
else
    echo "   ⚠️ Core Audio may need restart"
    echo "   Run: sudo killall coreaudiod"
fi

# Step 4: Test audio output
echo ""
echo "4️⃣ Testing audio output..."
echo "   Playing test sound (you should hear a beep)..."
afplay /System/Library/Sounds/Glass.aiff

echo ""
echo "5️⃣ Critical AG06 Hardware Checks:"
echo ""
echo "FRONT PANEL - Check these settings:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎚️ GAIN KNOBS (Channels 1 & 2):"
echo "   • Turn clockwise from fully left"
echo "   • Start at 12 o'clock position"
echo "   • For dynamic mics: 2-3 o'clock"
echo "   • For condenser mics: 11-1 o'clock"
echo ""
echo "🔴 MONITOR KNOB (Output Level):"
echo "   ⚠️ THIS IS OFTEN THE ISSUE!"
echo "   • Make sure it's NOT at minimum (fully left)"
echo "   • Turn to 9-10 o'clock to start"
echo "   • This controls speaker output volume"
echo ""
echo "🎧 PHONES KNOB (Headphone Level):"
echo "   • If using headphones, turn up to 10 o'clock"
echo "   • Independent from monitor output"
echo ""
echo "⚡ +48V PHANTOM POWER:"
echo "   • ON (LED lit) = Condenser mics"
echo "   • OFF (LED dark) = Dynamic mics"
echo "   • OFF for line inputs"
echo ""
echo "🔄 TO PC SWITCH:"
echo "   • DRY CH 1-2G = Normal recording"
echo "   • INPUT MIX = Monitor all inputs"
echo "   • LOOPBACK = Include computer audio"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "6️⃣ BACK PANEL - Verify connections:"
echo ""
echo "🔌 MONITOR OUT (Red/White RCA):"
echo "   • Connected to JBL 310 speakers"
echo "   • Red = Right, White = Left"
echo "   • Cables firmly inserted"
echo ""
echo "🔌 USB:"
echo "   • Connected to computer"
echo "   • Try different USB port if issues persist"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "7️⃣ Quick Test Procedure:"
echo ""
echo "For MICROPHONE testing:"
echo "  1. Connect mic to Channel 1 (XLR or 1/4\")"
echo "  2. Set GAIN to 12 o'clock"
echo "  3. Set MONITOR knob to 9 o'clock"
echo "  4. If condenser mic: Turn +48V ON"
echo "  5. Speak into mic while adjusting GAIN"
echo "  6. PEAK LED should flicker on loud sounds"
echo ""
echo "For PLAYBACK testing:"
echo "  1. Ensure MONITOR knob is NOT at minimum"
echo "  2. Play audio from computer"
echo "  3. Slowly increase MONITOR knob"
echo "  4. Check speaker power and connections"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚨 MOST COMMON ISSUES:"
echo ""
echo "1. MONITOR knob at minimum (no speaker output)"
echo "2. Wrong input selected in recording software"
echo "3. Phantom power OFF for condenser mic"
echo "4. GAIN too low for dynamic mic"
echo "5. TO PC switch in wrong position"
echo ""

# Check current audio levels
echo "8️⃣ Current System Volume:"
osascript -e "output volume of (get volume settings)" | awk '{print "   System volume: " $1 "%"}'

echo ""
echo "✅ Troubleshooting complete!"
echo ""
echo "If still no audio:"
echo "  • Restart the AG06 (unplug/replug USB)"
echo "  • Check recording software input settings"
echo "  • Verify mic cable is good (test with another)"
echo "  • Try headphones to isolate speaker issues"
echo ""