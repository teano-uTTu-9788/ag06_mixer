# 🎤 Shure SM58 + JBL 310 Setup Guide for AG06

## ⚡ IMMEDIATE FIX - Follow These Steps IN ORDER:

### Step 1: Set AG06 as Default Audio Device
1. Open **System Settings > Sound**
2. Under **Output**, select **AG06/AG03**
3. Under **Input**, select **AG06/AG03**

### Step 2: AG06 Front Panel Settings for SM58

**CRITICAL - Your Exact Settings:**

```
🎚️ CHANNEL 1 GAIN: 2-3 o'clock position (SM58 needs high gain!)
⚡ +48V PHANTOM: OFF (LED must be DARK - SM58 doesn't need it)
🔴 MONITOR: 9-10 o'clock (NOT at minimum!)
🎧 PHONES: 10 o'clock if using headphones
🔄 TO PC: Set to "DRY CH 1-2G"
```

### Step 3: Physical Connections Check

**MICROPHONE (SM58):**
- [ ] XLR cable connected to Channel 1 input
- [ ] Cable clicked/locked into mic
- [ ] Cable clicked/locked into AG06
- [ ] No visible damage to cable

**SPEAKERS (JBL 310):**
- [ ] RCA cables connected to MONITOR OUT on AG06 back
- [ ] Red RCA → Red (R) output
- [ ] White RCA → White (L) output  
- [ ] JBL 310 power switch ON
- [ ] JBL 310 volume knob at 12 o'clock
- [ ] Blue power LED on JBL 310 is lit

### Step 4: Test Audio Output First
1. Keep AG06 MONITOR knob at 9 o'clock
2. Play this test sound:
   ```bash
   afplay /System/Library/Sounds/Glass.aiff
   ```
3. You should hear it through JBL 310s
4. If NO sound:
   - Turn MONITOR knob slowly clockwise
   - Check JBL 310 power
   - Check RCA connections

### Step 5: Test SM58 Microphone
1. **GAIN Setting**: Start at 2 o'clock (SM58 needs more gain than condenser mics)
2. Tap the mic gently or speak "Test 1-2-3"
3. Watch the PEAK LED - it should flicker red on taps/loud speech
4. If no PEAK LED activity:
   - Increase GAIN to 3 o'clock
   - Check XLR cable connection
   - Try Channel 2 input instead

## 🔍 DIAGNOSTIC COMMANDS

Run these to verify system configuration:

```bash
# Check if AG06 is recognized
system_profiler SPAudioDataType | grep -A5 "AG06"

# Test system audio
afplay /System/Library/Sounds/Glass.aiff

# Check system volume
osascript -e "output volume of (get volume settings)"
```

## ⚠️ SM58-SPECIFIC REQUIREMENTS

The Shure SM58 is a **dynamic microphone** that:
- ❌ Does NOT need phantom power (keep +48V OFF)
- ✅ Needs MORE gain than condenser mics (2-3 o'clock)
- ✅ Is very durable and handles high SPL
- ✅ Has built-in pop filter (the ball grille)
- ✅ Best pickup 2-6 inches from mouth

## 🔊 JBL 310 MONITOR SETTINGS

Your JBL LSR310S is a studio subwoofer. For proper setup:
- Volume knob: Start at 12 o'clock
- Crossover: 80-120Hz (if adjustable)
- Phase: 0° unless you hear cancellation
- Power: Blue LED should be on

## 🚨 TROUBLESHOOTING CHECKLIST

### No Audio from Speakers:
1. **MONITOR knob** - #1 issue! Must NOT be at minimum
2. JBL 310 powered on (blue LED)
3. RCA cables properly connected
4. AG06 selected as system output

### No Microphone Signal:
1. **GAIN at 2-3 o'clock** for SM58
2. **Phantom power OFF** (no LED)
3. XLR cable fully inserted and locked
4. Try different XLR cable
5. Test on Channel 2

### Signal Present but Distorted:
1. Reduce GAIN if PEAK LED stays red
2. Check TO PC switch (should be DRY CH 1-2G)
3. Lower system volume if clipping

## ✅ CORRECT SIGNAL FLOW

```
SM58 Mic → XLR Cable → AG06 Ch.1 Input → 
GAIN (2-3 o'clock) → AG06 Processing → 
MONITOR OUT → RCA Cables → JBL 310 → Sound!
```

## 📊 QUICK REFERENCE CARD

| Component | Setting | Why |
|-----------|---------|-----|
| Ch.1 GAIN | 2-3 o'clock | SM58 is dynamic, needs gain |
| +48V | OFF | SM58 doesn't need phantom |
| MONITOR | 9-10 o'clock | Controls speaker volume |
| TO PC | DRY CH 1-2G | Clean recording signal |
| JBL 310 | 12 o'clock | Good starting volume |

## 🎯 TEST RIGHT NOW:

1. Set GAIN to 2 o'clock
2. Set MONITOR to 10 o'clock  
3. Phantom power OFF
4. Tap the SM58 - you should see PEAK LED flicker
5. Speak into SM58 - you should hear yourself through JBL 310

**If this doesn't work, the issue is likely:**
- MONITOR knob still too low
- JBL 310 not powered on
- Wrong audio device selected in macOS