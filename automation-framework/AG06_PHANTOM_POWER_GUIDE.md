# AG06 Phantom Power Guide & Settings Verification

## 🎤 What is Phantom Power?

**Phantom Power (+48V)** is a DC electrical power supply that runs through microphone cables to power certain types of microphones. It's called "phantom" because the power is sent through the same cables that carry the audio signal, making it invisible or "phantom-like."

### When You Need Phantom Power:
- **Condenser Microphones**: Require phantom power to operate (e.g., Audio-Technica AT2020, Rode NT1)
- **Active DI Boxes**: Some direct input boxes need phantom power
- **Active Ribbon Mics**: Certain modern ribbon microphones with built-in preamps

### When You DON'T Need Phantom Power:
- **Dynamic Microphones**: Don't need phantom power (e.g., Shure SM58, SM57, SM7B)
- **Passive DI Boxes**: Work without phantom power
- **Line-level Instruments**: Keyboards, audio interfaces, etc.

### ⚠️ Important Safety Notes:
- **Safe for most equipment**: Modern dynamic mics are designed to handle phantom power without damage
- **DANGER for vintage ribbon mics**: Can damage older ribbon microphones permanently
- **Check your manual**: Always verify your microphone's specifications

## 🎛️ AG06 Phantom Power Settings

### Location on AG06:
The **+48V** button is located on the top panel, near the input gain knobs.

### Correct Settings for Common Scenarios:

#### 1. **Condenser Microphone Setup**
```
+48V Button: ON (LED lit)
Gain: Start at 12 o'clock, adjust as needed
PAD: OFF (unless mic signal is too hot)
```

#### 2. **Dynamic Microphone Setup**
```
+48V Button: OFF (LED dark)
Gain: Usually needs more than condenser (2-3 o'clock)
PAD: OFF
```

#### 3. **Line Input/Instrument**
```
+48V Button: OFF (LED dark)
Gain: Lower (9-11 o'clock)
PAD: ON if signal is too strong
```

## ✅ AG06 Settings Verification Checklist

### Audio Input Settings:
- [ ] **Channel 1 Gain**: Set appropriately for your mic type
- [ ] **Channel 2 Gain**: Adjusted for second input if used
- [ ] **+48V Phantom Power**: 
  - ON for condenser mics
  - OFF for dynamic mics and line inputs
- [ ] **PAD Switches**: OFF unless input is clipping
- [ ] **COMP/EQ**: Adjust to taste (start flat)

### Output Settings:
- [ ] **Monitor Output Level**: Set to comfortable listening level
- [ ] **Phones Level**: Adjusted for headphone monitoring
- [ ] **TO PC Switch**: Set to "DRY CH 1-2G" for most recording scenarios

### USB/Computer Settings:
- [ ] **USB Cable**: Connected to computer
- [ ] **System Preferences**: AG06 selected as input/output device
- [ ] **Sample Rate**: Matching your DAW (typically 48kHz)
- [ ] **Buffer Size**: 256 samples for balance of latency/stability

### Physical Connections:
- [ ] **Monitor Out**: Connected to speakers (JBL 310s)
- [ ] **Microphone**: Connected to Channel 1 or 2
- [ ] **Headphones**: Connected if monitoring needed
- [ ] **Power**: USB powered from computer

## 🔧 Troubleshooting Phantom Power Issues

### No Signal from Condenser Mic:
1. Check +48V button is ON (LED lit)
2. Verify XLR cable is fully connected
3. Increase gain slowly while speaking
4. Check mic requires phantom power (see manual)

### Distorted/Clipping Signal:
1. Engage PAD button if available
2. Reduce gain on channel
3. Check if mic has internal pad switch
4. Move further from mic if very loud source

### Noise/Hum with Phantom Power:
1. Check XLR cable quality (use balanced cable)
2. Verify proper grounding
3. Keep cables away from power sources
4. Try different USB port on computer

## 🎯 Best Practices

1. **Turn phantom power ON/OFF with monitors muted** to avoid pops
2. **Wait 10-15 seconds** after enabling phantom power before adjusting gain
3. **Label your inputs** if using multiple mics (condenser vs dynamic)
4. **Document your settings** for consistent recordings
5. **Use quality XLR cables** for best phantom power delivery

## 📊 Current AG06 Status Check

Based on your connection at 2:35:31 PM:
- ✅ AG06 Mixer connected successfully
- ✅ USB communication established
- ✅ Ready for audio processing

### Recommended Next Steps:
1. Verify phantom power setting matches your microphone type
2. Check gain staging (peaks around -12dB to -6dB)
3. Test monitoring through both headphones and speakers
4. Confirm TO PC switch position for your workflow

## 🎙️ Quick Reference

| Microphone Type | +48V Setting | Typical Gain | Notes |
|----------------|--------------|--------------|--------|
| Condenser | ON | 12-2 o'clock | Requires phantom power |
| Dynamic | OFF | 2-4 o'clock | No phantom power needed |
| Ribbon (Modern) | Check Manual | Variable | Some need +48V |
| Ribbon (Vintage) | NEVER ON | Variable | Can cause damage |
| Line/Instrument | OFF | 9-12 o'clock | Use Hi-Z input if available |

Remember: When in doubt, check your microphone's manual for phantom power requirements!