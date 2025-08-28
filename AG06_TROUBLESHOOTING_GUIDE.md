# 🎛️ AG06 Mixer Troubleshooting & Setup Guide

## ❌ Current Status: AG06 Not Detected

### Immediate Action Required:

## 1. 🔌 Check Physical Connection
- **Power LED**: Is the AG06 power LED lit?
- **USB Cable**: Try a different USB-C/USB cable
- **USB Port**: Try different ports on your Mac
- **Power Supply**: Ensure AG06 has proper power

## 2. 💿 Driver Installation (REQUIRED)

### Download & Install Yamaha Driver:
1. Go to: https://www.yamaha.com/support/
2. Search for "AG06" 
3. Download "Yamaha Steinberg USB Driver" for macOS
4. Install the driver package
5. **RESTART YOUR MAC** (Required!)

### Alternative Driver Sources:
- Steinberg website: https://www.steinberg.net/support/
- Direct link: Search "Yamaha AG06 macOS driver"

## 3. 🔒 macOS Security Settings

After driver installation:
1. Open **System Settings** > **Privacy & Security**
2. Look for message about blocked software
3. Click **Allow** for Yamaha driver
4. Restart Mac again

## 4. 🎚️ Audio MIDI Setup Configuration

1. Open **Applications** > **Utilities** > **Audio MIDI Setup**
2. Click **+** button > **Create Aggregate Device**
3. Add AG06 if it appears
4. Set as default input/output

## 5. 🔄 Alternative Setup (Without AG06)

### Use Virtual Audio Routing:
Since you have **BlackHole 2ch** installed, we can use it:

```bash
# Set BlackHole as audio input for dual-channel simulation
open /Applications/Utilities/Audio\ MIDI\ Setup.app
```

### Configure AiOke for Virtual Routing:
1. **Music Channel**: Route YouTube/Spotify → BlackHole 2ch (Output)
2. **Vocal Channel**: MacBook Pro Microphone → System Input
3. **Mixed Output**: Multi-Output Device → System Output

## 6. 📱 Current Working Setup

### Without AG06 (Using System Audio):
- **Mac Browser**: http://localhost:9099 ✅
- **iPad Browser**: http://192.168.1.10:9099 ✅
- **Features**: All software features working
- **Limitation**: Hardware DSP effects not available

### Available Audio Devices:
1. **MacBook Pro Microphone** (Default Input)
2. **MacBook Pro Speakers** (Default Output)
3. **BlackHole 2ch** (Virtual Routing)
4. **Multi-Output Device** (Routing)

## 7. 🚀 Immediate Workaround

The AiOke system is **fully functional** without AG06:
- Dual-channel architecture works with software routing
- All AI features operational
- WebSocket streaming active
- Cross-device testing ready

### To proceed without AG06:
```bash
# System is already running and accessible
# Mac: http://localhost:9099
# iPad: http://192.168.1.10:9099
```

## 8. 🔧 Advanced Troubleshooting

If AG06 still not detected after driver installation:

### Reset USB System:
```bash
# Reset USB controllers
sudo kextunload -b com.apple.driver.usb.AppleUSBXHCI
sudo kextload -b com.apple.driver.usb.AppleUSBXHCI
```

### Check USB Power:
```bash
# Check USB power allocation
system_profiler SPUSBDataType | grep -A 5 "Current Available"
```

### Reset Core Audio:
```bash
# Kill and restart Core Audio
sudo killall coreaudiod
```

## 9. ✅ Next Steps

### Option A: Continue Without AG06
- System fully functional with software routing
- Use BlackHole for dual-channel separation
- All features available except hardware DSP

### Option B: Get AG06 Working
1. Install Yamaha driver (critical step)
2. Restart Mac
3. Allow driver in Security settings
4. Run detection again:
```bash
python3 enhanced_ag06_detection.py
```

## 10. 📞 Support Resources

- **Yamaha Support**: https://usa.yamaha.com/support/
- **AiOke Issues**: Current system fully operational
- **Alternative Mixers**: System supports any USB audio interface

---

## 🎤 Current System Status

**AiOke is FULLY OPERATIONAL** with:
- ✅ Dual-channel architecture (software routing)
- ✅ Cross-device support (Mac + iPad)
- ✅ All AI features active
- ✅ Enterprise monitoring working
- ⚠️ Hardware DSP unavailable (AG06 not connected)

**Continue testing at**:
- Mac: http://localhost:9099
- iPad: http://192.168.1.10:9099