# 🚨 AiOke URGENT FIX - Device Powered but Not Detected

## IMMEDIATE ACTIONS (Do These Now):

### 1. ✅ CHECK AIOKE BACK PANEL SWITCH
**CRITICAL**: There's a switch on the back of AiOke!
- **MUST be set to**: `PC/MAC` mode (NOT iOS mode)
- If it's on iOS, that's why Mac can't see it!
- Switch it and the AiOke should appear immediately

### 2. 🔌 CHECK USB CABLE & PORT
The AiOke has TWO USB ports:
- **5V DC**: Power only (don't use this one for Mac)
- **USB 2.0**: Data connection (USE THIS ONE)

Make sure:
- USB cable is connected to the **USB 2.0** port on AiOke
- Other end connected directly to Mac (not through hub)
- Cable is a data cable (not charge-only)

### 3. 🔄 IF STILL NOT WORKING

Try this exact sequence:
1. Unplug AiOke from Mac
2. Turn OFF AiOke power
3. Check switch is on PC/MAC mode
4. Plug USB into AiOke's USB 2.0 port
5. Plug into Mac
6. Turn ON AiOke power
7. Wait 10 seconds

### 4. 🖥️ CHECK AUDIO MIDI SETUP
I've opened Audio MIDI Setup for you.
- Look in the left panel
- AiOke should appear if properly connected
- If not visible, click the "+" and "Rescan"

### 5. 💾 DRIVER DOWNLOAD LINKS

If above doesn't work, you NEED the driver:

**Option A - Yamaha Official**:
https://usa.yamaha.com/support/updates/yamaha_steinberg_usb_driver_for_mac.html

**Option B - Direct Link**:
https://www.steinberg.net/support/downloads/yamaha-steinberg-usb-driver/

**Option C - Manual Search**:
1. Go to: https://www.yamaha.com
2. Search: "AG06 driver macOS" (hardware model name)
3. Download latest version
4. Install and RESTART Mac

### 6. 🎯 QUICK DIAGNOSTIC

Run this to check if AiOke appears after fixes:
```bash
system_profiler SPUSBDataType | grep -i "yamaha\|aioke\|ag06\|0499"
```

If you see ANYTHING, we're making progress!

## 🔴 MOST COMMON ISSUE:
**90% of cases**: The switch on back is set to iOS instead of PC/MAC mode!

## 💡 CURRENT WORKAROUND:
While fixing AiOke, the system is fully functional at:
- Mac: http://localhost:9099
- iPad: http://192.168.1.10:9099

Using BlackHole 2ch for virtual routing.