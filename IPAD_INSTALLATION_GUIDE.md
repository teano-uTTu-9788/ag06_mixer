# 📱 AiOke iPad Installation Guide

## Quick Install (Local Network)

If AiOke is running on your local network:

1. **Connect to Same WiFi**
   - Ensure iPad is on the same network as the computer running AiOke

2. **Open in Safari** 
   - Launch Safari (MUST be Safari, not Chrome/Firefox)
   - Enter the local URL: `http://192.168.1.10:9090`
   - (Replace with your actual IP and port)

3. **Install as App**
   - Tap Share button (□ with ↑ arrow)
   - Scroll down and tap "Add to Home Screen"
   - Name it "AiOke"
   - Tap "Add"

4. **Launch**
   - Find AiOke icon on home screen
   - Tap to open in full-screen mode

---

## Cloud Installation (Works Anywhere)

### Step 1: Deploy to Google Cloud

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"

# Get YouTube API key from Google Cloud Console
export YOUTUBE_API_KEY="your-api-key"

# Deploy
./deploy_to_gcp.sh
```

You'll get a URL like: `https://aioke-production-xxxxx.run.app`

### Step 2: Install on iPad

1. **Open Cloud URL in Safari**
   - Enter: `https://aioke-production-xxxxx.run.app`

2. **Add to Home Screen**
   - Tap Share → Add to Home Screen → Add

3. **Enable Permissions**
   - When prompted, allow:
     - Microphone (for voice commands)
     - Camera (for gesture control)

---

## 🎯 Features Available After Installation

### Offline Capabilities
- ✅ Interface loads offline
- ✅ Previous searches cached
- ✅ Settings preserved
- ⚠️ YouTube videos need internet

### Voice Commands
- "Play [song name]"
- "Skip song"
- "Volume up/down"
- "Add reverb"
- "Remove vocals"

### Gesture Controls
- ✋ Hand up = Pause
- 👉 Swipe right = Next song
- 👈 Swipe left = Previous
- ✌️ Peace sign = Add to favorites

---

## Troubleshooting

### "Cannot Connect" Error
- Check WiFi connection
- Verify server is running: `ps aux | grep aioke`
- Try IP address instead of hostname

### No Full Screen
- Must use Safari (not Chrome)
- iOS 12.2 or later required
- Clear Safari cache and retry

### Videos Won't Play
- YouTube API key needed for search
- Check internet connection
- Some videos region-restricted

### Voice Commands Not Working
- Enable microphone in Settings → Safari
- Speak clearly after tapping mic button
- English language set in iPad settings

---

## Pro Tips

### 1. Quick Access
- Add to dock for faster access
- Use Siri: "Open AiOke"

### 2. Performance
- Close other apps for better performance
- Enable "Reduce Motion" in Accessibility

### 3. Battery Life
- Lower screen brightness during karaoke
- Use wired headphones vs Bluetooth

### 4. Group Karaoke
- Connect to TV via AirPlay
- Use external Bluetooth speaker
- Enable "Guided Access" to lock app

---

## Need Help?

### Check Server Status
```bash
curl http://192.168.1.10:9090/health
```

### View Logs
```bash
tail -f aioke.log
```

### Restart Server
```bash
pkill -f aioke_production_server
python3 aioke_production_server.py
```

---

## 🎤 Enjoy AiOke!

Your personal karaoke system is ready. Sing your heart out!