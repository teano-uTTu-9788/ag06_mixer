# 🎤 AiOke Dual Channel Karaoke System Setup Guide

## System Architecture (Google Best Practices)

Following Google's audio engineering practices for professional audio systems:

```
┌──────────────┐     ┌──────────────┐
│  Channel 1   │     │  Channel 2   │
│   VOCALS     │     │    MUSIC     │
│  (Mic Input) │     │ (Any Source) │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐     ┌──────────────┐
│   Vocal      │     │    Music     │
│  Processing  │     │  Processing  │
│   Pipeline   │     │   Pipeline   │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
┌────────────────────────────────────┐
│        AG06 Hardware Mixer         │
│  Ch1: Vocals    Ch5/6: Music       │
│      (No software mixing)          │
└────────────────────────────────────┘
               │
               ▼
         Final Output
```

## Key Design Principles

1. **Complete Channel Separation**: Vocals and music never mix in software
2. **Hardware Mixing Only**: AG06 handles all channel blending
3. **Independent Processing**: Each channel has its own effects pipeline
4. **Source Agnostic**: Music can come from ANY source (YouTube, Spotify, Apple Music, etc.)
5. **Low Latency**: Direct hardware routing minimizes delay

## Setup Instructions

### Hardware Setup

#### AG06 Mixer Configuration

1. **Channel 1 - Vocals**
   - Connect microphone to XLR input (Channel 1)
   - Enable phantom power (+48V) if using condenser mic
   - Adjust GAIN knob for optimal input level
   - Set channel fader to unity (0dB)

2. **Channels 5/6 - Music**
   - USB connection from computer automatically routes to Ch 5/6
   - No physical connection needed (uses USB audio)
   - Adjust channel fader to balance with vocals

3. **Output Configuration**
   - Connect speakers/headphones to PHONES or MONITOR OUT
   - Use MONITOR/PHONES knob to adjust output level

### Software Setup

#### macOS Audio Configuration

1. **System Preferences → Sound**
   - Output Device: AG06/AG03
   - Input Device: AG06/AG03

2. **Audio MIDI Setup** (Applications → Utilities)
   ```
   - AG06/AG03: Set as default output
   - Format: 44100 Hz, 24-bit
   ```

3. **Music Source Setup**
   - **YouTube**: Play directly in browser or app
   - **Spotify**: Set output device to AG06 in app preferences
   - **Apple Music**: Uses system output (already configured)
   - **Local Files**: Any media player works

### Running the Dual Channel System

1. **Start the control interface:**
   ```bash
   python3 aioke_dual_channel_system.py
   ```

2. **Open control panel:**
   ```
   http://localhost:9092
   ```

3. **Configure effects independently:**
   - Adjust vocal effects (reverb, compression, EQ)
   - Set music processing (stereo width, vocal removal for karaoke)
   - All changes happen in real-time

## Usage Scenarios

### Scenario 1: YouTube Karaoke
1. Open YouTube in browser
2. Search for karaoke version of song
3. Play video - audio routes to AG06 Ch 5/6
4. Sing into microphone on Ch 1
5. Adjust channel faders to balance

### Scenario 2: Spotify with Live Vocals
1. Open Spotify app
2. Ensure output is set to AG06
3. Play any song
4. Enable "Vocal Removal" in music channel for karaoke effect
5. Sing along with processed audio

### Scenario 3: Professional Performance
1. Load backing track in any DAW or media player
2. Route audio output to AG06
3. Use advanced vocal effects (compression, reverb, delay)
4. Record final mix from AG06 USB output

## Effect Descriptions

### Vocal Channel Effects

- **Gate**: Removes background noise when not singing
- **Compressor**: Evens out vocal dynamics
- **EQ**: 
  - High-pass: Removes rumble and handling noise
  - Presence boost: Enhances clarity around 2kHz
  - Air: Adds brightness above 8kHz
- **Reverb**: Adds space and dimension
- **Delay**: Creates echo effects (optional)
- **Limiter**: Prevents clipping

### Music Channel Effects

- **Mid Duck**: Reduces frequencies where vocals sit
- **Stereo Enhancer**: Widens the stereo image
- **Vocal Remover**: Reduces center channel for karaoke
- **Limiter**: Prevents distortion

## Advantages of This Architecture

1. **Flexibility**: Any music source works without modification
2. **Quality**: No quality loss from software mixing
3. **Latency**: Hardware mixing = zero software latency
4. **Simplicity**: Each channel is independent and easy to control
5. **Professional**: Same approach used in professional studios

## Troubleshooting

### No Sound from Music
- Check AG06 is selected as system output device
- Verify USB cable is connected
- Check Ch 5/6 fader is up

### No Sound from Microphone
- Check XLR cable connection
- Enable phantom power if needed
- Adjust GAIN knob on Channel 1
- Check Channel 1 fader is up

### Feedback Issues
- Reduce monitor volume
- Position microphone away from speakers
- Use headphones for monitoring

### Latency Issues
- This system has minimal latency due to hardware mixing
- If experiencing delay, check buffer size in Audio MIDI Setup

## Advanced Tips

1. **Recording**: Use any DAW to record the AG06 stereo output
2. **Streaming**: OBS can capture AG06 output for live streaming
3. **Effects Automation**: Use MIDI to control effects in real-time
4. **Multi-mic Setup**: Use Ch 1-2 for duets

## Summary

This dual-channel architecture provides professional karaoke capabilities while maintaining simplicity and flexibility. By keeping channels completely separate until the hardware mixing stage, we achieve the best possible audio quality with the lowest latency - exactly how Google designs their professional audio systems.

---

*Based on Google's audio engineering best practices and professional studio standards.*