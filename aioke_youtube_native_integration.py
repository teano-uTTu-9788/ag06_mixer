#!/usr/bin/env python3
"""
AiOke YouTube Native App Integration for AG06 Mixer
Separates YouTube music playback from vocal mixing for true karaoke experience
"""

import asyncio
import pyaudio
import numpy as np
from aiohttp import web
import json
import subprocess
import platform
from typing import Dict, Optional
import sounddevice as sd

class YouTubeNativeKaraokeSystem:
    """
    Manages YouTube native app as music source with separate vocal processing
    
    Audio Routing:
    - YouTube App → System Audio → AG06 Ch 5/6 (USB input)
    - Microphone → AG06 Ch 1 (XLR input with phantom power)
    - AG06 Output → Speakers/Headphones
    """
    
    def __init__(self):
        # AG06 channel configuration
        self.channels = {
            'mic': 1,           # Channel 1: XLR microphone input
            'youtube': (5, 6),  # Channels 5/6: USB audio from computer
        }
        
        # Separate mixer settings for vocals and music
        self.vocal_mix = {
            'reverb': 0.3,      # Vocal reverb
            'echo': 0.15,       # Vocal echo/delay
            'compression': 0.4,  # Vocal compression
            'eq_low': 0,        # Low frequency cut for vocals
            'eq_mid': 0.2,      # Mid boost for presence
            'eq_high': 0.1,     # Slight high boost for clarity
            'level': 0.8        # Vocal level
        }
        
        self.music_mix = {
            'level': 0.9,       # Music level
            'vocal_reduction': 0.5,  # Center channel reduction for karaoke
            'eq_low': 0,        # Music bass
            'eq_mid': -0.2,     # Reduce mids to make room for vocals
            'eq_high': 0        # Music treble
        }
        
        # System audio routing
        self.audio_devices = self.detect_audio_devices()
        self.youtube_controller = YouTubeNativeController()
        
    def detect_audio_devices(self) -> Dict:
        """Detect AG06 and system audio devices"""
        devices = {}
        
        # List all audio devices
        device_list = sd.query_devices()
        
        for idx, device in enumerate(device_list):
            name = device['name'].lower()
            # Look for AG06 mixer
            if 'ag06' in name or 'yamaha' in name:
                if device['max_input_channels'] > 0:
                    devices['ag06_input'] = idx
                if device['max_output_channels'] > 0:
                    devices['ag06_output'] = idx
            # Look for system audio (for YouTube playback)
            elif 'soundflower' in name or 'blackhole' in name or 'loopback' in name:
                devices['virtual_audio'] = idx
                
        return devices
    
    async def setup_audio_routing(self):
        """Configure audio routing for karaoke setup"""
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # Route YouTube audio to AG06 USB channels
            await self.setup_macos_routing()
        elif system == 'Windows':
            await self.setup_windows_routing()
        else:
            print("⚠️ Manual audio routing setup required for Linux")
    
    async def setup_macos_routing(self):
        """Setup audio routing on macOS"""
        commands = [
            # Create aggregate device if needed
            ['sudo', 'killall', 'coreaudiod'],  # Restart audio system
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, check=False, capture_output=True)
            except:
                pass
        
        print("""
        📱 macOS Audio Setup Instructions:
        
        1. Open Audio MIDI Setup (in /Applications/Utilities/)
        2. Create Multi-Output Device:
           - Click '+' → Create Multi-Output Device
           - Check: AG06/AG03 (for mixer output)
           - Check: Built-in Output (for monitoring)
        
        3. Set System Output:
           - System Preferences → Sound → Output
           - Select: Multi-Output Device
        
        4. YouTube App:
           - Play music in YouTube app
           - Audio will route to AG06 Ch 5/6 via USB
        
        5. AG06 Mixer Settings:
           - Ch 1: Microphone (XLR input, phantom power ON if needed)
           - Ch 5/6: USB input from computer (YouTube audio)
           - Adjust faders to balance music vs vocals
        """)
    
    async def setup_windows_routing(self):
        """Setup audio routing on Windows"""
        print("""
        🖥️ Windows Audio Setup Instructions:
        
        1. Sound Settings:
           - Right-click speaker icon → Sound settings
           - Output: AG06/AG03
        
        2. AG06 Settings:
           - Ch 1: Microphone
           - Ch 5/6: USB audio from PC
        
        3. YouTube App:
           - Play music normally
           - Audio routes through USB to mixer
        """)
    
    def apply_vocal_effects(self, audio_data: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Apply effects to vocal channel only"""
        processed = audio_data.copy()
        
        # Apply compression
        if self.vocal_mix['compression'] > 0:
            threshold = 0.7
            ratio = 4.0
            processed = self.compress(processed, threshold, ratio, self.vocal_mix['compression'])
        
        # Apply EQ
        processed = self.apply_eq(processed, sample_rate,
                                 self.vocal_mix['eq_low'],
                                 self.vocal_mix['eq_mid'], 
                                 self.vocal_mix['eq_high'])
        
        # Apply reverb
        if self.vocal_mix['reverb'] > 0:
            processed = self.add_reverb(processed, sample_rate, self.vocal_mix['reverb'])
        
        # Apply echo/delay
        if self.vocal_mix['echo'] > 0:
            processed = self.add_delay(processed, sample_rate, self.vocal_mix['echo'])
        
        # Apply level
        processed *= self.vocal_mix['level']
        
        return processed
    
    def compress(self, audio: np.ndarray, threshold: float, ratio: float, amount: float) -> np.ndarray:
        """Dynamic range compression for vocals"""
        output = audio.copy()
        
        # Simple compressor
        mask = np.abs(audio) > threshold
        if np.any(mask):
            excess = np.abs(audio[mask]) - threshold
            compressed = threshold + excess / ratio
            output[mask] = np.sign(audio[mask]) * compressed
        
        # Mix with original based on amount
        return audio * (1 - amount) + output * amount
    
    def apply_eq(self, audio: np.ndarray, sr: int, low: float, mid: float, high: float) -> np.ndarray:
        """3-band EQ for vocals"""
        from scipy import signal
        
        output = audio.copy()
        
        # Low shelf (80Hz)
        if low != 0:
            sos = signal.butter(2, 80, btype='low', fs=sr, output='sos')
            low_band = signal.sosfilt(sos, audio)
            output += low_band * low
        
        # Mid bell (2kHz)
        if mid != 0:
            sos = signal.butter(2, [800, 3000], btype='band', fs=sr, output='sos')
            mid_band = signal.sosfilt(sos, audio)
            output += mid_band * mid
        
        # High shelf (8kHz)  
        if high != 0:
            sos = signal.butter(2, 8000, btype='high', fs=sr, output='sos')
            high_band = signal.sosfilt(sos, audio)
            output += high_band * high
            
        return output
    
    def add_reverb(self, audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
        """Add reverb to vocals"""
        from scipy import signal
        
        # Simple reverb using comb filters
        delays_ms = [29, 31, 37, 41]  # Prime numbers for natural sound
        output = audio.copy()
        
        for delay_ms in delays_ms:
            delay_samples = int(delay_ms * sr / 1000)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * 0.5
                output += delayed * (amount / len(delays_ms))
        
        return output
    
    def add_delay(self, audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
        """Add echo/delay to vocals"""
        delay_ms = 250  # Quarter note at 120 BPM
        feedback = 0.4
        
        delay_samples = int(delay_ms * sr / 1000)
        output = audio.copy()
        
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples]
            
            # Add feedback
            for i in range(3):  # 3 echoes
                delayed[delay_samples:] = delayed[:-delay_samples] * feedback
                output += delayed * (amount * (0.7 ** i))
        
        return output
    
    def reduce_vocals_from_music(self, stereo_audio: np.ndarray) -> np.ndarray:
        """
        Reduce vocals from YouTube music for karaoke
        Works on stereo signal by removing center channel
        """
        if stereo_audio.ndim != 2 or stereo_audio.shape[1] != 2:
            return stereo_audio
        
        left = stereo_audio[:, 0]
        right = stereo_audio[:, 1]
        
        # Calculate center (vocals) and sides (instruments)
        center = (left + right) / 2
        sides = (left - right) / 2
        
        # Reduce center channel based on setting
        reduction = self.music_mix['vocal_reduction']
        center *= (1 - reduction)
        
        # Reconstruct stereo with reduced vocals
        new_left = center + sides
        new_right = center - sides
        
        return np.column_stack([new_left, new_right])


class YouTubeNativeController:
    """Controls YouTube native app for music playback"""
    
    def __init__(self):
        self.system = platform.system()
        
    async def open_youtube_app(self):
        """Open YouTube native app"""
        if self.system == 'Darwin':
            subprocess.run(['open', '-a', 'YouTube'])
        elif self.system == 'Windows':
            subprocess.run(['start', 'youtube:'], shell=True)
        else:
            print("Please open YouTube app manually")
    
    async fun control_playback(self, action: str):
        """Control YouTube playback using system commands"""
        if self.system == 'Darwin':
            # Use AppleScript for macOS
            scripts = {
                'play': 'tell application "YouTube" to play',
                'pause': 'tell application "YouTube" to pause',
                'next': 'tell application "YouTube" to next track',
                'previous': 'tell application "YouTube" to previous track'
            }
            
            if action in scripts:
                subprocess.run(['osascript', '-e', scripts[action]])
                
    async def search_and_play(self, query: str):
        """Open YouTube app with search query"""
        search_url = f"youtube://results?search_query={query}"
        
        if self.system == 'Darwin':
            subprocess.run(['open', search_url])
        elif self.system == 'Windows':  
            subprocess.run(['start', search_url], shell=True)
        else:
            print(f"Open YouTube and search for: {query}")


async def create_karaoke_control_interface():
    """Create web interface for controlling karaoke system"""
    
    karaoke_system = YouTubeNativeKaraokeSystem()
    await karaoke_system.setup_audio_routing()
    
    async def handle_index(request):
        """Serve karaoke control interface"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AiOke - YouTube Native Karaoke</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    min-height: 100vh;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                }
                h1 {
                    text-align: center;
                    font-size: 48px;
                    margin-bottom: 10px;
                }
                .subtitle {
                    text-align: center;
                    font-size: 18px;
                    opacity: 0.9;
                    margin-bottom: 30px;
                }
                .section {
                    background: rgba(255,255,255,0.1);
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 20px;
                    backdrop-filter: blur(10px);
                }
                h2 {
                    font-size: 24px;
                    margin-bottom: 20px;
                    color: #4ade80;
                }
                .control-group {
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    opacity: 0.9;
                }
                input[type="range"] {
                    width: 100%;
                    height: 40px;
                    -webkit-appearance: none;
                    appearance: none;
                    background: transparent;
                    cursor: pointer;
                }
                input[type="range"]::-webkit-slider-track {
                    background: rgba(255,255,255,0.2);
                    height: 8px;
                    border-radius: 4px;
                }
                input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    background: #4ade80;
                    height: 24px;
                    width: 24px;
                    border-radius: 50%;
                    cursor: pointer;
                    margin-top: -8px;
                }
                .value {
                    float: right;
                    background: rgba(74, 222, 128, 0.2);
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                }
                .search-box {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                input[type="text"] {
                    flex: 1;
                    padding: 15px;
                    border: none;
                    border-radius: 10px;
                    background: rgba(255,255,255,0.2);
                    color: white;
                    font-size: 16px;
                }
                input[type="text"]::placeholder {
                    color: rgba(255,255,255,0.6);
                }
                button {
                    padding: 15px 30px;
                    border: none;
                    border-radius: 10px;
                    background: linear-gradient(45deg, #4ade80, #3b82f6);
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: transform 0.2s;
                }
                button:hover {
                    transform: scale(1.05);
                }
                .playback-controls {
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    margin-top: 20px;
                }
                .status {
                    background: rgba(74, 222, 128, 0.2);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    margin-top: 20px;
                }
                .instructions {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 30px;
                }
                .instructions h3 {
                    color: #4ade80;
                    margin-bottom: 15px;
                }
                .instructions ol {
                    padding-left: 20px;
                    line-height: 1.8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 AiOke</h1>
                <p class="subtitle">YouTube Native Karaoke System</p>
                
                <div class="section">
                    <h2>🎵 YouTube Music Control</h2>
                    <div class="search-box">
                        <input type="text" id="search" placeholder="Search for karaoke songs...">
                        <button onclick="searchYouTube()">Search</button>
                    </div>
                    <div class="playback-controls">
                        <button onclick="control('previous')">⏮️</button>
                        <button onclick="control('play')">▶️</button>
                        <button onclick="control('pause')">⏸️</button>
                        <button onclick="control('next')">⏭️</button>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎙️ Vocal Effects</h2>
                    <div class="control-group">
                        <label>Reverb <span class="value" id="reverb-val">30%</span></label>
                        <input type="range" id="reverb" min="0" max="100" value="30" oninput="updateVocal('reverb', this.value)">
                    </div>
                    <div class="control-group">
                        <label>Echo <span class="value" id="echo-val">15%</span></label>
                        <input type="range" id="echo" min="0" max="100" value="15" oninput="updateVocal('echo', this.value)">
                    </div>
                    <div class="control-group">
                        <label>Compression <span class="value" id="compression-val">40%</span></label>
                        <input type="range" id="compression" min="0" max="100" value="40" oninput="updateVocal('compression', this.value)">
                    </div>
                    <div class="control-group">
                        <label>Vocal Level <span class="value" id="level-val">80%</span></label>
                        <input type="range" id="level" min="0" max="100" value="80" oninput="updateVocal('level', this.value)">
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎛️ Music Mix</h2>
                    <div class="control-group">
                        <label>Music Level <span class="value" id="music-level-val">90%</span></label>
                        <input type="range" id="music-level" min="0" max="100" value="90" oninput="updateMusic('level', this.value)">
                    </div>
                    <div class="control-group">
                        <label>Vocal Reduction (Karaoke Mode) <span class="value" id="vocal-reduction-val">50%</span></label>
                        <input type="range" id="vocal-reduction" min="0" max="100" value="50" oninput="updateMusic('vocal_reduction', this.value)">
                    </div>
                </div>
                
                <div class="status">
                    ✅ System Ready - YouTube App as Music Source | AG06 Mixer for Vocals
                </div>
                
                <div class="instructions">
                    <h3>📋 Setup Instructions</h3>
                    <ol>
                        <li><strong>AG06 Mixer Setup:</strong>
                            <ul>
                                <li>Connect microphone to Channel 1 (XLR input)</li>
                                <li>Enable phantom power if using condenser mic</li>
                                <li>USB cable connects AG06 to computer</li>
                            </ul>
                        </li>
                        <li><strong>Audio Routing:</strong>
                            <ul>
                                <li>System audio output → AG06/AG03</li>
                                <li>YouTube audio plays through USB to Ch 5/6</li>
                                <li>Mic audio on Ch 1 mixes with music</li>
                            </ul>
                        </li>
                        <li><strong>YouTube App:</strong>
                            <ul>
                                <li>Search for karaoke versions of songs</li>
                                <li>Use "karaoke" or "instrumental" in search</li>
                                <li>Playback controlled from this interface</li>
                            </ul>
                        </li>
                        <li><strong>Performance Tips:</strong>
                            <ul>
                                <li>Adjust Ch 1 gain for optimal vocal level</li>
                                <li>Use Ch 5/6 fader to control music volume</li>
                                <li>Monitor through headphones on AG06</li>
                            </ul>
                        </li>
                    </ol>
                </div>
            </div>
            
            <script>
                async function searchYouTube() {
                    const query = document.getElementById('search').value;
                    if (query) {
                        await fetch('/api/youtube/search', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({query: query + ' karaoke'})
                        });
                    }
                }
                
                async function control(action) {
                    await fetch('/api/youtube/control', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({action: action})
                    });
                }
                
                async function updateVocal(param, value) {
                    document.getElementById(param + '-val').textContent = value + '%';
                    await fetch('/api/vocal/mix', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({[param]: value / 100})
                    });
                }
                
                async function updateMusic(param, value) {
                    document.getElementById(param.replace('_', '-') + '-val').textContent = value + '%';
                    await fetch('/api/music/mix', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({[param]: value / 100})
                    });
                }
                
                // Enter key searches
                document.getElementById('search').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') searchYouTube();
                });
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def handle_youtube_search(request):
        """Open YouTube app with search query"""
        data = await request.json()
        query = data.get('query', '')
        await karaoke_system.youtube_controller.search_and_play(query)
        return web.json_response({'status': 'ok', 'query': query})
    
    async def handle_youtube_control(request):
        """Control YouTube playback"""
        data = await request.json()
        action = data.get('action', '')
        await karaoke_system.youtube_controller.control_playback(action)
        return web.json_response({'status': 'ok', 'action': action})
    
    async def handle_vocal_mix(request):
        """Update vocal effects settings"""
        data = await request.json()
        karaoke_system.vocal_mix.update(data)
        return web.json_response({'status': 'ok', 'vocal_mix': karaoke_system.vocal_mix})
    
    async def handle_music_mix(request):
        """Update music mix settings"""
        data = await request.json()
        karaoke_system.music_mix.update(data)
        return web.json_response({'status': 'ok', 'music_mix': karaoke_system.music_mix})
    
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_post('/api/youtube/search', handle_youtube_search)
    app.router.add_post('/api/youtube/control', handle_youtube_control)
    app.router.add_post('/api/vocal/mix', handle_vocal_mix)
    app.router.add_post('/api/music/mix', handle_music_mix)
    
    return app


if __name__ == '__main__':
    print("""
    🎤 AiOke YouTube Native Karaoke System
    =====================================
    
    This system uses:
    - YouTube native app for music playback
    - AG06 mixer for separate vocal processing
    - Web interface for control
    
    Starting server on http://localhost:9091
    """)
    
    app = asyncio.run(create_karaoke_control_interface())
    web.run_app(app, host='0.0.0.0', port=9091)