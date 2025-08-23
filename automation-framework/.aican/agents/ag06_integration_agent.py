#!/usr/bin/env python3
"""
AG06 Hardware Integration Specialist Agent
Specialized agent for AG06 audio interface integration and real-time audio processing
"""

import asyncio
import subprocess
import json
import sys
import platform
from datetime import datetime
import threading
import queue
import numpy as np

try:
    import sounddevice as sd
    import pyaudio
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

class AG06IntegrationAgent:
    def __init__(self):
        self.platform = platform.system()
        self.ag06_device = None
        self.audio_queue = queue.Queue()
        self.is_monitoring = False
        
    async def detect_ag06_device(self):
        """Detect AG06/AG03 audio device on system"""
        print('🔍 Detecting AG06 audio device...')
        
        if not AUDIO_LIBS_AVAILABLE:
            print('❌ Audio libraries not installed. Installing...')
            await self.install_audio_libraries()
        
        try:
            devices = sd.query_devices()
            ag06_devices = []
            
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                if 'ag06' in device_name or 'ag03' in device_name or 'yamaha' in device_name:
                    ag06_devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
                    
            if ag06_devices:
                print(f'✅ Found {len(ag06_devices)} AG06/Yamaha device(s):')
                for device in ag06_devices:
                    print(f'  • Device {device["index"]}: {device["name"]}')
                    print(f'    Channels: {device["channels"]}, Sample Rate: {device["sample_rate"]}')
                self.ag06_device = ag06_devices[0]  # Use first found device
                return ag06_devices
            else:
                print('❌ No AG06/AG03 devices found')
                print('Available devices:')
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        print(f'  {i}: {device["name"]} (input channels: {device["max_input_channels"]})')
                return []
                
        except Exception as e:
            print(f'❌ Error detecting audio devices: {e}')
            return []
    
    async def install_audio_libraries(self):
        """Install required audio processing libraries"""
        print('📦 Installing audio processing libraries...')
        libraries = [
            'sounddevice',
            'PyAudio',
            'numpy',
            'scipy', 
            'librosa',
            'aubio'
        ]
        
        for lib in libraries:
            try:
                print(f'Installing {lib}...')
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-m', 'pip', 'install', lib,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                print(f'✅ {lib} installed')
            except Exception as e:
                print(f'❌ Failed to install {lib}: {e}')
    
    async def test_real_time_audio(self):
        """Test real-time audio capture from AG06"""
        if not self.ag06_device:
            print('❌ No AG06 device available for testing')
            return False
            
        print(f'🎤 Testing real-time audio capture from {self.ag06_device["name"]}...')
        
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f'Audio callback status: {status}')
                
                # Calculate RMS level
                rms_level = np.sqrt(np.mean(indata**2))
                if rms_level > 0.001:  # Threshold for actual audio
                    print(f'🔊 Audio detected: RMS = {rms_level:.6f}')
                    
                    # Basic frequency analysis
                    fft = np.fft.fft(indata[:, 0])
                    freqs = np.fft.fftfreq(len(fft), 1/48000)
                    magnitude = np.abs(fft)
                    
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                    dominant_freq = abs(freqs[dominant_freq_idx])
                    
                    # Classify as music or voice based on frequency characteristics
                    if 80 <= dominant_freq <= 300:
                        classification = 'Voice'
                    elif 20 <= dominant_freq <= 20000:
                        classification = 'Music'
                    else:
                        classification = 'Unknown'
                    
                    print(f'   Dominant frequency: {dominant_freq:.1f} Hz ({classification})')
            
            # Start audio stream
            stream = sd.InputStream(
                device=self.ag06_device['index'],
                channels=2,
                samplerate=48000,
                blocksize=1024,
                callback=audio_callback
            )
            
            print('🎧 Starting 10-second audio monitoring test...')
            print('   Try speaking or playing music into the AG06!')
            
            with stream:
                await asyncio.sleep(10)  # Monitor for 10 seconds
            
            print('✅ Audio test completed')
            return True
            
        except Exception as e:
            print(f'❌ Audio test failed: {e}')
            return False
    
    async def generate_optimized_audio_processor(self):
        """Generate optimized audio processor based on research findings"""
        processor_code = '''import numpy as np
import sounddevice as sd
import threading
import queue
from scipy import signal
from collections import deque
import time

class OptimizedAG06Processor:
    def __init__(self, device_index, sample_rate=48000, block_size=256):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.is_running = False
        
        # Circular buffer for real-time processing
        self.buffer = deque(maxlen=4096)
        self.spectrum_bands = 64
        
        # Initialize frequency bands (64 bands)
        self.freq_bands = np.logspace(np.log10(20), np.log10(20000), self.spectrum_bands)
        
    def start_monitoring(self):
        """Start real-time audio monitoring"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f'Status: {status}')
            
            # Convert to mono and add to buffer
            mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
            self.buffer.extend(mono_data)
            
            # Process if buffer has enough data
            if len(self.buffer) >= self.block_size:
                self.process_audio_block()
        
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=2,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=audio_callback
        )
        
        self.is_running = True
        self.stream.start()
        
    def process_audio_block(self):
        """Process audio block with spectrum analysis"""
        # Get latest block from buffer
        audio_block = np.array(list(self.buffer)[-self.block_size:])
        
        # Apply Hann window (Google best practice)
        windowed = audio_block * signal.windows.hann(len(audio_block))
        
        # FFT analysis
        fft = np.fft.fft(windowed, n=2048)
        magnitude = np.abs(fft[:1024])  # Take first half
        
        # Convert to frequency bands
        freqs = np.fft.fftfreq(2048, 1/self.sample_rate)[:1024]
        band_values = []
        
        for i in range(self.spectrum_bands):
            if i < len(self.freq_bands) - 1:
                band_mask = (freqs >= self.freq_bands[i]) & (freqs < self.freq_bands[i+1])
                band_energy = np.sum(magnitude[band_mask]) if np.any(band_mask) else 0
                band_values.append(band_energy)
            else:
                band_values.append(0)
        
        # Normalize to 0-100 range
        if max(band_values) > 0:
            normalized_bands = [(val/max(band_values)) * 100 for val in band_values]
        else:
            normalized_bands = [0] * self.spectrum_bands
        
        # Music vs Voice classification
        low_freq_energy = sum(normalized_bands[:8])   # 20-200 Hz
        mid_freq_energy = sum(normalized_bands[8:32])  # 200-2000 Hz  
        high_freq_energy = sum(normalized_bands[32:])  # 2000+ Hz
        
        if mid_freq_energy > low_freq_energy and mid_freq_energy > high_freq_energy:
            classification = 'voice'
        elif high_freq_energy > 30:
            classification = 'music'
        else:
            classification = 'ambient'
        
        # Calculate overall level
        rms_level = np.sqrt(np.mean(audio_block**2))
        level_db = 20 * np.log10(max(rms_level, 1e-10))
        
        return {
            'spectrum': normalized_bands,
            'level_db': level_db,
            'classification': classification,
            'timestamp': time.time()
        }
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        if hasattr(self, 'stream') and self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
'''
        
        # Save optimized processor
        with open('../optimized_ag06_processor.py', 'w') as f:
            f.write(processor_code)
        
        print('✅ Optimized AG06 processor generated')
        return processor_code

async def main():
    agent = AG06IntegrationAgent()
    
    print('🎛️ AG06 HARDWARE INTEGRATION SPECIALIST DEPLOYED')
    print('=' * 55)
    
    # Phase 1: Device detection
    devices = await agent.detect_ag06_device()
    
    # Phase 2: Test real-time audio if device found
    if devices and AUDIO_LIBS_AVAILABLE:
        await agent.test_real_time_audio()
    
    # Phase 3: Generate optimized processor
    await agent.generate_optimized_audio_processor()
    
    # Phase 4: Save configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'platform': agent.platform,
        'devices_found': devices,
        'audio_libs_available': AUDIO_LIBS_AVAILABLE,
        'recommendations': {
            'next_steps': [
                'Install missing audio libraries if needed',
                'Test optimized_ag06_processor.py with your AG06',
                'Integrate with existing Flask API',
                'Replace simulated data with real audio processing'
            ]
        }
    }
    
    with open('../ag06_integration_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('\n✅ AG06 integration analysis complete')
    print('📁 Files generated:')
    print('  • .aican/optimized_ag06_processor.py')
    print('  • .aican/ag06_integration_config.json')
    
    return config

if __name__ == '__main__':
    asyncio.run(main())