import numpy as np
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
