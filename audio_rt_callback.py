"""
Real-time audio processing with callback mode and proper windowing
Following big-tech low-latency patterns: small buffers, callback processing, Hann window
"""

import numpy as np
import pyaudio
import queue
import threading
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class AudioConfig:
    """Audio configuration with sane defaults"""
    rate: int = 44100
    block_size: int = 512  # Small for low latency
    channels: int = 1
    format: int = pyaudio.paInt16
    device_index: Optional[int] = None

class RealTimeAudioProcessor:
    """
    Low-latency audio processor using callback mode
    Core Audio/PortAudio backend on macOS for minimal latency
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Lock-free queue for audio blocks
        self.audio_queue = queue.Queue(maxsize=8)
        
        # Pre-compute Hann window for spectral analysis
        self.hann_window = np.hanning(self.config.block_size)
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        
        # Callback for processed results
        self.result_callback: Optional[Callable] = None
        
        # Performance metrics
        self.drops = 0
        self.processed = 0
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PortAudio callback - runs in audio thread, must be fast!
        No heavy processing here, just queue the data
        """
        if status:
            # Log input overflow/underflow
            if status & pyaudio.paInputOverflow:
                self.drops += 1
        
        # Convert and queue if space available
        if not self.audio_queue.full():
            try:
                audio_array = np.frombuffer(in_data, dtype=np.int16)
                self.audio_queue.put_nowait(audio_array)
            except queue.Full:
                self.drops += 1
        else:
            self.drops += 1
        
        return (None, pyaudio.paContinue)
    
    def _process_loop(self):
        """
        Processing thread - pulls from queue and analyzes
        Separate from audio callback for stability
        """
        while self.running:
            try:
                # Get audio block with timeout
                audio_block = self.audio_queue.get(timeout=1.0)
                
                # Convert to float and normalize
                x = audio_block.astype(np.float32) / 32768.0
                
                # Apply Hann window for clean FFT
                x_windowed = x * self.hann_window
                
                # FFT for spectrum
                spectrum = np.abs(np.fft.rfft(x_windowed))
                
                # Find peak frequency with quadratic interpolation
                peak_hz = self._find_peak_frequency(spectrum)
                
                # Calculate RMS and peak
                rms = np.sqrt(np.mean(x**2))
                peak = np.max(np.abs(x))
                
                # Convert to dB
                rms_db = 20 * np.log10(rms + 1e-12)
                peak_db = 20 * np.log10(peak + 1e-12)
                
                # Simple classification
                classification = self._classify_audio(rms_db, peak_hz, spectrum)
                
                # 64-band spectrum for visualization
                spectrum_64 = self._compress_spectrum(spectrum, 64)
                
                # Deliver results via callback
                if self.result_callback:
                    self.result_callback({
                        "rms_db": rms_db,
                        "peak_db": peak_db,
                        "peak_hz": peak_hz,
                        "classification": classification,
                        "spectrum": spectrum_64.tolist(),
                        "drops": self.drops,
                        "processed": self.processed,
                        "audio_buffer": x  # Include raw audio for AI processing
                    })
                
                self.processed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _find_peak_frequency(self, spectrum: np.ndarray) -> float:
        """
        Find peak frequency with quadratic interpolation
        More accurate than just using bin center
        """
        # Skip DC and find peak
        spec = spectrum[1:]
        if len(spec) < 3:
            return 0.0
        
        k = np.argmax(spec[:-1]) + 1
        
        # Quadratic interpolation around peak
        if k > 0 and k < len(spec) - 1:
            alpha = spec[k - 1]
            beta = spec[k]
            gamma = spec[k + 1]
            
            # Parabola vertex offset
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-12)
            
            # Interpolated bin
            peak_bin = k + p
        else:
            peak_bin = k
        
        # Convert to Hz
        peak_hz = peak_bin * self.config.rate / self.config.block_size
        return peak_hz
    
    def _classify_audio(self, rms_db: float, peak_hz: float, spectrum: np.ndarray) -> str:
        """
        Simple audio classification
        Replace with WebRTC VAD or ML model for production
        """
        if rms_db < -50:
            return "silence"
        elif 80 < peak_hz < 250:  # Human voice fundamental range
            return "speech"
        else:
            return "music"
    
    def _compress_spectrum(self, spectrum: np.ndarray, bands: int) -> np.ndarray:
        """
        Compress spectrum to N bands for visualization
        Log-scale grouping for perceptual relevance
        """
        n_bins = len(spectrum)
        
        # Log-scale band edges
        log_freqs = np.logspace(np.log10(1), np.log10(n_bins), bands + 1)
        band_edges = log_freqs.astype(int)
        
        # Average within each band
        result = np.zeros(bands)
        for i in range(bands):
            start = band_edges[i]
            end = band_edges[i + 1]
            if start < n_bins and end <= n_bins:
                result[i] = np.mean(spectrum[start:end])
        
        return result
    
    def start(self, result_callback: Callable):
        """Start real-time audio processing"""
        self.result_callback = result_callback
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream with callback
        self.stream = self.pa.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            input_device_index=self.config.device_index,
            frames_per_buffer=self.config.block_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.pa.terminate()
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "processed_blocks": self.processed,
            "dropped_blocks": self.drops,
            "drop_rate": self.drops / (self.processed + self.drops + 1),
            "queue_size": self.audio_queue.qsize()
        }

# Example usage with SSE integration
if __name__ == "__main__":
    from sse_streaming import audio_state
    
    def update_sse_state(results):
        """Bridge between audio processor and SSE state"""
        audio_state.update(**results)
        print(f"RMS: {results['rms_db']:.1f} dB, Peak: {results['peak_hz']:.1f} Hz, Class: {results['classification']}")
    
    # Start processor
    processor = RealTimeAudioProcessor()
    processor.start(update_sse_state)
    
    try:
        # Run for 30 seconds
        time.sleep(30)
    finally:
        processor.stop()
        print(f"Stats: {processor.get_stats()}")