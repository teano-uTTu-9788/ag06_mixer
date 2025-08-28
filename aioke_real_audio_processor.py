#!/usr/bin/env python3
"""
AiOke Real Audio Processor - Actual Audio Processing
Processes real audio from microphone and music channels, no mock data
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioMetrics:
    """Real-time audio metrics from actual input"""
    rms_level: float
    peak_level: float
    frequency_centroid: float
    zero_crossing_rate: float
    spectral_rolloff: float
    is_clipping: bool
    is_silent: bool

class RealAudioProcessor:
    """Processes real audio from AiOke mixer channels"""
    
    def __init__(self, device_name: str = "AiOke/AG06/AG03"):
        self.device_name = device_name
        self.sample_rate = 44100
        self.block_size = 512
        self.channels = 2
        
        # Audio queues for real-time processing
        self.vocal_queue = queue.Queue(maxsize=100)
        self.music_queue = queue.Queue(maxsize=100)
        
        # Real-time metrics
        self.vocal_metrics = None
        self.music_metrics = None
        
        # Processing state
        self.is_running = False
        self.audio_stream = None
        
        # Audio processing parameters
        self.noise_floor = -60  # dB
        self.clipping_threshold = 0.95
        
        # Find AG06 device
        self.device_info = self._find_ag06_device()
        
    def _find_ag06_device(self) -> Optional[Dict]:
        """Find AiOke device in available audio devices"""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            # Check for AiOke or AG06 in device name
            if any(name in device['name'] for name in ['AiOke', 'AG06', 'AG03']) and device['max_input_channels'] >= 2:
                logger.info(f"Found AiOke device: {device['name']} at index {idx}")
                return {'index': idx, 'info': device}
        logger.warning("AiOke device not found, using default input")
        return None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Real-time audio callback - processes actual audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if indata is None:
            return
        
        try:
            # Process stereo channels separately
            # Channel 0: Vocal (from microphone)
            # Channel 1: Music (from system/YouTube)
            
            if indata.shape[1] >= 1:
                vocal_data = indata[:, 0].copy()
                self.vocal_queue.put_nowait(vocal_data)
                
            if indata.shape[1] >= 2:
                music_data = indata[:, 1].copy()
                self.music_queue.put_nowait(music_data)
            
        except queue.Full:
            # Drop oldest samples if queue is full
            try:
                self.vocal_queue.get_nowait()
                self.vocal_queue.put_nowait(indata[:, 0].copy())
            except:
                pass
            
            try:
                self.music_queue.get_nowait()
                self.music_queue.put_nowait(indata[:, 1].copy())
            except:
                pass
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
    
    def _calculate_audio_metrics(self, audio_data: np.ndarray) -> AudioMetrics:
        """Calculate real audio metrics from actual audio data"""
        
        # RMS level (root mean square - average power)
        rms = np.sqrt(np.mean(audio_data**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Peak level
        peak = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Frequency domain analysis
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        # Spectral centroid (brightness)
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
        
        # Zero crossing rate (percussion/speech detection)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
        zcr = zero_crossings / len(audio_data)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
            rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
        else:
            rolloff = 0
        
        # Detection flags
        is_clipping = peak >= self.clipping_threshold
        is_silent = rms_db < self.noise_floor
        
        return AudioMetrics(
            rms_level=float(rms),
            peak_level=float(peak),
            frequency_centroid=float(centroid),
            zero_crossing_rate=float(zcr),
            spectral_rolloff=float(rolloff),
            is_clipping=is_clipping,
            is_silent=is_silent
        )
    
    def _process_audio_queue(self):
        """Process audio from queues and calculate metrics"""
        while self.is_running:
            try:
                # Process vocal channel
                if not self.vocal_queue.empty():
                    vocal_samples = []
                    # Collect samples for analysis
                    while not self.vocal_queue.empty() and len(vocal_samples) < 10:
                        vocal_samples.append(self.vocal_queue.get_nowait())
                    
                    if vocal_samples:
                        vocal_data = np.concatenate(vocal_samples)
                        self.vocal_metrics = self._calculate_audio_metrics(vocal_data)
                
                # Process music channel
                if not self.music_queue.empty():
                    music_samples = []
                    while not self.music_queue.empty() and len(music_samples) < 10:
                        music_samples.append(self.music_queue.get_nowait())
                    
                    if music_samples:
                        music_data = np.concatenate(music_samples)
                        self.music_metrics = self._calculate_audio_metrics(music_data)
                
                time.sleep(0.05)  # 20Hz processing rate
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def start(self):
        """Start real audio processing"""
        if self.is_running:
            logger.warning("Audio processor already running")
            return
        
        try:
            # Use AG06 device if found, otherwise default
            device_index = self.device_info['index'] if self.device_info else None
            
            # Start audio stream with real audio callback
            self.audio_stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._audio_callback
            )
            
            self.audio_stream.start()
            self.is_running = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_audio_queue)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            logger.info(f"Real audio processing started on device: {device_index}")
            
        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}")
            raise
    
    def stop(self):
        """Stop audio processing"""
        self.is_running = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)
        
        logger.info("Audio processing stopped")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from actual audio input"""
        result = {
            'timestamp': time.time(),
            'vocal': None,
            'music': None,
            'status': 'processing' if self.is_running else 'stopped'
        }
        
        if self.vocal_metrics:
            result['vocal'] = {
                'rms_level': self.vocal_metrics.rms_level,
                'peak_level': self.vocal_metrics.peak_level,
                'frequency_centroid': self.vocal_metrics.frequency_centroid,
                'zero_crossing_rate': self.vocal_metrics.zero_crossing_rate,
                'spectral_rolloff': self.vocal_metrics.spectral_rolloff,
                'is_clipping': self.vocal_metrics.is_clipping,
                'is_silent': self.vocal_metrics.is_silent,
                'status': 'silent' if self.vocal_metrics.is_silent else 
                         ('clipping' if self.vocal_metrics.is_clipping else 'good')
            }
        
        if self.music_metrics:
            result['music'] = {
                'rms_level': self.music_metrics.rms_level,
                'peak_level': self.music_metrics.peak_level,
                'frequency_centroid': self.music_metrics.frequency_centroid,
                'zero_crossing_rate': self.music_metrics.zero_crossing_rate,
                'spectral_rolloff': self.music_metrics.spectral_rolloff,
                'is_clipping': self.music_metrics.is_clipping,
                'is_silent': self.music_metrics.is_silent,
                'status': 'silent' if self.music_metrics.is_silent else
                         ('clipping' if self.music_metrics.is_clipping else 'good')
            }
        
        return result
    
    def apply_ai_mixing(self, vocal_metrics: AudioMetrics, music_metrics: AudioMetrics) -> Dict[str, Any]:
        """Apply AI mixing decisions based on real audio metrics"""
        
        mixing_decisions = {
            'vocal_adjustments': {},
            'music_adjustments': {},
            'effects': {},
            'reasoning': []
        }
        
        # Vocal processing decisions based on real metrics
        if vocal_metrics:
            # Volume adjustment
            if vocal_metrics.is_clipping:
                mixing_decisions['vocal_adjustments']['gain'] = -6.0
                mixing_decisions['reasoning'].append("Vocal clipping detected, reducing gain")
            elif vocal_metrics.rms_level < 0.05:
                mixing_decisions['vocal_adjustments']['gain'] = 3.0
                mixing_decisions['reasoning'].append("Vocal too quiet, increasing gain")
            
            # EQ based on frequency content
            if vocal_metrics.frequency_centroid < 200:
                mixing_decisions['vocal_adjustments']['high_shelf'] = 3.0
                mixing_decisions['reasoning'].append("Vocal lacks brightness, adding high shelf")
            elif vocal_metrics.frequency_centroid > 4000:
                mixing_decisions['vocal_adjustments']['low_shelf'] = 2.0
                mixing_decisions['reasoning'].append("Vocal too bright, adding warmth")
            
            # Compression based on dynamics
            if vocal_metrics.peak_level / (vocal_metrics.rms_level + 1e-10) > 10:
                mixing_decisions['vocal_adjustments']['compression'] = {
                    'ratio': 4.0,
                    'threshold': -20.0,
                    'attack': 5.0,
                    'release': 100.0
                }
                mixing_decisions['reasoning'].append("High vocal dynamics, applying compression")
        
        # Music processing decisions
        if music_metrics:
            # Duck music when vocals are present
            if vocal_metrics and not vocal_metrics.is_silent:
                music_duck = -3.0 if vocal_metrics.rms_level > 0.1 else -1.5
                mixing_decisions['music_adjustments']['ducking'] = music_duck
                mixing_decisions['reasoning'].append(f"Ducking music by {music_duck}dB for vocal clarity")
            
            # Prevent music clipping
            if music_metrics.is_clipping:
                mixing_decisions['music_adjustments']['limiter'] = {
                    'threshold': -1.0,
                    'release': 50.0
                }
                mixing_decisions['reasoning'].append("Music clipping detected, applying limiter")
        
        # Overall mix decisions
        if vocal_metrics and music_metrics:
            # Reverb based on mix density
            total_energy = vocal_metrics.rms_level + music_metrics.rms_level
            if total_energy < 0.2:
                mixing_decisions['effects']['reverb'] = {
                    'wet': 0.3,
                    'room_size': 0.7
                }
                mixing_decisions['reasoning'].append("Sparse mix, adding reverb for fullness")
        
        return mixing_decisions


def test_real_audio():
    """Test real audio processing"""
    processor = RealAudioProcessor()
    
    try:
        # List available devices
        print("\n🎤 Available Audio Devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{idx}: {device['name']} ({device['max_input_channels']} channels)")
        
        print("\n🎵 Starting Real Audio Processing...")
        processor.start()
        
        print("\n📊 Monitoring Real Audio (Press Ctrl+C to stop):")
        print("-" * 50)
        
        while True:
            metrics = processor.get_real_time_metrics()
            
            # Display real metrics
            print(f"\n⏰ Time: {time.strftime('%H:%M:%S')}")
            
            if metrics['vocal']:
                v = metrics['vocal']
                print(f"🎤 Vocal: Level={v['rms_level']:.3f}, Peak={v['peak_level']:.3f}, "
                      f"Status={v['status']}, Freq={v['frequency_centroid']:.0f}Hz")
            else:
                print("🎤 Vocal: No signal")
            
            if metrics['music']:
                m = metrics['music']
                print(f"🎵 Music: Level={m['rms_level']:.3f}, Peak={m['peak_level']:.3f}, "
                      f"Status={m['status']}, Freq={m['frequency_centroid']:.0f}Hz")
            else:
                print("🎵 Music: No signal")
            
            # Apply AI mixing if both channels have data
            if metrics['vocal'] and metrics['music']:
                if processor.vocal_metrics and processor.music_metrics:
                    mixing = processor.apply_ai_mixing(
                        processor.vocal_metrics,
                        processor.music_metrics
                    )
                    if mixing['reasoning']:
                        print(f"🤖 AI Mixing: {', '.join(mixing['reasoning'][:2])}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Stopping audio processing...")
    finally:
        processor.stop()
        print("✅ Audio processing stopped")


if __name__ == "__main__":
    test_real_audio()