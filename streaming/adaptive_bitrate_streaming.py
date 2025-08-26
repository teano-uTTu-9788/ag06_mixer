"""
YouTube-style Adaptive Bitrate Streaming (HLS/DASH)
Following YouTube, Netflix, and Twitch streaming practices
"""

import asyncio
import json
import time
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import hashlib
import math

class StreamingProtocol(Enum):
    """Streaming protocols"""
    HLS = "hls"  # Apple HTTP Live Streaming
    DASH = "dash"  # MPEG-DASH
    CMAF = "cmaf"  # Common Media Application Format
    WEBRTC = "webrtc"  # Real-time
    RTMP = "rtmp"  # Legacy

class QualityLevel(Enum):
    """Stream quality levels (YouTube-style)"""
    AUDIO_ONLY = "audio"  # Audio only
    Q144P = "144p"  # 256kbps
    Q240P = "240p"  # 400kbps
    Q360P = "360p"  # 700kbps
    Q480P = "480p"  # 1.2Mbps
    Q720P = "720p"  # 2.5Mbps
    Q720P60 = "720p60"  # 3.5Mbps
    Q1080P = "1080p"  # 4.5Mbps
    Q1080P60 = "1080p60"  # 6Mbps
    Q1440P = "1440p"  # 10Mbps
    Q2160P = "4k"  # 15Mbps
    Q2160P60 = "4k60"  # 25Mbps

@dataclass
class BitrateProfile:
    """Bitrate profile for quality level"""
    quality: QualityLevel
    video_bitrate: int  # kbps
    audio_bitrate: int  # kbps
    resolution: Tuple[int, int]
    fps: int
    codec: str
    profile: str  # baseline, main, high

@dataclass
class StreamSegment:
    """Media segment"""
    segment_id: int
    quality: QualityLevel
    duration: float  # seconds
    size_bytes: int
    url: str
    timestamp: float
    keyframe: bool = False
    discontinuity: bool = False

@dataclass
class StreamManifest:
    """Stream manifest (HLS/DASH)"""
    stream_id: str
    duration: float
    segments: Dict[QualityLevel, List[StreamSegment]]
    bitrate_profiles: List[BitrateProfile]
    created_at: float
    updated_at: float
    live: bool = False
    dvr_window: float = 0  # seconds of DVR

@dataclass
class BufferState:
    """Client buffer state"""
    current_level: QualityLevel
    buffer_size: float  # seconds
    bandwidth_estimate: float  # Mbps
    dropped_frames: int
    stall_count: int
    switch_count: int

class ABRAlgorithm(Enum):
    """Adaptive Bitrate algorithms"""
    BUFFER_BASED = "buffer"  # Netflix's BBA
    THROUGHPUT = "throughput"  # Traditional
    HYBRID = "hybrid"  # YouTube's approach
    ML_BASED = "ml"  # Neural network
    BOLA = "bola"  # Lyapunov optimization
    MPC = "mpc"  # Model Predictive Control

class AdaptiveBitrateEngine:
    """YouTube-style ABR streaming engine"""
    
    def __init__(self):
        self.bitrate_profiles = self._initialize_profiles()
        self.algorithm = ABRAlgorithm.HYBRID
        self.bandwidth_estimator = BandwidthEstimator()
        self.quality_controller = QualityController()
        self.buffer_manager = BufferManager()
        self.analytics = StreamingAnalytics()
    
    def _initialize_profiles(self) -> Dict[QualityLevel, BitrateProfile]:
        """Initialize YouTube-style bitrate profiles"""
        return {
            QualityLevel.AUDIO_ONLY: BitrateProfile(
                QualityLevel.AUDIO_ONLY, 0, 128, (0, 0), 0, "opus", "audio"
            ),
            QualityLevel.Q144P: BitrateProfile(
                QualityLevel.Q144P, 128, 64, (256, 144), 30, "h264", "baseline"
            ),
            QualityLevel.Q240P: BitrateProfile(
                QualityLevel.Q240P, 336, 64, (426, 240), 30, "h264", "baseline"
            ),
            QualityLevel.Q360P: BitrateProfile(
                QualityLevel.Q360P, 636, 96, (640, 360), 30, "h264", "main"
            ),
            QualityLevel.Q480P: BitrateProfile(
                QualityLevel.Q480P, 1104, 128, (854, 480), 30, "h264", "main"
            ),
            QualityLevel.Q720P: BitrateProfile(
                QualityLevel.Q720P, 2372, 192, (1280, 720), 30, "h264", "high"
            ),
            QualityLevel.Q720P60: BitrateProfile(
                QualityLevel.Q720P60, 3308, 192, (1280, 720), 60, "h264", "high"
            ),
            QualityLevel.Q1080P: BitrateProfile(
                QualityLevel.Q1080P, 4308, 192, (1920, 1080), 30, "h264", "high"
            ),
            QualityLevel.Q1080P60: BitrateProfile(
                QualityLevel.Q1080P60, 5808, 256, (1920, 1080), 60, "h264", "high"
            ),
            QualityLevel.Q1440P: BitrateProfile(
                QualityLevel.Q1440P, 9808, 256, (2560, 1440), 30, "h265", "main10"
            ),
            QualityLevel.Q2160P: BitrateProfile(
                QualityLevel.Q2160P, 14808, 256, (3840, 2160), 30, "vp9", "profile2"
            ),
            QualityLevel.Q2160P60: BitrateProfile(
                QualityLevel.Q2160P60, 24808, 320, (3840, 2160), 60, "av1", "main"
            )
        }
    
    async def create_manifest(self, stream_id: str, duration: float,
                             segment_duration: float = 4.0) -> StreamManifest:
        """Create streaming manifest with all quality levels"""
        
        num_segments = math.ceil(duration / segment_duration)
        segments = {}
        
        for quality, profile in self.bitrate_profiles.items():
            quality_segments = []
            
            for i in range(num_segments):
                segment = StreamSegment(
                    segment_id=i,
                    quality=quality,
                    duration=min(segment_duration, duration - i * segment_duration),
                    size_bytes=int((profile.video_bitrate + profile.audio_bitrate) * 
                                  segment_duration * 1024 / 8),
                    url=f"/segments/{stream_id}/{quality.value}/segment_{i:05d}.ts",
                    timestamp=i * segment_duration,
                    keyframe=(i % 10 == 0)  # Keyframe every 10 segments
                )
                quality_segments.append(segment)
            
            segments[quality] = quality_segments
        
        manifest = StreamManifest(
            stream_id=stream_id,
            duration=duration,
            segments=segments,
            bitrate_profiles=list(self.bitrate_profiles.values()),
            created_at=time.time(),
            updated_at=time.time()
        )
        
        return manifest
    
    async def select_quality(self, buffer_state: BufferState,
                           network_state: Dict[str, float]) -> QualityLevel:
        """Select optimal quality level using ABR algorithm"""
        
        if self.algorithm == ABRAlgorithm.BUFFER_BASED:
            return await self._buffer_based_selection(buffer_state)
        elif self.algorithm == ABRAlgorithm.THROUGHPUT:
            return await self._throughput_based_selection(network_state)
        elif self.algorithm == ABRAlgorithm.HYBRID:
            return await self._hybrid_selection(buffer_state, network_state)
        elif self.algorithm == ABRAlgorithm.ML_BASED:
            return await self._ml_based_selection(buffer_state, network_state)
        elif self.algorithm == ABRAlgorithm.MPC:
            return await self._mpc_selection(buffer_state, network_state)
        else:
            return buffer_state.current_level
    
    async def _hybrid_selection(self, buffer_state: BufferState,
                               network_state: Dict[str, float]) -> QualityLevel:
        """YouTube-style hybrid ABR (buffer + throughput)"""
        
        bandwidth = network_state.get("bandwidth", 1.0)  # Mbps
        buffer_size = buffer_state.buffer_size
        
        # Phase detection
        if buffer_size < 5:  # Startup phase
            # Conservative quality to build buffer
            target_bitrate = bandwidth * 0.5 * 1000  # 50% of bandwidth
        elif buffer_size < 15:  # Steady state
            # Match bandwidth
            target_bitrate = bandwidth * 0.8 * 1000  # 80% of bandwidth
        else:  # Abundant buffer
            # Can be aggressive
            target_bitrate = bandwidth * 1.2 * 1000  # 120% of bandwidth
        
        # Find best matching quality
        best_quality = QualityLevel.Q360P  # Default
        
        for quality, profile in self.bitrate_profiles.items():
            total_bitrate = profile.video_bitrate + profile.audio_bitrate
            
            if total_bitrate <= target_bitrate:
                best_quality = quality
            else:
                break
        
        # Smoothing to avoid oscillation
        if buffer_state.switch_count > 5 and buffer_size > 10:
            # Too many switches, stay at current
            return buffer_state.current_level
        
        return best_quality
    
    async def _buffer_based_selection(self, buffer_state: BufferState) -> QualityLevel:
        """Netflix BBA algorithm"""
        buffer_size = buffer_state.buffer_size
        
        # Buffer zones (Netflix paper values)
        if buffer_size < 5:
            return QualityLevel.Q240P
        elif buffer_size < 10:
            return QualityLevel.Q360P
        elif buffer_size < 15:
            return QualityLevel.Q480P
        elif buffer_size < 20:
            return QualityLevel.Q720P
        elif buffer_size < 30:
            return QualityLevel.Q1080P
        else:
            return QualityLevel.Q1080P60
    
    async def _throughput_based_selection(self, network_state: Dict[str, float]) -> QualityLevel:
        """Traditional throughput-based selection"""
        bandwidth = network_state.get("bandwidth", 1.0) * 1000  # Convert to kbps
        safety_factor = 0.8  # Use 80% of available bandwidth
        
        target_bitrate = bandwidth * safety_factor
        
        selected = QualityLevel.Q240P
        for quality, profile in self.bitrate_profiles.items():
            if profile.video_bitrate + profile.audio_bitrate <= target_bitrate:
                selected = quality
            else:
                break
        
        return selected
    
    async def _ml_based_selection(self, buffer_state: BufferState,
                                 network_state: Dict[str, float]) -> QualityLevel:
        """ML-based selection using neural network (simulated)"""
        
        # Extract features
        features = [
            buffer_state.buffer_size / 60,  # Normalized buffer
            network_state.get("bandwidth", 1.0) / 25,  # Normalized bandwidth
            buffer_state.stall_count / 10,  # Normalized stalls
            buffer_state.switch_count / 20,  # Normalized switches
            network_state.get("rtt", 50) / 200,  # Normalized RTT
            network_state.get("loss", 0) / 5  # Normalized packet loss
        ]
        
        # Simulate neural network output
        score = sum(features) / len(features)
        
        # Map score to quality
        if score < 0.2:
            return QualityLevel.Q240P
        elif score < 0.4:
            return QualityLevel.Q480P
        elif score < 0.6:
            return QualityLevel.Q720P
        elif score < 0.8:
            return QualityLevel.Q1080P
        else:
            return QualityLevel.Q1440P
    
    async def _mpc_selection(self, buffer_state: BufferState,
                            network_state: Dict[str, float]) -> QualityLevel:
        """Model Predictive Control (simplified)"""
        
        # Look ahead horizon
        horizon = 5  # segments
        
        # Predict future bandwidth (simplified)
        current_bandwidth = network_state.get("bandwidth", 1.0)
        predicted_bandwidth = [current_bandwidth * (0.9 + 0.2 * i/horizon) 
                              for i in range(horizon)]
        
        # Calculate QoE for each quality option
        best_quality = buffer_state.current_level
        best_qoe = -float('inf')
        
        for quality in QualityLevel:
            profile = self.bitrate_profiles.get(quality)
            if not profile:
                continue
            
            # Simulate future buffer evolution
            future_buffer = buffer_state.buffer_size
            qoe = 0
            
            for bw in predicted_bandwidth:
                download_time = (profile.video_bitrate + profile.audio_bitrate) / (bw * 1000)
                future_buffer += 4 - download_time  # 4 second segments
                
                if future_buffer < 0:
                    qoe -= 100  # Heavy stall penalty
                    break
                
                # QoE = quality - switching - stalls
                qoe += math.log(profile.video_bitrate + 1)  # Quality term
                
                if quality != buffer_state.current_level:
                    qoe -= 5  # Switch penalty
            
            if qoe > best_qoe:
                best_qoe = qoe
                best_quality = quality
        
        return best_quality

class BandwidthEstimator:
    """Bandwidth estimation using EWMA and harmonic mean"""
    
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha  # EWMA weight
        self.samples: deque = deque(maxlen=20)
        self.estimate = 1.0  # Mbps
    
    def add_sample(self, bytes_downloaded: int, time_taken: float):
        """Add bandwidth sample"""
        if time_taken > 0:
            bandwidth_mbps = (bytes_downloaded * 8) / (time_taken * 1000000)
            self.samples.append(bandwidth_mbps)
            
            # EWMA update
            self.estimate = self.alpha * bandwidth_mbps + (1 - self.alpha) * self.estimate
    
    def get_estimate(self) -> float:
        """Get bandwidth estimate"""
        if not self.samples:
            return self.estimate
        
        # Harmonic mean for conservative estimate
        harmonic = len(self.samples) / sum(1/s for s in self.samples if s > 0)
        
        # Blend EWMA and harmonic mean
        return 0.7 * self.estimate + 0.3 * harmonic

class QualityController:
    """Quality switching controller with smoothing"""
    
    def __init__(self):
        self.quality_history: deque = deque(maxlen=10)
        self.switch_penalty = 0
        self.last_switch_time = 0
    
    def should_switch(self, current: QualityLevel, target: QualityLevel,
                      buffer_size: float) -> bool:
        """Decide if quality switch should happen"""
        
        # Don't switch too frequently
        if time.time() - self.last_switch_time < 2:  # 2 second cooldown
            return False
        
        # Don't switch if buffer is low
        if buffer_size < 5 and target > current:
            return False
        
        # Allow immediate downgrade if buffer critical
        if buffer_size < 2 and target < current:
            return True
        
        # Check if consistent upgrade is warranted
        if target > current:
            # Need consistent better conditions
            recent_qualities = list(self.quality_history)[-5:]
            if len(recent_qualities) >= 5 and all(q >= target for q in recent_qualities):
                return True
            return False
        
        return True
    
    def record_switch(self, new_quality: QualityLevel):
        """Record quality switch"""
        self.quality_history.append(new_quality)
        self.last_switch_time = time.time()
        self.switch_penalty = min(self.switch_penalty + 1, 10)

class BufferManager:
    """Buffer management and monitoring"""
    
    def __init__(self, target_buffer: float = 20.0):
        self.target_buffer = target_buffer  # seconds
        self.buffer_level = 0.0
        self.stall_events: List[float] = []
        self.buffer_history: deque = deque(maxlen=100)
    
    def update_buffer(self, downloaded: float, consumed: float):
        """Update buffer level"""
        self.buffer_level += downloaded - consumed
        self.buffer_level = max(0, self.buffer_level)
        
        self.buffer_history.append(self.buffer_level)
        
        if self.buffer_level == 0:
            self.stall_events.append(time.time())
    
    def get_buffer_health(self) -> str:
        """Get buffer health status"""
        if self.buffer_level < 2:
            return "critical"
        elif self.buffer_level < 5:
            return "low"
        elif self.buffer_level < 15:
            return "normal"
        else:
            return "healthy"
    
    def predict_stall(self, download_rate: float, playback_rate: float) -> float:
        """Predict time to stall"""
        if download_rate >= playback_rate:
            return float('inf')
        
        deficit_rate = playback_rate - download_rate
        return self.buffer_level / deficit_rate

class StreamingAnalytics:
    """YouTube-style analytics"""
    
    def __init__(self):
        self.metrics = {
            "startup_time": 0,
            "total_stalls": 0,
            "stall_duration": 0,
            "quality_switches": 0,
            "average_bitrate": 0,
            "qoe_score": 100
        }
        self.quality_time: Dict[QualityLevel, float] = {}
        self.session_start = time.time()
    
    def calculate_qoe(self, buffer_state: BufferState) -> float:
        """Calculate Quality of Experience score"""
        
        # YouTube's QoE formula (simplified)
        # QoE = α*quality - β*stalls - γ*switches
        
        quality_score = self._get_quality_score(buffer_state.current_level)
        stall_penalty = buffer_state.stall_count * 20
        switch_penalty = buffer_state.switch_count * 5
        
        qoe = 100 * quality_score - stall_penalty - switch_penalty
        
        return max(0, min(100, qoe))
    
    def _get_quality_score(self, quality: QualityLevel) -> float:
        """Map quality to score"""
        quality_scores = {
            QualityLevel.Q240P: 0.3,
            QualityLevel.Q360P: 0.4,
            QualityLevel.Q480P: 0.5,
            QualityLevel.Q720P: 0.7,
            QualityLevel.Q1080P: 0.85,
            QualityLevel.Q1440P: 0.95,
            QualityLevel.Q2160P: 1.0
        }
        return quality_scores.get(quality, 0.5)

# Example usage
async def main():
    """Demonstrate adaptive bitrate streaming"""
    
    print("📺 YouTube-style Adaptive Bitrate Streaming")
    print("=" * 60)
    
    # Initialize ABR engine
    abr = AdaptiveBitrateEngine()
    
    # Create manifest for a 10-minute video
    manifest = await abr.create_manifest("video_001", 600.0)
    
    print(f"\n📋 Stream Manifest Created:")
    print(f"Stream ID: {manifest.stream_id}")
    print(f"Duration: {manifest.duration}s")
    print(f"Quality Levels: {len(manifest.bitrate_profiles)}")
    
    # Simulate streaming session
    print("\n🎬 Simulating Streaming Session:")
    print("-" * 40)
    
    # Initial buffer state
    buffer_state = BufferState(
        current_level=QualityLevel.Q360P,
        buffer_size=10.0,
        bandwidth_estimate=5.0,
        dropped_frames=0,
        stall_count=0,
        switch_count=0
    )
    
    # Simulate network conditions
    network_states = [
        {"bandwidth": 2.0, "rtt": 50, "loss": 0},   # Poor
        {"bandwidth": 5.0, "rtt": 30, "loss": 0},   # Medium
        {"bandwidth": 10.0, "rtt": 20, "loss": 0},  # Good
        {"bandwidth": 25.0, "rtt": 10, "loss": 0},  # Excellent
        {"bandwidth": 1.0, "rtt": 100, "loss": 2},  # Degraded
    ]
    
    print("\n📊 ABR Decisions:")
    
    for i, network in enumerate(network_states):
        # Update bandwidth estimate
        abr.bandwidth_estimator.add_sample(1000000, 1.0/network["bandwidth"])
        
        # Select quality
        selected = await abr.select_quality(buffer_state, network)
        
        print(f"\nTime {i*10}s:")
        print(f"  Network: {network['bandwidth']:.1f} Mbps")
        print(f"  Buffer: {buffer_state.buffer_size:.1f}s")
        print(f"  Selected: {selected.value}")
        
        # Update buffer state for next iteration
        buffer_state.current_level = selected
        buffer_state.bandwidth_estimate = network["bandwidth"]
        
        # Simulate buffer evolution
        if network["bandwidth"] < 3:
            buffer_state.buffer_size = max(0, buffer_state.buffer_size - 2)
            if buffer_state.buffer_size == 0:
                buffer_state.stall_count += 1
        else:
            buffer_state.buffer_size = min(30, buffer_state.buffer_size + 3)
        
        if selected != buffer_state.current_level:
            buffer_state.switch_count += 1
    
    # Calculate final QoE
    analytics = StreamingAnalytics()
    qoe = analytics.calculate_qoe(buffer_state)
    
    print("\n📈 Session Analytics:")
    print("-" * 40)
    print(f"Quality Switches: {buffer_state.switch_count}")
    print(f"Stalls: {buffer_state.stall_count}")
    print(f"QoE Score: {qoe:.1f}/100")
    
    print("\n✅ Adaptive bitrate streaming operational!")
    
    return abr

if __name__ == "__main__":
    asyncio.run(main())