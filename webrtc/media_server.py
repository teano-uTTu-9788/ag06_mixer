"""
WebRTC Media Server with AI Audio Processing
Integrates WebRTC streams with the AI mixing engine
"""

import asyncio
import json
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from av import AudioFrame

# Import our AI mixing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_mixing_brain import AutonomousMixingEngine
from studio_dsp_chain import StudioDSPChain
from complete_ai_mixer import CompleteMixingSystem

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Audio stream configuration"""
    sample_rate: int = 48000
    channels: int = 2
    frame_size: int = 960  # 20ms at 48kHz
    genre: Optional[str] = None

class AudioProcessingTrack(MediaStreamTrack):
    """
    Custom audio track that processes audio through AI mixing engine
    """
    kind = "audio"
    
    def __init__(self, track: MediaStreamTrack, mixing_engine: CompleteMixingSystem, config: StreamConfig):
        super().__init__()
        self.track = track
        self.mixing_engine = mixing_engine
        self.config = config
        self.frame_count = 0
        
    async def recv(self):
        """Receive and process audio frame"""
        frame = await self.track.recv()
        
        # Convert to numpy array
        audio_array = frame.to_ndarray()
        
        # Ensure stereo
        if audio_array.ndim == 1:
            audio_array = np.stack([audio_array, audio_array])
        elif audio_array.shape[0] == 1:
            audio_array = np.repeat(audio_array, 2, axis=0)
            
        # Process through AI mixing engine
        try:
            # Transpose for mixing engine (expects samples x channels)
            audio_input = audio_array.T
            
            # Apply AI processing
            processed_audio = self.mixing_engine.process(
                audio_input,
                sample_rate=self.config.sample_rate
            )
            
            # Transpose back for frame
            processed_audio = processed_audio.T
            
            # Create new frame with processed audio
            new_frame = AudioFrame.from_ndarray(
                processed_audio.astype(np.int16),
                format='s16',
                layout='stereo'
            )
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            self.frame_count += 1
            
            return new_frame
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return frame

class WebRTCMediaServer:
    """
    WebRTC media server that handles peer connections and audio processing
    """
    
    def __init__(self):
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.audio_tracks: Dict[str, AudioProcessingTrack] = {}
        self.mixing_engines: Dict[str, CompleteMixingSystem] = {}
        self.stream_configs: Dict[str, StreamConfig] = {}
        
    async def create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create a new peer connection"""
        pc = RTCPeerConnection()
        self.peer_connections[peer_id] = pc
        
        # Create mixing engine for this peer
        config = StreamConfig()
        self.stream_configs[peer_id] = config
        self.mixing_engines[peer_id] = CompleteMixingSystem(config.sample_rate)
        
        # Handle track events
        @pc.on("track")
        async def on_track(track):
            logger.info(f"Track received: {track.kind} for peer {peer_id}")
            
            if track.kind == "audio":
                # Create processed track
                processed_track = AudioProcessingTrack(
                    track,
                    self.mixing_engines[peer_id],
                    self.stream_configs[peer_id]
                )
                self.audio_tracks[peer_id] = processed_track
                
                # Add processed track to peer connection for sending back
                pc.addTrack(processed_track)
                
                # Optional: Record the processed audio
                # recorder = MediaRecorder(f"recordings/{peer_id}.wav")
                # asyncio.create_task(recorder.start(processed_track))
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed to {pc.connectionState} for peer {peer_id}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.cleanup_peer(peer_id)
        
        return pc
    
    async def handle_offer(self, peer_id: str, offer: dict) -> dict:
        """Handle WebRTC offer and return answer"""
        pc = await self.create_peer_connection(peer_id)
        
        # Set remote description
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    async def handle_answer(self, peer_id: str, answer: dict):
        """Handle WebRTC answer"""
        if peer_id in self.peer_connections:
            pc = self.peer_connections[peer_id]
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
    
    async def add_ice_candidate(self, peer_id: str, candidate: dict):
        """Add ICE candidate to peer connection"""
        if peer_id in self.peer_connections:
            pc = self.peer_connections[peer_id]
            await pc.addIceCandidate(candidate)
    
    async def update_stream_config(self, peer_id: str, config: dict):
        """Update stream configuration including genre"""
        if peer_id in self.stream_configs:
            if "genre" in config:
                self.stream_configs[peer_id].genre = config["genre"]
                
                # Update mixing engine profile
                if peer_id in self.mixing_engines:
                    engine = self.mixing_engines[peer_id]
                    # The mixing engine will auto-detect or use the specified genre
                    logger.info(f"Updated genre to {config['genre']} for peer {peer_id}")
            
            if "sample_rate" in config:
                self.stream_configs[peer_id].sample_rate = config["sample_rate"]
            
            if "channels" in config:
                self.stream_configs[peer_id].channels = config["channels"]
    
    async def get_stream_stats(self, peer_id: str) -> dict:
        """Get statistics for a peer's stream"""
        stats = {
            "peer_id": peer_id,
            "connected": False,
            "audio_processing": False
        }
        
        if peer_id in self.peer_connections:
            pc = self.peer_connections[peer_id]
            stats["connected"] = pc.connectionState == "connected"
            
            if peer_id in self.audio_tracks:
                track = self.audio_tracks[peer_id]
                stats["audio_processing"] = True
                stats["frames_processed"] = track.frame_count
                
                # Get mixing engine stats
                if peer_id in self.mixing_engines:
                    engine = self.mixing_engines[peer_id]
                    # Add any available stats from the mixing engine
                    stats["genre"] = self.stream_configs[peer_id].genre
        
        return stats
    
    async def cleanup_peer(self, peer_id: str):
        """Clean up resources for a disconnected peer"""
        logger.info(f"Cleaning up peer {peer_id}")
        
        # Close peer connection
        if peer_id in self.peer_connections:
            pc = self.peer_connections[peer_id]
            await pc.close()
            del self.peer_connections[peer_id]
        
        # Remove tracks
        if peer_id in self.audio_tracks:
            del self.audio_tracks[peer_id]
        
        # Remove mixing engine
        if peer_id in self.mixing_engines:
            del self.mixing_engines[peer_id]
        
        # Remove config
        if peer_id in self.stream_configs:
            del self.stream_configs[peer_id]
    
    async def cleanup_all(self):
        """Clean up all peer connections"""
        peer_ids = list(self.peer_connections.keys())
        for peer_id in peer_ids:
            await self.cleanup_peer(peer_id)


class MediaServerIntegration:
    """
    Integration layer between WebRTC signaling and media server
    """
    
    def __init__(self, media_server: WebRTCMediaServer):
        self.media_server = media_server
        
    async def on_offer(self, peer_id: str, offer: dict) -> dict:
        """Handle offer from signaling server"""
        return await self.media_server.handle_offer(peer_id, offer)
    
    async def on_answer(self, peer_id: str, answer: dict):
        """Handle answer from signaling server"""
        await self.media_server.handle_answer(peer_id, answer)
    
    async def on_ice_candidate(self, peer_id: str, candidate: dict):
        """Handle ICE candidate from signaling server"""
        await self.media_server.add_ice_candidate(peer_id, candidate)
    
    async def on_stream_config(self, peer_id: str, config: dict):
        """Handle stream configuration update"""
        await self.media_server.update_stream_config(peer_id, config)
    
    async def on_disconnect(self, peer_id: str):
        """Handle peer disconnection"""
        await self.media_server.cleanup_peer(peer_id)
    
    async def get_stats(self, peer_id: str) -> dict:
        """Get stream statistics"""
        return await self.media_server.get_stream_stats(peer_id)


# Example usage
async def main():
    """Example of running the media server"""
    logging.basicConfig(level=logging.INFO)
    
    # Create media server
    media_server = WebRTCMediaServer()
    integration = MediaServerIntegration(media_server)
    
    # This would normally be integrated with the signaling server
    # For testing, you can create mock offers/answers
    
    logger.info("WebRTC Media Server with AI Processing ready")
    
    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await media_server.cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())