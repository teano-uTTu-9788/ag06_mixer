"""
WebRTC Signaling Server for Real-Time Audio Streaming
Following Google's WebRTC best practices and architecture
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import aioredis
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
import socketio

# Import our audio processing
from ai_mixing_brain import AutonomousMixingEngine
from complete_ai_mixer import CompleteMixingSystem

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Socket.IO server with Redis adapter for scaling
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    async_mode='aiohttp',
    client_manager=socketio.AsyncRedisManager('redis://localhost:6379'),
    logger=True,
    engineio_logger=True
)

app = web.Application()
sio.attach(app)

@dataclass
class PeerConnection:
    """Represents a WebRTC peer connection"""
    id: str
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    ice_candidates: list
    audio_config: dict
    is_publisher: bool = False
    is_subscriber: bool = False
    room_id: Optional[str] = None
    
class SignalingServer:
    """WebRTC signaling server with room management"""
    
    def __init__(self):
        self.connections: Dict[str, PeerConnection] = {}
        self.rooms: Dict[str, Set[str]] = {}
        self.redis_client = None
        self.mixing_engine = CompleteMixingSystem(44100)
        
    async def setup(self):
        """Initialize Redis connection for scaling"""
        self.redis_client = await aioredis.create_redis_pool(
            'redis://localhost:6379',
            encoding='utf-8'
        )
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()

# Global server instance
signaling_server = SignalingServer()

# WebRTC signaling events
@sio.event
async def connect(sid, environ):
    """Handle new WebRTC connection"""
    logger.info(f"Client connected: {sid}")
    
    # Create peer connection
    peer = PeerConnection(
        id=str(uuid.uuid4()),
        session_id=sid,
        user_id=None,
        created_at=datetime.utcnow(),
        ice_candidates=[],
        audio_config={
            'sampleRate': 48000,
            'channels': 2,
            'bitDepth': 16,
            'codec': 'opus'
        }
    )
    
    signaling_server.connections[sid] = peer
    
    # Send connection confirmation
    await sio.emit('connected', {
        'peerId': peer.id,
        'sessionId': sid,
        'audioConfig': peer.audio_config
    }, to=sid)
    
    return True

@sio.event
async def disconnect(sid):
    """Handle WebRTC disconnection"""
    logger.info(f"Client disconnected: {sid}")
    
    if sid in signaling_server.connections:
        peer = signaling_server.connections[sid]
        
        # Leave room if in one
        if peer.room_id:
            await leave_room_handler(sid, peer.room_id)
        
        # Cleanup
        del signaling_server.connections[sid]
    
    return True

@sio.event
async def offer(sid, data):
    """Handle WebRTC offer"""
    logger.info(f"Received offer from {sid}")
    
    if sid not in signaling_server.connections:
        await sio.emit('error', {'message': 'Connection not found'}, to=sid)
        return
    
    peer = signaling_server.connections[sid]
    peer.is_publisher = True
    
    # Store offer in Redis for scaling
    if signaling_server.redis_client:
        await signaling_server.redis_client.setex(
            f"offer:{peer.id}",
            300,  # 5 minute TTL
            json.dumps(data)
        )
    
    # Forward offer to subscribers in the same room
    if peer.room_id and peer.room_id in signaling_server.rooms:
        for subscriber_sid in signaling_server.rooms[peer.room_id]:
            if subscriber_sid != sid:
                await sio.emit('offer', {
                    'from': peer.id,
                    'offer': data['offer']
                }, to=subscriber_sid)
    
    return True

@sio.event
async def answer(sid, data):
    """Handle WebRTC answer"""
    logger.info(f"Received answer from {sid}")
    
    if sid not in signaling_server.connections:
        await sio.emit('error', {'message': 'Connection not found'}, to=sid)
        return
    
    peer = signaling_server.connections[sid]
    peer.is_subscriber = True
    
    # Find the publisher and forward answer
    target_peer_id = data.get('to')
    for conn_sid, conn in signaling_server.connections.items():
        if conn.id == target_peer_id:
            await sio.emit('answer', {
                'from': peer.id,
                'answer': data['answer']
            }, to=conn_sid)
            break
    
    return True

@sio.event
async def ice_candidate(sid, data):
    """Handle ICE candidate exchange"""
    logger.info(f"Received ICE candidate from {sid}")
    
    if sid not in signaling_server.connections:
        return
    
    peer = signaling_server.connections[sid]
    peer.ice_candidates.append(data['candidate'])
    
    # Forward to target peer
    target_peer_id = data.get('to')
    if target_peer_id:
        for conn_sid, conn in signaling_server.connections.items():
            if conn.id == target_peer_id:
                await sio.emit('ice_candidate', {
                    'from': peer.id,
                    'candidate': data['candidate']
                }, to=conn_sid)
                break
    
    return True

@sio.event
async def join_room(sid, room_id):
    """Join a room for group audio streaming"""
    logger.info(f"Client {sid} joining room {room_id}")
    
    if sid not in signaling_server.connections:
        return False
    
    peer = signaling_server.connections[sid]
    peer.room_id = room_id
    
    # Create room if doesn't exist
    if room_id not in signaling_server.rooms:
        signaling_server.rooms[room_id] = set()
    
    signaling_server.rooms[room_id].add(sid)
    
    # Join Socket.IO room
    sio.enter_room(sid, room_id)
    
    # Notify others in room
    await sio.emit('peer_joined', {
        'peerId': peer.id,
        'sessionId': sid
    }, room=room_id, skip_sid=sid)
    
    # Send current room members to new peer
    room_members = []
    for member_sid in signaling_server.rooms[room_id]:
        if member_sid != sid and member_sid in signaling_server.connections:
            member = signaling_server.connections[member_sid]
            room_members.append({
                'peerId': member.id,
                'sessionId': member_sid,
                'isPublisher': member.is_publisher
            })
    
    await sio.emit('room_members', {
        'roomId': room_id,
        'members': room_members
    }, to=sid)
    
    return True

async def leave_room_handler(sid, room_id):
    """Leave a room"""
    if room_id in signaling_server.rooms:
        signaling_server.rooms[room_id].discard(sid)
        
        # Clean up empty rooms
        if not signaling_server.rooms[room_id]:
            del signaling_server.rooms[room_id]
        
        # Leave Socket.IO room
        sio.leave_room(sid, room_id)
        
        # Notify others
        await sio.emit('peer_left', {
            'sessionId': sid
        }, room=room_id)

@sio.event
async def leave_room(sid, room_id):
    """Handle leave room request"""
    await leave_room_handler(sid, room_id)
    return True

@sio.event
async def audio_stream_metadata(sid, data):
    """Handle audio stream metadata updates"""
    if sid not in signaling_server.connections:
        return
    
    peer = signaling_server.connections[sid]
    
    # Update audio configuration
    peer.audio_config.update(data.get('config', {}))
    
    # Process audio settings through AI
    if 'genre' in data:
        # This would normally process through the mixing engine
        logger.info(f"Audio stream genre detected: {data['genre']}")
    
    return True

# REST API endpoints
async def health_check(request):
    """Health check endpoint for Kubernetes"""
    return web.json_response({
        'status': 'healthy',
        'connections': len(signaling_server.connections),
        'rooms': len(signaling_server.rooms),
        'timestamp': datetime.utcnow().isoformat()
    })

async def get_stats(request):
    """Get server statistics"""
    stats = {
        'connections': {
            'total': len(signaling_server.connections),
            'publishers': sum(1 for c in signaling_server.connections.values() if c.is_publisher),
            'subscribers': sum(1 for c in signaling_server.connections.values() if c.is_subscriber)
        },
        'rooms': {
            'total': len(signaling_server.rooms),
            'members': {room: len(members) for room, members in signaling_server.rooms.items()}
        }
    }
    return web.json_response(stats)

async def create_room(request):
    """Create a new room"""
    data = await request.json()
    room_id = data.get('roomId', str(uuid.uuid4()))
    
    if room_id not in signaling_server.rooms:
        signaling_server.rooms[room_id] = set()
    
    return web.json_response({
        'roomId': room_id,
        'created': True
    })

# Setup routes
app.router.add_get('/health', health_check)
app.router.add_get('/api/stats', get_stats)
app.router.add_post('/api/rooms', create_room)

# Setup CORS
cors = cors_setup(app, defaults={
    "*": ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*"
    )
})

# Startup and cleanup
async def on_startup(app):
    """Initialize server on startup"""
    await signaling_server.setup()
    logger.info("WebRTC signaling server started")

async def on_cleanup(app):
    """Cleanup on shutdown"""
    await signaling_server.cleanup()
    logger.info("WebRTC signaling server stopped")

app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8080)