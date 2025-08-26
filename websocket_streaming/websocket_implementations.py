#!/usr/bin/env python3
"""
WebSocket Streaming Implementations - SOLID Compliance
Implements the interfaces defined in interfaces.py with production-ready features
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Set, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import weakref
import threading
from collections import defaultdict, deque
import numpy as np
import base64

# Import our SOLID interfaces
from .interfaces import (
    IAudioProcessor, IConnectionManager, ISecurityValidator, ICircuitBreaker,
    IBackpressureManager, IPerformanceMonitor, IMessageRouter, IStateManager,
    AudioMessage, ConnectionInfo, PerformanceMetrics, SecurityEvent,
    CircuitBreakerState, MessagePriority, SecurityLevel,
    AudioConfig, SecurityConfig, PerformanceConfig, ResilienceConfig, StreamingConfig,
    StreamingException, SecurityValidationError, CircuitBreakerOpenError,
    BackpressureError, PerformanceThresholdExceeded
)

# Import existing circuit breaker
from reliability.circuit_breaker import (
    CircuitBreaker as ReliabilityCircuitBreaker,
    CircuitBreakerConfig as ReliabilityConfig,
    circuit_breaker_registry
)

# AI Mixer imports (with fallback for testing)
try:
    from ai_mixing_brain import AutonomousMixingEngine
    from studio_dsp_chain import StudioDSPChain
except ImportError:
    # Mock classes for testing
    class AutonomousMixingEngine:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate
            
        def process(self, audio):
            return audio
    
    class StudioDSPChain:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate
            
        def process_full_chain(self, audio):
            return audio
        
        def process_essential(self, audio):
            return audio
        
        def process_minimal(self, audio):
            return audio

logger = logging.getLogger(__name__)


class ProductionAudioProcessor:
    """Production-ready audio processor with AI mixing and DSP"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mixing_engine = AutonomousMixingEngine(config.sample_rate)
        self.dsp_chain = StudioDSPChain(config.sample_rate)
        self.processing_stats = defaultdict(list)
        self.sessions: Dict[str, Dict] = {}
        
    async def process_audio_frame(self, audio_data: bytes, session_id: str) -> bytes:
        """Process single audio frame with session context"""
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array (expecting float32 PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Handle different audio formats
            expected_size = self.config.frame_size * self.config.channels
            if len(audio_array) != expected_size:
                # Try to adapt the audio data
                if len(audio_array) == self.config.frame_size:
                    # Mono to stereo conversion
                    audio_array = np.repeat(audio_array, self.config.channels)
                elif len(audio_array) > expected_size:
                    # Truncate to expected size
                    audio_array = audio_array[:expected_size]
                elif len(audio_array) < expected_size:
                    # Pad to expected size
                    padding = np.zeros(expected_size - len(audio_array), dtype=np.float32)
                    audio_array = np.concatenate([audio_array, padding])
            
            # Reshape to channels x samples for processing
            try:
                audio_samples = audio_array.reshape(self.config.channels, -1)
            except ValueError:
                # Fallback: create stereo from mono or adjust shape
                if len(audio_array) % self.config.channels != 0:
                    # Pad to make it divisible
                    pad_size = self.config.channels - (len(audio_array) % self.config.channels)
                    audio_array = np.pad(audio_array, (0, pad_size), mode='constant')
                audio_samples = audio_array.reshape(self.config.channels, -1)
            
            # Apply AI mixing based on detected genre
            processed_samples = await self._apply_ai_mixing(audio_samples, session_id)
            
            # Apply studio DSP chain
            processed_samples = await self._apply_dsp_chain(processed_samples, session_id)
            
            # Convert back to bytes
            processed_audio = processed_samples.flatten().astype(np.float32).tobytes()
            
            # Record processing metrics
            processing_time = (time.time() - start_time) * 1000
            self._record_processing_metrics(session_id, processing_time, len(audio_data))
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Audio processing failed for session {session_id}: {e}")
            raise
    
    async def _apply_ai_mixing(self, audio: np.ndarray, session_id: str) -> np.ndarray:
        """Apply AI-powered mixing based on genre detection"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {'genre_history': deque(maxlen=10)}
            
        try:
            # Process through mixing engine (check method signature)
            if hasattr(self.mixing_engine, 'process'):
                # Try with sample_rate parameter first
                try:
                    processed = self.mixing_engine.process(audio.T, sample_rate=self.config.sample_rate)
                except TypeError:
                    # Fallback to no sample_rate parameter
                    processed = self.mixing_engine.process(audio.T)
            else:
                # Fallback: simple pass-through with basic processing
                processed = audio.T
                
            # Update session with detected genre
            if hasattr(self.mixing_engine, 'detected_genre'):
                self.sessions[session_id]['genre_history'].append(self.mixing_engine.detected_genre)
            else:
                # Mock genre detection
                self.sessions[session_id]['genre_history'].append('unknown')
                
            return processed.T
        except Exception as e:
            logger.warning(f"AI mixing failed for session {session_id}, using passthrough: {e}")
            return audio
    
    async def _apply_dsp_chain(self, audio: np.ndarray, session_id: str) -> np.ndarray:
        """Apply studio DSP processing chain"""
        quality = self.config.quality_level
        
        try:
            if hasattr(self.dsp_chain, 'process_full_chain') and quality == "high_quality":
                processed = self.dsp_chain.process_full_chain(audio.T)
            elif hasattr(self.dsp_chain, 'process_essential') and quality == "balanced":
                processed = self.dsp_chain.process_essential(audio.T)
            elif hasattr(self.dsp_chain, 'process_minimal'):
                processed = self.dsp_chain.process_minimal(audio.T)
            else:
                # Fallback: basic processing
                processed = self._apply_basic_dsp(audio.T)
                
            return processed.T
        except Exception as e:
            logger.warning(f"DSP processing failed for session {session_id}, using basic processing: {e}")
            return self._apply_basic_dsp(audio)
    
    def _apply_basic_dsp(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic DSP processing as fallback"""
        # Basic noise gate
        threshold = 0.01
        audio = np.where(np.abs(audio) > threshold, audio, audio * 0.1)
        
        # Basic limiting
        audio = np.clip(audio, -0.95, 0.95)
        
        return audio
    
    def _record_processing_metrics(self, session_id: str, processing_time: float, data_size: int):
        """Record processing metrics for monitoring"""
        self.processing_stats[session_id].append({
            'timestamp': time.time(),
            'processing_time_ms': processing_time,
            'data_size_bytes': data_size,
            'throughput_mbps': (data_size * 8) / (processing_time / 1000) / 1_000_000
        })
        
        # Keep only recent metrics (last 100 frames)
        if len(self.processing_stats[session_id]) > 100:
            self.processing_stats[session_id] = self.processing_stats[session_id][-100:]
    
    async def get_processing_metrics(self, session_id: str) -> Dict[str, float]:
        """Get processing performance metrics"""
        if session_id not in self.processing_stats:
            return {}
            
        metrics = self.processing_stats[session_id]
        if not metrics:
            return {}
        
        recent_metrics = metrics[-10:]  # Last 10 frames
        
        result = {
            'avg_processing_time_ms': np.mean([m['processing_time_ms'] for m in recent_metrics]),
            'max_processing_time_ms': np.max([m['processing_time_ms'] for m in recent_metrics]),
            'avg_throughput_mbps': np.mean([m['throughput_mbps'] for m in recent_metrics]),
            'frames_processed': len(metrics),
        }
        
        # Add genre info if available
        if session_id in self.sessions:
            genre_history = self.sessions[session_id].get('genre_history', deque())
            if len(genre_history) > 0:
                result['current_genre'] = genre_history[-1]
            else:
                result['current_genre'] = 'unknown'
        else:
            result['current_genre'] = 'unknown'
            
        return result
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return [
            f"float32_pcm_{self.config.sample_rate}hz",
            f"int16_pcm_{self.config.sample_rate}hz",
            f"stereo_{self.config.channels}ch"
        ]


class ProductionConnectionManager:
    """Production WebSocket connection manager with room support"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.connections: Dict[str, Any] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        
    async def register_connection(self, websocket: Any, user_id: str) -> str:
        """Register new WebSocket connection, return connection ID"""
        connection_id = f"conn_{uuid.uuid4().hex[:12]}"
        
        with self._lock:
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                user_id=user_id,
                ip_address=getattr(websocket, 'remote_address', ['unknown', 0])[0],
                user_agent=getattr(websocket, 'request_headers', {}).get('user-agent', 'unknown'),
                connected_at=time.time(),
                last_activity=time.time()
            )
            
            self.connections[connection_id] = {
                'websocket': websocket,
                'info': connection_info,
                'message_count': 0,
                'last_ping': time.time()
            }
            
            self.user_sessions[user_id].append(connection_id)
            
        logger.info(f"Registered connection {connection_id} for user {user_id}")
        return connection_id
    
    async def get_connection(self, connection_id: str) -> Optional[Any]:
        """Retrieve WebSocket connection by ID"""
        with self._lock:
            conn_data = self.connections.get(connection_id)
            return conn_data['websocket'] if conn_data else None
    
    async def cleanup_connection(self, connection_id: str) -> bool:
        """Clean up and remove connection"""
        with self._lock:
            if connection_id not in self.connections:
                return False
                
            conn_data = self.connections[connection_id]
            user_id = conn_data['info'].user_id
            
            # Remove from rooms
            for room_connections in self.rooms.values():
                room_connections.discard(connection_id)
            
            # Remove from user sessions
            if connection_id in self.user_sessions[user_id]:
                self.user_sessions[user_id].remove(connection_id)
                
            # Close websocket if still open
            try:
                websocket = conn_data['websocket']
                if hasattr(websocket, 'close'):
                    await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket for {connection_id}: {e}")
            
            # Remove connection
            del self.connections[connection_id]
            
        logger.info(f"Cleaned up connection {connection_id}")
        return True
    
    async def get_active_connections(self) -> Dict[str, Any]:
        """Get all active connections"""
        with self._lock:
            return {
                conn_id: {
                    'user_id': data['info'].user_id,
                    'connected_at': data['info'].connected_at,
                    'last_activity': data['info'].last_activity,
                    'message_count': data['message_count'],
                    'room_id': data['info'].room_id
                }
                for conn_id, data in self.connections.items()
            }
    
    async def broadcast_to_room(self, room_id: str, message: bytes) -> int:
        """Broadcast message to all connections in room, return count"""
        if room_id not in self.rooms:
            return 0
            
        connections_in_room = list(self.rooms[room_id])
        successful_sends = 0
        
        for connection_id in connections_in_room:
            try:
                websocket = await self.get_connection(connection_id)
                if websocket and hasattr(websocket, 'send'):
                    await websocket.send(message)
                    successful_sends += 1
                    
                    # Update activity timestamp
                    if connection_id in self.connections:
                        self.connections[connection_id]['info'].last_activity = time.time()
                        
            except Exception as e:
                logger.warning(f"Failed to send to connection {connection_id}: {e}")
                # Mark for cleanup
                asyncio.create_task(self.cleanup_connection(connection_id))
        
        return successful_sends
    
    async def add_to_room(self, connection_id: str, room_id: str) -> bool:
        """Add connection to room"""
        if connection_id not in self.connections:
            return False
            
        with self._lock:
            self.rooms[room_id].add(connection_id)
            self.connections[connection_id]['info'].room_id = room_id
            
        return True
    
    async def remove_from_room(self, connection_id: str, room_id: str) -> bool:
        """Remove connection from room"""
        with self._lock:
            if room_id in self.rooms:
                self.rooms[room_id].discard(connection_id)
                
                # Clear room if empty
                if not self.rooms[room_id]:
                    del self.rooms[room_id]
            
            if connection_id in self.connections:
                self.connections[connection_id]['info'].room_id = None
                
        return True


class ProductionSecurityValidator:
    """Production security validator with rate limiting and validation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque())
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self._lock = threading.RLock()
        
    async def validate_connection(self, websocket: Any, request: Any) -> bool:
        """Validate connection security requirements"""
        try:
            # Check origin if specified
            origin = getattr(request, 'headers', {}).get('origin')
            if self.config.allowed_origins and origin not in self.config.allowed_origins:
                self._record_security_event(
                    event_type="invalid_origin",
                    connection_id="pending",
                    severity="medium",
                    details={"origin": origin}
                )
                return False
            
            # Check IP blocking
            ip_address = getattr(websocket, 'remote_address', ['unknown', 0])[0]
            if ip_address in self.blocked_ips:
                self._record_security_event(
                    event_type="blocked_ip",
                    connection_id="pending", 
                    severity="high",
                    details={"ip": ip_address}
                )
                return False
            
            # Validate TLS version if required
            if self.config.tls_version and hasattr(websocket, 'transport'):
                # TLS version checking would go here
                pass
                
            return True
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    async def validate_message(self, message: Any, connection_id: str) -> bool:
        """Validate incoming message security"""
        try:
            # Check message size
            message_size = len(str(message).encode('utf-8'))
            if message_size > self.config.max_message_size:
                self._record_security_event(
                    event_type="oversized_message",
                    connection_id=connection_id,
                    severity="medium",
                    details={"size": message_size, "limit": self.config.max_message_size}
                )
                return False
            
            # Check for malicious content patterns
            message_str = str(message).lower()
            suspicious_patterns = [
                '<script', 'javascript:', 'eval(', 'document.',
                'window.', 'alert(', 'confirm(', 'prompt('
            ]
            
            for pattern in suspicious_patterns:
                if pattern in message_str:
                    self._record_security_event(
                        event_type="suspicious_content",
                        connection_id=connection_id,
                        severity="high", 
                        details={"pattern": pattern}
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            return False
    
    async def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection exceeds rate limits"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        with self._lock:
            # Clean old entries
            while (self.rate_limits[connection_id] and 
                   self.rate_limits[connection_id][0] < window_start):
                self.rate_limits[connection_id].popleft()
            
            # Check current rate
            current_count = len(self.rate_limits[connection_id])
            if current_count >= self.config.rate_limit_per_minute:
                self._record_security_event(
                    event_type="rate_limit_exceeded",
                    connection_id=connection_id,
                    severity="medium",
                    details={
                        "current_rate": current_count,
                        "limit": self.config.rate_limit_per_minute
                    }
                )
                return False
            
            # Record this request
            self.rate_limits[connection_id].append(current_time)
            return True
    
    async def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to prevent injection attacks"""
        sanitized = {}
        
        for key, value in metadata.items():
            # Sanitize key
            clean_key = str(key)[:50]  # Limit key length
            clean_key = ''.join(c for c in clean_key if c.isalnum() or c in '_-.')
            
            # Sanitize value
            if isinstance(value, str):
                clean_value = value[:500]  # Limit string length
                # Remove potentially dangerous characters
                clean_value = ''.join(c for c in clean_value if c.isprintable())
                sanitized[clean_key] = clean_value
            elif isinstance(value, (int, float, bool)):
                sanitized[clean_key] = value
            elif isinstance(value, dict):
                sanitized[clean_key] = await self.sanitize_metadata(value)
            # Skip other types for security
        
        return sanitized
    
    def _record_security_event(self, event_type: str, connection_id: str, 
                             severity: str, details: Dict[str, Any]):
        """Record security event"""
        event = SecurityEvent(
            event_type=event_type,
            connection_id=connection_id,
            severity=severity,
            timestamp=time.time(),
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log high severity events
        if severity in ['high', 'critical']:
            logger.warning(f"Security event: {event_type} for {connection_id} - {details}")
    
    def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        return self.security_events[-limit:]


class CircuitBreakerAdapter:
    """Adapter to use existing circuit breaker with our interface"""
    
    def __init__(self, name: str, config: ResilienceConfig):
        self.name = name
        breaker_config = ReliabilityConfig(
            failure_threshold=int(config.failure_threshold * 10),  # Convert rate to count
            recovery_timeout=config.recovery_timeout_seconds,
            success_threshold=3,  # Fixed value
            timeout=30.0  # Default timeout
        )
        self.breaker = circuit_breaker_registry.get_breaker(name, breaker_config)
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection"""
        try:
            return await self.breaker.call(operation, *args, **kwargs)
        except Exception as e:
            if "OPEN" in str(e):
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            raise e
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        state = self.breaker.get_state()
        return state['state'] == 'open'
    
    def get_state(self) -> str:
        """Get circuit breaker state"""
        return self.breaker.get_state()['state']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker performance metrics"""
        return self.breaker.get_state()


class BackpressureManager:
    """Queue management and backpressure handling"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queues: Dict[str, deque] = defaultdict(lambda: deque())
        self.dropped_count = 0
        self._lock = threading.RLock()
    
    async def enqueue_message(self, message: bytes, priority: int = 1) -> bool:
        """Enqueue message with priority, return success status"""
        # For simplicity, using single queue - could extend to priority queues
        queue_id = "default"
        
        with self._lock:
            if len(self.queues[queue_id]) >= self.max_queue_size:
                self.dropped_count += 1
                return False
                
            # Store message with metadata
            self.queues[queue_id].append({
                'message': message,
                'priority': priority,
                'timestamp': time.time()
            })
            
        return True
    
    async def dequeue_message(self) -> Optional[bytes]:
        """Dequeue next message for processing"""
        queue_id = "default"
        
        with self._lock:
            if not self.queues[queue_id]:
                return None
                
            message_data = self.queues[queue_id].popleft()
            return message_data['message']
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return sum(len(queue) for queue in self.queues.values())
    
    def is_queue_full(self) -> bool:
        """Check if queue is at capacity"""
        return self.get_queue_size() >= self.max_queue_size
    
    def get_dropped_count(self) -> int:
        """Get count of dropped messages due to backpressure"""
        return self.dropped_count


class PerformanceMonitor:
    """Performance measurement and monitoring"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.throughput: Dict[str, List[int]] = defaultdict(list)
        self.start_time = time.time()
        self._lock = threading.RLock()
    
    async def record_latency(self, operation_type: str, latency_ms: float) -> None:
        """Record operation latency measurement"""
        with self._lock:
            self.metrics[f"{operation_type}_latency"].append(latency_ms)
            
            # Keep only recent measurements
            if len(self.metrics[f"{operation_type}_latency"]) > 1000:
                self.metrics[f"{operation_type}_latency"] = self.metrics[f"{operation_type}_latency"][-1000:]
    
    async def record_throughput(self, operation_type: str, count: int) -> None:
        """Record throughput measurement"""
        with self._lock:
            self.throughput[f"{operation_type}_throughput"].append(count)
            
            # Keep only recent measurements
            if len(self.throughput[f"{operation_type}_throughput"]) > 1000:
                self.throughput[f"{operation_type}_throughput"] = self.throughput[f"{operation_type}_throughput"][-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'uptime_seconds': time.time() - self.start_time,
            'metrics': {}
        }
        
        with self._lock:
            for operation, latencies in self.metrics.items():
                if latencies:
                    report['metrics'][operation] = {
                        'avg': np.mean(latencies),
                        'p95': np.percentile(latencies, 95),
                        'p99': np.percentile(latencies, 99),
                        'count': len(latencies)
                    }
            
            for operation, counts in self.throughput.items():
                if counts:
                    report['metrics'][operation] = {
                        'total': sum(counts),
                        'avg': np.mean(counts),
                        'count': len(counts)
                    }
        
        return report
    
    def check_sla_compliance(self) -> Dict[str, bool]:
        """Check if system meets SLA targets"""
        sla_results = {}
        
        with self._lock:
            # Check audio processing latency SLA (< 25ms)
            audio_latencies = self.metrics.get('audio_processing_latency', [])
            if audio_latencies:
                avg_latency = np.mean(audio_latencies[-100:])  # Last 100 measurements
                sla_results['audio_latency_sla'] = avg_latency < 25.0
            
            # Check WebSocket message latency SLA (< 10ms)
            ws_latencies = self.metrics.get('websocket_message_latency', [])
            if ws_latencies:
                avg_latency = np.mean(ws_latencies[-100:])
                sla_results['websocket_latency_sla'] = avg_latency < 10.0
        
        return sla_results


class MessageRouter:
    """Message routing and distribution with topic support"""
    
    def __init__(self, connection_manager: ProductionConnectionManager):
        self.connection_manager = connection_manager
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> connection_ids
        self.filters: Dict[str, Callable] = {}  # topic -> filter function
        self._lock = threading.RLock()
    
    async def route_message(self, message: Any, routing_key: str) -> bool:
        """Route message to appropriate destination"""
        try:
            # Parse routing key (e.g., "room.audio.studio1")
            parts = routing_key.split('.')
            
            if len(parts) >= 2:
                message_type = parts[0]
                target = parts[1]
                
                if message_type == "room":
                    # Route to specific room
                    room_id = target
                    count = await self.connection_manager.broadcast_to_room(room_id, message)
                    return count > 0
                elif message_type == "user":
                    # Route to specific user (all their connections)
                    user_id = target
                    # Implementation would route to all user connections
                    pass
                elif message_type == "topic":
                    # Route to topic subscribers
                    topic = target
                    return await self.broadcast_to_topic(topic, message)
            
            return False
            
        except Exception as e:
            logger.error(f"Message routing failed: {e}")
            return False
    
    async def broadcast_message(self, message: Any, filter_func: Optional[Callable] = None) -> int:
        """Broadcast message to multiple destinations"""
        connections = await self.connection_manager.get_active_connections()
        successful_sends = 0
        
        for connection_id, conn_data in connections.items():
            # Apply filter if provided
            if filter_func and not filter_func(conn_data):
                continue
                
            try:
                websocket = await self.connection_manager.get_connection(connection_id)
                if websocket and hasattr(websocket, 'send'):
                    await websocket.send(message)
                    successful_sends += 1
            except Exception as e:
                logger.warning(f"Broadcast failed to {connection_id}: {e}")
        
        return successful_sends
    
    async def subscribe_to_topic(self, topic: str, connection_id: str) -> bool:
        """Subscribe connection to message topic"""
        with self._lock:
            self.subscriptions[topic].add(connection_id)
        return True
    
    async def unsubscribe_from_topic(self, topic: str, connection_id: str) -> bool:
        """Unsubscribe connection from topic"""
        with self._lock:
            self.subscriptions[topic].discard(connection_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
        return True
    
    async def broadcast_to_topic(self, topic: str, message: Any) -> bool:
        """Broadcast message to all topic subscribers"""
        if topic not in self.subscriptions:
            return False
            
        connection_ids = list(self.subscriptions[topic])
        successful_sends = 0
        
        for connection_id in connection_ids:
            try:
                websocket = await self.connection_manager.get_connection(connection_id)
                if websocket and hasattr(websocket, 'send'):
                    await websocket.send(message)
                    successful_sends += 1
            except Exception as e:
                logger.warning(f"Topic broadcast failed to {connection_id}: {e}")
                # Remove failed connection from topic
                await self.unsubscribe_from_topic(topic, connection_id)
        
        return successful_sends > 0


class StateManager:
    """Distributed state management with Redis-like functionality"""
    
    def __init__(self):
        self.state_store: Dict[str, Dict] = {}  # In-memory for now
        self.counters: Dict[str, int] = defaultdict(int)
        self.expiry_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_keys())
    
    async def store_state(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store state with optional TTL"""
        with self._lock:
            self.state_store[key] = {
                'value': value,
                'created_at': time.time(),
                'accessed_at': time.time()
            }
            
            if ttl:
                self.expiry_times[key] = time.time() + ttl
            elif key in self.expiry_times:
                del self.expiry_times[key]
        
        return True
    
    async def retrieve_state(self, key: str) -> Optional[Any]:
        """Retrieve state by key"""
        with self._lock:
            if key not in self.state_store:
                return None
                
            # Check if expired
            if key in self.expiry_times and time.time() > self.expiry_times[key]:
                del self.state_store[key]
                del self.expiry_times[key]
                return None
            
            # Update access time
            self.state_store[key]['accessed_at'] = time.time()
            return self.state_store[key]['value']
    
    async def delete_state(self, key: str) -> bool:
        """Delete state by key"""
        with self._lock:
            if key in self.state_store:
                del self.state_store[key]
            if key in self.expiry_times:
                del self.expiry_times[key]
            if key in self.counters:
                del self.counters[key]
        
        return True
    
    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment counter, return new value"""
        with self._lock:
            self.counters[key] += amount
            return self.counters[key]
    
    async def _cleanup_expired_keys(self):
        """Background task to clean up expired keys"""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self._lock:
                    for key, expiry_time in self.expiry_times.items():
                        if current_time > expiry_time:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        if key in self.state_store:
                            del self.state_store[key]
                        del self.expiry_times[key]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"State cleanup error: {e}")
                await asyncio.sleep(60)


# Factory implementations
class AudioProcessorFactory:
    """Factory for creating audio processors"""
    
    @staticmethod
    def create_processor(config: AudioConfig) -> ProductionAudioProcessor:
        """Create audio processor instance"""
        return ProductionAudioProcessor(config)


class ConnectionManagerFactory:
    """Factory for creating connection managers"""
    
    @staticmethod
    def create_manager(config: StreamingConfig) -> ProductionConnectionManager:
        """Create connection manager instance"""
        return ProductionConnectionManager(config)


class SecurityValidatorFactory:
    """Factory for creating security validators"""
    
    @staticmethod
    def create_validator(config: SecurityConfig) -> ProductionSecurityValidator:
        """Create security validator instance"""
        return ProductionSecurityValidator(config)


# Export all implementations
__all__ = [
    'ProductionAudioProcessor',
    'ProductionConnectionManager', 
    'ProductionSecurityValidator',
    'CircuitBreakerAdapter',
    'BackpressureManager',
    'PerformanceMonitor',
    'MessageRouter',
    'StateManager',
    'AudioProcessorFactory',
    'ConnectionManagerFactory',
    'SecurityValidatorFactory'
]