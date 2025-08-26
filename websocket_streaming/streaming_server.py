#!/usr/bin/env python3
"""
Production WebSocket Streaming Server
Orchestrates all components for real-time audio streaming with MANU compliance
"""

import asyncio
import websockets
import json
import logging
import time
import signal
import sys
from typing import Dict, Any, Optional, Set, Callable
from dataclasses import asdict
from datetime import datetime
import threading

# SOLID-compliant imports
from .interfaces import (
    AudioMessage, StreamingConfig, AudioConfig, SecurityConfig,
    PerformanceConfig, ResilienceConfig, MessagePriority, SecurityLevel,
    StreamingException, SecurityValidationError
)
from .websocket_implementations import (
    ProductionAudioProcessor, ProductionConnectionManager,
    ProductionSecurityValidator, CircuitBreakerAdapter,
    BackpressureManager, PerformanceMonitor, MessageRouter,
    StateManager, AudioProcessorFactory, ConnectionManagerFactory,
    SecurityValidatorFactory
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingOrchestrator:
    """
    Main orchestrator for WebSocket streaming system
    Coordinates all SOLID-compliant components
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.running = False
        self.server = None
        
        # Initialize all components using factories (Dependency Injection)
        self.audio_processor = AudioProcessorFactory.create_processor(config.audio_config)
        self.connection_manager = ConnectionManagerFactory.create_manager(config)
        self.security_validator = SecurityValidatorFactory.create_validator(config.security_config)
        
        # Initialize supporting components
        self.circuit_breaker = CircuitBreakerAdapter("streaming_server", config.resilience_config)
        self.backpressure_manager = BackpressureManager(max_queue_size=1000)
        self.performance_monitor = PerformanceMonitor()
        self.message_router = MessageRouter(self.connection_manager)
        self.state_manager = StateManager()
        
        # Connection tracking
        self.active_streams: Dict[str, Dict] = {}
        self.message_handlers = self._setup_message_handlers()
        
        # Performance tracking
        self.stats = {
            'connections_total': 0,
            'messages_processed': 0,
            'audio_frames_processed': 0,
            'errors_total': 0,
            'start_time': time.time()
        }
        
        logger.info("StreamingOrchestrator initialized with SOLID architecture")
    
    def _setup_message_handlers(self) -> Dict[str, Callable]:
        """Setup message type handlers"""
        return {
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'join_room': self._handle_join_room,
            'leave_room': self._handle_leave_room,
            'ping': self._handle_ping,
            'get_stats': self._handle_get_stats
        }
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket streaming server"""
        try:
            logger.info(f"Starting WebSocket streaming server on {host}:{port}")
            
            # Start supporting tasks
            asyncio.create_task(self._health_check_task())
            asyncio.create_task(self._performance_monitoring_task())
            asyncio.create_task(self._cleanup_task())
            
            # Setup signal handlers for graceful shutdown
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                host,
                port,
                max_size=self.config.security_config.max_message_size,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.running = True
            self.stats['start_time'] = time.time()
            
            logger.info(f"✅ WebSocket streaming server started successfully")
            logger.info(f"📊 Configuration: {self.config.max_connections} max connections")
            logger.info(f"🔊 Audio: {self.config.audio_config.sample_rate}Hz, {self.config.audio_config.channels} channels")
            logger.info(f"🔒 Security: {self.config.security_config.security_level.value} level")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Failed to start streaming server: {e}")
            raise
    
    async def stop_server(self):
        """Gracefully stop the streaming server"""
        logger.info("Stopping WebSocket streaming server...")
        
        self.running = False
        
        # Close all connections
        connections = await self.connection_manager.get_active_connections()
        for connection_id in connections:
            await self.connection_manager.cleanup_connection(connection_id)
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("✅ WebSocket streaming server stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.stop_server())
    
    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        connection_id = None
        start_time = time.time()
        
        try:
            # Security validation
            if not await self.security_validator.validate_connection(websocket, None):
                logger.warning(f"Connection rejected: security validation failed")
                await websocket.close(code=1008, reason="Security validation failed")
                return
            
            # Extract user ID (in production, get from authentication token)
            user_id = self._extract_user_id(websocket)
            
            # Check connection limits
            active_connections = await self.connection_manager.get_active_connections()
            if len(active_connections) >= self.config.max_connections:
                logger.warning(f"Connection rejected: max connections reached")
                await websocket.close(code=1013, reason="Server at capacity")
                return
            
            # Register connection
            connection_id = await self.connection_manager.register_connection(websocket, user_id)
            self.stats['connections_total'] += 1
            
            # Record connection establishment latency
            connection_latency = (time.time() - start_time) * 1000
            await self.performance_monitor.record_latency('connection_establishment', connection_latency)
            
            logger.info(f"✅ New connection established: {connection_id} for user {user_id}")
            
            # Send welcome message
            welcome_message = {
                'type': 'welcome',
                'connection_id': connection_id,
                'server_info': {
                    'version': '1.0.0',
                    'audio_config': asdict(self.config.audio_config),
                    'supported_features': [
                        'audio_processing',
                        'genre_detection',
                        'real_time_dsp',
                        'room_broadcasting',
                        'topic_subscriptions'
                    ]
                }
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Handle messages for this connection
            await self._handle_connection_messages(websocket, connection_id)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection {connection_id} closed normally")
        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e}")
            self.stats['errors_total'] += 1
        finally:
            # Clean up connection
            if connection_id:
                await self.connection_manager.cleanup_connection(connection_id)
                if connection_id in self.active_streams:
                    del self.active_streams[connection_id]
    
    def _extract_user_id(self, websocket) -> str:
        """Extract user ID from WebSocket connection"""
        # In production, extract from JWT token in query params or headers
        # For now, use a simple fallback
        remote_addr = getattr(websocket, 'remote_address', ('unknown', 0))[0]
        return f"user_{hash(remote_addr) % 10000}"
    
    async def _handle_connection_messages(self, websocket, connection_id: str):
        """Handle messages for a specific connection"""
        async for message in websocket:
            message_start_time = time.time()
            
            try:
                # Check rate limiting
                if not await self.security_validator.check_rate_limit(connection_id):
                    logger.warning(f"Rate limit exceeded for {connection_id}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'code': 'RATE_LIMIT_EXCEEDED',
                        'message': 'Too many requests'
                    }))
                    continue
                
                # Validate message
                if not await self.security_validator.validate_message(message, connection_id):
                    logger.warning(f"Invalid message from {connection_id}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'code': 'INVALID_MESSAGE',
                        'message': 'Message validation failed'
                    }))
                    continue
                
                # Parse message
                try:
                    if isinstance(message, bytes):
                        # Binary message (audio data)
                        await self._handle_binary_message(websocket, connection_id, message)
                    else:
                        # Text message (JSON)
                        msg_data = json.loads(message)
                        await self._handle_json_message(websocket, connection_id, msg_data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {connection_id}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'code': 'INVALID_JSON',
                        'message': 'Invalid JSON format'
                    }))
                    continue
                
                # Record message processing latency
                message_latency = (time.time() - message_start_time) * 1000
                await self.performance_monitor.record_latency('message_processing', message_latency)
                
                self.stats['messages_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing message from {connection_id}: {e}")
                self.stats['errors_total'] += 1
                try:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'code': 'PROCESSING_ERROR',
                        'message': 'Internal processing error'
                    }))
                except:
                    pass  # Connection might be closed
    
    async def _handle_binary_message(self, websocket, connection_id: str, data: bytes):
        """Handle binary audio data"""
        try:
            # Use circuit breaker for audio processing
            processed_audio = await self.circuit_breaker.execute(
                self.audio_processor.process_audio_frame,
                data,
                connection_id
            )
            
            # Get processing metrics
            metrics = await self.audio_processor.get_processing_metrics(connection_id)
            
            # Send processed audio back
            response = {
                'type': 'audio_processed',
                'connection_id': connection_id,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            # Send JSON response first, then binary data
            await websocket.send(json.dumps(response))
            await websocket.send(processed_audio)
            
            self.stats['audio_frames_processed'] += 1
            
            # Record audio processing metrics
            if 'avg_processing_time_ms' in metrics:
                await self.performance_monitor.record_latency(
                    'audio_processing',
                    metrics['avg_processing_time_ms']
                )
                
        except Exception as e:
            logger.error(f"Audio processing failed for {connection_id}: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'code': 'AUDIO_PROCESSING_FAILED',
                'message': f'Audio processing error: {str(e)}'
            }))
    
    async def _handle_json_message(self, websocket, connection_id: str, msg_data: Dict[str, Any]):
        """Handle JSON message"""
        message_type = msg_data.get('type')
        
        if message_type in self.message_handlers:
            try:
                response = await self.message_handlers[message_type](
                    connection_id, msg_data, websocket
                )
                if response:
                    await websocket.send(json.dumps(response))
            except Exception as e:
                logger.error(f"Handler error for {message_type}: {e}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'code': 'HANDLER_ERROR',
                    'message': f'Handler error: {str(e)}'
                }))
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'code': 'UNKNOWN_MESSAGE_TYPE',
                'message': f'Unknown message type: {message_type}'
            }))
    
    async def _handle_subscribe(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle topic subscription"""
        topic = msg_data.get('topic')
        if not topic:
            return {'type': 'error', 'message': 'Topic required'}
        
        success = await self.message_router.subscribe_to_topic(topic, connection_id)
        return {
            'type': 'subscription_result',
            'topic': topic,
            'subscribed': success
        }
    
    async def _handle_unsubscribe(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle topic unsubscription"""
        topic = msg_data.get('topic')
        if not topic:
            return {'type': 'error', 'message': 'Topic required'}
        
        success = await self.message_router.unsubscribe_from_topic(topic, connection_id)
        return {
            'type': 'unsubscription_result',
            'topic': topic,
            'unsubscribed': success
        }
    
    async def _handle_join_room(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle room join"""
        room_id = msg_data.get('room_id')
        if not room_id:
            return {'type': 'error', 'message': 'Room ID required'}
        
        success = await self.connection_manager.add_to_room(connection_id, room_id)
        return {
            'type': 'room_join_result',
            'room_id': room_id,
            'joined': success
        }
    
    async def _handle_leave_room(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle room leave"""
        room_id = msg_data.get('room_id')
        if not room_id:
            return {'type': 'error', 'message': 'Room ID required'}
        
        success = await self.connection_manager.remove_from_room(connection_id, room_id)
        return {
            'type': 'room_leave_result',
            'room_id': room_id,
            'left': success
        }
    
    async def _handle_ping(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle ping message"""
        return {
            'type': 'pong',
            'timestamp': time.time(),
            'server_time': datetime.utcnow().isoformat()
        }
    
    async def _handle_get_stats(self, connection_id: str, msg_data: Dict, websocket) -> Dict:
        """Handle statistics request"""
        performance_report = self.performance_monitor.get_performance_report()
        sla_compliance = self.performance_monitor.check_sla_compliance()
        circuit_breaker_state = self.circuit_breaker.get_metrics()
        
        return {
            'type': 'stats',
            'server_stats': self.stats,
            'performance': performance_report,
            'sla_compliance': sla_compliance,
            'circuit_breaker': circuit_breaker_state,
            'active_connections': len(await self.connection_manager.get_active_connections())
        }
    
    async def _health_check_task(self):
        """Background health check task"""
        while self.running:
            try:
                # Check system health
                connections = await self.connection_manager.get_active_connections()
                
                # Log health status
                logger.info(f"Health check: {len(connections)} active connections, "
                          f"CB state: {self.circuit_breaker.get_state()}")
                
                # Check SLA compliance
                sla_results = self.performance_monitor.check_sla_compliance()
                if sla_results and not all(sla_results.values()):
                    logger.warning(f"SLA violation detected: {sla_results}")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_task(self):
        """Background performance monitoring task"""
        while self.running:
            try:
                # Record throughput metrics
                await self.performance_monitor.record_throughput(
                    'messages_per_second',
                    self.stats['messages_processed']
                )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.running:
            try:
                # Clean up inactive connections (implement connection timeout logic)
                current_time = time.time()
                connections = await self.connection_manager.get_active_connections()
                
                inactive_connections = []
                for conn_id, conn_data in connections.items():
                    if current_time - conn_data['last_activity'] > 300:  # 5 minutes inactive
                        inactive_connections.append(conn_id)
                
                for conn_id in inactive_connections:
                    logger.info(f"Cleaning up inactive connection: {conn_id}")
                    await self.connection_manager.cleanup_connection(conn_id)
                
                await asyncio.sleep(120)  # Cleanup every 2 minutes
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(120)


def create_production_config() -> StreamingConfig:
    """Create production streaming configuration"""
    return StreamingConfig(
        audio_config=AudioConfig(
            sample_rate=48000,
            channels=2,
            bit_depth=16,
            frame_size=960,  # 20ms at 48kHz
            max_processing_time_ms=20,
            quality_level="balanced"
        ),
        security_config=SecurityConfig(
            allowed_origins=["*"],  # Configure for production
            rate_limit_per_minute=60,
            max_message_size=1048576,  # 1MB
            require_authentication=False,  # For demo purposes
            security_level=SecurityLevel.STANDARD,
            tls_version="1.3"
        ),
        performance_config=PerformanceConfig(
            max_latency_ms=25,
            max_cpu_usage_percent=80.0,
            max_memory_per_connection_kb=50,
            min_throughput_ops_per_sec=1000,
            uptime_target_percent=99.95
        ),
        resilience_config=ResilienceConfig(
            failure_threshold=0.5,
            recovery_timeout_seconds=30,
            half_open_max_requests=10,
            retry_attempts=3,
            backoff_multiplier=2.0
        ),
        max_connections=1000,
        max_rooms=100,
        enable_clustering=False,  # Single instance for now
        redis_url="redis://localhost:6379"
    )


async def main():
    """Main entry point for the streaming server"""
    config = create_production_config()
    orchestrator = StreamingOrchestrator(config)
    
    try:
        await orchestrator.start_server(host="0.0.0.0", port=8765)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await orchestrator.stop_server()


if __name__ == "__main__":
    asyncio.run(main())