#!/usr/bin/env python3
"""
WebSocket Streaming Interfaces - SOLID Compliance

Based on latest research and industry best practices from:
- Microsoft's 2024 WebSocket architectural guidelines  
- Netflix's Pushy platform patterns
- Google's Autosocket distribution
- Meta's real-time architecture research
- OWASP WebSocket Security Guidelines 2024

All interfaces follow SOLID principles with single responsibility,
dependency inversion, and interface segregation.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

# SOLID Principle: Interface Segregation
# Each interface has a single, focused responsibility

class IAudioProcessor(Protocol):
    """Single Responsibility: Audio processing operations only"""
    
    @abstractmethod
    async def process_audio_frame(self, audio_data: bytes, session_id: str) -> bytes:
        """Process single audio frame with session context"""
        pass
    
    @abstractmethod
    async def get_processing_metrics(self, session_id: str) -> Dict[str, float]:
        """Get processing performance metrics"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        pass

class IConnectionManager(Protocol):
    """Single Responsibility: WebSocket connection lifecycle management"""
    
    @abstractmethod
    async def register_connection(self, websocket: Any, user_id: str) -> str:
        """Register new WebSocket connection, return connection ID"""
        pass
    
    @abstractmethod
    async def get_connection(self, connection_id: str) -> Optional[Any]:
        """Retrieve WebSocket connection by ID"""
        pass
    
    @abstractmethod
    async def cleanup_connection(self, connection_id: str) -> bool:
        """Clean up and remove connection"""
        pass
    
    @abstractmethod
    async def get_active_connections(self) -> Dict[str, Any]:
        """Get all active connections"""
        pass
    
    @abstractmethod
    async def broadcast_to_room(self, room_id: str, message: bytes) -> int:
        """Broadcast message to all connections in room, return count"""
        pass

class ISecurityValidator(Protocol):
    """Single Responsibility: Security validation and protection"""
    
    @abstractmethod
    async def validate_connection(self, websocket: Any, request: Any) -> bool:
        """Validate connection security requirements"""
        pass
    
    @abstractmethod
    async def validate_message(self, message: Any, connection_id: str) -> bool:
        """Validate incoming message security"""
        pass
    
    @abstractmethod
    async def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection exceeds rate limits"""
        pass
    
    @abstractmethod
    async def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to prevent injection attacks"""
        pass

class ICircuitBreaker(Protocol):
    """Single Responsibility: Failure protection and resilience"""
    
    @abstractmethod
    async def execute(self, operation: Any, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection"""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        pass
    
    @abstractmethod
    def get_state(self) -> str:
        """Get circuit breaker state: CLOSED, OPEN, HALF_OPEN"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker performance metrics"""
        pass

class IBackpressureManager(Protocol):
    """Single Responsibility: Queue management and backpressure handling"""
    
    @abstractmethod
    async def enqueue_message(self, message: bytes, priority: int = 1) -> bool:
        """Enqueue message with priority, return success status"""
        pass
    
    @abstractmethod
    async def dequeue_message(self) -> Optional[bytes]:
        """Dequeue next message for processing"""
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size"""
        pass
    
    @abstractmethod
    def is_queue_full(self) -> bool:
        """Check if queue is at capacity"""
        pass
    
    @abstractmethod
    def get_dropped_count(self) -> int:
        """Get count of dropped messages due to backpressure"""
        pass

class IPerformanceMonitor(Protocol):
    """Single Responsibility: Performance measurement and monitoring"""
    
    @abstractmethod
    async def record_latency(self, operation_type: str, latency_ms: float) -> None:
        """Record operation latency measurement"""
        pass
    
    @abstractmethod
    async def record_throughput(self, operation_type: str, count: int) -> None:
        """Record throughput measurement"""
        pass
    
    @abstractmethod
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        pass
    
    @abstractmethod
    def check_sla_compliance(self) -> Dict[str, bool]:
        """Check if system meets SLA targets"""
        pass

class IMessageRouter(Protocol):
    """Single Responsibility: Message routing and distribution"""
    
    @abstractmethod
    async def route_message(self, message: Any, routing_key: str) -> bool:
        """Route message to appropriate destination"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Any, filter_func: Any = None) -> int:
        """Broadcast message to multiple destinations"""
        pass
    
    @abstractmethod
    async def subscribe_to_topic(self, topic: str, connection_id: str) -> bool:
        """Subscribe connection to message topic"""
        pass
    
    @abstractmethod
    async def unsubscribe_from_topic(self, topic: str, connection_id: str) -> bool:
        """Unsubscribe connection from topic"""
        pass

class IStateManager(Protocol):
    """Single Responsibility: Distributed state management"""
    
    @abstractmethod
    async def store_state(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store state with optional TTL"""
        pass
    
    @abstractmethod
    async def retrieve_state(self, key: str) -> Optional[Any]:
        """Retrieve state by key"""
        pass
    
    @abstractmethod
    async def delete_state(self, key: str) -> bool:
        """Delete state by key"""
        pass
    
    @abstractmethod
    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment counter, return new value"""
        pass

# Data Transfer Objects (DTOs) for type safety

@dataclass
class AudioMessage:
    """Audio message data structure"""
    connection_id: str
    user_id: str
    audio_data: bytes
    timestamp: float
    metadata: Dict[str, Any]
    message_id: str
    room_id: Optional[str] = None

@dataclass
class ConnectionInfo:
    """WebSocket connection information"""
    connection_id: str
    user_id: str
    ip_address: str
    user_agent: str
    connected_at: float
    last_activity: float
    room_id: Optional[str] = None
    subscription_tier: str = "free"

@dataclass
class PerformanceMetrics:
    """Performance measurement data structure"""
    latency_ms: float
    throughput_ops_per_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_connections: int
    dropped_messages: int
    error_rate_percent: float

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: str  # rate_limit_exceeded, invalid_origin, etc.
    connection_id: str
    severity: str    # low, medium, high, critical
    timestamp: float
    details: Dict[str, Any]

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"  
    HALF_OPEN = "half_open"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class SecurityLevel(Enum):
    """Security validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

# Configuration data classes

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 16
    frame_size: int = 960  # samples
    max_processing_time_ms: int = 20
    quality_level: str = "balanced"  # battery_saver, balanced, high_quality

@dataclass
class SecurityConfig:
    """Security configuration"""
    allowed_origins: List[str]
    rate_limit_per_minute: int = 60
    max_message_size: int = 1048576  # 1MB
    require_authentication: bool = True
    security_level: SecurityLevel = SecurityLevel.STANDARD
    tls_version: str = "1.3"

@dataclass
class PerformanceConfig:
    """Performance target configuration"""
    max_latency_ms: int = 25
    max_cpu_usage_percent: float = 80.0
    max_memory_per_connection_kb: int = 50
    min_throughput_ops_per_sec: int = 1000
    uptime_target_percent: float = 99.95

@dataclass
class ResilienceConfig:
    """Circuit breaker and resilience configuration"""
    failure_threshold: float = 0.5  # 50% failure rate triggers open
    recovery_timeout_seconds: int = 30
    half_open_max_requests: int = 10
    retry_attempts: int = 3
    backoff_multiplier: float = 2.0

@dataclass
class StreamingConfig:
    """Complete streaming system configuration"""
    audio_config: AudioConfig
    security_config: SecurityConfig
    performance_config: PerformanceConfig
    resilience_config: ResilienceConfig
    max_connections: int = 10000
    max_rooms: int = 1000
    enable_clustering: bool = True
    redis_url: str = "redis://localhost:6379"

# Factory interfaces for dependency injection

class IAudioProcessorFactory(Protocol):
    """Factory for creating audio processors"""
    
    @abstractmethod
    def create_processor(self, config: AudioConfig) -> IAudioProcessor:
        """Create audio processor instance"""
        pass

class IConnectionManagerFactory(Protocol):
    """Factory for creating connection managers"""
    
    @abstractmethod
    def create_manager(self, config: StreamingConfig) -> IConnectionManager:
        """Create connection manager instance"""
        pass

class ISecurityValidatorFactory(Protocol):
    """Factory for creating security validators"""
    
    @abstractmethod
    def create_validator(self, config: SecurityConfig) -> ISecurityValidator:
        """Create security validator instance"""
        pass

# Exception types for proper error handling

class StreamingException(Exception):
    """Base exception for streaming operations"""
    pass

class SecurityValidationError(StreamingException):
    """Security validation failed"""
    pass

class CircuitBreakerOpenError(StreamingException):
    """Circuit breaker is open"""
    pass

class BackpressureError(StreamingException):
    """Backpressure limit exceeded"""
    pass

class PerformanceThresholdExceeded(StreamingException):
    """Performance threshold exceeded"""
    pass

# This interface design follows SOLID principles:
# - Single Responsibility: Each interface has one clear purpose
# - Open/Closed: Easily extensible through composition
# - Liskov Substitution: All implementations are substitutable
# - Interface Segregation: No fat interfaces, highly focused
# - Dependency Inversion: Depend on abstractions, not concretions