"""
Cloud-native audio processing microservice
Following Google SRE practices and cloud-native principles
"""

import grpc
from concurrent import futures
import time
import logging
import os
from typing import Iterator, Optional
import numpy as np
from dataclasses import dataclass
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter, metrics_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Import our audio processing modules
from ai_mixing_brain import AutonomousMixingEngine
from studio_dsp_chain import StudioDSPChain
from complete_ai_mixer import CompleteMixingSystem

# Setup structured logging (like Google uses)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Setup OpenTelemetry (Google Cloud Trace compatible)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Prometheus metrics (Google uses these extensively)
audio_processed_counter = Counter(
    'audio_chunks_processed_total',
    'Total number of audio chunks processed',
    ['service', 'genre', 'status']
)

processing_latency_histogram = Histogram(
    'audio_processing_duration_seconds',
    'Audio processing duration in seconds',
    ['operation', 'genre'],
    buckets=(.001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0)
)

active_streams_gauge = Gauge(
    'active_audio_streams',
    'Number of active audio streams'
)

model_version_counter = Counter(
    'model_version_requests_total',
    'Model version requests for A/B testing',
    ['version', 'genre']
)

# Circuit breaker pattern (Netflix Hystrix style)
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("circuit_breaker_half_open", function=func.__name__)
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("circuit_breaker_closed", function=func.__name__)
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error("circuit_breaker_opened", 
                           function=func.__name__, 
                           failures=self.failure_count)
            raise e

# Feature flags (like Google uses for gradual rollouts)
class FeatureFlags:
    """Simple feature flag system for A/B testing and gradual rollouts"""
    
    def __init__(self):
        self.flags = {
            "use_ai_mixing_v2": float(os.getenv("FF_AI_MIXING_V2", "0.0")),
            "enable_advanced_effects": float(os.getenv("FF_ADVANCED_EFFECTS", "1.0")),
            "use_ml_genre_detection": float(os.getenv("FF_ML_GENRE", "1.0")),
            "enable_stereo_processing": float(os.getenv("FF_STEREO", "0.5")),
        }
    
    def is_enabled(self, flag: str, user_id: str = None) -> bool:
        """Check if feature is enabled, with optional user bucketing"""
        if flag not in self.flags:
            return False
        
        rollout_percentage = self.flags[flag]
        
        if user_id:
            # Consistent bucketing based on user ID
            bucket = hash(f"{flag}:{user_id}") % 100
            return bucket < (rollout_percentage * 100)
        else:
            # Random rollout
            import random
            return random.random() < rollout_percentage

# Model versioning for A/B testing
class ModelRegistry:
    """Model registry for versioning and A/B testing"""
    
    def __init__(self):
        self.models = {
            "mixing_v1": CompleteMixingSystem(44100),
            "mixing_v2": CompleteMixingSystem(44100),  # Would be different model
        }
        self.default_model = "mixing_v1"
        
    def get_model(self, version: str = None):
        """Get specific model version or default"""
        if version and version in self.models:
            model_version_counter.labels(version=version, genre="unknown").inc()
            return self.models[version]
        return self.models[self.default_model]

# Health check service (Kubernetes/GKE style)
class HealthCheck:
    """Health check implementation for Kubernetes probes"""
    
    def __init__(self, audio_processor):
        self.audio_processor = audio_processor
        self.start_time = time.time()
        
    def liveness(self) -> tuple[bool, str]:
        """Liveness probe - is the service alive?"""
        try:
            # Basic check - can we allocate memory?
            _ = np.zeros(1024)
            return True, "OK"
        except Exception as e:
            return False, str(e)
    
    def readiness(self) -> tuple[bool, str]:
        """Readiness probe - can we handle requests?"""
        try:
            # Check if audio processor is initialized
            if not self.audio_processor:
                return False, "Audio processor not initialized"
            
            # Check if we can process a small buffer
            test_buffer = np.zeros(512, dtype=np.float32)
            _ = self.audio_processor.process_realtime(test_buffer)
            
            return True, "Ready"
        except Exception as e:
            return False, str(e)
    
    def startup(self) -> tuple[bool, str]:
        """Startup probe - for slow starting containers"""
        uptime = time.time() - self.start_time
        if uptime < 10:  # Give 10 seconds to start
            return False, f"Starting up... {uptime:.1f}s"
        return self.readiness()

# Rate limiting (Google Cloud Armor style)
class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int = 100, burst: int = 200):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self.tokens = burst
        self.last_update = time.time()
        
    def allow(self, tokens: int = 1) -> bool:
        """Check if request is allowed"""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Add tokens based on elapsed time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Audio processing service implementation
class AudioProcessingServicer:
    """gRPC service implementation for audio processing"""
    
    def __init__(self):
        self.mixing_system = CompleteMixingSystem(44100)
        self.circuit_breaker = CircuitBreaker()
        self.feature_flags = FeatureFlags()
        self.model_registry = ModelRegistry()
        self.rate_limiter = RateLimiter()
        self.health_check = HealthCheck(self.mixing_system)
        
        # Stream management
        self.active_streams = {}
        
        logger.info("audio_service_initialized", 
                   features=self.feature_flags.flags)
    
    def StreamAudio(self, request_iterator: Iterator, context) -> Iterator:
        """Bidirectional streaming for real-time audio processing"""
        
        stream_id = None
        with tracer.start_as_current_span("stream_audio") as span:
            try:
                for audio_chunk in request_iterator:
                    # Rate limiting
                    if not self.rate_limiter.allow():
                        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                    "Rate limit exceeded")
                    
                    # Extract stream ID
                    if not stream_id:
                        stream_id = audio_chunk.stream_id
                        self.active_streams[stream_id] = time.time()
                        active_streams_gauge.inc()
                        span.set_attribute("stream.id", stream_id)
                    
                    # Process audio
                    with processing_latency_histogram.labels(
                        operation="stream", genre="unknown").time():
                        
                        # Convert PCM to numpy
                        audio_array = np.frombuffer(
                            audio_chunk.pcm_data, 
                            dtype=np.int16
                        ).astype(np.float32) / 32768.0
                        
                        # Process based on feature flags
                        if self.feature_flags.is_enabled("use_ai_mixing_v2", stream_id):
                            model = self.model_registry.get_model("mixing_v2")
                        else:
                            model = self.model_registry.get_model("mixing_v1")
                        
                        # Process through circuit breaker
                        result = self.circuit_breaker.call(
                            model.process_complete,
                            audio_array,
                            auto_mode=True
                        )
                        
                        # Track metrics
                        audio_processed_counter.labels(
                            service="audio_processing",
                            genre=result.get("genre", "unknown"),
                            status="success"
                        ).inc()
                        
                        # Create response
                        processed_audio = self._create_processed_audio(result)
                        yield processed_audio
                        
            except Exception as e:
                logger.error("stream_processing_error", 
                           stream_id=stream_id, 
                           error=str(e))
                audio_processed_counter.labels(
                    service="audio_processing",
                    genre="unknown",
                    status="error"
                ).inc()
                context.abort(grpc.StatusCode.INTERNAL, str(e))
            finally:
                if stream_id and stream_id in self.active_streams:
                    del self.active_streams[stream_id]
                    active_streams_gauge.dec()
    
    def ProcessAudio(self, request, context):
        """Process a single audio buffer"""
        
        with tracer.start_as_current_span("process_audio") as span:
            span.set_attribute("audio.sample_rate", request.audio.sample_rate)
            span.set_attribute("audio.channels", request.audio.channels)
            
            try:
                # Rate limiting
                if not self.rate_limiter.allow(tokens=5):  # Higher cost for batch
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                "Rate limit exceeded")
                
                # Process audio
                audio_array = np.frombuffer(
                    request.audio.pcm_data,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                
                # Get model based on request
                model_version = request.options.model_version if request.options else None
                model = self.model_registry.get_model(model_version)
                
                # Process
                with processing_latency_histogram.labels(
                    operation="batch", genre="unknown").time():
                    
                    result = model.process_complete(
                        audio_array,
                        auto_mode=request.options.enable_ai_mixing if request.options else True
                    )
                
                # Create response
                return self._create_process_response(result, request)
                
            except Exception as e:
                logger.error("batch_processing_error", error=str(e))
                context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def GetMetrics(self, request, context):
        """Get current processing metrics"""
        
        metrics = {
            "active_streams": len(self.active_streams),
            "uptime_seconds": time.time() - self.health_check.start_time,
            "circuit_breaker_state": self.circuit_breaker.state,
            "rate_limit_tokens": self.rate_limiter.tokens,
        }
        
        # Add feature flag states
        for flag, value in self.feature_flags.flags.items():
            metrics[f"feature_{flag}"] = value
        
        # Health status
        is_healthy, message = self.health_check.readiness()
        health_status = "HEALTHY" if is_healthy else "UNHEALTHY"
        
        return {
            "metrics": metrics,
            "health": {
                "status": health_status,
                "message": message
            }
        }
    
    def _create_processed_audio(self, result: dict):
        """Create ProcessedAudio message from processing result"""
        # This would create the protobuf message
        # Simplified for illustration
        return {
            "pcm_data": result["audio"]["left"].tobytes(),
            "profile_name": result.get("genre", "unknown"),
            "metrics": {
                "rms_db": result["metrics"].get("rms_db", -60),
                "peak_db": result["metrics"].get("peak_db", -60),
                "lufs": result["metrics"].get("lufs", -23),
                "genre": result.get("genre", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "dsp_metrics": result.get("dsp", {})
            }
        }
    
    def _create_process_response(self, result: dict, request):
        """Create ProcessAudioResponse from result"""
        import uuid
        return {
            "audio": self._create_processed_audio(result),
            "request_id": str(uuid.uuid4())
        }

# Graceful shutdown handler
class GracefulShutdown:
    """Handle graceful shutdown for Kubernetes SIGTERM"""
    
    def __init__(self, server):
        self.server = server
        self.shutdown_event = futures.ThreadPoolExecutor(max_workers=1)
        
    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM for graceful shutdown"""
        logger.info("received_sigterm", signal=signum)
        
        # Stop accepting new requests
        self.server.stop(grace=30)  # 30 second grace period
        
        # Wait for existing requests to complete
        logger.info("waiting_for_requests_to_complete")
        time.sleep(5)
        
        logger.info("shutdown_complete")
        
def serve():
    """Start the gRPC server with all cloud-native features"""
    
    # Start Prometheus metrics server
    start_http_server(9090)
    logger.info("prometheus_metrics_started", port=9090)
    
    # Create gRPC server with interceptors
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
        ]
    )
    
    # Add OpenTelemetry instrumentation
    GrpcInstrumentorServer().instrument(server)
    
    # Add service
    servicer = AudioProcessingServicer()
    # Would add generated servicer here
    # audio_service_pb2_grpc.add_AudioProcessingServicer_to_server(servicer, server)
    
    # Add health check service
    # health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Start server
    port = os.getenv("GRPC_PORT", "50051")
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info("grpc_server_started", 
               port=port,
               workers=10,
               features=servicer.feature_flags.flags)
    
    # Setup graceful shutdown
    shutdown_handler = GracefulShutdown(server)
    import signal
    signal.signal(signal.SIGTERM, shutdown_handler.handle_sigterm)
    
    # Keep alive
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()