#!/usr/bin/env python3
"""
gRPC Service Mesh Implementation - Google/Netflix Best Practices 2025
High-performance RPC with service discovery and load balancing
"""

import grpc
from grpc import aio
import asyncio
from typing import AsyncIterator, Optional
import numpy as np
from dataclasses import dataclass
import time
import logging
from concurrent import futures
import consul
import circuit_breaker
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
import prometheus_client

# Proto definitions (would normally be in .proto files)
from google.protobuf import empty_pb2, timestamp_pb2
import struct

logger = logging.getLogger(__name__)

# Service mesh configuration
SERVICE_MESH_CONFIG = {
    "consul_host": "localhost",
    "consul_port": 8500,
    "service_name": "aioke-audio",
    "health_check_interval": "10s",
    "deregister_critical_after": "1m",
    "load_balancing": "round_robin",
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "expected_exception": grpc.RpcError
    },
    "retry_policy": {
        "max_attempts": 3,
        "initial_backoff": 0.1,
        "max_backoff": 1.0,
        "backoff_multiplier": 2.0,
        "retryable_status_codes": [
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.RESOURCE_EXHAUSTED
        ]
    }
}


class AudioRequest:
    """Audio processing request message"""
    def __init__(self, audio_data: bytes, sample_rate: int, channels: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.channels = channels
        self.request_id = str(time.time())
        

class AudioResponse:
    """Audio processing response message"""
    def __init__(self, vocal_data: bytes, music_data: bytes, latency_ms: float):
        self.vocal_data = vocal_data
        self.music_data = music_data
        self.latency_ms = latency_ms
        self.processed_at = time.time()


class StreamingAudioRequest:
    """Streaming audio request"""
    def __init__(self, chunk: bytes, chunk_id: int):
        self.chunk = chunk
        self.chunk_id = chunk_id
        

class ServiceDiscovery:
    """Consul-based service discovery (Netflix pattern)"""
    
    def __init__(self):
        self.consul = consul.Consul(
            host=SERVICE_MESH_CONFIG["consul_host"],
            port=SERVICE_MESH_CONFIG["consul_port"]
        )
        self._service_cache = {}
        
    async def register_service(self, name: str, host: str, port: int):
        """Register service with Consul"""
        service_def = {
            "Name": name,
            "ID": f"{name}-{host}-{port}",
            "Address": host,
            "Port": port,
            "Check": {
                "GRPC": f"{host}:{port}",
                "Interval": SERVICE_MESH_CONFIG["health_check_interval"],
                "DeregisterCriticalServiceAfter": SERVICE_MESH_CONFIG["deregister_critical_after"]
            }
        }
        
        self.consul.agent.service.register(service_def)
        logger.info(f"Registered service {name} at {host}:{port}")
        
    async def discover_service(self, name: str) -> list:
        """Discover healthy service instances"""
        _, services = self.consul.health.service(name, passing=True)
        
        instances = []
        for service in services:
            instances.append({
                "host": service["Service"]["Address"],
                "port": service["Service"]["Port"],
                "id": service["Service"]["ID"]
            })
            
        self._service_cache[name] = instances
        return instances
    
    async def deregister_service(self, service_id: str):
        """Deregister service from Consul"""
        self.consul.agent.service.deregister(service_id)


class LoadBalancer:
    """Client-side load balancing (Google pattern)"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self._round_robin_counter = 0
        self._endpoint_stats = {}  # For least_connections strategy
        
    def select_endpoint(self, endpoints: list) -> dict:
        """Select endpoint based on load balancing strategy"""
        if not endpoints:
            raise ValueError("No endpoints available")
            
        if self.strategy == "round_robin":
            endpoint = endpoints[self._round_robin_counter % len(endpoints)]
            self._round_robin_counter += 1
            return endpoint
            
        elif self.strategy == "least_connections":
            # Select endpoint with least active connections
            min_conn = float('inf')
            selected = endpoints[0]
            
            for endpoint in endpoints:
                endpoint_id = f"{endpoint['host']}:{endpoint['port']}"
                connections = self._endpoint_stats.get(endpoint_id, 0)
                if connections < min_conn:
                    min_conn = connections
                    selected = endpoint
                    
            return selected
            
        elif self.strategy == "random":
            import random
            return random.choice(endpoints)
            
        else:
            return endpoints[0]  # Default to first endpoint


class CircuitBreakerInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """Circuit breaker for gRPC calls (Netflix Hystrix pattern)"""
    
    def __init__(self):
        self.breaker = circuit_breaker.CircuitBreaker(
            failure_threshold=SERVICE_MESH_CONFIG["circuit_breaker"]["failure_threshold"],
            recovery_timeout=SERVICE_MESH_CONFIG["circuit_breaker"]["recovery_timeout"],
            expected_exception=SERVICE_MESH_CONFIG["circuit_breaker"]["expected_exception"]
        )
        
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept unary calls with circuit breaker"""
        
        @self.breaker
        async def protected_call():
            return await continuation(client_call_details, request)
            
        try:
            return await protected_call()
        except circuit_breaker.CircuitBreakerError:
            # Return degraded response when circuit is open
            logger.warning(f"Circuit breaker open for {client_call_details.method}")
            raise grpc.aio.AioRpcError(
                grpc.StatusCode.UNAVAILABLE,
                "Service temporarily unavailable - circuit breaker open"
            )


class RetryInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """Retry interceptor with exponential backoff (Google pattern)"""
    
    def __init__(self):
        self.config = SERVICE_MESH_CONFIG["retry_policy"]
        
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept calls with retry logic"""
        
        for attempt in range(self.config["max_attempts"]):
            try:
                response = await continuation(client_call_details, request)
                return response
                
            except grpc.aio.AioRpcError as e:
                if e.code() not in self.config["retryable_status_codes"]:
                    raise
                    
                if attempt == self.config["max_attempts"] - 1:
                    raise
                    
                # Exponential backoff
                backoff = min(
                    self.config["initial_backoff"] * (self.config["backoff_multiplier"] ** attempt),
                    self.config["max_backoff"]
                )
                
                logger.info(f"Retrying {client_call_details.method} after {backoff}s (attempt {attempt + 1})")
                await asyncio.sleep(backoff)
                
        raise Exception("Max retries exceeded")


class AudioProcessingServicer:
    """gRPC service implementation for audio processing"""
    
    def __init__(self):
        self.processing_count = prometheus_client.Counter(
            'grpc_audio_processing_total',
            'Total audio processing requests'
        )
        self.processing_latency = prometheus_client.Histogram(
            'grpc_audio_processing_latency_seconds',
            'Audio processing latency'
        )
        
    async def ProcessAudio(self, request: AudioRequest, context) -> AudioResponse:
        """Process single audio request"""
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(request.audio_data, dtype=np.float32)
            audio_data = audio_data.reshape(-1, request.channels)
            
            # Process audio (simplified)
            vocal_data = audio_data[:, 0] * 0.8  # Simulate vocal extraction
            music_data = audio_data[:, 1] * 0.8  # Simulate music extraction
            
            # Convert back to bytes
            vocal_bytes = vocal_data.astype(np.float32).tobytes()
            music_bytes = music_data.astype(np.float32).tobytes()
            
            latency = (time.time() - start_time) * 1000
            
            self.processing_count.inc()
            self.processing_latency.observe(time.time() - start_time)
            
            return AudioResponse(
                vocal_data=vocal_bytes,
                music_data=music_bytes,
                latency_ms=latency
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Processing failed: {str(e)}")
            raise
            
    async def StreamProcessAudio(self, request_iterator: AsyncIterator[StreamingAudioRequest], context):
        """Process streaming audio"""
        
        chunks = []
        async for request in request_iterator:
            chunks.append(request.chunk)
            
            # Process every 10 chunks
            if len(chunks) >= 10:
                combined_audio = b''.join(chunks)
                
                # Process combined audio
                audio_array = np.frombuffer(combined_audio, dtype=np.float32)
                processed = audio_array * 0.9  # Simple processing
                
                yield AudioResponse(
                    vocal_data=processed.tobytes(),
                    music_data=processed.tobytes(),
                    latency_ms=10.0
                )
                
                chunks = []  # Reset buffer


class HealthServicer(health_pb2_grpc.HealthServicer):
    """Health check service (Kubernetes/Istio pattern)"""
    
    def __init__(self):
        self.status = health_pb2.HealthCheckResponse.SERVING
        
    async def Check(self, request, context):
        if request.service == "aioke.audio":
            return health_pb2.HealthCheckResponse(status=self.status)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return health_pb2.HealthCheckResponse()
            
    async def Watch(self, request, context):
        """Stream health status changes"""
        while True:
            yield health_pb2.HealthCheckResponse(status=self.status)
            await asyncio.sleep(5)


class GrpcAudioClient:
    """gRPC client with service mesh features"""
    
    def __init__(self):
        self.discovery = ServiceDiscovery()
        self.load_balancer = LoadBalancer(SERVICE_MESH_CONFIG["load_balancing"])
        self.channels = {}
        self.stubs = {}
        
    async def connect(self, service_name: str):
        """Connect to service with discovery and load balancing"""
        
        # Discover service endpoints
        endpoints = await self.discovery.discover_service(service_name)
        
        if not endpoints:
            raise Exception(f"No healthy endpoints for {service_name}")
            
        # Select endpoint
        endpoint = self.load_balancer.select_endpoint(endpoints)
        endpoint_addr = f"{endpoint['host']}:{endpoint['port']}"
        
        # Create channel with interceptors
        if endpoint_addr not in self.channels:
            interceptors = [
                CircuitBreakerInterceptor(),
                RetryInterceptor()
            ]
            
            self.channels[endpoint_addr] = grpc.aio.insecure_channel(
                endpoint_addr,
                interceptors=interceptors,
                options=[
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                ]
            )
            
        return self.channels[endpoint_addr]
        
    async def process_audio(self, audio_data: np.ndarray):
        """Process audio through gRPC service"""
        
        channel = await self.connect("aioke-audio")
        
        # Create request
        request = AudioRequest(
            audio_data=audio_data.tobytes(),
            sample_rate=44100,
            channels=2
        )
        
        # Make RPC call
        # In real implementation, would use generated stub
        async with channel as ch:
            # Simulated call
            response = await self._simulate_call(request)
            
        return response
        
    async def stream_audio(self, audio_stream: AsyncIterator[np.ndarray]):
        """Stream audio processing"""
        
        channel = await self.connect("aioke-audio")
        
        async def request_generator():
            chunk_id = 0
            async for audio_chunk in audio_stream:
                yield StreamingAudioRequest(
                    chunk=audio_chunk.tobytes(),
                    chunk_id=chunk_id
                )
                chunk_id += 1
                
        # Stream processing
        async with channel as ch:
            # In real implementation, would use generated stub
            async for response in self._simulate_stream(request_generator()):
                yield response


async def serve():
    """Start gRPC server with service mesh integration"""
    
    # Create server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.so_reuseport', 1),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Add services
    audio_servicer = AudioProcessingServicer()
    health_servicer = HealthServicer()
    
    # Add health check
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Enable reflection for debugging
    service_names = (
        health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    
    # Start server
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    
    # Register with service discovery
    discovery = ServiceDiscovery()
    await discovery.register_service("aioke-audio", "localhost", port)
    
    await server.start()
    logger.info(f"gRPC server started on port {port}")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await discovery.deregister_service(f"aioke-audio-localhost-{port}")
        await server.stop(5)


if __name__ == "__main__":
    asyncio.run(serve())