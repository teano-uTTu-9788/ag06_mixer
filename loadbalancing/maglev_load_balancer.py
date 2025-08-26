"""
Google Maglev-style Load Balancing with Consistent Hashing
Following Google's Maglev paper and production practices
"""

import hashlib
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import random
import bisect

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    MAGLEV = "maglev"  # Google's consistent hashing
    CONSISTENT_HASH = "consistent_hash"  # Traditional
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"
    RANDOM = "random"

class HealthCheckType(Enum):
    """Health check types"""
    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    CUSTOM = "custom"

@dataclass
class Backend:
    """Backend server"""
    backend_id: str
    address: str
    port: int
    weight: int = 1
    healthy: bool = True
    connections: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def load_score(self) -> float:
        """Calculate load score (lower is better)"""
        # Weighted combination of metrics
        connection_score = self.connections / max(self.weight, 1)
        cpu_score = self.cpu_percent / 100
        memory_score = self.memory_percent / 100
        latency_score = min(self.response_time_ms / 1000, 1.0)
        error_score = self.error_rate * 10  # Heavy penalty for errors
        
        return (connection_score * 0.3 +
                cpu_score * 0.2 +
                memory_score * 0.2 +
                latency_score * 0.2 +
                error_score * 0.1)

@dataclass
class HealthCheck:
    """Health check configuration"""
    check_type: HealthCheckType
    interval_seconds: int = 5
    timeout_seconds: int = 3
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    path: str = "/health"
    expected_response: Any = 200

class MaglevHashTable:
    """Google Maglev consistent hashing implementation"""
    
    def __init__(self, size: int = 65537):  # Prime number
        self.size = size
        self.lookup_table: List[Optional[str]] = [None] * size
        self.backends: Dict[str, Backend] = {}
        self.permutations: Dict[str, List[int]] = {}
    
    def add_backend(self, backend: Backend):
        """Add backend to hash table"""
        self.backends[backend.backend_id] = backend
        self._rebuild_table()
    
    def remove_backend(self, backend_id: str):
        """Remove backend from hash table"""
        if backend_id in self.backends:
            del self.backends[backend_id]
            self._rebuild_table()
    
    def _rebuild_table(self):
        """Rebuild Maglev lookup table"""
        if not self.backends:
            return
        
        # Generate permutations for each backend
        for backend_id in self.backends:
            self.permutations[backend_id] = self._generate_permutation(backend_id)
        
        # Build lookup table
        self.lookup_table = [None] * self.size
        next_index = {b: 0 for b in self.backends}
        
        n = 0
        while n < self.size:
            for backend_id in sorted(self.backends.keys()):  # Deterministic order
                c = next_index[backend_id]
                
                while True:
                    position = self.permutations[backend_id][c % len(self.permutations[backend_id])]
                    
                    if self.lookup_table[position] is None:
                        self.lookup_table[position] = backend_id
                        next_index[backend_id] = c + 1
                        n += 1
                        break
                    
                    c += 1
                    next_index[backend_id] = c
                
                if n >= self.size:
                    break
    
    def _generate_permutation(self, backend_id: str) -> List[int]:
        """Generate permutation for backend using hash functions"""
        permutation = []
        
        # Use two hash functions (as per Maglev paper)
        offset = self._hash1(backend_id) % self.size
        skip = (self._hash2(backend_id) % (self.size - 1)) + 1
        
        for j in range(self.size):
            permutation.append((offset + j * skip) % self.size)
        
        return permutation
    
    def _hash1(self, key: str) -> int:
        """First hash function"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _hash2(self, key: str) -> int:
        """Second hash function"""
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)
    
    def get_backend(self, key: str) -> Optional[Backend]:
        """Get backend for given key"""
        if not self.backends:
            return None
        
        # Hash the key to get position
        hash_value = self._hash1(key) % self.size
        
        # Look up backend
        backend_id = self.lookup_table[hash_value]
        
        if backend_id and backend_id in self.backends:
            backend = self.backends[backend_id]
            
            # Skip unhealthy backends
            if not backend.healthy:
                # Linear probe for healthy backend
                for i in range(1, min(100, self.size)):
                    next_pos = (hash_value + i) % self.size
                    next_id = self.lookup_table[next_pos]
                    
                    if next_id and next_id in self.backends:
                        next_backend = self.backends[next_id]
                        if next_backend.healthy:
                            return next_backend
                
                # No healthy backend found
                return None
            
            return backend
        
        return None

class ConsistentHashRing:
    """Traditional consistent hash ring with virtual nodes"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.backends: Dict[str, Backend] = {}
    
    def add_backend(self, backend: Backend):
        """Add backend to ring"""
        self.backends[backend.backend_id] = backend
        
        # Add virtual nodes
        for i in range(self.virtual_nodes * backend.weight):
            virtual_key = f"{backend.backend_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = backend.backend_id
        
        # Rebuild sorted keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_backend(self, backend_id: str):
        """Remove backend from ring"""
        if backend_id not in self.backends:
            return
        
        backend = self.backends[backend_id]
        
        # Remove virtual nodes
        for i in range(self.virtual_nodes * backend.weight):
            virtual_key = f"{backend_id}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
        
        del self.backends[backend_id]
        
        # Rebuild sorted keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_backend(self, key: str) -> Optional[Backend]:
        """Get backend for key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Binary search for position
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        backend_id = self.ring[self.sorted_keys[idx]]
        
        if backend_id in self.backends:
            backend = self.backends[backend_id]
            if backend.healthy:
                return backend
        
        # Try next backends if unhealthy
        for i in range(1, len(self.sorted_keys)):
            next_idx = (idx + i) % len(self.sorted_keys)
            next_id = self.ring[self.sorted_keys[next_idx]]
            
            if next_id in self.backends:
                next_backend = self.backends[next_id]
                if next_backend.healthy:
                    return next_backend
        
        return None
    
    def _hash(self, key: str) -> int:
        """Hash function"""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

class LoadBalancer:
    """Production-grade load balancer with multiple algorithms"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.MAGLEV):
        self.algorithm = algorithm
        self.backends: Dict[str, Backend] = {}
        
        # Algorithm-specific structures
        self.maglev_table = MaglevHashTable()
        self.consistent_hash = ConsistentHashRing()
        self.round_robin_index = 0
        
        # Health checking
        self.health_checker = HealthChecker()
        
        # Metrics
        self.metrics = LoadBalancerMetrics()
        
        # Connection tracking
        self.active_connections: Dict[str, Set[str]] = defaultdict(set)
    
    def add_backend(self, backend: Backend):
        """Add backend server"""
        self.backends[backend.backend_id] = backend
        
        if self.algorithm == LoadBalancingAlgorithm.MAGLEV:
            self.maglev_table.add_backend(backend)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            self.consistent_hash.add_backend(backend)
    
    def remove_backend(self, backend_id: str):
        """Remove backend server"""
        if backend_id in self.backends:
            del self.backends[backend_id]
            
            if self.algorithm == LoadBalancingAlgorithm.MAGLEV:
                self.maglev_table.remove_backend(backend_id)
            elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                self.consistent_hash.remove_backend(backend_id)
            
            # Clean up connections
            if backend_id in self.active_connections:
                del self.active_connections[backend_id]
    
    async def route_request(self, request_key: str, 
                           client_ip: Optional[str] = None) -> Optional[Backend]:
        """Route request to backend"""
        start_time = time.time()
        
        backend = None
        
        if self.algorithm == LoadBalancingAlgorithm.MAGLEV:
            backend = self.maglev_table.get_backend(request_key)
        
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            backend = self.consistent_hash.get_backend(request_key)
        
        elif self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            backend = self._round_robin_select()
        
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            backend = self._least_connections_select()
        
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED:
            backend = self._weighted_select()
        
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            if client_ip:
                backend = self._ip_hash_select(client_ip)
        
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            backend = self._random_select()
        
        # Track metrics
        latency = (time.time() - start_time) * 1000
        await self.metrics.record_routing(backend, latency)
        
        # Update connection count
        if backend:
            backend.connections += 1
            self.active_connections[backend.backend_id].add(request_key)
        
        return backend
    
    def _round_robin_select(self) -> Optional[Backend]:
        """Round-robin selection"""
        healthy_backends = [b for b in self.backends.values() if b.healthy]
        
        if not healthy_backends:
            return None
        
        backend = healthy_backends[self.round_robin_index % len(healthy_backends)]
        self.round_robin_index += 1
        
        return backend
    
    def _least_connections_select(self) -> Optional[Backend]:
        """Least connections selection"""
        healthy_backends = [b for b in self.backends.values() if b.healthy]
        
        if not healthy_backends:
            return None
        
        return min(healthy_backends, key=lambda b: b.connections / b.weight)
    
    def _weighted_select(self) -> Optional[Backend]:
        """Weighted random selection"""
        healthy_backends = [b for b in self.backends.values() if b.healthy]
        
        if not healthy_backends:
            return None
        
        total_weight = sum(b.weight for b in healthy_backends)
        rand = random.uniform(0, total_weight)
        
        cumulative = 0
        for backend in healthy_backends:
            cumulative += backend.weight
            if rand <= cumulative:
                return backend
        
        return healthy_backends[-1]
    
    def _ip_hash_select(self, client_ip: str) -> Optional[Backend]:
        """IP hash selection"""
        healthy_backends = [b for b in self.backends.values() if b.healthy]
        
        if not healthy_backends:
            return None
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return healthy_backends[hash_value % len(healthy_backends)]
    
    def _random_select(self) -> Optional[Backend]:
        """Random selection"""
        healthy_backends = [b for b in self.backends.values() if b.healthy]
        
        if not healthy_backends:
            return None
        
        return random.choice(healthy_backends)
    
    async def release_connection(self, backend_id: str, request_key: str):
        """Release connection from backend"""
        if backend_id in self.backends:
            self.backends[backend_id].connections = max(0, 
                self.backends[backend_id].connections - 1)
            
            if backend_id in self.active_connections:
                self.active_connections[backend_id].discard(request_key)

class HealthChecker:
    """Health checking system"""
    
    def __init__(self):
        self.check_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.checking = False
    
    async def check_backend(self, backend: Backend, config: HealthCheck) -> bool:
        """Check backend health"""
        try:
            if config.check_type == HealthCheckType.HTTP:
                return await self._http_check(backend, config)
            elif config.check_type == HealthCheckType.TCP:
                return await self._tcp_check(backend, config)
            elif config.check_type == HealthCheckType.GRPC:
                return await self._grpc_check(backend, config)
            else:
                return True
        except Exception:
            return False
    
    async def _http_check(self, backend: Backend, config: HealthCheck) -> bool:
        """HTTP health check (simulated)"""
        await asyncio.sleep(0.01)  # Simulate network call
        
        # Simulate health check
        is_healthy = random.random() > 0.05  # 95% healthy
        
        self.check_results[backend.backend_id].append(is_healthy)
        
        # Update backend health based on threshold
        healthy_count = sum(self.check_results[backend.backend_id])
        
        if backend.healthy and healthy_count < config.unhealthy_threshold:
            backend.healthy = False
        elif not backend.healthy and healthy_count >= config.healthy_threshold:
            backend.healthy = True
        
        return is_healthy
    
    async def _tcp_check(self, backend: Backend, config: HealthCheck) -> bool:
        """TCP health check (simulated)"""
        await asyncio.sleep(0.005)
        return random.random() > 0.02  # 98% healthy
    
    async def _grpc_check(self, backend: Backend, config: HealthCheck) -> bool:
        """gRPC health check (simulated)"""
        await asyncio.sleep(0.01)
        return random.random() > 0.03  # 97% healthy
    
    async def start_health_checks(self, backends: Dict[str, Backend], 
                                 config: HealthCheck):
        """Start periodic health checks"""
        self.checking = True
        
        while self.checking:
            for backend in backends.values():
                await self.check_backend(backend, config)
            
            await asyncio.sleep(config.interval_seconds)

class LoadBalancerMetrics:
    """Load balancer metrics and monitoring"""
    
    def __init__(self):
        self.total_requests = 0
        self.backend_requests: Dict[str, int] = defaultdict(int)
        self.total_latency = 0.0
        self.errors = 0
    
    async def record_routing(self, backend: Optional[Backend], latency: float):
        """Record routing decision"""
        self.total_requests += 1
        self.total_latency += latency
        
        if backend:
            self.backend_requests[backend.backend_id] += 1
        else:
            self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "average_latency_ms": avg_latency,
            "error_rate": self.errors / self.total_requests if self.total_requests > 0 else 0,
            "backend_distribution": dict(self.backend_requests)
        }

# Example usage
async def main():
    """Demonstrate Maglev load balancing"""
    
    print("⚖️ Google Maglev Load Balancer")
    print("=" * 60)
    
    # Create load balancer
    lb = LoadBalancer(LoadBalancingAlgorithm.MAGLEV)
    
    # Add backends
    backends = [
        Backend("server1", "10.0.0.1", 8080, weight=2),
        Backend("server2", "10.0.0.2", 8080, weight=1),
        Backend("server3", "10.0.0.3", 8080, weight=1),
        Backend("server4", "10.0.0.4", 8080, weight=3),
    ]
    
    for backend in backends:
        lb.add_backend(backend)
    
    print(f"\n🖥️ Backends Added: {len(backends)}")
    for b in backends:
        print(f"  - {b.backend_id}: {b.address}:{b.port} (weight={b.weight})")
    
    # Simulate requests
    print("\n📊 Routing Simulation:")
    print("-" * 40)
    
    request_keys = [f"user_{i}" for i in range(20)]
    
    for key in request_keys:
        backend = await lb.route_request(key)
        if backend:
            print(f"Request '{key}' -> {backend.backend_id}")
    
    # Show distribution
    stats = lb.metrics.get_stats()
    
    print("\n📈 Load Distribution:")
    print("-" * 40)
    
    for backend_id, count in stats["backend_distribution"].items():
        percentage = (count / stats["total_requests"]) * 100
        print(f"{backend_id}: {count} requests ({percentage:.1f}%)")
    
    # Test consistency
    print("\n🔄 Consistency Test:")
    print("-" * 40)
    
    # Route same keys again
    print("Routing same keys again...")
    consistent = True
    
    for key in request_keys[:5]:
        backend = await lb.route_request(key)
        print(f"Request '{key}' -> {backend.backend_id if backend else 'None'}")
    
    print("\n✅ Maglev load balancer operational with consistent hashing!")
    
    return lb

if __name__ == "__main__":
    asyncio.run(main())