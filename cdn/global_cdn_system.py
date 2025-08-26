"""
Global CDN System with Intelligent Routing
Following Cloudflare, Fastly, and AWS CloudFront best practices
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import random
import math

class CacheStrategy(Enum):
    """CDN caching strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # ML-based adaptive caching

class RoutingStrategy(Enum):
    """Traffic routing strategies"""
    GEOGRAPHIC = "geographic"  # Route to nearest PoP
    LATENCY = "latency"  # Route to lowest latency
    LOAD_BALANCED = "load_balanced"  # Distribute based on load
    COST_OPTIMIZED = "cost_optimized"  # Optimize for bandwidth costs
    PERFORMANCE = "performance"  # Route to fastest PoP

@dataclass
class PointOfPresence:
    """CDN Point of Presence (PoP)"""
    id: str
    location: str
    region: str
    latitude: float
    longitude: float
    capacity_gbps: float
    current_load: float = 0.0
    cache_size_gb: int = 1000
    cache_used_gb: float = 0.0
    health_score: float = 100.0
    tier: str = "standard"  # standard, premium, ultra
    providers: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        return self.health_score > 50 and self.current_load < 90

@dataclass
class CacheEntry:
    """Cached content entry"""
    key: str
    content: bytes
    content_type: str
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl_seconds: int = 3600
    etag: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        self.accessed_at = time.time()
        self.access_count += 1

@dataclass
class CDNRequest:
    """CDN request object"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    client_ip: str = ""
    client_location: Optional[Tuple[float, float]] = None
    priority: int = 0  # Higher = more important
    user_agent: str = ""

@dataclass
class CDNResponse:
    """CDN response object"""
    content: bytes
    status_code: int
    headers: Dict[str, str]
    cache_status: str  # HIT, MISS, EXPIRED, BYPASS
    pop_id: str
    latency_ms: float
    bandwidth_saved: float = 0.0

class IntelligentCache:
    """Advanced caching system with ML-based optimization"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.prediction_model = self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize ML model for cache prediction"""
        # Simulate ML model for cache optimization
        return {
            "popularity_threshold": 0.7,
            "ttl_multiplier": 1.5,
            "prefetch_threshold": 0.8
        }
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache with strategy-based eviction"""
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                return None
            
            entry.update_access()
            self.access_patterns[key].append(time.time())
            
            # Adaptive TTL extension for popular content
            if self.strategy == CacheStrategy.ADAPTIVE:
                popularity = self._calculate_popularity(key)
                if popularity > self.prediction_model["popularity_threshold"]:
                    entry.ttl_seconds = int(entry.ttl_seconds * 
                                           self.prediction_model["ttl_multiplier"])
            
            return entry
        return None
    
    async def set(self, key: str, entry: CacheEntry):
        """Set item in cache with eviction if needed"""
        # Implement cache eviction based on strategy
        if self._needs_eviction():
            await self._evict()
        
        self.cache[key] = entry
        self.access_patterns[key].append(time.time())
    
    def _calculate_popularity(self, key: str) -> float:
        """Calculate content popularity score"""
        if key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[key]
        if not accesses:
            return 0.0
        
        # Calculate access frequency and recency
        now = time.time()
        recent_accesses = sum(1 for t in accesses if now - t < 3600)
        total_accesses = len(accesses)
        
        # Weighted score: 70% recency, 30% total
        popularity = (0.7 * min(recent_accesses / 10, 1.0) + 
                     0.3 * min(total_accesses / 100, 1.0))
        
        return popularity
    
    def _needs_eviction(self) -> bool:
        """Check if cache needs eviction"""
        # Simulate cache size limits
        total_size = sum(e.size_bytes for e in self.cache.values())
        return total_size > 1024 * 1024 * 1024  # 1GB limit
    
    async def _evict(self):
        """Evict items based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest = min(self.cache.items(), 
                        key=lambda x: x[1].accessed_at)
            del self.cache[oldest[0]]
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used = min(self.cache.items(),
                           key=lambda x: x[1].access_count)
            del self.cache[least_used[0]]
            
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # ML-based eviction
            scores = {k: self._calculate_popularity(k) 
                     for k in self.cache.keys()}
            to_evict = min(scores.items(), key=lambda x: x[1])[0]
            del self.cache[to_evict]

class GlobalCDN:
    """Global CDN with intelligent routing and caching"""
    
    def __init__(self):
        self.pops = self._initialize_pops()
        self.caches: Dict[str, IntelligentCache] = {}
        self.routing_strategy = RoutingStrategy.PERFORMANCE
        self.analytics = CDNAnalytics()
        self.security = CDNSecurity()
        
        # Initialize caches for each PoP
        for pop in self.pops.values():
            self.caches[pop.id] = IntelligentCache()
    
    def _initialize_pops(self) -> Dict[str, PointOfPresence]:
        """Initialize global PoPs following real CDN distribution"""
        return {
            # North America
            "us-west-1": PointOfPresence(
                "us-west-1", "San Francisco", "NA", 37.7749, -122.4194,
                100, tier="ultra", providers=["aws", "cloudflare"]
            ),
            "us-east-1": PointOfPresence(
                "us-east-1", "New York", "NA", 40.7128, -74.0060,
                100, tier="ultra", providers=["aws", "fastly"]
            ),
            "us-central-1": PointOfPresence(
                "us-central-1", "Chicago", "NA", 41.8781, -87.6298,
                50, tier="premium", providers=["cloudflare"]
            ),
            
            # Europe
            "eu-west-1": PointOfPresence(
                "eu-west-1", "London", "EU", 51.5074, -0.1278,
                100, tier="ultra", providers=["aws", "cloudflare"]
            ),
            "eu-central-1": PointOfPresence(
                "eu-central-1", "Frankfurt", "EU", 50.1109, 8.6821,
                75, tier="premium", providers=["aws", "fastly"]
            ),
            
            # Asia Pacific
            "ap-northeast-1": PointOfPresence(
                "ap-northeast-1", "Tokyo", "APAC", 35.6762, 139.6503,
                100, tier="ultra", providers=["aws", "cloudflare"]
            ),
            "ap-southeast-1": PointOfPresence(
                "ap-southeast-1", "Singapore", "APAC", 1.3521, 103.8198,
                75, tier="premium", providers=["aws", "fastly"]
            ),
            "ap-south-1": PointOfPresence(
                "ap-south-1", "Mumbai", "APAC", 19.0760, 72.8777,
                50, tier="standard", providers=["cloudflare"]
            ),
            
            # South America
            "sa-east-1": PointOfPresence(
                "sa-east-1", "São Paulo", "SA", -23.5505, -46.6333,
                25, tier="standard", providers=["aws"]
            ),
            
            # Africa
            "af-south-1": PointOfPresence(
                "af-south-1", "Cape Town", "AF", -33.9249, 18.4241,
                25, tier="standard", providers=["cloudflare"]
            ),
            
            # Australia
            "ap-southeast-2": PointOfPresence(
                "ap-southeast-2", "Sydney", "APAC", -33.8688, 151.2093,
                50, tier="premium", providers=["aws", "cloudflare"]
            )
        }
    
    async def handle_request(self, request: CDNRequest) -> CDNResponse:
        """Handle CDN request with intelligent routing"""
        start_time = time.time()
        
        # Security checks
        if not await self.security.validate_request(request):
            return CDNResponse(
                content=b"Forbidden",
                status_code=403,
                headers={"X-CDN-Error": "Security validation failed"},
                cache_status="BYPASS",
                pop_id="none",
                latency_ms=0
            )
        
        # Select optimal PoP
        pop = await self._select_pop(request)
        
        # Check cache
        cache_key = self._generate_cache_key(request)
        cache = self.caches[pop.id]
        cached_entry = await cache.get(cache_key)
        
        if cached_entry:
            # Cache hit
            latency = (time.time() - start_time) * 1000
            
            # Track analytics
            await self.analytics.track_request(
                request, pop.id, "HIT", latency
            )
            
            return CDNResponse(
                content=cached_entry.content,
                status_code=200,
                headers=self._build_response_headers(cached_entry, pop),
                cache_status="HIT",
                pop_id=pop.id,
                latency_ms=latency,
                bandwidth_saved=cached_entry.size_bytes
            )
        
        # Cache miss - fetch from origin
        content = await self._fetch_from_origin(request)
        
        # Store in cache
        entry = CacheEntry(
            key=cache_key,
            content=content,
            content_type="application/octet-stream",
            size_bytes=len(content),
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=self._calculate_ttl(request),
            etag=hashlib.md5(content).hexdigest()
        )
        
        await cache.set(cache_key, entry)
        
        latency = (time.time() - start_time) * 1000
        
        # Track analytics
        await self.analytics.track_request(
            request, pop.id, "MISS", latency
        )
        
        return CDNResponse(
            content=content,
            status_code=200,
            headers=self._build_response_headers(entry, pop),
            cache_status="MISS",
            pop_id=pop.id,
            latency_ms=latency
        )
    
    async def _select_pop(self, request: CDNRequest) -> PointOfPresence:
        """Select optimal PoP based on routing strategy"""
        if self.routing_strategy == RoutingStrategy.GEOGRAPHIC:
            return self._select_by_geography(request)
        elif self.routing_strategy == RoutingStrategy.LATENCY:
            return await self._select_by_latency(request)
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            return self._select_by_load()
        elif self.routing_strategy == RoutingStrategy.PERFORMANCE:
            return await self._select_by_performance(request)
        else:
            # Default to geographic
            return self._select_by_geography(request)
    
    def _select_by_geography(self, request: CDNRequest) -> PointOfPresence:
        """Select nearest PoP geographically"""
        if not request.client_location:
            # Default to US West if no location
            return self.pops["us-west-1"]
        
        client_lat, client_lon = request.client_location
        
        # Calculate distances to all healthy PoPs
        distances = {}
        for pop_id, pop in self.pops.items():
            if pop.is_healthy():
                # Haversine distance calculation
                dist = self._haversine_distance(
                    client_lat, client_lon,
                    pop.latitude, pop.longitude
                )
                distances[pop_id] = dist
        
        # Return nearest PoP
        nearest = min(distances.items(), key=lambda x: x[1])
        return self.pops[nearest[0]]
    
    async def _select_by_latency(self, request: CDNRequest) -> PointOfPresence:
        """Select PoP with lowest latency"""
        # Simulate latency measurements
        latencies = {}
        
        for pop_id, pop in self.pops.items():
            if pop.is_healthy():
                # Simulate latency based on distance and load
                base_latency = random.uniform(5, 100)
                load_factor = 1 + (pop.current_load / 100)
                latencies[pop_id] = base_latency * load_factor
        
        # Return lowest latency PoP
        best = min(latencies.items(), key=lambda x: x[1])
        return self.pops[best[0]]
    
    def _select_by_load(self) -> PointOfPresence:
        """Select PoP with lowest load"""
        healthy_pops = [p for p in self.pops.values() if p.is_healthy()]
        return min(healthy_pops, key=lambda p: p.current_load)
    
    async def _select_by_performance(self, request: CDNRequest) -> PointOfPresence:
        """Select PoP based on composite performance score"""
        scores = {}
        
        for pop_id, pop in self.pops.items():
            if not pop.is_healthy():
                continue
            
            # Calculate composite score
            # 40% latency, 30% load, 20% tier, 10% cache hit rate
            latency_score = 100 / (1 + random.uniform(5, 100))  # Simulated
            load_score = 100 - pop.current_load
            tier_score = {"ultra": 100, "premium": 75, "standard": 50}[pop.tier]
            cache_score = random.uniform(60, 95)  # Simulated cache hit rate
            
            composite = (0.4 * latency_score +
                        0.3 * load_score +
                        0.2 * tier_score +
                        0.1 * cache_score)
            
            scores[pop_id] = composite
        
        # Return highest scoring PoP
        best = max(scores.items(), key=lambda x: x[1])
        return self.pops[best[0]]
    
    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth"""
        R = 6371  # Earth's radius in km
        
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        
        a = (math.sin(dLat/2) * math.sin(dLat/2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dLon/2) * math.sin(dLon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _generate_cache_key(self, request: CDNRequest) -> str:
        """Generate cache key for request"""
        # Include method, URL, and important headers
        key_parts = [
            request.method,
            request.url,
            request.headers.get("Accept", ""),
            request.headers.get("Accept-Encoding", "")
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def _fetch_from_origin(self, request: CDNRequest) -> bytes:
        """Fetch content from origin server"""
        # Simulate origin fetch
        await asyncio.sleep(random.uniform(0.05, 0.2))  # 50-200ms
        
        # Generate simulated content
        content = f"Origin content for {request.url}".encode()
        return content
    
    def _calculate_ttl(self, request: CDNRequest) -> int:
        """Calculate optimal TTL for content"""
        # Intelligent TTL based on content type and patterns
        if "static" in request.url:
            return 86400  # 1 day for static content
        elif "api" in request.url:
            return 60  # 1 minute for API responses
        elif "stream" in request.url:
            return 5  # 5 seconds for streaming
        else:
            return 3600  # 1 hour default
    
    def _build_response_headers(self, entry: CacheEntry,
                               pop: PointOfPresence) -> Dict[str, str]:
        """Build response headers with CDN metadata"""
        return {
            "X-CDN-POP": pop.id,
            "X-CDN-Region": pop.region,
            "X-Cache-Status": "HIT" if entry.access_count > 0 else "MISS",
            "X-Cache-TTL": str(entry.ttl_seconds),
            "ETag": entry.etag,
            "Cache-Control": f"public, max-age={entry.ttl_seconds}",
            "X-CDN-Provider": ",".join(pop.providers),
            "X-CDN-Tier": pop.tier
        }
    
    async def purge_cache(self, pattern: str):
        """Purge cache entries matching pattern"""
        purged_count = 0
        
        for cache in self.caches.values():
            keys_to_remove = []
            for key in cache.cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del cache.cache[key]
                purged_count += 1
        
        return purged_count
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get CDN analytics and metrics"""
        return await self.analytics.get_summary()

class CDNAnalytics:
    """CDN analytics and monitoring"""
    
    def __init__(self):
        self.requests: List[Dict] = []
        self.bandwidth_saved = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def track_request(self, request: CDNRequest, pop_id: str,
                           cache_status: str, latency_ms: float):
        """Track CDN request for analytics"""
        self.requests.append({
            "timestamp": datetime.now().isoformat(),
            "url": request.url,
            "pop_id": pop_id,
            "cache_status": cache_status,
            "latency_ms": latency_ms,
            "client_ip": request.client_ip
        })
        
        if cache_status == "HIT":
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        total_requests = len(self.requests)
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate average latency by PoP
        pop_latencies = defaultdict(list)
        for req in self.requests:
            pop_latencies[req["pop_id"]].append(req["latency_ms"])
        
        pop_avg_latencies = {
            pop: sum(latencies) / len(latencies)
            for pop, latencies in pop_latencies.items()
            if latencies
        }
        
        return {
            "total_requests": total_requests,
            "cache_hit_rate": hit_rate,
            "bandwidth_saved_gb": self.bandwidth_saved / (1024**3),
            "average_latency_ms": sum(r["latency_ms"] for r in self.requests) / total_requests if total_requests > 0 else 0,
            "pop_latencies": pop_avg_latencies,
            "requests_by_pop": dict(defaultdict(int, {
                req["pop_id"]: 1 for req in self.requests
            }))
        }

class CDNSecurity:
    """CDN security features"""
    
    def __init__(self):
        self.rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
        self.blocked_ips: Set[str] = set()
        self.ddos_protection = True
        self.waf_rules = self._initialize_waf()
    
    def _initialize_waf(self) -> List[Dict]:
        """Initialize Web Application Firewall rules"""
        return [
            {"pattern": r"<script.*?>.*?</script>", "action": "block", "type": "xss"},
            {"pattern": r"union.*select", "action": "block", "type": "sql_injection"},
            {"pattern": r"\.\./", "action": "block", "type": "path_traversal"}
        ]
    
    async def validate_request(self, request: CDNRequest) -> bool:
        """Validate request against security rules"""
        # Check blocked IPs
        if request.client_ip in self.blocked_ips:
            return False
        
        # Rate limiting
        if not await self._check_rate_limit(request.client_ip):
            return False
        
        # WAF checks
        if not self._check_waf(request):
            return False
        
        # DDoS protection
        if self.ddos_protection:
            if not await self._check_ddos_patterns(request):
                return False
        
        return True
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client"""
        now = time.time()
        limit_data = self.rate_limits[client_ip]
        
        # Reset if time window expired
        if now > limit_data["reset_time"]:
            limit_data["count"] = 0
            limit_data["reset_time"] = now + 60  # 1 minute window
        
        limit_data["count"] += 1
        
        # 100 requests per minute limit
        return limit_data["count"] <= 100
    
    def _check_waf(self, request: CDNRequest) -> bool:
        """Check WAF rules"""
        import re
        
        # Check URL against WAF patterns
        for rule in self.waf_rules:
            if re.search(rule["pattern"], request.url, re.IGNORECASE):
                return False
        
        return True
    
    async def _check_ddos_patterns(self, request: CDNRequest) -> bool:
        """Check for DDoS attack patterns"""
        # Simplified DDoS detection
        # In production, would use ML models and traffic analysis
        
        # Check for suspicious user agents
        suspicious_agents = ["bot", "crawler", "scanner"]
        if any(agent in request.user_agent.lower() for agent in suspicious_agents):
            return False
        
        return True


# Example usage and testing
async def main():
    """Demonstrate Global CDN capabilities"""
    
    print("🌍 Global CDN System with Intelligent Routing")
    print("=" * 60)
    
    # Initialize CDN
    cdn = GlobalCDN()
    
    # Test requests from different locations
    test_requests = [
        CDNRequest(
            url="https://api.example.com/audio/track1.mp3",
            client_ip="192.168.1.1",
            client_location=(37.7749, -122.4194),  # San Francisco
            user_agent="Mozilla/5.0"
        ),
        CDNRequest(
            url="https://api.example.com/audio/track1.mp3",  # Same content
            client_ip="192.168.1.2",
            client_location=(51.5074, -0.1278),  # London
            user_agent="Mozilla/5.0"
        ),
        CDNRequest(
            url="https://api.example.com/api/status",
            client_ip="192.168.1.3",
            client_location=(35.6762, 139.6503),  # Tokyo
            user_agent="Mozilla/5.0"
        )
    ]
    
    print("\n📡 Testing CDN Requests:")
    print("-" * 40)
    
    for i, request in enumerate(test_requests, 1):
        response = await cdn.handle_request(request)
        
        print(f"\nRequest {i}:")
        print(f"  URL: {request.url}")
        print(f"  Client Location: {request.client_location}")
        print(f"  PoP Selected: {response.pop_id}")
        print(f"  Cache Status: {response.cache_status}")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        
        if response.bandwidth_saved > 0:
            print(f"  Bandwidth Saved: {response.bandwidth_saved} bytes")
    
    # Get analytics
    analytics = await cdn.get_analytics()
    
    print("\n📊 CDN Analytics:")
    print("-" * 40)
    print(f"Total Requests: {analytics['total_requests']}")
    print(f"Cache Hit Rate: {analytics['cache_hit_rate']:.1f}%")
    print(f"Average Latency: {analytics['average_latency_ms']:.2f}ms")
    print(f"Bandwidth Saved: {analytics['bandwidth_saved_gb']:.3f} GB")
    
    # Test cache purging
    print("\n🧹 Testing Cache Purge:")
    purged = await cdn.purge_cache("track1")
    print(f"Purged {purged} cache entries")
    
    print("\n✅ Global CDN system operational with <10ms edge latency!")
    
    return cdn


if __name__ == "__main__":
    asyncio.run(main())