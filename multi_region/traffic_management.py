#!/usr/bin/env python3
"""
Global Traffic Management System for AI Mixer Multi-Region Deployment

Provides intelligent traffic routing, health monitoring, and failover
capabilities across multiple geographic regions.
"""

import asyncio
import aiohttp
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class RoutingStrategy(Enum):
    GEOLOCATION = "geolocation"
    LATENCY_BASED = "latency_based"
    WEIGHTED = "weighted"
    FAILOVER = "failover"

@dataclass
class RegionEndpoint:
    name: str
    url: str
    region: str
    priority: int
    weight: int
    status: RegionStatus = RegionStatus.HEALTHY
    last_check: float = 0.0
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0

@dataclass
class HealthMetrics:
    endpoint: str
    response_time_ms: float
    http_status: int
    timestamp: float
    error_message: Optional[str] = None

@dataclass
class RoutingRule:
    strategy: RoutingStrategy
    countries: List[str]
    regions: List[str]
    weight: int = 100
    enabled: bool = True

class GlobalTrafficManager:
    """Global traffic management and load balancing system"""
    
    def __init__(self, config_file: str = None):
        self.regions: Dict[str, RegionEndpoint] = {}
        self.routing_rules: List[RoutingRule] = []
        self.health_metrics: List[HealthMetrics] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.config = self._load_default_config()
        
        if config_file:
            self._load_config(config_file)
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            "health_check_interval": 30,
            "health_timeout": 10,
            "max_retries": 3,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_reset_time": 300,
            "latency_threshold_ms": 1000,
            "error_rate_threshold": 0.1,
        }
    
    def _load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
    
    async def initialize(self):
        """Initialize the traffic manager"""
        # Create HTTP session with timeouts
        timeout = aiohttp.ClientTimeout(total=self.config["health_timeout"])
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Add default regions
        await self._setup_default_regions()
        
        # Setup default routing rules
        self._setup_default_routing_rules()
        
        # Start health monitoring
        self.monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("Global Traffic Manager initialized with %d regions", len(self.regions))
    
    async def _setup_default_regions(self):
        """Setup default regional endpoints"""
        default_regions = [
            RegionEndpoint(
                name="us-west-1",
                url="https://us-west-1.aimixer.com",
                region="us-west",
                priority=1,
                weight=100
            ),
            RegionEndpoint(
                name="us-west-2", 
                url="https://us-west-2.aimixer.com",
                region="us-west",
                priority=2,
                weight=100
            ),
            RegionEndpoint(
                name="us-east-1",
                url="https://us-east-1.aimixer.com",
                region="us-east",
                priority=1,
                weight=100
            ),
            RegionEndpoint(
                name="us-east-2",
                url="https://us-east-2.aimixer.com", 
                region="us-east",
                priority=2,
                weight=100
            ),
            RegionEndpoint(
                name="eu-west-1",
                url="https://eu-west-1.aimixer.com",
                region="eu-west",
                priority=1,
                weight=100
            ),
            RegionEndpoint(
                name="eu-west-2",
                url="https://eu-west-2.aimixer.com",
                region="eu-west", 
                priority=2,
                weight=100
            ),
            RegionEndpoint(
                name="ap-southeast-1",
                url="https://ap-southeast-1.aimixer.com",
                region="asia-pacific",
                priority=1,
                weight=100
            ),
            RegionEndpoint(
                name="ap-northeast-1",
                url="https://ap-northeast-1.aimixer.com",
                region="asia-pacific",
                priority=2,
                weight=100
            )
        ]
        
        for region in default_regions:
            self.regions[region.name] = region
    
    def _setup_default_routing_rules(self):
        """Setup default routing rules"""
        self.routing_rules = [
            RoutingRule(
                strategy=RoutingStrategy.GEOLOCATION,
                countries=["US", "CA", "MX"],
                regions=["us-west", "us-east"]
            ),
            RoutingRule(
                strategy=RoutingStrategy.GEOLOCATION,
                countries=["GB", "FR", "DE", "NL", "ES", "IT"],
                regions=["eu-west"]
            ),
            RoutingRule(
                strategy=RoutingStrategy.GEOLOCATION,
                countries=["JP", "KR", "SG", "AU", "IN"],
                regions=["asia-pacific"]
            ),
            RoutingRule(
                strategy=RoutingStrategy.LATENCY_BASED,
                countries=["*"],  # Fallback for all countries
                regions=["us-west", "us-east", "eu-west", "asia-pacific"]
            )
        ]
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                await self._check_all_regions_health()
                await asyncio.sleep(self.config["health_check_interval"])
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_all_regions_health(self):
        """Check health of all regions concurrently"""
        tasks = []
        for endpoint in self.regions.values():
            task = asyncio.create_task(self._check_region_health(endpoint))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_region_health(self, endpoint: RegionEndpoint):
        """Check health of a single region endpoint"""
        start_time = time.time()
        
        try:
            # Make health check request
            async with self.session.get(f"{endpoint.url}/health") as response:
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Record metrics
                metrics = HealthMetrics(
                    endpoint=endpoint.name,
                    response_time_ms=response_time,
                    http_status=response.status,
                    timestamp=time.time()
                )
                
                if response.status == 200:
                    # Successful health check
                    endpoint.status = RegionStatus.HEALTHY
                    endpoint.response_time = response_time
                    endpoint.success_count += 1
                    endpoint.error_count = max(0, endpoint.error_count - 1)  # Gradual recovery
                    
                    # Check if response time indicates degradation
                    if response_time > self.config["latency_threshold_ms"]:
                        endpoint.status = RegionStatus.DEGRADED
                        logger.warning(f"Region {endpoint.name} degraded - high latency: {response_time:.1f}ms")
                    
                else:
                    # Failed health check
                    endpoint.error_count += 1
                    metrics.error_message = f"HTTP {response.status}"
                    
                    if endpoint.error_count >= self.config["circuit_breaker_threshold"]:
                        endpoint.status = RegionStatus.UNHEALTHY
                        logger.error(f"Region {endpoint.name} marked unhealthy - error count: {endpoint.error_count}")
                    else:
                        endpoint.status = RegionStatus.DEGRADED
                
                self.health_metrics.append(metrics)
                endpoint.last_check = time.time()
                
        except asyncio.TimeoutError:
            # Timeout
            response_time = (time.time() - start_time) * 1000
            endpoint.error_count += 1
            endpoint.status = RegionStatus.DEGRADED if endpoint.error_count < 3 else RegionStatus.UNHEALTHY
            
            metrics = HealthMetrics(
                endpoint=endpoint.name,
                response_time_ms=response_time,
                http_status=0,
                timestamp=time.time(),
                error_message="Timeout"
            )
            self.health_metrics.append(metrics)
            logger.warning(f"Region {endpoint.name} health check timeout")
            
        except Exception as e:
            # Other errors
            endpoint.error_count += 1
            endpoint.status = RegionStatus.OFFLINE
            
            metrics = HealthMetrics(
                endpoint=endpoint.name,
                response_time_ms=0,
                http_status=0,
                timestamp=time.time(),
                error_message=str(e)
            )
            self.health_metrics.append(metrics)
            logger.error(f"Region {endpoint.name} health check failed: {e}")
        
        # Limit metrics history
        if len(self.health_metrics) > 1000:
            self.health_metrics = self.health_metrics[-500:]
    
    def get_best_endpoints_for_country(self, country_code: str, strategy: RoutingStrategy = None) -> List[RegionEndpoint]:
        """Get best endpoints for a given country"""
        
        # Find applicable routing rules
        applicable_rules = []
        for rule in self.routing_rules:
            if rule.enabled and (country_code in rule.countries or "*" in rule.countries):
                if strategy is None or rule.strategy == strategy:
                    applicable_rules.append(rule)
        
        if not applicable_rules:
            # Fallback to all healthy regions
            return self._get_healthy_endpoints()
        
        # Get candidate endpoints from rules
        candidate_regions = set()
        for rule in applicable_rules:
            candidate_regions.update(rule.regions)
        
        # Filter to healthy endpoints in candidate regions
        candidates = []
        for endpoint in self.regions.values():
            if endpoint.region in candidate_regions and endpoint.status in [RegionStatus.HEALTHY, RegionStatus.DEGRADED]:
                candidates.append(endpoint)
        
        # Sort by priority and health
        candidates.sort(key=lambda e: (e.priority, e.response_time, -e.success_count))
        
        return candidates
    
    def _get_healthy_endpoints(self) -> List[RegionEndpoint]:
        """Get all healthy endpoints"""
        healthy = [e for e in self.regions.values() if e.status in [RegionStatus.HEALTHY, RegionStatus.DEGRADED]]
        healthy.sort(key=lambda e: (e.priority, e.response_time))
        return healthy
    
    def get_endpoint_for_request(self, country_code: str = "US", client_ip: str = None) -> Optional[RegionEndpoint]:
        """Get the best endpoint for a specific request"""
        
        # Try geolocation-based routing first
        candidates = self.get_best_endpoints_for_country(country_code, RoutingStrategy.GEOLOCATION)
        
        if not candidates:
            # Fallback to latency-based routing
            candidates = self.get_best_endpoints_for_country(country_code, RoutingStrategy.LATENCY_BASED)
        
        if not candidates:
            # Final fallback to any healthy endpoint
            candidates = self._get_healthy_endpoints()
        
        if not candidates:
            logger.error("No healthy endpoints available")
            return None
        
        # For now, return the best candidate (could add weighted selection here)
        return candidates[0]
    
    def get_region_statistics(self) -> Dict:
        """Get comprehensive region statistics"""
        stats = {
            "regions": {},
            "overall": {
                "total_endpoints": len(self.regions),
                "healthy_endpoints": 0,
                "degraded_endpoints": 0,
                "unhealthy_endpoints": 0,
                "offline_endpoints": 0
            }
        }
        
        for endpoint in self.regions.values():
            # Per-region stats
            region_stats = {
                "name": endpoint.name,
                "url": endpoint.url,
                "region": endpoint.region,
                "status": endpoint.status.value,
                "priority": endpoint.priority,
                "weight": endpoint.weight,
                "response_time_ms": round(endpoint.response_time, 2),
                "success_count": endpoint.success_count,
                "error_count": endpoint.error_count,
                "last_check": endpoint.last_check
            }
            
            # Recent metrics
            recent_metrics = [m for m in self.health_metrics if m.endpoint == endpoint.name and m.timestamp > time.time() - 300]
            if recent_metrics:
                response_times = [m.response_time_ms for m in recent_metrics if m.response_time_ms > 0]
                if response_times:
                    region_stats["avg_response_time_5min"] = round(statistics.mean(response_times), 2)
                    region_stats["p95_response_time_5min"] = round(statistics.quantiles(response_times, n=20)[18], 2) if len(response_times) > 5 else region_stats["response_time_ms"]
            
            stats["regions"][endpoint.name] = region_stats
            
            # Overall stats
            if endpoint.status == RegionStatus.HEALTHY:
                stats["overall"]["healthy_endpoints"] += 1
            elif endpoint.status == RegionStatus.DEGRADED:
                stats["overall"]["degraded_endpoints"] += 1
            elif endpoint.status == RegionStatus.UNHEALTHY:
                stats["overall"]["unhealthy_endpoints"] += 1
            else:
                stats["overall"]["offline_endpoints"] += 1
        
        return stats
    
    async def shutdown(self):
        """Shutdown the traffic manager"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("Global Traffic Manager shutdown complete")

class TrafficManagerAPI:
    """REST API for traffic management"""
    
    def __init__(self, traffic_manager: GlobalTrafficManager):
        self.traffic_manager = traffic_manager
    
    async def handle_route_request(self, request_data: Dict) -> Dict:
        """Handle routing request"""
        country = request_data.get("country", "US")
        client_ip = request_data.get("client_ip")
        
        endpoint = self.traffic_manager.get_endpoint_for_request(country, client_ip)
        
        if endpoint:
            return {
                "success": True,
                "endpoint": {
                    "name": endpoint.name,
                    "url": endpoint.url,
                    "region": endpoint.region,
                    "status": endpoint.status.value,
                    "response_time_ms": round(endpoint.response_time, 2)
                }
            }
        else:
            return {
                "success": False,
                "error": "No healthy endpoints available"
            }
    
    async def handle_health_request(self) -> Dict:
        """Handle health check request"""
        stats = self.traffic_manager.get_region_statistics()
        
        return {
            "status": "healthy" if stats["overall"]["healthy_endpoints"] > 0 else "degraded",
            "timestamp": time.time(),
            "statistics": stats
        }

# Example usage and testing
async def main():
    """Example usage of the Global Traffic Manager"""
    
    # Initialize traffic manager
    traffic_manager = GlobalTrafficManager()
    await traffic_manager.initialize()
    
    # Create API handler
    api = TrafficManagerAPI(traffic_manager)
    
    try:
        # Wait for initial health checks
        await asyncio.sleep(5)
        
        # Test routing for different countries
        test_countries = ["US", "GB", "JP", "AU"]
        
        for country in test_countries:
            request_data = {"country": country}
            response = await api.handle_route_request(request_data)
            print(f"Routing for {country}: {response}")
        
        # Get health statistics
        health_response = await api.handle_health_request()
        print(f"\nHealth Statistics:")
        print(json.dumps(health_response, indent=2))
        
        # Run for a minute to collect metrics
        print("\nMonitoring for 60 seconds...")
        await asyncio.sleep(60)
        
        # Final statistics
        final_stats = await api.handle_health_request()
        print(f"\nFinal Statistics:")
        print(json.dumps(final_stats, indent=2))
        
    finally:
        await traffic_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())