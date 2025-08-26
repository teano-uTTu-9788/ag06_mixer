#!/usr/bin/env python3
"""
AI Mixer Health Check Service

Provides comprehensive health monitoring for all AI Mixer components:
- Regional endpoint health checks
- Service availability monitoring
- Performance metric collection
- Automated failover triggering
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status for a service endpoint"""
    endpoint: str
    region: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    error_message: Optional[str] = None
    last_check: Optional[str] = None
    uptime_percentage: float = 100.0

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    total_endpoints: int
    healthy_endpoints: int
    degraded_endpoints: int
    unhealthy_endpoints: int
    average_response_time: float
    overall_status: str
    timestamp: str

class HealthCheckService:
    """Service for monitoring AI Mixer component health"""
    
    def __init__(self):
        self.endpoints = {
            "us-west": "https://us-west.aimixer.com",
            "us-east": "https://us-east.aimixer.com",
            "eu-west": "https://eu-west.aimixer.com",
            "asia-pacific": "https://ap.aimixer.com",
            "edge": "https://edge.aimixer.com",
            "global": "https://api.aimixer.com"
        }
        
        self.health_history: Dict[str, List[HealthStatus]] = {}
        self.alert_thresholds = {
            "response_time_warning": 100,  # ms
            "response_time_critical": 500,  # ms
            "uptime_warning": 95.0,  # percentage
            "uptime_critical": 90.0   # percentage
        }
        
        # Initialize history for each endpoint
        for region in self.endpoints.keys():
            self.health_history[region] = []
    
    async def check_endpoint_health(self, session: aiohttp.ClientSession, 
                                  region: str, endpoint: str) -> HealthStatus:
        """Check health of a single endpoint"""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with session.get(f"{endpoint}/health", timeout=timeout) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Determine status based on response time and service health
                    if response_time > self.alert_thresholds["response_time_critical"]:
                        status = "degraded"
                        error_msg = f"High latency: {response_time:.1f}ms"
                    elif response_time > self.alert_thresholds["response_time_warning"]:
                        status = "degraded"
                        error_msg = f"Elevated latency: {response_time:.1f}ms"
                    else:
                        # Check service-specific health indicators
                        service_status = data.get("status", "unknown")
                        if service_status == "healthy":
                            status = "healthy"
                            error_msg = None
                        else:
                            status = "degraded"
                            error_msg = f"Service reports: {service_status}"
                else:
                    response_time = (time.time() - start_time) * 1000
                    status = "unhealthy"
                    error_msg = f"HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            response_time = 10000  # Timeout time
            status = "unhealthy"
            error_msg = "Request timeout"
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = "unhealthy"
            error_msg = str(e)
        
        return HealthStatus(
            endpoint=endpoint,
            region=region,
            status=status,
            response_time_ms=response_time,
            error_message=error_msg,
            last_check=datetime.utcnow().isoformat(),
            uptime_percentage=self.calculate_uptime(region)
        )
    
    async def check_all_endpoints(self) -> List[HealthStatus]:
        """Check health of all endpoints"""
        results = []
        
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            
            for region, endpoint in self.endpoints.items():
                task = self.check_endpoint_health(session, region, endpoint)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            health_statuses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    region = list(self.endpoints.keys())[i]
                    logger.error(f"Health check failed for {region}: {result}")
                    
                    # Create unhealthy status for failed check
                    health_statuses.append(HealthStatus(
                        endpoint=self.endpoints[region],
                        region=region,
                        status="unhealthy",
                        response_time_ms=0,
                        error_message=f"Check failed: {str(result)}",
                        last_check=datetime.utcnow().isoformat(),
                        uptime_percentage=self.calculate_uptime(region)
                    ))
                else:
                    health_statuses.append(result)
        
        # Update history
        for status in health_statuses:
            self.health_history[status.region].append(status)
            
            # Keep only last 1000 records per region
            if len(self.health_history[status.region]) > 1000:
                self.health_history[status.region] = self.health_history[status.region][-1000:]
        
        return health_statuses
    
    def calculate_uptime(self, region: str) -> float:
        """Calculate uptime percentage for a region"""
        if not self.health_history.get(region):
            return 100.0
        
        # Consider last 100 checks or 24 hours, whichever is smaller
        recent_checks = self.health_history[region][-100:]
        
        if not recent_checks:
            return 100.0
        
        # Filter checks from last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_checks = [
            check for check in recent_checks
            if datetime.fromisoformat(check.last_check.replace('Z', '+00:00')) > cutoff_time
        ]
        
        if not recent_checks:
            return 100.0
        
        healthy_checks = len([check for check in recent_checks if check.status == "healthy"])
        return (healthy_checks / len(recent_checks)) * 100.0
    
    def generate_system_metrics(self, health_statuses: List[HealthStatus]) -> SystemMetrics:
        """Generate system-wide metrics"""
        total = len(health_statuses)
        healthy = len([s for s in health_statuses if s.status == "healthy"])
        degraded = len([s for s in health_statuses if s.status == "degraded"])
        unhealthy = len([s for s in health_statuses if s.status == "unhealthy"])
        
        # Calculate average response time (excluding failed requests)
        response_times = [s.response_time_ms for s in health_statuses if s.response_time_ms > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Determine overall status
        if unhealthy > 0:
            overall_status = "unhealthy"
        elif degraded > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return SystemMetrics(
            total_endpoints=total,
            healthy_endpoints=healthy,
            degraded_endpoints=degraded,
            unhealthy_endpoints=unhealthy,
            average_response_time=avg_response_time,
            overall_status=overall_status,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def check_alert_conditions(self, health_statuses: List[HealthStatus]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        for status in health_statuses:
            # High latency alerts
            if status.response_time_ms > self.alert_thresholds["response_time_critical"]:
                alerts.append({
                    "severity": "critical",
                    "type": "high_latency",
                    "region": status.region,
                    "message": f"Critical latency: {status.response_time_ms:.1f}ms",
                    "threshold": self.alert_thresholds["response_time_critical"],
                    "value": status.response_time_ms
                })
            elif status.response_time_ms > self.alert_thresholds["response_time_warning"]:
                alerts.append({
                    "severity": "warning",
                    "type": "high_latency",
                    "region": status.region,
                    "message": f"High latency: {status.response_time_ms:.1f}ms",
                    "threshold": self.alert_thresholds["response_time_warning"],
                    "value": status.response_time_ms
                })
            
            # Uptime alerts
            if status.uptime_percentage < self.alert_thresholds["uptime_critical"]:
                alerts.append({
                    "severity": "critical",
                    "type": "low_uptime",
                    "region": status.region,
                    "message": f"Critical uptime: {status.uptime_percentage:.1f}%",
                    "threshold": self.alert_thresholds["uptime_critical"],
                    "value": status.uptime_percentage
                })
            elif status.uptime_percentage < self.alert_thresholds["uptime_warning"]:
                alerts.append({
                    "severity": "warning",
                    "type": "low_uptime",
                    "region": status.region,
                    "message": f"Low uptime: {status.uptime_percentage:.1f}%",
                    "threshold": self.alert_thresholds["uptime_warning"],
                    "value": status.uptime_percentage
                })
            
            # Service down alerts
            if status.status == "unhealthy":
                alerts.append({
                    "severity": "critical",
                    "type": "service_down",
                    "region": status.region,
                    "message": f"Service down: {status.error_message}",
                    "threshold": None,
                    "value": status.status
                })
        
        return alerts
    
    async def export_metrics_to_file(self, health_statuses: List[HealthStatus], 
                                   metrics: SystemMetrics, alerts: List[Dict[str, Any]]):
        """Export metrics to JSON file for Prometheus scraping"""
        export_data = {
            "health_statuses": [asdict(status) for status in health_statuses],
            "system_metrics": asdict(metrics),
            "alerts": alerts,
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        # Write to file
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/health_check.json", "w") as f:
            json.dump(export_data, f, indent=2)
    
    async def run_health_check_cycle(self):
        """Run a complete health check cycle"""
        logger.info("Starting health check cycle")
        
        # Check all endpoints
        health_statuses = await self.check_all_endpoints()
        
        # Generate metrics
        metrics = self.generate_system_metrics(health_statuses)
        
        # Check for alerts
        alerts = self.check_alert_conditions(health_statuses)
        
        # Export metrics
        await self.export_metrics_to_file(health_statuses, metrics, alerts)
        
        # Log summary
        logger.info(f"Health check completed: {metrics.overall_status} "
                   f"({metrics.healthy_endpoints}/{metrics.total_endpoints} healthy, "
                   f"avg response time: {metrics.average_response_time:.1f}ms)")
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                logger.warning(f"  {alert['severity'].upper()}: {alert['message']}")
        
        return health_statuses, metrics, alerts
    
    async def run_continuous_monitoring(self, interval_seconds: int = 30):
        """Run continuous health monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                await self.run_health_check_cycle()
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Health check cycle failed: {e}")
                await asyncio.sleep(interval_seconds)

async def main():
    """Main function"""
    service = HealthCheckService()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run single health check
        await service.run_health_check_cycle()
    else:
        # Run continuous monitoring
        interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        await service.run_continuous_monitoring(interval)

if __name__ == "__main__":
    asyncio.run(main())