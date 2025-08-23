#!/usr/bin/env python3
"""
AG06 Microservices Architecture 2025
Netflix-Inspired Microservices Implementation

Key Principles:
- Single Responsibility per service
- Independent deployment capability  
- Event-driven communication
- Circuit breaker resilience
- Service discovery and health checks
- Distributed tracing and observability
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Protocol
import aiohttp
from aiohttp import web, ClientSession
import weakref

# Service Discovery and Registry (Netflix Eureka-inspired)
class ServiceRegistryProtocol(Protocol):
    async def register_service(self, service_info: 'ServiceInfo') -> bool: ...
    async def discover_services(self, service_type: str) -> List['ServiceInfo']: ...
    async def deregister_service(self, service_id: str) -> bool: ...

@dataclass
class ServiceInfo:
    """Service registration information"""
    service_id: str
    service_type: str
    host: str
    port: int
    health_endpoint: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)

class ServiceRegistry:
    """Netflix Eureka-inspired service registry"""
    
    def __init__(self, heartbeat_interval: int = 30):
        self.services: Dict[str, ServiceInfo] = {}
        self.heartbeat_interval = heartbeat_interval
        self.is_running = False
        
    async def start(self):
        """Start the service registry"""
        self.is_running = True
        asyncio.create_task(self._cleanup_stale_services())
        
    async def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service instance"""
        try:
            self.services[service_info.service_id] = service_info
            logging.info(f"Registered service: {service_info.service_type} ({service_info.service_id})")
            return True
        except Exception as e:
            logging.error(f"Failed to register service {service_info.service_id}: {e}")
            return False
    
    async def discover_services(self, service_type: str) -> List[ServiceInfo]:
        """Discover services by type"""
        return [
            service for service in self.services.values()
            if service.service_type == service_type
        ]
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        if service_id in self.services:
            del self.services[service_id]
            logging.info(f"Deregistered service: {service_id}")
            return True
        return False
    
    async def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat"""
        if service_id in self.services:
            self.services[service_id].last_heartbeat = datetime.now()
            return True
        return False
    
    async def _cleanup_stale_services(self):
        """Remove services that haven't sent heartbeats"""
        while self.is_running:
            try:
                stale_cutoff = datetime.now() - timedelta(seconds=self.heartbeat_interval * 3)
                stale_services = [
                    service_id for service_id, service in self.services.items()
                    if service.last_heartbeat < stale_cutoff
                ]
                
                for service_id in stale_services:
                    await self.deregister_service(service_id)
                    logging.warning(f"Removed stale service: {service_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logging.error(f"Error in service cleanup: {e}")
                await asyncio.sleep(5)

# Base Microservice Framework
class BaseService(ABC):
    """Abstract base class for all microservices"""
    
    def __init__(self, 
                 service_type: str,
                 port: int,
                 registry: ServiceRegistry,
                 health_check_interval: int = 30):
        
        self.service_id = f"{service_type}-{uuid.uuid4().hex[:8]}"
        self.service_type = service_type
        self.port = port
        self.registry = registry
        self.health_check_interval = health_check_interval
        
        # Service state
        self.is_healthy = True
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.request_count = 0
        self.error_count = 0
        
        # HTTP server
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/info', self.service_info)
        self.app.router.add_post('/shutdown', self.shutdown)
        
        # Add service-specific routes
        self.setup_service_routes()
    
    @abstractmethod
    def setup_service_routes(self):
        """Setup service-specific routes"""
        pass
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process service-specific request"""
        pass
    
    async def start(self):
        """Start the microservice"""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            # Register with service registry
            service_info = ServiceInfo(
                service_id=self.service_id,
                service_type=self.service_type,
                host="localhost",
                port=self.port,
                health_endpoint=f"http://localhost:{self.port}/health",
                metadata={
                    "start_time": self.start_time.isoformat(),
                    "version": "1.0.0"
                }
            )
            
            await self.registry.register_service(service_info)
            
            # Start health check heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start HTTP server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logging.info(f"Started {self.service_type} service on port {self.port}")
            
        except Exception as e:
            logging.error(f"Failed to start service {self.service_type}: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the microservice"""
        self.is_running = False
        await self.registry.deregister_service(self.service_id)
        logging.info(f"Stopped {self.service_type} service")
    
    async def health_check(self, request):
        """Health check endpoint"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        health_data = {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "status": "healthy" if self.is_healthy else "unhealthy",
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add service-specific health metrics
        service_health = await self.get_service_health()
        health_data.update(service_health)
        
        status_code = 200 if self.is_healthy else 503
        return web.json_response(health_data, status=status_code)
    
    async def service_info(self, request):
        """Service information endpoint"""
        return web.json_response({
            "service_id": self.service_id,
            "service_type": self.service_type,
            "port": self.port,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "version": "1.0.0"
        })
    
    async def shutdown(self, request):
        """Graceful shutdown endpoint"""
        asyncio.create_task(self.stop())
        return web.json_response({"status": "shutting_down"})
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to registry"""
        while self.is_running:
            try:
                await self.registry.update_heartbeat(self.service_id)
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Heartbeat failed for {self.service_id}: {e}")
                await asyncio.sleep(5)
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service-specific health metrics"""
        pass

# Audio Processing Microservice
class AudioProcessingService(BaseService):
    """Dedicated service for audio processing operations"""
    
    def __init__(self, registry: ServiceRegistry):
        super().__init__("audio-processing", 8001, registry)
        self.processing_queue = asyncio.Queue(maxsize=100)
        self.active_sessions: Set[str] = set()
        
    def setup_service_routes(self):
        """Setup audio processing routes"""
        self.app.router.add_post('/process', self.handle_audio_process)
        self.app.router.add_post('/enhance', self.handle_audio_enhance)
        self.app.router.add_get('/sessions', self.get_active_sessions)
        
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio-related requests"""
        operation = request_data.get("operation")
        
        if operation == "process":
            return await self._process_audio(request_data)
        elif operation == "enhance":
            return await self._enhance_audio(request_data)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def handle_audio_process(self, request):
        """Handle audio processing requests"""
        try:
            self.request_count += 1
            data = await request.json()
            result = await self._process_audio(data)
            return web.json_response(result)
        except Exception as e:
            self.error_count += 1
            logging.error(f"Audio processing error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_audio_enhance(self, request):
        """Handle audio enhancement requests"""
        try:
            self.request_count += 1
            data = await request.json()
            result = await self._enhance_audio(data)
            return web.json_response(result)
        except Exception as e:
            self.error_count += 1
            logging.error(f"Audio enhancement error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_active_sessions(self, request):
        """Get active processing sessions"""
        return web.json_response({
            "active_sessions": list(self.active_sessions),
            "queue_size": self.processing_queue.qsize()
        })
    
    async def _process_audio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Core audio processing logic"""
        session_id = data.get("session_id", str(uuid.uuid4()))
        self.active_sessions.add(session_id)
        
        try:
            # Simulate audio processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                "session_id": session_id,
                "status": "processed",
                "processing_time_ms": 100,
                "output_format": "48kHz/24bit",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.active_sessions.discard(session_id)
    
    async def _enhance_audio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Audio enhancement with AI processing"""
        session_id = data.get("session_id", str(uuid.uuid4()))
        enhancement_type = data.get("type", "general")
        
        self.active_sessions.add(session_id)
        
        try:
            # Simulate AI enhancement
            await asyncio.sleep(0.2)  # AI processing takes longer
            
            return {
                "session_id": session_id,
                "status": "enhanced",
                "enhancement_type": enhancement_type,
                "processing_time_ms": 200,
                "improvements": {
                    "noise_reduction": "12dB",
                    "clarity_boost": "15%",
                    "dynamic_range": "expanded"
                },
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.active_sessions.discard(session_id)
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Audio service specific health metrics"""
        return {
            "queue_size": self.processing_queue.qsize(),
            "active_sessions": len(self.active_sessions),
            "queue_full": self.processing_queue.full(),
            "processing_capacity": "normal" if len(self.active_sessions) < 10 else "high"
        }

# Hardware Control Microservice
class HardwareControlService(BaseService):
    """Dedicated service for AG06 hardware control"""
    
    def __init__(self, registry: ServiceRegistry):
        super().__init__("hardware-control", 8002, registry)
        self.device_status = {
            "connected": True,
            "monitor_mute": False,
            "levels": {"usb": 3.0, "channel_1": 3.0, "channel_2": 3.0, "monitor": 3.0},
            "to_pc_switch": "DRY CH1-2"
        }
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        
    def setup_service_routes(self):
        """Setup hardware control routes"""
        self.app.router.add_get('/status', self.get_hardware_status)
        self.app.router.add_post('/configure', self.configure_hardware)
        self.app.router.add_post('/diagnostics', self.run_diagnostics)
        self.app.router.add_post('/troubleshoot', self.get_troubleshooting)
        
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process hardware control requests"""
        operation = request_data.get("operation")
        
        if operation == "status":
            return await self._get_status()
        elif operation == "configure":
            return await self._configure(request_data.get("config", {}))
        elif operation == "diagnostics":
            return await self._run_diagnostics()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def get_hardware_status(self, request):
        """Get current hardware status"""
        return web.json_response(self.device_status)
    
    async def configure_hardware(self, request):
        """Configure hardware settings"""
        try:
            self.request_count += 1
            config = await request.json()
            result = await self._configure(config)
            return web.json_response(result)
        except Exception as e:
            self.error_count += 1
            return web.json_response({"error": str(e)}, status=500)
    
    async def run_diagnostics(self, request):
        """Run hardware diagnostics"""
        try:
            self.request_count += 1
            result = await self._run_diagnostics()
            self.last_diagnostics = result
            return web.json_response(result)
        except Exception as e:
            self.error_count += 1
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_troubleshooting(self, request):
        """Get troubleshooting recommendations"""
        try:
            data = await request.json()
            issue = data.get("issue", "general")
            recommendations = await self._get_troubleshooting_steps(issue)
            return web.json_response(recommendations)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get detailed hardware status"""
        return {
            "device_status": self.device_status,
            "last_update": datetime.now().isoformat(),
            "diagnostics_available": self.last_diagnostics is not None
        }
    
    async def _configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hardware configuration"""
        # Update device status with new configuration
        if "levels" in config:
            self.device_status["levels"].update(config["levels"])
        
        if "monitor_mute" in config:
            self.device_status["monitor_mute"] = config["monitor_mute"]
        
        if "to_pc_switch" in config:
            self.device_status["to_pc_switch"] = config["to_pc_switch"]
        
        return {
            "status": "configured",
            "applied_config": config,
            "current_status": self.device_status,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive hardware diagnostics"""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Check 1: Monitor Mute
        mute_check = {
            "name": "Monitor Mute Status",
            "status": "failed" if self.device_status["monitor_mute"] else "passed",
            "message": "Monitor is muted - no audio output" if self.device_status["monitor_mute"] else "Monitor active",
            "critical": self.device_status["monitor_mute"]
        }
        diagnostics["checks"].append(mute_check)
        
        # Check 2: Level Settings
        levels = self.device_status["levels"]
        low_levels = {k: v for k, v in levels.items() if v < 2.0}
        level_check = {
            "name": "Output Levels",
            "status": "warning" if low_levels else "passed",
            "message": f"Low levels detected: {list(low_levels.keys())}" if low_levels else "All levels adequate",
            "critical": False
        }
        diagnostics["checks"].append(level_check)
        
        # Check 3: Hardware Connection
        connection_check = {
            "name": "Hardware Connection",
            "status": "passed" if self.device_status["connected"] else "failed",
            "message": "AG06 connected via USB" if self.device_status["connected"] else "AG06 not detected",
            "critical": not self.device_status["connected"]
        }
        diagnostics["checks"].append(connection_check)
        
        # Determine overall status
        has_critical = any(check["critical"] for check in diagnostics["checks"])
        has_warnings = any(check["status"] == "warning" for check in diagnostics["checks"])
        
        if has_critical:
            diagnostics["overall_status"] = "critical"
        elif has_warnings:
            diagnostics["overall_status"] = "warning"
        else:
            diagnostics["overall_status"] = "healthy"
        
        # Generate recommendations
        if self.device_status["monitor_mute"]:
            diagnostics["recommendations"].append("Press Monitor Mute button to enable audio output")
        
        if low_levels:
            diagnostics["recommendations"].append(f"Increase these levels to 3-5 range: {', '.join(low_levels.keys())}")
        
        if not self.device_status["connected"]:
            diagnostics["recommendations"].extend([
                "Check USB connection to AG06",
                "Verify AG06 power connection",
                "Try different USB port"
            ])
        
        return diagnostics
    
    async def _get_troubleshooting_steps(self, issue: str) -> Dict[str, Any]:
        """Get specific troubleshooting steps"""
        troubleshooting_db = {
            "no_sound": [
                "Check Monitor Mute button is NOT engaged",
                "Verify speakers connected to Monitor Out L/R",
                "Check speaker power and volume",
                "Confirm AG06 is selected in macOS Sound settings"
            ],
            "low_volume": [
                "Check Monitor knob level (turn to 3-5)",
                "Check USB/Computer knob level",
                "Check Channel 1-2 knob levels",
                "Verify input source levels"
            ],
            "connection_issues": [
                "Check USB cable connection",
                "Try different USB port",
                "Verify AG06 power connection",
                "Restart AG06 (power cycle)"
            ],
            "audio_quality": [
                "Check sample rate settings (48kHz recommended)",
                "Verify TO PC switch position",
                "Check for ground loops (hum)",
                "Update AG06 drivers if available"
            ]
        }
        
        steps = troubleshooting_db.get(issue, troubleshooting_db["no_sound"])
        
        return {
            "issue": issue,
            "troubleshooting_steps": steps,
            "estimated_time": "5-10 minutes",
            "difficulty": "beginner",
            "success_rate": "95%"
        }
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Hardware service specific health metrics"""
        return {
            "device_connected": self.device_status["connected"],
            "last_diagnostics": self.last_diagnostics["timestamp"] if self.last_diagnostics else None,
            "diagnostics_status": self.last_diagnostics["overall_status"] if self.last_diagnostics else "unknown"
        }

# Monitoring and Observability Service
class MonitoringService(BaseService):
    """Dedicated service for system monitoring and observability"""
    
    def __init__(self, registry: ServiceRegistry):
        super().__init__("monitoring", 8003, registry)
        self.metrics: Dict[str, List[tuple]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.max_metrics_history = 1000
        
    def setup_service_routes(self):
        """Setup monitoring routes"""
        self.app.router.add_post('/metrics', self.record_metric)
        self.app.router.add_get('/metrics', self.get_metrics)
        self.app.router.add_get('/alerts', self.get_alerts)
        self.app.router.add_get('/dashboard', self.get_dashboard_data)
        
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process monitoring requests"""
        operation = request_data.get("operation")
        
        if operation == "record_metric":
            return await self._record_metric(
                request_data["name"], 
                request_data["value"], 
                request_data.get("tags", {})
            )
        elif operation == "get_metrics":
            return await self._get_metrics()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def record_metric(self, request):
        """Record a metric value"""
        try:
            self.request_count += 1
            data = await request.json()
            result = await self._record_metric(
                data["name"],
                data["value"],
                data.get("tags", {})
            )
            return web.json_response(result)
        except Exception as e:
            self.error_count += 1
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_metrics(self, request):
        """Get metrics data"""
        metrics = await self._get_metrics()
        return web.json_response(metrics)
    
    async def get_alerts(self, request):
        """Get active alerts"""
        return web.json_response({"alerts": self.alerts})
    
    async def get_dashboard_data(self, request):
        """Get dashboard data"""
        dashboard_data = {
            "system_health": await self._get_system_health(),
            "service_status": await self._get_service_status(),
            "recent_metrics": await self._get_recent_metrics(),
            "active_alerts": len(self.alerts),
            "timestamp": datetime.now().isoformat()
        }
        return web.json_response(dashboard_data)
    
    async def _record_metric(self, name: str, value: float, tags: Dict[str, str]) -> Dict[str, Any]:
        """Record a metric with timestamp"""
        timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((timestamp, value, tags))
        
        # Limit history
        if len(self.metrics[name]) > self.max_metrics_history:
            self.metrics[name].pop(0)
        
        # Check for alerts
        await self._check_metric_alerts(name, value, tags)
        
        return {
            "status": "recorded",
            "metric": name,
            "value": value,
            "timestamp": timestamp
        }
    
    async def _get_metrics(self) -> Dict[str, Any]:
        """Get all metrics data"""
        summary = {}
        for name, data_points in self.metrics.items():
            if data_points:
                values = [dp[1] for dp in data_points]
                summary[name] = {
                    "latest": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        return {"metrics": summary}
    
    async def _check_metric_alerts(self, name: str, value: float, tags: Dict[str, str]):
        """Check if metric triggers alerts"""
        # Define alert thresholds
        thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "error_rate": 0.05,
            "response_time": 1000.0
        }
        
        if name in thresholds and value > thresholds[name]:
            alert = {
                "id": str(uuid.uuid4()),
                "metric": name,
                "value": value,
                "threshold": thresholds[name],
                "tags": tags,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning"
            }
            
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts.pop(0)
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        # Discover all services
        all_services = []
        for service_type in ["audio-processing", "hardware-control", "monitoring"]:
            services = await self.registry.discover_services(service_type)
            all_services.extend(services)
        
        healthy_services = sum(1 for s in all_services if (datetime.now() - s.last_heartbeat).seconds < 60)
        
        return {
            "total_services": len(all_services),
            "healthy_services": healthy_services,
            "health_percentage": (healthy_services / len(all_services) * 100) if all_services else 0
        }
    
    async def _get_service_status(self) -> List[Dict[str, Any]]:
        """Get status of all services"""
        status_list = []
        
        for service in self.registry.services.values():
            time_since_heartbeat = (datetime.now() - service.last_heartbeat).seconds
            status_list.append({
                "service_id": service.service_id,
                "service_type": service.service_type,
                "host": service.host,
                "port": service.port,
                "status": "healthy" if time_since_heartbeat < 60 else "unhealthy",
                "last_heartbeat": service.last_heartbeat.isoformat()
            })
        
        return status_list
    
    async def _get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent metrics for dashboard"""
        recent = {}
        for name, data_points in self.metrics.items():
            if data_points:
                # Get last 10 data points
                recent_points = data_points[-10:]
                recent[name] = [
                    {"timestamp": dp[0], "value": dp[1], "tags": dp[2]}
                    for dp in recent_points
                ]
        return recent
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Monitoring service specific health metrics"""
        return {
            "total_metrics": len(self.metrics),
            "total_data_points": sum(len(points) for points in self.metrics.values()),
            "active_alerts": len(self.alerts),
            "memory_usage_mb": sum(len(points) for points in self.metrics.values()) * 0.1  # Rough estimate
        }

# Service Orchestrator (Main Entry Point)
class MicroservicesOrchestrator:
    """Orchestrator for managing all microservices"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.services: List[BaseService] = []
        self.is_running = False
        
    async def start_all_services(self):
        """Start all microservices"""
        try:
            # Start service registry
            await self.registry.start()
            logging.info("Service registry started")
            
            # Create and start services
            audio_service = AudioProcessingService(self.registry)
            hardware_service = HardwareControlService(self.registry)
            monitoring_service = MonitoringService(self.registry)
            
            self.services = [audio_service, hardware_service, monitoring_service]
            
            # Start all services
            start_tasks = [service.start() for service in self.services]
            await asyncio.gather(*start_tasks)
            
            self.is_running = True
            logging.info("All microservices started successfully")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start microservices: {e}")
            return False
    
    async def stop_all_services(self):
        """Stop all microservices"""
        self.is_running = False
        
        # Stop all services
        stop_tasks = [service.stop() for service in self.services]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logging.info("All microservices stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "orchestrator": {
                "running": self.is_running,
                "service_count": len(self.services)
            },
            "registry": {
                "registered_services": len(self.registry.services)
            },
            "services": []
        }
        
        # Get individual service status via HTTP
        for service in self.services:
            try:
                async with ClientSession() as session:
                    async with session.get(f"http://localhost:{service.port}/health", timeout=5) as resp:
                        service_status = await resp.json()
                        status["services"].append(service_status)
            except Exception as e:
                status["services"].append({
                    "service_type": service.service_type,
                    "status": "unreachable",
                    "error": str(e)
                })
        
        return status

# Example usage and testing
async def main():
    """Example usage of the microservices architecture"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = MicroservicesOrchestrator()
    
    try:
        # Start all microservices
        if await orchestrator.start_all_services():
            print("✅ All microservices started successfully")
            
            # Run for demonstration
            await asyncio.sleep(5)
            
            # Test inter-service communication
            async with ClientSession() as session:
                # Test audio processing service
                async with session.post(
                    "http://localhost:8001/process",
                    json={"session_id": "test-123", "input_format": "48kHz"}
                ) as resp:
                    result = await resp.json()
                    print(f"Audio processing result: {result}")
                
                # Test hardware diagnostics
                async with session.post("http://localhost:8002/diagnostics") as resp:
                    diagnostics = await resp.json()
                    print(f"Hardware diagnostics: {diagnostics['overall_status']}")
                
                # Test monitoring metrics
                await session.post(
                    "http://localhost:8003/metrics",
                    json={"name": "test_metric", "value": 42.0}
                )
                
                async with session.get("http://localhost:8003/dashboard") as resp:
                    dashboard = await resp.json()
                    print(f"System health: {dashboard['system_health']['health_percentage']:.1f}%")
            
            # Get comprehensive system status
            status = await orchestrator.get_system_status()
            print(f"\n📊 System Status: {len(status['services'])} services running")
            
        else:
            print("❌ Failed to start microservices")
    
    finally:
        await orchestrator.stop_all_services()

if __name__ == "__main__":
    asyncio.run(main())