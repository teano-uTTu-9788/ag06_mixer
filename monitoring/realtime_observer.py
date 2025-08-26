#!/usr/bin/env python3
"""
Real-Time Workflow Observability System
Production-grade monitoring with Prometheus integration
"""

import json
import time
import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import psutil

# Graceful fallback if prometheus_client not available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available, metrics will use fallback")

@dataclass
class WorkflowMetric:
    timestamp: str
    workflow_id: str
    step_name: str
    duration_ms: float
    status: str
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, float]] = None
    correlation_id: Optional[str] = None

class PrometheusMetrics:
    """Prometheus metrics with fallback to in-memory storage"""
    
    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.workflow_counter = Counter('workflows_total', 'Total workflows executed', ['status', 'workflow_type'])
            self.workflow_duration = Histogram('workflow_duration_seconds', 'Workflow duration', ['workflow_type'])
            self.step_duration = Histogram('workflow_step_duration_seconds', 'Step duration', ['step_name'])
            self.error_counter = Counter('workflow_errors_total', 'Total workflow errors', ['error_type'])
            self.resource_gauge = Gauge('workflow_resource_usage', 'Resource usage', ['resource_type'])
            self.active_workflows = Gauge('workflows_active', 'Currently active workflows')
        else:
            # Fallback to in-memory counters
            self.counters = defaultdict(int)
            self.gauges = defaultdict(float)
            self.histograms = defaultdict(list)
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        if PROMETHEUS_AVAILABLE:
            if name == 'workflows_total':
                self.workflow_counter.labels(**labels).inc()
            elif name == 'workflow_errors_total':
                self.error_counter.labels(**labels).inc()
        else:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            self.counters[key] += 1
    
    def record_duration(self, name: str, duration: float, labels: Dict[str, str] = None):
        if PROMETHEUS_AVAILABLE:
            if name == 'workflow_duration_seconds':
                self.workflow_duration.labels(**labels).observe(duration)
            elif name == 'workflow_step_duration_seconds':
                self.step_duration.labels(**labels).observe(duration)
        else:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            self.histograms[key].append(duration)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        if PROMETHEUS_AVAILABLE:
            if name == 'workflow_resource_usage':
                self.resource_gauge.labels(**labels).set(value)
            elif name == 'workflows_active':
                self.active_workflows.set(value)
        else:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            self.gauges[key] = value

class RealtimeObserver:
    """Production-grade real-time workflow observer"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.metrics = PrometheusMetrics()
        self.events: List[WorkflowMetric] = []
        self.active_workflows: Dict[str, Dict] = {}
        self.enable_prometheus = enable_prometheus
        self.correlation_storage: Dict[str, List[str]] = {}
        
        # Setup structured logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(correlation_id)s | %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE and enable_prometheus:
            try:
                start_http_server(9090)
                print("✅ Prometheus metrics server started on :9090")
            except Exception as e:
                print(f"Warning: Could not start Prometheus server: {e}")
    
    def generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracing"""
        return str(uuid.uuid4())[:8]
    
    def start_workflow(self, workflow_id: str, workflow_type: str = "default") -> str:
        """Start tracking a workflow"""
        correlation_id = self.generate_correlation_id()
        
        self.active_workflows[workflow_id] = {
            "start_time": time.time(),
            "workflow_type": workflow_type,
            "correlation_id": correlation_id,
            "steps": []
        }
        
        self.metrics.set_gauge("workflows_active", len(self.active_workflows))
        
        self.logger.info(
            f"Started workflow {workflow_id}",
            extra={"correlation_id": correlation_id}
        )
        
        return correlation_id
    
    def record_step(self, workflow_id: str, step_name: str, duration_ms: float, 
                   status: str = "success", error_message: str = None):
        """Record a workflow step"""
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Unknown workflow: {workflow_id}")
            return
        
        correlation_id = self.active_workflows[workflow_id]["correlation_id"]
        
        # Collect resource usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            resource_usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "memory_available_mb": memory_info.available / 1024 / 1024
            }
        except Exception:
            resource_usage = None
        
        # Create metric record
        metric = WorkflowMetric(
            timestamp=datetime.now().isoformat(),
            workflow_id=workflow_id,
            step_name=step_name,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            resource_usage=resource_usage,
            correlation_id=correlation_id
        )
        
        self.events.append(metric)
        self.active_workflows[workflow_id]["steps"].append(step_name)
        
        # Update metrics
        self.metrics.record_duration(
            "workflow_step_duration_seconds", 
            duration_ms / 1000.0,
            {"step_name": step_name}
        )
        
        if status == "error":
            self.metrics.increment_counter(
                "workflow_errors_total",
                {"error_type": error_message or "unknown"}
            )
        
        # Update resource metrics
        if resource_usage:
            self.metrics.set_gauge("workflow_resource_usage", cpu_percent, {"resource_type": "cpu"})
            self.metrics.set_gauge("workflow_resource_usage", resource_usage["memory_percent"], {"resource_type": "memory"})
        
        self.logger.info(
            f"Step {step_name}: {status} ({duration_ms:.2f}ms)",
            extra={"correlation_id": correlation_id}
        )
    
    def complete_workflow(self, workflow_id: str, final_status: str = "success"):
        """Complete workflow tracking"""
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Unknown workflow: {workflow_id}")
            return
        
        workflow = self.active_workflows[workflow_id]
        total_duration = time.time() - workflow["start_time"]
        correlation_id = workflow["correlation_id"]
        
        self.metrics.increment_counter(
            "workflows_total",
            {"status": final_status, "workflow_type": workflow["workflow_type"]}
        )
        
        self.metrics.record_duration(
            "workflow_duration_seconds",
            total_duration,
            {"workflow_type": workflow["workflow_type"]}
        )
        
        del self.active_workflows[workflow_id]
        self.metrics.set_gauge("workflows_active", len(self.active_workflows))
        
        self.logger.info(
            f"Completed workflow {workflow_id}: {final_status} ({total_duration:.2f}s)",
            extra={"correlation_id": correlation_id}
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate health score
            health_score = 100
            if cpu_usage > 80:
                health_score -= 30
            if memory.percent > 85:
                health_score -= 30
            if disk.percent > 90:
                health_score -= 20
            if len(self.active_workflows) > 10:
                health_score -= 10
            
            return {
                "status": "healthy" if health_score >= 70 else "degraded" if health_score >= 40 else "unhealthy",
                "health_score": max(0, health_score),
                "system": {
                    "cpu_percent": cpu_usage,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / 1024 / 1024 / 1024
                },
                "workflows": {
                    "active_count": len(self.active_workflows),
                    "total_events": len(self.events),
                    "active_ids": list(self.active_workflows.keys())
                },
                "observability": {
                    "prometheus_enabled": PROMETHEUS_AVAILABLE,
                    "metrics_endpoint": "http://localhost:9090/metrics" if PROMETHEUS_AVAILABLE else None,
                    "correlation_tracking": True
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "health_score": 0,
                "error": str(e)
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        if not PROMETHEUS_AVAILABLE:
            return {
                "counters": dict(self.metrics.counters),
                "gauges": dict(self.metrics.gauges),
                "histograms": {k: {"count": len(v), "avg": sum(v)/len(v) if v else 0} 
                              for k, v in self.metrics.histograms.items()}
            }
        return {"message": "Use Prometheus endpoint /metrics for full metrics"}
    
    def export_events(self, filename: str = None) -> str:
        """Export all events to JSON file"""
        if filename is None:
            filename = f"workflow_events_{int(time.time())}.json"
        
        events_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_events": len(self.events),
            "active_workflows": len(self.active_workflows),
            "events": [asdict(event) for event in self.events]
        }
        
        with open(filename, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        return filename

# Singleton instance for global usage
_observer_instance = None

def get_observer() -> RealtimeObserver:
    """Get global observer instance"""
    global _observer_instance
    if _observer_instance is None:
        _observer_instance = RealtimeObserver()
    return _observer_instance

async def demo_workflow():
    """Demo workflow to show observability in action"""
    observer = get_observer()
    
    # Start workflow
    workflow_id = "demo_workflow_001"
    correlation_id = observer.start_workflow(workflow_id, "demo")
    
    try:
        # Step 1: Data validation
        await asyncio.sleep(0.1)  # Simulate work
        observer.record_step(workflow_id, "data_validation", 150.0, "success")
        
        # Step 2: Processing
        await asyncio.sleep(0.2)
        observer.record_step(workflow_id, "data_processing", 220.0, "success")
        
        # Step 3: Output generation
        await asyncio.sleep(0.1)
        observer.record_step(workflow_id, "output_generation", 180.0, "success")
        
        observer.complete_workflow(workflow_id, "success")
        print(f"✅ Demo workflow completed (correlation: {correlation_id})")
        
    except Exception as e:
        observer.record_step(workflow_id, "error_handler", 0, "error", str(e))
        observer.complete_workflow(workflow_id, "failed")
        print(f"❌ Demo workflow failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Real-Time Workflow Observer")
    
    # Run demo
    asyncio.run(demo_workflow())
    
    # Show health status
    observer = get_observer()
    health = observer.get_health_status()
    print(f"📊 System Health: {health['status']} (Score: {health['health_score']})")
    
    # Export events
    filename = observer.export_events()
    print(f"📁 Events exported to: {filename}")
    
    print("✅ Observer demo complete")