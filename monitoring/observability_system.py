"""
Observability System for AG06 Mixer
Research-driven monitoring and telemetry implementation
Based on 2025 OpenTelemetry and distributed tracing research
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import numpy as np
from enum import Enum
from contextlib import asynccontextmanager

# OpenTelemetry imports (simulated for demonstration)
from typing import Protocol


class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass
class Span:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"


@dataclass
class HealthStatus:
    """System health status"""
    healthy: bool
    latency_ms: float
    error_rate: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    timestamp: float = field(default_factory=time.time)


class IMetricsCollector(Protocol):
    """Metrics collector interface"""
    
    async def collect(self, metric: Metric) -> None:
        """Collect a metric"""
        ...
    
    async def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """Get collected metrics"""
        ...


class ITracer(Protocol):
    """Distributed tracer interface"""
    
    async def start_span(self, operation_name: str, parent_span: Optional[Span] = None) -> Span:
        """Start a new span"""
        ...
    
    async def finish_span(self, span: Span) -> None:
        """Finish a span"""
        ...


class IHealthChecker(Protocol):
    """Health checker interface"""
    
    async def check_health(self) -> HealthStatus:
        """Check system health"""
        ...


class PrometheusMetricsCollector:
    """Prometheus-compatible metrics collector"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._metric_registry: Dict[str, MetricType] = {}
        self._buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    
    async def collect(self, metric: Metric) -> None:
        """Collect a metric"""
        self._metrics[metric.name].append(metric)
        self._metric_registry[metric.name] = metric.type
        
        # Maintain sliding window (last 1000 metrics per name)
        if len(self._metrics[metric.name]) > 1000:
            self._metrics[metric.name] = self._metrics[metric.name][-1000:]
    
    async def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """Get collected metrics"""
        if name:
            return self._metrics.get(name, [])
        
        all_metrics = []
        for metrics_list in self._metrics.values():
            all_metrics.extend(metrics_list)
        return all_metrics
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for name, metric_type in self._metric_registry.items():
            lines.append(f"# TYPE {name} {metric_type.value}")
            
            metrics = self._metrics[name]
            if not metrics:
                continue
            
            if metric_type == MetricType.COUNTER:
                total = sum(m.value for m in metrics)
                labels = self._format_labels(metrics[-1].labels)
                lines.append(f"{name}{labels} {total}")
            
            elif metric_type == MetricType.GAUGE:
                latest = metrics[-1]
                labels = self._format_labels(latest.labels)
                lines.append(f"{name}{labels} {latest.value}")
            
            elif metric_type == MetricType.HISTOGRAM:
                values = [m.value for m in metrics]
                for bucket in self._buckets:
                    count = sum(1 for v in values if v <= bucket)
                    lines.append(f"{name}_bucket{{le=\"{bucket}\"}} {count}")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {len(values)}")
                lines.append(f"{name}_sum {sum(values)}")
                lines.append(f"{name}_count {len(values)}")
        
        return "\n".join(lines)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"


class JaegerTracer:
    """Jaeger-compatible distributed tracer"""
    
    def __init__(self, service_name: str = "ag06-mixer"):
        """Initialize tracer"""
        self._service_name = service_name
        self._spans: Dict[str, Span] = {}
        self._finished_spans: List[Span] = []
        self._trace_counter = 0
        self._span_counter = 0
    
    async def start_span(self, operation_name: str, parent_span: Optional[Span] = None) -> Span:
        """Start a new span"""
        self._trace_counter += 1
        self._span_counter += 1
        
        trace_id = f"trace_{self._trace_counter}" if not parent_span else parent_span.trace_id
        span_id = f"span_{self._span_counter}"
        parent_span_id = parent_span.span_id if parent_span else None
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self._spans[span_id] = span
        return span
    
    async def finish_span(self, span: Span) -> None:
        """Finish a span"""
        span.end_time = time.time()
        span.status = "completed"
        
        if span.span_id in self._spans:
            del self._spans[span.span_id]
        
        self._finished_spans.append(span)
        
        # Keep only last 1000 finished spans
        if len(self._finished_spans) > 1000:
            self._finished_spans = self._finished_spans[-1000:]
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        return [s for s in self._finished_spans if s.trace_id == trace_id]
    
    def export_jaeger_format(self) -> Dict[str, Any]:
        """Export traces in Jaeger format"""
        traces = defaultdict(list)
        
        for span in self._finished_spans:
            traces[span.trace_id].append({
                "traceID": span.trace_id,
                "spanID": span.span_id,
                "parentSpanID": span.parent_span_id or "",
                "operationName": span.operation_name,
                "startTime": int(span.start_time * 1000000),  # microseconds
                "duration": int((span.end_time - span.start_time) * 1000000) if span.end_time else 0,
                "tags": [{"key": k, "value": v} for k, v in span.tags.items()],
                "logs": span.logs,
                "process": {
                    "serviceName": self._service_name,
                    "tags": []
                }
            })
        
        return {
            "data": [
                {
                    "traceID": trace_id,
                    "spans": spans,
                    "processes": {
                        self._service_name: {
                            "serviceName": self._service_name,
                            "tags": []
                        }
                    }
                }
                for trace_id, spans in traces.items()
            ]
        }


class SystemHealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        """Initialize health checker"""
        self._latency_history = deque(maxlen=100)
        self._error_count = 0
        self._request_count = 0
        self._start_time = time.time()
        self._memory_baseline = 150.0  # MB
        self._last_health_status: Optional[HealthStatus] = None
    
    async def check_health(self) -> HealthStatus:
        """Check system health"""
        # Calculate metrics
        avg_latency = sum(self._latency_history) / len(self._latency_history) if self._latency_history else 0
        error_rate = self._error_count / self._request_count if self._request_count > 0 else 0
        elapsed = time.time() - self._start_time
        throughput = self._request_count / elapsed if elapsed > 0 else 0
        
        # Simulated resource metrics (would use psutil in production)
        memory_usage = self._memory_baseline + np.random.normal(0, 10)
        cpu_usage = 25 + np.random.normal(0, 5)
        active_connections = np.random.randint(10, 50)
        
        # Determine health
        healthy = (
            avg_latency < 20 and  # <20ms latency
            error_rate < 0.01 and  # <1% error rate
            memory_usage < 500 and  # <500MB memory
            cpu_usage < 80  # <80% CPU
        )
        
        status = HealthStatus(
            healthy=healthy,
            latency_ms=avg_latency,
            error_rate=error_rate,
            throughput=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            active_connections=active_connections
        )
        
        self._last_health_status = status
        return status
    
    def record_request(self, latency_ms: float, error: bool = False) -> None:
        """Record a request for health metrics"""
        self._latency_history.append(latency_ms)
        self._request_count += 1
        if error:
            self._error_count += 1


class AlertManager:
    """Alert management system"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize alert manager"""
        self._webhook_url = webhook_url
        self._alerts: List[Dict[str, Any]] = []
        self._alert_rules: List[Dict[str, Any]] = []
        self._cooldown_periods: Dict[str, float] = {}
    
    def add_rule(self, 
                 name: str,
                 condition: Callable[[HealthStatus], bool],
                 severity: str = "warning",
                 cooldown_minutes: int = 5) -> None:
        """Add an alert rule"""
        self._alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "cooldown_minutes": cooldown_minutes
        })
    
    async def check_alerts(self, health_status: HealthStatus) -> List[Dict[str, Any]]:
        """Check alert conditions"""
        triggered_alerts = []
        current_time = time.time()
        
        for rule in self._alert_rules:
            rule_name = rule["name"]
            
            # Check cooldown
            if rule_name in self._cooldown_periods:
                if current_time < self._cooldown_periods[rule_name]:
                    continue
            
            # Check condition
            if rule["condition"](health_status):
                alert = {
                    "name": rule_name,
                    "severity": rule["severity"],
                    "timestamp": current_time,
                    "health_status": health_status.__dict__
                }
                
                triggered_alerts.append(alert)
                self._alerts.append(alert)
                
                # Set cooldown
                self._cooldown_periods[rule_name] = current_time + (rule["cooldown_minutes"] * 60)
                
                # Send webhook if configured
                if self._webhook_url:
                    await self._send_webhook(alert)
        
        return triggered_alerts
    
    async def _send_webhook(self, alert: Dict[str, Any]) -> None:
        """Send alert via webhook"""
        # Would implement actual webhook sending
        print(f"🚨 Alert: {alert['name']} - {alert['severity']}")


class ObservabilitySystem:
    """Complete observability system for AG06"""
    
    def __init__(self):
        """Initialize observability system"""
        self.metrics_collector = PrometheusMetricsCollector()
        self.tracer = JaegerTracer()
        self.health_checker = SystemHealthChecker()
        self.alert_manager = AlertManager()
        
        # Setup default alert rules
        self._setup_alert_rules()
    
    def _setup_alert_rules(self) -> None:
        """Setup default alert rules"""
        # High latency alert
        self.alert_manager.add_rule(
            name="high_latency",
            condition=lambda h: h.latency_ms > 50,
            severity="critical",
            cooldown_minutes=5
        )
        
        # High error rate alert
        self.alert_manager.add_rule(
            name="high_error_rate",
            condition=lambda h: h.error_rate > 0.05,
            severity="critical",
            cooldown_minutes=10
        )
        
        # High memory usage alert
        self.alert_manager.add_rule(
            name="high_memory",
            condition=lambda h: h.memory_usage_mb > 400,
            severity="warning",
            cooldown_minutes=15
        )
        
        # System unhealthy alert
        self.alert_manager.add_rule(
            name="system_unhealthy",
            condition=lambda h: not h.healthy,
            severity="critical",
            cooldown_minutes=5
        )
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_span: Optional[Span] = None):
        """Context manager for tracing operations"""
        span = await self.tracer.start_span(operation_name, parent_span)
        try:
            yield span
        finally:
            await self.tracer.finish_span(span)
    
    async def record_metric(self, 
                           name: str,
                           value: float,
                           metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            unit=""
        )
        await self.metrics_collector.collect(metric)
    
    async def check_and_alert(self) -> HealthStatus:
        """Check health and trigger alerts"""
        health_status = await self.health_checker.check_health()
        await self.alert_manager.check_alerts(health_status)
        
        # Record health metrics
        await self.record_metric("system_health", 1.0 if health_status.healthy else 0.0)
        await self.record_metric("latency_ms", health_status.latency_ms)
        await self.record_metric("error_rate", health_status.error_rate)
        await self.record_metric("throughput_rps", health_status.throughput)
        await self.record_metric("memory_usage_mb", health_status.memory_usage_mb)
        await self.record_metric("cpu_usage_percent", health_status.cpu_usage_percent)
        
        return health_status
    
    def get_metrics_export(self) -> str:
        """Get Prometheus format metrics export"""
        return self.metrics_collector.export_prometheus_format()
    
    def get_traces_export(self) -> Dict[str, Any]:
        """Get Jaeger format traces export"""
        return self.tracer.export_jaeger_format()


# Usage example
async def monitor_audio_processing():
    """Example of monitoring audio processing"""
    observability = ObservabilitySystem()
    
    async with observability.trace_operation("audio_processing") as span:
        span.tags["input_size"] = 4096
        span.tags["sample_rate"] = 48000
        
        # Simulate processing
        start = time.time()
        await asyncio.sleep(0.01)  # Simulate 10ms processing
        latency = (time.time() - start) * 1000
        
        # Record metrics
        await observability.record_metric("audio_latency_ms", latency, MetricType.HISTOGRAM)
        await observability.record_metric("audio_buffers_processed", 1, MetricType.COUNTER)
        
        # Record request
        observability.health_checker.record_request(latency)
        
        # Check health and alerts
        health = await observability.check_and_alert()
        
        if not health.healthy:
            span.tags["error"] = True
            span.logs.append({
                "timestamp": time.time(),
                "message": "System unhealthy",
                "health_status": health.__dict__
            })
    
    return observability.get_metrics_export()