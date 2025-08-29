#!/usr/bin/env python3
"""
Production Monitoring System for Terminal Automation Framework
Following Google SRE Four Golden Signals and Meta monitoring patterns
"""

import asyncio
import json
import time
import psutil
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler


class SignalType(Enum):
    """Google SRE Four Golden Signals."""
    LATENCY = "latency"
    TRAFFIC = "traffic"
    ERRORS = "errors"
    SATURATION = "saturation"


@dataclass
class Metric:
    """Individual metric data point."""
    timestamp: float
    signal_type: SignalType
    value: float
    unit: str
    labels: Dict[str, str]


@dataclass
class Alert:
    """Alert configuration."""
    name: str
    signal_type: SignalType
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_seconds: int
    severity: str  # 'info', 'warning', 'critical'


class ProductionMonitor:
    """Production monitoring system with Google SRE patterns."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize monitoring system."""
        self.metrics: Dict[SignalType, deque] = {
            signal: deque(maxlen=max_history)
            for signal in SignalType
        }
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=100)
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        
        # Initialize default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Set up default alert configurations."""
        self.alerts = [
            Alert(
                name="High Latency",
                signal_type=SignalType.LATENCY,
                threshold=200,  # ms
                comparison="gt",
                duration_seconds=60,
                severity="warning"
            ),
            Alert(
                name="Critical Latency",
                signal_type=SignalType.LATENCY,
                threshold=500,  # ms
                comparison="gt",
                duration_seconds=30,
                severity="critical"
            ),
            Alert(
                name="High Error Rate",
                signal_type=SignalType.ERRORS,
                threshold=5,  # %
                comparison="gt",
                duration_seconds=60,
                severity="warning"
            ),
            Alert(
                name="High CPU Saturation",
                signal_type=SignalType.SATURATION,
                threshold=80,  # %
                comparison="gt",
                duration_seconds=120,
                severity="warning"
            ),
            Alert(
                name="Critical Memory Saturation",
                signal_type=SignalType.SATURATION,
                threshold=90,  # %
                comparison="gt",
                duration_seconds=60,
                severity="critical"
            )
        ]
    
    def record_metric(self, signal_type: SignalType, value: float, 
                      unit: str = "", labels: Optional[Dict[str, str]] = None):
        """Record a metric data point."""
        with self._lock:
            metric = Metric(
                timestamp=time.time(),
                signal_type=signal_type,
                value=value,
                unit=unit,
                labels=labels or {}
            )
            self.metrics[signal_type].append(metric)
    
    def record_latency(self, operation: str, duration_ms: float):
        """Record latency metric."""
        self.record_metric(
            SignalType.LATENCY,
            duration_ms,
            unit="ms",
            labels={"operation": operation}
        )
    
    def record_request(self, endpoint: str, status_code: int):
        """Record traffic metric."""
        self.record_metric(
            SignalType.TRAFFIC,
            1,
            unit="requests",
            labels={"endpoint": endpoint, "status": str(status_code)}
        )
    
    def record_error(self, error_type: str, operation: str):
        """Record error metric."""
        self.record_metric(
            SignalType.ERRORS,
            1,
            unit="errors",
            labels={"type": error_type, "operation": operation}
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self._lock:
            current = {}
            
            # Latency (p50, p95, p99)
            latency_values = [m.value for m in self.metrics[SignalType.LATENCY]]
            if latency_values:
                latency_values.sort()
                n = len(latency_values)
                current['latency_p50'] = latency_values[int(n * 0.5)]
                current['latency_p95'] = latency_values[int(n * 0.95)]
                current['latency_p99'] = latency_values[int(n * 0.99)]
            
            # Traffic (requests per second)
            now = time.time()
            recent_traffic = [
                m for m in self.metrics[SignalType.TRAFFIC]
                if now - m.timestamp < 60
            ]
            current['requests_per_second'] = len(recent_traffic) / 60.0
            
            # Errors (error rate)
            recent_errors = [
                m for m in self.metrics[SignalType.ERRORS]
                if now - m.timestamp < 300  # Last 5 minutes
            ]
            current['error_rate'] = len(recent_errors) / 5.0  # Per minute
            
            # Saturation
            current['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            current['memory_percent'] = psutil.virtual_memory().percent
            current['disk_percent'] = psutil.disk_usage('/').percent
            
            # Record saturation metrics
            self.record_metric(SignalType.SATURATION, current['cpu_percent'], 
                             unit="%", labels={"resource": "cpu"})
            self.record_metric(SignalType.SATURATION, current['memory_percent'],
                             unit="%", labels={"resource": "memory"})
            
            return current
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        triggered_alerts = []
        current = self.get_current_metrics()
        now = time.time()
        
        for alert in self.alerts:
            # Get recent metrics for this signal
            recent_metrics = [
                m for m in self.metrics[alert.signal_type]
                if now - m.timestamp < alert.duration_seconds
            ]
            
            if not recent_metrics:
                continue
            
            # Calculate average value
            avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Check threshold
            triggered = False
            if alert.comparison == "gt" and avg_value > alert.threshold:
                triggered = True
            elif alert.comparison == "lt" and avg_value < alert.threshold:
                triggered = True
            elif alert.comparison == "eq" and avg_value == alert.threshold:
                triggered = True
            
            if triggered:
                alert_data = {
                    'name': alert.name,
                    'severity': alert.severity,
                    'signal': alert.signal_type.value,
                    'threshold': alert.threshold,
                    'current_value': avg_value,
                    'timestamp': now
                }
                triggered_alerts.append(alert_data)
                self.alert_history.append(alert_data)
        
        return triggered_alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        current = self.get_current_metrics()
        alerts = self.check_alerts()
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        # Get historical data for charts (last 100 points)
        history = {
            'latency': [],
            'traffic': [],
            'errors': [],
            'saturation': []
        }
        
        with self._lock:
            for signal_type in SignalType:
                recent = list(self.metrics[signal_type])[-100:]
                history[signal_type.value] = [
                    {'timestamp': m.timestamp, 'value': m.value}
                    for m in recent
                ]
        
        return {
            'current': current,
            'alerts': alerts,
            'uptime': uptime_str,
            'history': history,
            'timestamp': time.time()
        }
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                self.get_current_metrics()
                
                # Check alerts
                alerts = self.check_alerts()
                if alerts:
                    for alert in alerts:
                        print(f"🚨 Alert: {alert['name']} - {alert['severity'].upper()}")
                        print(f"   {alert['signal']}: {alert['current_value']:.2f} > {alert['threshold']}")
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with self._lock:
            data = {
                'start_time': self.start_time,
                'export_time': time.time(),
                'metrics': {
                    signal.value: [asdict(m) for m in metrics]
                    for signal, metrics in self.metrics.items()
                },
                'alerts': [asdict(a) for a in self.alerts],
                'alert_history': list(self.alert_history)
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class MonitoringDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for monitoring dashboard."""
    
    monitor: Optional[ProductionMonitor] = None
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_dashboard()
        elif self.path == '/metrics':
            self.send_metrics()
        elif self.path == '/health':
            self.send_health()
        else:
            self.send_error(404, "Not Found")
    
    def send_dashboard(self):
        """Send HTML dashboard."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Terminal Automation - Production Monitoring</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               margin: 0; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                  gap: 20px; margin: 20px 0; }
        .metric { background: white; padding: 20px; border-radius: 8px; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-label { color: #666; font-size: 12px; text-transform: uppercase; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; margin: 5px 0; }
        .metric-unit { color: #999; font-size: 14px; }
        .alert { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; 
                border-radius: 4px; margin: 10px 0; }
        .alert.critical { background: #f8d7da; border-color: #dc3545; }
        .status { display: inline-block; width: 10px; height: 10px; 
                 border-radius: 50%; margin-right: 5px; }
        .status.green { background: #28a745; }
        .status.yellow { background: #ffc107; }
        .status.red { background: #dc3545; }
    </style>
    <script>
        async function updateMetrics() {
            try {
                const response = await fetch('/metrics');
                const data = await response.json();
                
                // Update metric values
                document.getElementById('latency-p50').textContent = 
                    data.current.latency_p50?.toFixed(1) || '0';
                document.getElementById('latency-p95').textContent = 
                    data.current.latency_p95?.toFixed(1) || '0';
                document.getElementById('rps').textContent = 
                    data.current.requests_per_second?.toFixed(2) || '0';
                document.getElementById('error-rate').textContent = 
                    data.current.error_rate?.toFixed(2) || '0';
                document.getElementById('cpu').textContent = 
                    data.current.cpu_percent?.toFixed(1) || '0';
                document.getElementById('memory').textContent = 
                    data.current.memory_percent?.toFixed(1) || '0';
                document.getElementById('uptime').textContent = data.uptime || '0:00:00';
                
                // Update alerts
                const alertsDiv = document.getElementById('alerts');
                if (data.alerts.length > 0) {
                    alertsDiv.innerHTML = data.alerts.map(alert => 
                        `<div class="alert ${alert.severity}">
                            <strong>${alert.name}</strong>: ${alert.signal} = ${alert.current_value.toFixed(2)}
                        </div>`
                    ).join('');
                } else {
                    alertsDiv.innerHTML = '<div style="color: #28a745;">✅ No active alerts</div>';
                }
                
            } catch (error) {
                console.error('Failed to update metrics:', error);
            }
        }
        
        // Update every 5 seconds
        setInterval(updateMetrics, 5000);
        updateMetrics();
    </script>
</head>
<body>
    <h1>🎯 Terminal Automation - Production Monitoring</h1>
    <p>Following Google SRE Four Golden Signals</p>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-label">Latency P50</div>
            <div class="metric-value"><span id="latency-p50">0</span></div>
            <div class="metric-unit">ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">Latency P95</div>
            <div class="metric-value"><span id="latency-p95">0</span></div>
            <div class="metric-unit">ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">Traffic</div>
            <div class="metric-value"><span id="rps">0</span></div>
            <div class="metric-unit">req/s</div>
        </div>
        <div class="metric">
            <div class="metric-label">Error Rate</div>
            <div class="metric-value"><span id="error-rate">0</span></div>
            <div class="metric-unit">errors/min</div>
        </div>
        <div class="metric">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value"><span id="cpu">0</span></div>
            <div class="metric-unit">%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value"><span id="memory">0</span></div>
            <div class="metric-unit">%</div>
        </div>
    </div>
    
    <h2>⏱️ Uptime: <span id="uptime">0:00:00</span></h2>
    
    <h2>🚨 Alerts</h2>
    <div id="alerts">
        <div style="color: #28a745;">✅ No active alerts</div>
    </div>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_metrics(self):
        """Send metrics JSON."""
        if not self.monitor:
            self.send_error(500, "Monitor not initialized")
            return
        
        data = self.monitor.get_dashboard_data()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_health(self):
        """Send health check response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'healthy'}).encode())
    
    def send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_monitoring_dashboard(monitor: ProductionMonitor, port: int = 8080):
    """Start HTTP monitoring dashboard."""
    MonitoringDashboardHandler.monitor = monitor
    
    server = HTTPServer(('localhost', port), MonitoringDashboardHandler)
    print(f"📊 Monitoring dashboard started at http://localhost:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring dashboard stopped")
        server.shutdown()


async def demo_monitoring():
    """Demonstrate monitoring capabilities."""
    monitor = ProductionMonitor()
    monitor.start_monitoring(interval_seconds=2)
    
    print("🚀 Production Monitoring Demo")
    print("="*50)
    
    # Simulate some operations
    for i in range(10):
        # Record latency
        latency = 50 + (i * 10)  # Increasing latency
        monitor.record_latency("workflow_execution", latency)
        
        # Record traffic
        monitor.record_request("/execute", 200)
        
        # Occasionally record errors
        if i % 3 == 0:
            monitor.record_error("ValidationError", "execute_workflow")
        
        await asyncio.sleep(1)
        
        # Print current metrics
        if i % 5 == 0:
            metrics = monitor.get_current_metrics()
            print(f"\n📊 Current Metrics (iteration {i}):")
            print(f"  Latency P50: {metrics.get('latency_p50', 0):.1f}ms")
            print(f"  Requests/sec: {metrics.get('requests_per_second', 0):.2f}")
            print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
            print(f"  Memory: {metrics.get('memory_percent', 0):.1f}%")
    
    # Export metrics
    monitor.export_metrics('production_metrics.json')
    print("\n📄 Metrics exported to production_metrics.json")
    
    monitor.stop_monitoring()
    print("✅ Monitoring demo complete")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_monitoring())
    
    # Start dashboard
    monitor = ProductionMonitor()
    monitor.start_monitoring()
    start_monitoring_dashboard(monitor, port=8080)