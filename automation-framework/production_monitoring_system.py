#!/usr/bin/env python3
"""
Production Monitoring System for Aioke Advanced Enterprise
Provides real-time monitoring, alerting, and performance optimization
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import threading
import requests

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    component: str
    metric_type: str

@dataclass
class Alert:
    """Alert configuration"""
    name: str
    component: str
    metric: str
    threshold: float
    condition: str  # 'above', 'below', 'equals'
    severity: str  # 'info', 'warning', 'critical'
    cooldown: int = 300  # seconds between alerts
    last_triggered: Optional[float] = None

class MonitoringSystem:
    """Production monitoring system with alerting"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.metrics_history: Dict[str, deque] = {}
        self.alerts: List[Alert] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.collection_interval = 5  # seconds
        self.history_size = 1000  # keep last 1000 points per metric
        
        # Performance baselines
        self.baselines = {
            'response_time': 100,  # ms
            'error_rate': 0.01,  # 1%
            'throughput': 10,  # events/sec
            'memory_usage': 100,  # MB
            'cpu_usage': 50  # %
        }
        
        # Initialize metric storage
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize metric storage structures"""
        metric_names = [
            'uptime', 'total_events', 'error_count', 'error_rate',
            'throughput', 'response_time', 'borg_jobs', 'borg_utilization',
            'cells_total', 'cells_healthy', 'workflows_active',
            'kafka_messages', 'finagle_requests', 'airflow_tasks'
        ]
        
        for metric in metric_names:
            self.metrics_history[metric] = deque(maxlen=self.history_size)
    
    def configure_alerts(self):
        """Configure production alerting thresholds"""
        self.alerts = [
            # Critical alerts
            Alert('High Error Rate', 'system', 'error_rate', 0.05, 'above', 'critical'),
            Alert('System Down', 'system', 'uptime', 0, 'equals', 'critical'),
            Alert('No Throughput', 'system', 'throughput', 1, 'below', 'critical'),
            
            # Warning alerts
            Alert('Elevated Errors', 'system', 'error_rate', 0.01, 'above', 'warning'),
            Alert('Low Throughput', 'system', 'throughput', 5, 'below', 'warning'),
            Alert('High Response Time', 'system', 'response_time', 200, 'above', 'warning'),
            
            # Component alerts
            Alert('Borg Overloaded', 'borg', 'utilization', 90, 'above', 'warning'),
            Alert('Cells Unhealthy', 'cells', 'health_ratio', 0.8, 'below', 'warning'),
            Alert('Kafka Lag', 'kafka', 'consumer_lag', 1000, 'above', 'warning'),
            
            # Info alerts
            Alert('High Traffic', 'system', 'throughput', 50, 'above', 'info'),
            Alert('Many Workflows', 'cadence', 'active_workflows', 10, 'above', 'info')
        ]
    
    async def collect_metrics(self):
        """Collect metrics from production server"""
        try:
            # Fetch health metrics
            health_response = requests.get(f"{self.api_url}/health", timeout=5)
            health_data = health_response.json()
            
            # Fetch detailed metrics
            metrics_response = requests.get(f"{self.api_url}/metrics", timeout=5)
            metrics_data = metrics_response.json()
            
            timestamp = time.time()
            
            # Store system metrics
            self._store_metric('uptime', health_data['uptime'], timestamp)
            self._store_metric('total_events', health_data['total_events'], timestamp)
            self._store_metric('error_count', health_data['error_count'], timestamp)
            
            # Calculate derived metrics
            error_rate = health_data['error_count'] / max(health_data['total_events'], 1)
            self._store_metric('error_rate', error_rate, timestamp)
            
            # Calculate throughput
            if len(self.metrics_history['total_events']) > 1:
                prev_events = self.metrics_history['total_events'][-2].value
                prev_time = self.metrics_history['total_events'][-2].timestamp
                throughput = (health_data['total_events'] - prev_events) / (timestamp - prev_time)
                self._store_metric('throughput', throughput, timestamp)
            
            # Store component metrics
            components = metrics_data.get('components', {})
            if components:
                self._store_metric('borg_jobs', components['borg']['jobs'], timestamp)
                self._store_metric('cells_total', components['cells']['total'], timestamp)
                self._store_metric('cells_healthy', components['cells']['healthy'], timestamp)
                self._store_metric('workflows_active', components['workflows']['active'], timestamp)
            
            # Simulate response time (would be real in production)
            response_time = health_response.elapsed.total_seconds() * 1000
            self._store_metric('response_time', response_time, timestamp)
            
            return True
            
        except Exception as e:
            print(f"❌ Metrics collection error: {e}")
            return False
    
    def _store_metric(self, name: str, value: float, timestamp: float):
        """Store a metric data point"""
        point = MetricPoint(timestamp, value, 'system', name)
        self.metrics_history[name].append(point)
    
    def check_alerts(self):
        """Check all configured alerts"""
        current_time = time.time()
        triggered_alerts = []
        
        for alert in self.alerts:
            # Skip if in cooldown period
            if alert.last_triggered and (current_time - alert.last_triggered) < alert.cooldown:
                continue
            
            # Get metric value
            metric_value = self.get_current_metric(alert.metric)
            if metric_value is None:
                continue
            
            # Check alert condition
            triggered = False
            if alert.condition == 'above' and metric_value > alert.threshold:
                triggered = True
            elif alert.condition == 'below' and metric_value < alert.threshold:
                triggered = True
            elif alert.condition == 'equals' and metric_value == alert.threshold:
                triggered = True
            
            if triggered:
                alert.last_triggered = current_time
                alert_info = {
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert.name,
                    'severity': alert.severity,
                    'component': alert.component,
                    'metric': alert.metric,
                    'value': metric_value,
                    'threshold': alert.threshold,
                    'condition': alert.condition
                }
                triggered_alerts.append(alert_info)
                self.alert_history.append(alert_info)
                self._send_alert(alert_info)
        
        return triggered_alerts
    
    def _send_alert(self, alert_info: Dict[str, Any]):
        """Send alert notification"""
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨'
        }
        
        emoji = severity_emoji.get(alert_info['severity'], '📢')
        print(f"{emoji} ALERT: {alert_info['alert']} - {alert_info['metric']}={alert_info['value']:.2f} ({alert_info['condition']} {alert_info['threshold']})")
    
    def get_current_metric(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        if metric_name not in self.metrics_history:
            return None
        
        history = self.metrics_history[metric_name]
        if not history:
            return None
        
        return history[-1].value
    
    def get_metric_stats(self, metric_name: str, window_seconds: int = 300) -> Dict[str, float]:
        """Get statistics for a metric over a time window"""
        if metric_name not in self.metrics_history:
            return {}
        
        history = self.metrics_history[metric_name]
        if not history:
            return {}
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Filter data points within window
        window_values = [
            point.value for point in history
            if point.timestamp >= window_start
        ]
        
        if not window_values:
            return {}
        
        return {
            'current': window_values[-1],
            'min': min(window_values),
            'max': max(window_values),
            'mean': statistics.mean(window_values),
            'median': statistics.median(window_values),
            'stddev': statistics.stdev(window_values) if len(window_values) > 1 else 0,
            'count': len(window_values)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'healthy',
            'metrics_summary': {},
            'alerts_triggered': len(self.alert_history),
            'recommendations': []
        }
        
        # Analyze key metrics
        key_metrics = ['throughput', 'error_rate', 'response_time', 'uptime']
        
        for metric in key_metrics:
            stats = self.get_metric_stats(metric)
            if stats:
                report['metrics_summary'][metric] = stats
                
                # Check against baselines
                if metric in self.baselines:
                    baseline = self.baselines[metric]
                    current = stats.get('current', 0)
                    
                    if metric == 'error_rate' and current > baseline:
                        report['system_health'] = 'degraded'
                        report['recommendations'].append(
                            f"Error rate ({current:.2%}) exceeds baseline ({baseline:.2%}). Investigate error sources."
                        )
                    elif metric == 'response_time' and current > baseline:
                        report['recommendations'].append(
                            f"Response time ({current:.0f}ms) exceeds baseline ({baseline}ms). Consider scaling or optimization."
                        )
                    elif metric == 'throughput' and current < baseline:
                        report['recommendations'].append(
                            f"Throughput ({current:.1f}/s) below baseline ({baseline}/s). Check for bottlenecks."
                        )
        
        # Add optimization recommendations
        if not report['recommendations']:
            report['recommendations'].append("System performing optimally. No immediate actions required.")
        
        return report
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        print("🔍 Starting production monitoring system")
        self.monitoring_active = True
        
        while self.monitoring_active:
            # Collect metrics
            success = await self.collect_metrics()
            
            if success:
                # Check alerts
                triggered = self.check_alerts()
                
                # Generate periodic reports
                if int(time.time()) % 60 == 0:  # Every minute
                    report = self.get_performance_report()
                    print(f"📊 Performance: Health={report['system_health']}, Alerts={report['alerts_triggered']}")
            
            await asyncio.sleep(self.collection_interval)
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        report = self.get_performance_report()
        metrics = report.get('metrics_summary', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aioke Production Monitoring</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }}
                .metric {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; margin-top: 10px; }}
                .status-healthy {{ color: #27ae60; }}
                .status-degraded {{ color: #f39c12; }}
                .status-critical {{ color: #e74c3c; }}
                .alerts {{ background: white; padding: 20px; border-radius: 8px; margin-top: 20px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-left: 4px solid; }}
                .alert-info {{ border-color: #3498db; background: #ecf0f1; }}
                .alert-warning {{ border-color: #f39c12; background: #fcf8e3; }}
                .alert-critical {{ border-color: #e74c3c; background: #f2dede; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Aioke Advanced Enterprise - Production Monitoring</h1>
                <p>Status: <span class="status-{report['system_health']}">{report['system_health'].upper()}</span></p>
                <p>Last Updated: {report['timestamp']}</p>
            </div>
            
            <div class="metrics">
        """
        
        # Add metric cards
        for metric_name, stats in metrics.items():
            current = stats.get('current', 0)
            unit = self._get_unit(metric_name)
            formatted_value = self._format_value(current, metric_name)
            
            html += f"""
                <div class="metric">
                    <div class="metric-value">{formatted_value}{unit}</div>
                    <div class="metric-label">{metric_name.replace('_', ' ').title()}</div>
                    <small>Min: {self._format_value(stats.get('min', 0), metric_name)}{unit} | 
                           Max: {self._format_value(stats.get('max', 0), metric_name)}{unit}</small>
                </div>
            """
        
        # Add recent alerts
        html += """
            <div class="alerts">
                <h2>Recent Alerts</h2>
        """
        
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        if recent_alerts:
            for alert in reversed(recent_alerts):
                html += f"""
                    <div class="alert alert-{alert['severity']}">
                        <strong>{alert['alert']}</strong> - {alert['timestamp']}<br>
                        {alert['metric']}: {alert['value']:.2f} ({alert['condition']} {alert['threshold']})
                    </div>
                """
        else:
            html += "<p>No recent alerts</p>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_unit(self, metric_name: str) -> str:
        """Get unit for metric display"""
        units = {
            'throughput': '/s',
            'response_time': 'ms',
            'error_rate': '%',
            'uptime': 's',
            'memory_usage': 'MB',
            'cpu_usage': '%'
        }
        return units.get(metric_name, '')
    
    def _format_value(self, value: float, metric_name: str) -> str:
        """Format metric value for display"""
        if metric_name == 'error_rate':
            return f"{value * 100:.2f}"
        elif metric_name in ['throughput', 'response_time']:
            return f"{value:.1f}"
        elif metric_name == 'uptime':
            hours = value / 3600
            return f"{hours:.1f}h"
        else:
            return f"{value:.0f}"
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring_active = False
        print("🛑 Monitoring stopped")

async def main():
    """Main entry point for monitoring system"""
    monitor = MonitoringSystem()
    monitor.configure_alerts()
    
    # Generate initial dashboard
    with open('monitoring_dashboard.html', 'w') as f:
        f.write(monitor.generate_dashboard_html())
    print("📊 Dashboard created: monitoring_dashboard.html")
    
    # Start monitoring
    try:
        await monitor.monitoring_loop()
    except KeyboardInterrupt:
        monitor.stop()
        
        # Generate final report
        final_report = monitor.get_performance_report()
        with open('performance_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        print("📋 Final report saved: performance_report.json")

if __name__ == '__main__':
    asyncio.run(main())