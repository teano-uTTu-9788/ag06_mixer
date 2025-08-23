#!/usr/bin/env python3

"""
Metrics Collection System - Inspired by Google's Borgmon
Collects, aggregates, and exposes metrics for monitoring
"""

import json
import time
import sqlite3
import psutil
import threading
import http.server
import socketserver
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

@dataclass
class Metric:
    """Represents a single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    metric_type: str  # counter, gauge, histogram

class MetricsCollector:
    """Main metrics collection system"""
    
    def __init__(self, db_path: str = "/tmp/metrics.db", retention_days: int = 7):
        self.db_path = db_path
        self.retention_days = retention_days
        self.metrics_buffer = deque(maxlen=10000)
        self.aggregations = defaultdict(list)
        self.running = False
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # System metrics collectors
        self.collectors = {
            'system': self._collect_system_metrics,
            'process': self._collect_process_metrics,
            'workflow': self._collect_workflow_metrics,
            'custom': self._collect_custom_metrics
        }
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                labels TEXT,
                metric_type TEXT,
                INDEX idx_name_timestamp (name, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggregations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                period TEXT NOT NULL,
                min_value REAL,
                max_value REAL,
                avg_value REAL,
                sum_value REAL,
                count INTEGER,
                timestamp REAL NOT NULL,
                INDEX idx_name_period_timestamp (name, period, timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, 
                     metric_type: str = "gauge"):
        """Record a single metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            
        # Trigger flush if buffer is getting full
        if len(self.metrics_buffer) > 8000:
            self._flush_metrics()
    
    def _flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_to_insert = []
        with self.lock:
            while self.metrics_buffer:
                metric = self.metrics_buffer.popleft()
                metrics_to_insert.append((
                    metric.name,
                    metric.value,
                    metric.timestamp,
                    json.dumps(metric.labels),
                    metric.metric_type
                ))
        
        cursor.executemany(
            'INSERT INTO metrics (name, value, timestamp, labels, metric_type) VALUES (?, ?, ?, ?, ?)',
            metrics_to_insert
        )
        
        conn.commit()
        conn.close()
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("system.cpu.usage", cpu_percent, {"unit": "percent"})
        
        cpu_count = psutil.cpu_count()
        self.record_metric("system.cpu.count", cpu_count, {"type": "logical"})
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric("system.memory.used", memory.used, {"unit": "bytes"})
        self.record_metric("system.memory.available", memory.available, {"unit": "bytes"})
        self.record_metric("system.memory.percent", memory.percent, {"unit": "percent"})
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_metric("system.disk.used", disk.used, {"unit": "bytes", "mount": "/"})
        self.record_metric("system.disk.free", disk.free, {"unit": "bytes", "mount": "/"})
        self.record_metric("system.disk.percent", disk.percent, {"unit": "percent", "mount": "/"})
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.record_metric("system.network.bytes_sent", net_io.bytes_sent, 
                          {"unit": "bytes"}, "counter")
        self.record_metric("system.network.bytes_recv", net_io.bytes_recv, 
                          {"unit": "bytes"}, "counter")
        
        # Load average
        load_avg = psutil.getloadavg()
        self.record_metric("system.load.1min", load_avg[0])
        self.record_metric("system.load.5min", load_avg[1])
        self.record_metric("system.load.15min", load_avg[2])
    
    def _collect_process_metrics(self):
        """Collect process-level metrics"""
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                info = proc.info
                if info['name'] in ['workflow_daemon', 'parallel_executor', 'python3']:
                    self.record_metric(
                        f"process.cpu.usage",
                        info['cpu_percent'],
                        {"pid": str(info['pid']), "name": info['name']}
                    )
                    self.record_metric(
                        f"process.memory.rss",
                        info['memory_info'].rss,
                        {"pid": str(info['pid']), "name": info['name']}
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def _collect_workflow_metrics(self):
        """Collect workflow-specific metrics"""
        # Check for workflow daemon state
        daemon_state_path = Path("/tmp/workflow_daemon/state.json")
        if daemon_state_path.exists():
            with open(daemon_state_path) as f:
                state = json.load(f)
                
            self.record_metric("workflow.active_count", len(state.get('workflows', [])))
            
            for workflow in state.get('workflows', []):
                self.record_metric(
                    "workflow.execution",
                    1 if workflow['status'] == 'running' else 0,
                    {"name": workflow['name'], "status": workflow['status']}
                )
        
        # Check for parallel execution stats
        parallel_stats_path = Path("/tmp/parallel_stats.json")
        if parallel_stats_path.exists():
            with open(parallel_stats_path) as f:
                stats = json.load(f)
                
            self.record_metric("parallel.total_tasks", stats.get('total_tasks', 0))
            self.record_metric("parallel.completed_tasks", stats.get('completed_tasks', 0))
            self.record_metric("parallel.failed_tasks", stats.get('failed_tasks', 0))
            self.record_metric("parallel.workers", stats.get('workers', 0))
    
    def _collect_custom_metrics(self):
        """Collect custom application metrics"""
        # Read custom metrics from file if exists
        custom_metrics_path = Path("/tmp/custom_metrics.json")
        if custom_metrics_path.exists():
            with open(custom_metrics_path) as f:
                metrics = json.load(f)
                
            for metric in metrics:
                self.record_metric(
                    metric['name'],
                    metric['value'],
                    metric.get('labels', {}),
                    metric.get('type', 'gauge')
                )
    
    def aggregate_metrics(self, period: str = "5m"):
        """Aggregate metrics over time periods"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Define period in seconds
        periods = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400
        }
        
        period_seconds = periods.get(period, 300)
        cutoff_time = time.time() - period_seconds
        
        # Get metrics for aggregation
        cursor.execute('''
            SELECT name, value, timestamp
            FROM metrics
            WHERE timestamp > ?
            ORDER BY name, timestamp
        ''', (cutoff_time,))
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for row in cursor.fetchall():
            metrics_by_name[row[0]].append(row[1])
        
        # Calculate aggregations
        aggregations = []
        for name, values in metrics_by_name.items():
            if values:
                aggregations.append((
                    name,
                    period,
                    min(values),
                    max(values),
                    sum(values) / len(values),
                    sum(values),
                    len(values),
                    time.time()
                ))
        
        # Store aggregations
        cursor.executemany('''
            INSERT INTO aggregations (name, period, min_value, max_value, avg_value, sum_value, count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', aggregations)
        
        conn.commit()
        conn.close()
        
        return aggregations
    
    def query_metrics(self, name: str, start_time: float = None, end_time: float = None) -> List[Dict]:
        """Query metrics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM metrics WHERE name = ?'
        params = [name]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT 1000'
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'value': row[2],
                'timestamp': row[3],
                'labels': json.loads(row[4]) if row[4] else {},
                'type': row[5]
            })
        
        conn.close()
        return results
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (self.retention_days * 86400)
        
        cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM aggregations WHERE timestamp < ?', (cutoff_time,))
        
        deleted_metrics = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_metrics
    
    def start_collection(self, interval: int = 10):
        """Start metrics collection loop"""
        self.running = True
        
        def collection_loop():
            while self.running:
                for collector_name, collector_func in self.collectors.items():
                    try:
                        collector_func()
                    except Exception as e:
                        print(f"Error in {collector_name} collector: {e}")
                
                # Flush metrics
                self._flush_metrics()
                
                # Periodic aggregation
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self.aggregate_metrics("5m")
                
                # Periodic cleanup
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.cleanup_old_metrics()
                
                time.sleep(interval)
        
        collection_thread = threading.Thread(target=collection_loop, daemon=True)
        collection_thread.start()
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        self._flush_metrics()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest metrics
        cursor.execute('''
            SELECT DISTINCT name, value, timestamp
            FROM metrics
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
        ''', (time.time() - 300,))  # Last 5 minutes
        
        latest_metrics = {}
        for row in cursor.fetchall():
            if row[0] not in latest_metrics:
                latest_metrics[row[0]] = {
                    'value': row[1],
                    'timestamp': row[2]
                }
        
        # Get aggregations
        cursor.execute('''
            SELECT name, period, avg_value, min_value, max_value
            FROM aggregations
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (time.time() - 3600,))  # Last hour
        
        aggregations = defaultdict(dict)
        for row in cursor.fetchall():
            aggregations[row[0]][row[1]] = {
                'avg': row[2],
                'min': row[3],
                'max': row[4]
            }
        
        conn.close()
        
        return {
            'latest_metrics': latest_metrics,
            'aggregations': dict(aggregations),
            'timestamp': time.time()
        }

class MetricsDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for metrics dashboard"""
    
    def __init__(self, *args, collector: MetricsCollector = None, **kwargs):
        self.collector = collector
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = self.collector.get_dashboard_data()
            self.wfile.write(json.dumps(data, indent=2).encode())
        
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Serve dashboard HTML
            html = self._generate_dashboard_html()
            self.wfile.write(html.encode())
        
        else:
            self.send_error(404)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Metrics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { 
            display: inline-block; 
            margin: 10px; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            min-width: 200px;
        }
        .metric-name { font-weight: bold; color: #333; }
        .metric-value { font-size: 24px; color: #2196F3; }
        .timestamp { color: #666; font-size: 12px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .refresh-btn { 
            padding: 10px 20px; 
            background: #4CAF50; 
            color: white; 
            border: none; 
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Workflow Metrics Dashboard</h1>
    <button class="refresh-btn" onclick="refreshMetrics()">Refresh</button>
    
    <h2>System Metrics</h2>
    <div id="system-metrics"></div>
    
    <h2>Workflow Metrics</h2>
    <div id="workflow-metrics"></div>
    
    <h2>Performance Metrics</h2>
    <div id="performance-metrics"></div>
    
    <script>
        function refreshMetrics() {
            fetch('/metrics')
                .then(response => response.json())
                .then(data => {
                    updateMetrics('system-metrics', data.latest_metrics, 'system.');
                    updateMetrics('workflow-metrics', data.latest_metrics, 'workflow.');
                    updateMetrics('performance-metrics', data.latest_metrics, 'parallel.');
                });
        }
        
        function updateMetrics(elementId, metrics, prefix) {
            const container = document.getElementById(elementId);
            container.innerHTML = '';
            
            for (const [name, data] of Object.entries(metrics)) {
                if (name.startsWith(prefix)) {
                    const metricDiv = document.createElement('div');
                    metricDiv.className = 'metric';
                    
                    const displayName = name.replace(prefix, '').replace(/\\./g, ' ');
                    const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
                    
                    metricDiv.innerHTML = `
                        <div class="metric-name">${displayName}</div>
                        <div class="metric-value">${data.value.toFixed(2)}</div>
                        <div class="timestamp">${timestamp}</div>
                    `;
                    
                    container.appendChild(metricDiv);
                }
            }
        }
        
        // Auto-refresh every 5 seconds
        setInterval(refreshMetrics, 5000);
        
        // Initial load
        refreshMetrics();
    </script>
</body>
</html>
        '''

def start_dashboard_server(port: int = 8080, collector: MetricsCollector = None):
    """Start the metrics dashboard HTTP server"""
    handler = lambda *args, **kwargs: MetricsDashboardHandler(*args, collector=collector, **kwargs)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Metrics dashboard running at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Create metrics collector
    collector = MetricsCollector()
    
    # Start collection
    collector.start_collection(interval=5)
    
    print("Metrics collector started")
    print("Starting dashboard server...")
    
    # Start dashboard server
    try:
        start_dashboard_server(port=8080, collector=collector)
    except KeyboardInterrupt:
        print("\nShutting down...")
        collector.stop_collection()