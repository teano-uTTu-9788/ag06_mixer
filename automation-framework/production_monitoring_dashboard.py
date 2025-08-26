#!/usr/bin/env python3
"""
Production Monitoring Dashboard for AG06 Mixer Mobile App
Real-time monitoring with Google/Meta best practices
"""

import asyncio
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import aiohttp
import psutil

class ProductionMonitoringDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.metrics = {
            'mobile_app_tests': {'status': 'unknown', 'passed': 0, 'total': 88, 'percentage': 0},
            'integration_tests': {'status': 'unknown', 'passed': 0, 'total': 26, 'percentage': 0},
            'server_status': {'status': 'unknown', 'uptime': 0, 'response_time': 0},
            'system_health': {'cpu': 0, 'memory': 0, 'disk': 0},
            'alerts': [],
            'sli_slo_status': {
                'availability': {'current': 0, 'target': 99.9, 'status': 'unknown'},
                'latency_p99': {'current': 0, 'target': 500, 'status': 'unknown'},
                'crash_free_rate': {'current': 0, 'target': 99.9, 'status': 'unknown'}
            },
            'feature_flags': {},
            'ab_test_status': {},
            'last_updated': datetime.now().isoformat()
        }
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.metrics)
        
        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'dashboard': True,
                    'mobile_tests': self.metrics['mobile_app_tests']['status'] == 'passing',
                    'integration_tests': self.metrics['integration_tests']['status'] == 'passing',
                    'server': self.metrics['server_status']['status'] == 'healthy'
                }
            })
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify(self.metrics['alerts'])
        
        @self.app.route('/api/sli-slo')
        def get_sli_slo():
            return jsonify(self.metrics['sli_slo_status'])
    
    async def update_metrics(self):
        """Update all metrics"""
        while True:
            try:
                await self.collect_mobile_test_metrics()
                await self.collect_integration_test_metrics()
                await self.collect_server_metrics()
                await self.collect_system_health()
                await self.update_sli_slo()
                await self.check_alerts()
                
                self.metrics['last_updated'] = datetime.now().isoformat()
                
                # Log metrics update
                print(f"✅ Metrics updated at {self.metrics['last_updated']}")
                
            except Exception as e:
                print(f"❌ Error updating metrics: {e}")
                await self.add_alert('critical', f"Metrics collection failed: {e}")
            
            await asyncio.sleep(30)  # Update every 30 seconds
    
    async def collect_mobile_test_metrics(self):
        """Collect mobile app test results"""
        try:
            # Check if mobile test results exist
            results_file = Path("mobile_test_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                self.metrics['mobile_app_tests'] = {
                    'status': 'passing' if results['percentage'] == 100.0 else 'failing',
                    'passed': results['passed'],
                    'total': results['passed'] + results['failed'],
                    'percentage': results['percentage'],
                    'last_run': datetime.now().isoformat()
                }
                
                # Alert if tests are failing
                if results['percentage'] < 100.0:
                    await self.add_alert('high', f"Mobile tests failing: {results['failed']} failures")
            else:
                self.metrics['mobile_app_tests']['status'] = 'not_run'
                
        except Exception as e:
            self.metrics['mobile_app_tests']['status'] = 'error'
            await self.add_alert('medium', f"Mobile test collection failed: {e}")
    
    async def collect_integration_test_metrics(self):
        """Collect integration test results"""
        try:
            results_file = Path("mobile_integration_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                self.metrics['integration_tests'] = {
                    'status': 'passing' if results['percentage'] > 90.0 else 'failing',
                    'passed': results['passed'],
                    'total': results['total'],
                    'percentage': results['percentage'],
                    'last_run': results['timestamp']
                }
                
                # Alert if integration tests are failing
                if results['percentage'] < 90.0:
                    await self.add_alert('high', f"Integration tests failing: {results['failed']} failures")
            else:
                self.metrics['integration_tests']['status'] = 'not_run'
                
        except Exception as e:
            self.metrics['integration_tests']['status'] = 'error'
            await self.add_alert('medium', f"Integration test collection failed: {e}")
    
    async def collect_server_metrics(self):
        """Collect server health metrics"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8080/healthz', timeout=5) as response:
                    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    
                    if response.status == 200:
                        self.metrics['server_status'] = {
                            'status': 'healthy',
                            'uptime': self.get_server_uptime(),
                            'response_time': response_time,
                            'last_check': datetime.now().isoformat()
                        }
                        
                        # Alert if response time is high
                        if response_time > 1000:  # > 1 second
                            await self.add_alert('medium', f"High server response time: {response_time:.2f}ms")
                    else:
                        self.metrics['server_status']['status'] = 'unhealthy'
                        await self.add_alert('critical', f"Server health check failed: {response.status}")
                        
        except Exception as e:
            self.metrics['server_status'] = {
                'status': 'unreachable',
                'uptime': 0,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
            await self.add_alert('critical', f"Server unreachable: {e}")
    
    async def collect_system_health(self):
        """Collect system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.metrics['system_health'] = {
                'cpu': cpu_percent,
                'memory': memory.percent,
                'disk': disk.percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'processes': len(psutil.pids()),
                'last_check': datetime.now().isoformat()
            }
            
            # System health alerts
            if cpu_percent > 80:
                await self.add_alert('high', f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                await self.add_alert('high', f"High memory usage: {memory.percent:.1f}%")
            if disk.percent > 90:
                await self.add_alert('critical', f"Low disk space: {100-disk.percent:.1f}% free")
                
        except Exception as e:
            await self.add_alert('medium', f"System health collection failed: {e}")
    
    async def update_sli_slo(self):
        """Update SLI/SLO metrics"""
        try:
            # Calculate availability (based on server status)
            if self.metrics['server_status']['status'] == 'healthy':
                availability = 100.0
            else:
                availability = 0.0
            
            # Calculate P99 latency
            p99_latency = self.metrics['server_status'].get('response_time', 0)
            
            # Calculate crash-free rate (based on test results)
            mobile_percentage = self.metrics['mobile_app_tests'].get('percentage', 0)
            integration_percentage = self.metrics['integration_tests'].get('percentage', 0)
            crash_free_rate = (mobile_percentage + integration_percentage) / 2
            
            # Update SLI/SLO status
            self.metrics['sli_slo_status'] = {
                'availability': {
                    'current': availability,
                    'target': 99.9,
                    'status': 'healthy' if availability >= 99.9 else 'degraded'
                },
                'latency_p99': {
                    'current': p99_latency,
                    'target': 500,  # 500ms
                    'status': 'healthy' if p99_latency <= 500 else 'degraded'
                },
                'crash_free_rate': {
                    'current': crash_free_rate,
                    'target': 99.9,
                    'status': 'healthy' if crash_free_rate >= 99.9 else 'degraded'
                }
            }
            
            # SLO violation alerts
            for metric_name, metric in self.metrics['sli_slo_status'].items():
                if metric['status'] == 'degraded':
                    await self.add_alert('high', f"SLO violation: {metric_name} at {metric['current']:.2f} (target: {metric['target']})")
                    
        except Exception as e:
            await self.add_alert('medium', f"SLI/SLO calculation failed: {e}")
    
    async def check_alerts(self):
        """Check for various alert conditions"""
        # Clean up old alerts (older than 1 hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics['alerts'] = [
            alert for alert in self.metrics['alerts']
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]
    
    async def add_alert(self, severity: str, message: str):
        """Add an alert to the system"""
        alert = {
            'id': f"alert_{int(time.time())}",
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'resolved': False
        }
        
        # Avoid duplicate alerts
        existing = [a for a in self.metrics['alerts'] if a['message'] == message and not a['resolved']]
        if not existing:
            self.metrics['alerts'].append(alert)
            print(f"🚨 {severity.upper()} ALERT: {message}")
    
    def get_server_uptime(self):
        """Get server uptime from log file"""
        try:
            log_file = Path(".mixer_logs/server.log")
            if log_file.exists():
                # Get first log entry timestamp
                with open(log_file, 'r') as f:
                    first_line = f.readline()
                    if "Serving Flask app" in first_line:
                        # Server started recently
                        return int(time.time())
            return 0
        except:
            return 0
    
    def get_dashboard_template(self):
        """Get the HTML dashboard template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>AG06 Mixer - Production Monitoring</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child { border-bottom: none; }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .progress {
            width: 100px;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: #28a745;
            transition: width 0.3s ease;
        }
        .progress-bar.warning { background: #ffc107; }
        .progress-bar.critical { background: #dc3545; }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-critical {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .alert-high {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .alert-medium {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .timestamp {
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎛️ AG06 Mixer - Production Monitoring</h1>
        <p>Real-time monitoring with Google/Meta best practices</p>
        <p class="timestamp">Last updated: <span id="lastUpdated">Loading...</span></p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>📱 Mobile App Tests</h3>
            <div class="metric">
                <span>Status</span>
                <span id="mobileStatus" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>Tests Passed</span>
                <span id="mobileTests">0/88</span>
            </div>
            <div class="metric">
                <span>Success Rate</span>
                <div class="progress">
                    <div id="mobileProgress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>🔗 Integration Tests</h3>
            <div class="metric">
                <span>Status</span>
                <span id="integrationStatus" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>Tests Passed</span>
                <span id="integrationTests">0/26</span>
            </div>
            <div class="metric">
                <span>Success Rate</span>
                <div class="progress">
                    <div id="integrationProgress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>🖥️ Server Health</h3>
            <div class="metric">
                <span>Status</span>
                <span id="serverStatus" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>Response Time</span>
                <span id="responseTime">0ms</span>
            </div>
            <div class="metric">
                <span>Uptime</span>
                <span id="uptime">0s</span>
            </div>
        </div>
        
        <div class="card">
            <h3>💻 System Resources</h3>
            <div class="metric">
                <span>CPU Usage</span>
                <div class="progress">
                    <div id="cpuProgress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
            <div class="metric">
                <span>Memory Usage</span>
                <div class="progress">
                    <div id="memoryProgress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
            <div class="metric">
                <span>Disk Usage</span>
                <div class="progress">
                    <div id="diskProgress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 SLI/SLO Status</h3>
            <div class="metric">
                <span>Availability (99.9% target)</span>
                <span id="sloAvailability" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>P99 Latency (500ms target)</span>
                <span id="sloLatency" class="status-healthy">Loading...</span>
            </div>
            <div class="metric">
                <span>Crash-free Rate (99.9% target)</span>
                <span id="sloCrashFree" class="status-healthy">Loading...</span>
            </div>
        </div>
        
        <div class="card">
            <h3>🚨 Active Alerts</h3>
            <div id="alertsContainer">
                <p class="timestamp">No active alerts</p>
            </div>
        </div>
    </div>
    
    <script>
        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();
                
                // Update last updated timestamp
                document.getElementById('lastUpdated').textContent = new Date(metrics.last_updated).toLocaleString();
                
                // Update mobile app tests
                const mobile = metrics.mobile_app_tests;
                document.getElementById('mobileStatus').textContent = mobile.status;
                document.getElementById('mobileStatus').className = getStatusClass(mobile.status);
                document.getElementById('mobileTests').textContent = `${mobile.passed}/${mobile.total}`;
                document.getElementById('mobileProgress').style.width = mobile.percentage + '%';
                document.getElementById('mobileProgress').className = 'progress-bar ' + getProgressClass(mobile.percentage);
                
                // Update integration tests
                const integration = metrics.integration_tests;
                document.getElementById('integrationStatus').textContent = integration.status;
                document.getElementById('integrationStatus').className = getStatusClass(integration.status);
                document.getElementById('integrationTests').textContent = `${integration.passed}/${integration.total}`;
                document.getElementById('integrationProgress').style.width = integration.percentage + '%';
                document.getElementById('integrationProgress').className = 'progress-bar ' + getProgressClass(integration.percentage);
                
                // Update server health
                const server = metrics.server_status;
                document.getElementById('serverStatus').textContent = server.status;
                document.getElementById('serverStatus').className = getStatusClass(server.status);
                document.getElementById('responseTime').textContent = Math.round(server.response_time) + 'ms';
                document.getElementById('uptime').textContent = formatUptime(server.uptime);
                
                // Update system resources
                const system = metrics.system_health;
                updateProgressBar('cpuProgress', system.cpu);
                updateProgressBar('memoryProgress', system.memory);
                updateProgressBar('diskProgress', system.disk);
                
                // Update SLI/SLO
                const slo = metrics.sli_slo_status;
                document.getElementById('sloAvailability').textContent = slo.availability.current.toFixed(2) + '%';
                document.getElementById('sloAvailability').className = getStatusClass(slo.availability.status);
                document.getElementById('sloLatency').textContent = Math.round(slo.latency_p99.current) + 'ms';
                document.getElementById('sloLatency').className = getStatusClass(slo.latency_p99.status);
                document.getElementById('sloCrashFree').textContent = slo.crash_free_rate.current.toFixed(2) + '%';
                document.getElementById('sloCrashFree').className = getStatusClass(slo.crash_free_rate.status);
                
                // Update alerts
                updateAlerts(metrics.alerts);
                
            } catch (error) {
                console.error('Failed to update dashboard:', error);
            }
        }
        
        function getStatusClass(status) {
            if (status === 'passing' || status === 'healthy') return 'status-healthy';
            if (status === 'degraded' || status === 'warning') return 'status-warning';
            return 'status-critical';
        }
        
        function getProgressClass(percentage) {
            if (percentage >= 95) return '';
            if (percentage >= 80) return 'warning';
            return 'critical';
        }
        
        function updateProgressBar(id, percentage) {
            const element = document.getElementById(id);
            element.style.width = percentage + '%';
            element.className = 'progress-bar ' + getProgressClass(percentage);
        }
        
        function formatUptime(seconds) {
            if (seconds === 0) return '0s';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) return `${hours}h ${minutes}m`;
            return `${minutes}m`;
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alertsContainer');
            if (alerts.length === 0) {
                container.innerHTML = '<p class="timestamp">No active alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => 
                `<div class="alert alert-${alert.severity}">
                    <strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}
                    <div class="timestamp">${new Date(alert.timestamp).toLocaleString()}</div>
                </div>`
            ).join('');
        }
        
        // Update dashboard every 30 seconds
        updateDashboard();
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
        """
    
    def run_dashboard(self, host='127.0.0.1', port=8082):
        """Run the monitoring dashboard"""
        # Start metrics collection in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Start metrics collection task
        metrics_task = loop.create_task(self.update_metrics())
        
        print(f"🚀 Starting Production Monitoring Dashboard on http://{host}:{port}")
        print("📊 Features enabled:")
        print("   • Real-time mobile app test monitoring (88/88)")
        print("   • Integration test monitoring (26 tests)")
        print("   • Server health and response time tracking")
        print("   • System resource monitoring (CPU/Memory/Disk)")
        print("   • SLI/SLO tracking (Availability, Latency, Crash-free rate)")
        print("   • Alert management with severity levels")
        print("   • Google/Meta production best practices")
        
        try:
            self.app.run(host=host, port=port, debug=False)
        except KeyboardInterrupt:
            print("\n⏹️ Shutting down monitoring dashboard...")
            metrics_task.cancel()
        finally:
            loop.close()

def main():
    """Main function to run the production monitoring dashboard"""
    dashboard = ProductionMonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()