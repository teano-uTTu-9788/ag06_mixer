#!/usr/bin/env python3
"""
Aioke Monitoring and Alerting System
Real-time monitoring with health checks, metrics collection, and alerting
"""

import asyncio
import json
import logging
import time
import psutil
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheck:
    """Health check result data structure"""
    service: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    response_time: float
    timestamp: datetime
    message: str
    details: Dict[str, Any] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    uptime_seconds: int
    timestamp: datetime

class MonitoringSystem:
    """
    Comprehensive monitoring system for Aioke components
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.health_checks: List[HealthCheck] = []
        self.metrics_history: List[SystemMetrics] = []
        self.alert_history: List[Dict] = []
        self.start_time = time.time()
        
        # Monitoring intervals
        self.health_check_interval = 30  # seconds
        self.metrics_interval = 10      # seconds
        self.alert_cooldown = 300       # 5 minutes
        self.last_alerts = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        return {
            'services': {
                'backend': {
                    'url': 'http://localhost:8080/health',
                    'timeout': 10,
                    'critical_response_time': 5.0
                },
                'frontend': {
                    'url': 'http://localhost:3000/health',
                    'timeout': 15,
                    'critical_response_time': 3.0
                }
            },
            'thresholds': {
                'cpu_warning': 70,
                'cpu_critical': 90,
                'memory_warning': 80,
                'memory_critical': 95,
                'disk_warning': 85,
                'disk_critical': 95,
                'response_time_warning': 2.0,
                'response_time_critical': 5.0
            },
            'alerts': {
                'email': {
                    'enabled': False,  # Set to True when configured
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                },
                'webhook': {
                    'enabled': False,
                    'url': ''
                }
            }
        }
    
    async def check_service_health(self, service_name: str, config: Dict) -> HealthCheck:
        """Perform health check for a service"""
        start_time = time.time()
        
        try:
            response = requests.get(
                config['url'], 
                timeout=config['timeout'],
                headers={'User-Agent': 'Aioke-Monitor/1.0'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                status = 'healthy'
                message = 'Service responding normally'
                
                # Check response time thresholds
                if response_time > config.get('critical_response_time', 5.0):
                    status = 'critical'
                    message = f'Response time critical: {response_time:.2f}s'
                elif response_time > self.config['thresholds']['response_time_warning']:
                    status = 'warning'
                    message = f'Response time slow: {response_time:.2f}s'
            
            else:
                status = 'critical'
                message = f'HTTP {response.status_code}: {response.reason}'
                response_time = time.time() - start_time
            
            # Try to parse JSON response for additional details
            details = {}
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    details = response.json()
            except:
                details = {'raw_response': response.text[:200]}
        
        except requests.exceptions.Timeout:
            status = 'critical'
            message = f'Timeout after {config["timeout"]}s'
            response_time = config['timeout']
            details = {'error': 'timeout'}
        
        except requests.exceptions.ConnectionError:
            status = 'critical'
            message = 'Connection refused'
            response_time = time.time() - start_time
            details = {'error': 'connection_refused'}
        
        except Exception as e:
            status = 'unknown'
            message = f'Error: {str(e)}'
            response_time = time.time() - start_time
            details = {'error': str(e)}
        
        return HealthCheck(
            service=service_name,
            status=status,
            response_time=response_time,
            timestamp=datetime.now(),
            message=message,
            details=details
        )
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage (current directory)
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Active network connections
        connections = len(psutil.net_connections())
        
        # System uptime
        uptime_seconds = time.time() - psutil.boot_time()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_io,
            active_connections=connections,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now()
        )
    
    def analyze_metrics(self, metrics: SystemMetrics) -> List[Dict]:
        """Analyze metrics and generate alerts if needed"""
        alerts = []
        thresholds = self.config['thresholds']
        
        # CPU alerts
        if metrics.cpu_percent >= thresholds['cpu_critical']:
            alerts.append({
                'level': 'critical',
                'metric': 'cpu',
                'value': metrics.cpu_percent,
                'threshold': thresholds['cpu_critical'],
                'message': f'CPU usage critical: {metrics.cpu_percent}%'
            })
        elif metrics.cpu_percent >= thresholds['cpu_warning']:
            alerts.append({
                'level': 'warning',
                'metric': 'cpu',
                'value': metrics.cpu_percent,
                'threshold': thresholds['cpu_warning'],
                'message': f'CPU usage high: {metrics.cpu_percent}%'
            })
        
        # Memory alerts
        if metrics.memory_percent >= thresholds['memory_critical']:
            alerts.append({
                'level': 'critical',
                'metric': 'memory',
                'value': metrics.memory_percent,
                'threshold': thresholds['memory_critical'],
                'message': f'Memory usage critical: {metrics.memory_percent}%'
            })
        elif metrics.memory_percent >= thresholds['memory_warning']:
            alerts.append({
                'level': 'warning',
                'metric': 'memory',
                'value': metrics.memory_percent,
                'threshold': thresholds['memory_warning'],
                'message': f'Memory usage high: {metrics.memory_percent}%'
            })
        
        # Disk alerts
        if metrics.disk_percent >= thresholds['disk_critical']:
            alerts.append({
                'level': 'critical',
                'metric': 'disk',
                'value': metrics.disk_percent,
                'threshold': thresholds['disk_critical'],
                'message': f'Disk usage critical: {metrics.disk_percent:.1f}%'
            })
        elif metrics.disk_percent >= thresholds['disk_warning']:
            alerts.append({
                'level': 'warning',
                'metric': 'disk',
                'value': metrics.disk_percent,
                'threshold': thresholds['disk_warning'],
                'message': f'Disk usage high: {metrics.disk_percent:.1f}%'
            })
        
        return alerts
    
    async def send_alert(self, alert: Dict):
        """Send alert via configured channels"""
        alert_key = f"{alert['metric']}_{alert['level']}"
        current_time = time.time()
        
        # Check cooldown period
        if alert_key in self.last_alerts:
            if current_time - self.last_alerts[alert_key] < self.alert_cooldown:
                return  # Skip alert due to cooldown
        
        # Update last alert time
        self.last_alerts[alert_key] = current_time
        
        # Add timestamp to alert
        alert['timestamp'] = datetime.now().isoformat()
        
        # Log alert
        logger.warning(f"ALERT [{alert['level'].upper()}]: {alert['message']}")
        
        # Send email alert
        if self.config['alerts']['email']['enabled']:
            await self._send_email_alert(alert)
        
        # Send webhook alert
        if self.config['alerts']['webhook']['enabled']:
            await self._send_webhook_alert(alert)
        
        # Store in alert history
        self.alert_history.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    async def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            email_config = self.config['alerts']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Aioke Alert - {alert['level'].title()}: {alert['metric'].title()}"
            
            body = f"""
Aioke System Alert

Level: {alert['level'].upper()}
Metric: {alert['metric']}
Value: {alert['value']}
Threshold: {alert['threshold']}
Message: {alert['message']}
Timestamp: {alert['timestamp']}

System Status Dashboard: http://localhost:8080/health
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            text = msg.as_string()
            server.sendmail(email_config['username'], email_config['recipients'], text)
            server.quit()
            
            logger.info(f"Email alert sent for {alert['metric']} {alert['level']}")
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert"""
        try:
            webhook_url = self.config['alerts']['webhook']['url']
            
            payload = {
                'text': f"🚨 Aioke Alert: {alert['message']}",
                'alert': alert
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert['metric']} {alert['level']}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        current_time = datetime.now()
        
        # Latest health checks
        latest_health = {}
        for check in self.health_checks[-10:]:  # Last 10 checks
            if check.service not in latest_health:
                latest_health[check.service] = check
        
        # Latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Recent alerts (last hour)
        one_hour_ago = current_time - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > one_hour_ago
        ]
        
        # Calculate availability (last 24 hours)
        availability = {}
        for service in self.config['services']:
            service_checks = [
                check for check in self.health_checks
                if check.service == service and 
                check.timestamp > current_time - timedelta(hours=24)
            ]
            
            if service_checks:
                healthy_checks = sum(1 for check in service_checks if check.status == 'healthy')
                availability[service] = (healthy_checks / len(service_checks)) * 100
            else:
                availability[service] = 0
        
        return {
            'timestamp': current_time.isoformat(),
            'uptime': time.time() - self.start_time,
            'services': {
                service: asdict(check) for service, check in latest_health.items()
            },
            'system_metrics': asdict(latest_metrics) if latest_metrics else None,
            'availability_24h': availability,
            'recent_alerts': recent_alerts,
            'alert_summary': {
                'critical': sum(1 for a in recent_alerts if a['level'] == 'critical'),
                'warning': sum(1 for a in recent_alerts if a['level'] == 'warning')
            }
        }
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting Aioke monitoring system...")
        
        last_health_check = 0
        last_metrics_check = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Health checks
                if current_time - last_health_check >= self.health_check_interval:
                    logger.info("Performing health checks...")
                    
                    for service_name, config in self.config['services'].items():
                        health_check = await self.check_service_health(service_name, config)
                        self.health_checks.append(health_check)
                        
                        # Alert on service issues
                        if health_check.status in ['critical', 'unknown']:
                            await self.send_alert({
                                'level': 'critical',
                                'metric': 'service_health',
                                'service': service_name,
                                'value': health_check.status,
                                'threshold': 'healthy',
                                'message': f'Service {service_name} is {health_check.status}: {health_check.message}'
                            })
                        elif health_check.status == 'warning':
                            await self.send_alert({
                                'level': 'warning',
                                'metric': 'service_health',
                                'service': service_name,
                                'value': health_check.status,
                                'threshold': 'healthy',
                                'message': f'Service {service_name} has issues: {health_check.message}'
                            })
                    
                    # Keep only last 1000 health checks
                    if len(self.health_checks) > 1000:
                        self.health_checks = self.health_checks[-1000:]
                    
                    last_health_check = current_time
                
                # System metrics
                if current_time - last_metrics_check >= self.metrics_interval:
                    logger.debug("Collecting system metrics...")
                    
                    metrics = self.collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Analyze metrics and send alerts
                    alerts = self.analyze_metrics(metrics)
                    for alert in alerts:
                        await self.send_alert(alert)
                    
                    # Keep only last 8640 metrics (24 hours at 10s intervals)
                    if len(self.metrics_history) > 8640:
                        self.metrics_history = self.metrics_history[-8640:]
                    
                    last_metrics_check = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def save_status_report(self, filename: str = "monitoring_status.json"):
        """Save current status report to file"""
        report = self.generate_status_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Status report saved to {filename}")

async def main():
    """Main entry point for monitoring system"""
    monitor = MonitoringSystem()
    
    # Save status report every 5 minutes
    async def save_reports():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            try:
                monitor.save_status_report()
            except Exception as e:
                logger.error(f"Failed to save status report: {e}")
    
    # Run monitoring and report saving concurrently
    await asyncio.gather(
        monitor.monitoring_loop(),
        save_reports()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Monitoring system stopped")