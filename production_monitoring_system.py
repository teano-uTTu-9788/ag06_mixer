#!/usr/bin/env python3
"""
Production Monitoring & Alerting System
Real-time monitoring, alerting, and automated responses for AG06 Workflow System
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import production components
from integrated_workflow_system import IntegratedWorkflowSystem
from specialized_workflow_agent import SpecializedWorkflowAgent

@dataclass
class Alert:
    id: str
    level: str  # INFO, WARNING, CRITICAL
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class HealthMetrics:
    component: str
    status: str
    score: float
    response_time_ms: float
    error_count: int
    last_check: datetime

class ProductionMonitor:
    """Production monitoring system with automated alerts"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.metrics_history: Dict[str, List[HealthMetrics]] = {}
        self.system = None
        self.agents: Dict[str, SpecializedWorkflowAgent] = {}
        self.monitoring_active = False
        
        # Thresholds for alerting
        self.thresholds = {
            'response_time_ms': 5000,  # 5 seconds
            'error_rate_percent': 5.0,  # 5%
            'health_score_min': 70.0,  # Minimum health score
            'queue_size_max': 100,     # Maximum queue size
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('production_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize production monitoring system"""
        self.logger.info("🔧 Initializing Production Monitoring System...")
        
        # Initialize core system
        self.system = IntegratedWorkflowSystem()
        
        # Initialize production agents
        agent_configs = [
            ("primary_agent", "Primary workflow processing agent"),
            ("backup_agent", "Backup and failover agent"),
            ("monitor_agent", "System monitoring and health checks")
        ]
        
        for agent_id, description in agent_configs:
            agent = SpecializedWorkflowAgent(agent_id)
            await agent.initialize()
            self.agents[agent_id] = agent
            self.logger.info(f"✅ Agent {agent_id} initialized: {description}")
        
        self.monitoring_active = True
        self.logger.info("✅ Production Monitoring System initialized")
        
        return True
    
    async def collect_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Collect comprehensive health metrics from all components"""
        metrics = {}
        
        try:
            # System health
            start_time = time.time()
            system_health = await self.system.get_system_health()
            response_time = (time.time() - start_time) * 1000
            
            metrics['integrated_system'] = HealthMetrics(
                component='integrated_system',
                status=system_health.get('status', 'unknown'),
                score=float(system_health.get('score', 0)),
                response_time_ms=response_time,
                error_count=0,
                last_check=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system health: {e}")
            metrics['integrated_system'] = HealthMetrics(
                component='integrated_system',
                status='error',
                score=0.0,
                response_time_ms=999999,
                error_count=1,
                last_check=datetime.now()
            )
        
        # Agent health metrics
        for agent_id, agent in self.agents.items():
            try:
                start_time = time.time()
                agent_status = await agent.get_agent_status()
                response_time = (time.time() - start_time) * 1000
                
                metrics[agent_id] = HealthMetrics(
                    component=agent_id,
                    status=agent_status.get('status', 'unknown'),
                    score=float(agent_status.get('performance', {}).get('success_rate_percent', 0)),
                    response_time_ms=response_time,
                    error_count=0,
                    last_check=datetime.now()
                )
                
            except Exception as e:
                self.logger.error(f"Failed to collect {agent_id} health: {e}")
                metrics[agent_id] = HealthMetrics(
                    component=agent_id,
                    status='error',
                    score=0.0,
                    response_time_ms=999999,
                    error_count=1,
                    last_check=datetime.now()
                )
        
        return metrics
    
    def analyze_metrics(self, metrics: Dict[str, HealthMetrics]) -> List[Alert]:
        """Analyze metrics and generate alerts"""
        new_alerts = []
        
        for component_id, metric in metrics.items():
            # Response time alert
            if metric.response_time_ms > self.thresholds['response_time_ms']:
                alert = Alert(
                    id=f"{component_id}_response_time_{int(time.time())}",
                    level='WARNING',
                    component=component_id,
                    message=f"High response time: {metric.response_time_ms:.1f}ms (threshold: {self.thresholds['response_time_ms']}ms)",
                    timestamp=datetime.now()
                )
                new_alerts.append(alert)
            
            # Health score alert  
            if metric.score < self.thresholds['health_score_min']:
                level = 'CRITICAL' if metric.score < 50 else 'WARNING'
                alert = Alert(
                    id=f"{component_id}_health_score_{int(time.time())}",
                    level=level,
                    component=component_id,
                    message=f"Low health score: {metric.score:.1f} (threshold: {self.thresholds['health_score_min']})",
                    timestamp=datetime.now()
                )
                new_alerts.append(alert)
            
            # Error count alert
            if metric.error_count > 0:
                alert = Alert(
                    id=f"{component_id}_errors_{int(time.time())}",
                    level='CRITICAL',
                    component=component_id,
                    message=f"Component errors detected: {metric.error_count}",
                    timestamp=datetime.now()
                )
                new_alerts.append(alert)
            
            # Status alert
            if metric.status in ['error', 'failed', 'unknown']:
                alert = Alert(
                    id=f"{component_id}_status_{int(time.time())}",
                    level='CRITICAL',
                    component=component_id,
                    message=f"Component status: {metric.status}",
                    timestamp=datetime.now()
                )
                new_alerts.append(alert)
        
        return new_alerts
    
    async def handle_alerts(self, alerts: List[Alert]):
        """Handle and respond to alerts automatically"""
        for alert in alerts:
            self.alerts.append(alert)
            
            # Log alert
            log_level = getattr(logging, alert.level)
            self.logger.log(log_level, f"🚨 ALERT [{alert.level}] {alert.component}: {alert.message}")
            
            # Automatic responses based on alert level and type
            if alert.level == 'CRITICAL':
                await self.handle_critical_alert(alert)
            elif alert.level == 'WARNING':
                await self.handle_warning_alert(alert)
            
            # Store alert for dashboard
            await self.store_alert(alert)
    
    async def handle_critical_alert(self, alert: Alert):
        """Handle critical alerts with automatic responses"""
        self.logger.critical(f"🔴 CRITICAL ALERT: {alert.component} - {alert.message}")
        
        # Automatic responses
        if 'response_time' in alert.message and alert.component in self.agents:
            # Restart agent if response time is too high
            try:
                self.logger.info(f"🔄 Attempting to restart {alert.component}...")
                agent = self.agents[alert.component]
                await agent.initialize()  # Re-initialize agent
                self.logger.info(f"✅ Successfully restarted {alert.component}")
                
                # Mark alert as resolved
                alert.resolved = True
                alert.resolution_time = datetime.now()
                
            except Exception as e:
                self.logger.error(f"❌ Failed to restart {alert.component}: {e}")
        
        # If system health is critical, attempt recovery
        if alert.component == 'integrated_system' and 'health_score' in alert.message:
            await self.attempt_system_recovery()
    
    async def handle_warning_alert(self, alert: Alert):
        """Handle warning alerts with monitoring and logging"""
        self.logger.warning(f"🟡 WARNING ALERT: {alert.component} - {alert.message}")
        
        # Increase monitoring frequency for this component
        self.logger.info(f"📊 Increasing monitoring frequency for {alert.component}")
    
    async def attempt_system_recovery(self):
        """Attempt automatic system recovery"""
        self.logger.info("🔄 Attempting system recovery...")
        
        try:
            # Re-initialize system
            self.system = IntegratedWorkflowSystem()
            
            # Test system with simple workflow
            test_result = await self.system.execute_workflow(
                "recovery_test",
                "test",
                ["test_step"],
                {"recovery": True}
            )
            
            if test_result.get('status') == 'success':
                self.logger.info("✅ System recovery successful")
                return True
            else:
                self.logger.error("❌ System recovery failed - workflow execution failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ System recovery failed: {e}")
            return False
    
    async def store_alert(self, alert: Alert):
        """Store alert for dashboard and historical tracking"""
        alert_data = asdict(alert)
        alert_data['timestamp'] = alert.timestamp.isoformat()
        if alert.resolution_time:
            alert_data['resolution_time'] = alert.resolution_time.isoformat()
        
        # Store to file for dashboard
        alerts_file = Path('production_alerts.json')
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
        else:
            alerts_data = []
        
        alerts_data.append(alert_data)
        
        # Keep only last 1000 alerts
        if len(alerts_data) > 1000:
            alerts_data = alerts_data[-1000:]
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        metrics = await self.collect_health_metrics()
        
        # Calculate overall system health
        total_score = sum(m.score for m in metrics.values())
        avg_score = total_score / len(metrics) if metrics else 0
        
        # Recent alerts summary
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]
        critical_count = len([a for a in recent_alerts if a.level == 'CRITICAL'])
        warning_count = len([a for a in recent_alerts if a.level == 'WARNING'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health_score': avg_score,
            'system_status': 'healthy' if avg_score > 80 else 'degraded' if avg_score > 50 else 'critical',
            'components': {name: asdict(metric) for name, metric in metrics.items()},
            'alerts_24h': {
                'total': len(recent_alerts),
                'critical': critical_count,
                'warning': warning_count,
                'resolved': len([a for a in recent_alerts if a.resolved])
            },
            'monitoring_active': self.monitoring_active
        }
        
        return report
    
    async def start_continuous_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring loop"""
        self.logger.info(f"🔄 Starting continuous monitoring (interval: {interval_seconds}s)...")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self.collect_health_metrics()
                
                # Store metrics history
                for component_id, metric in metrics.items():
                    if component_id not in self.metrics_history:
                        self.metrics_history[component_id] = []
                    
                    self.metrics_history[component_id].append(metric)
                    
                    # Keep only last 1000 metrics per component
                    if len(self.metrics_history[component_id]) > 1000:
                        self.metrics_history[component_id] = self.metrics_history[component_id][-1000:]
                
                # Analyze for alerts
                new_alerts = self.analyze_metrics(metrics)
                if new_alerts:
                    await self.handle_alerts(new_alerts)
                
                # Generate and store health report
                report = await self.generate_health_report()
                with open('production_health_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Log status
                if len(metrics) > 0:
                    avg_score = sum(m.score for m in metrics.values()) / len(metrics)
                    self.logger.info(f"📊 System Health: {avg_score:.1f}/100 | Components: {len(metrics)} | Alerts: {len(new_alerts)}")
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring cycle error: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(interval_seconds)
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("⏹️ Production monitoring stopped")

async def main():
    """Main production monitoring entry point"""
    monitor = ProductionMonitor()
    
    try:
        # Initialize monitoring system
        await monitor.initialize()
        
        # Generate initial health report
        initial_report = await monitor.generate_health_report()
        print("\n🏥 INITIAL PRODUCTION HEALTH REPORT:")
        print(f"Overall Health Score: {initial_report['overall_health_score']:.1f}/100")
        print(f"System Status: {initial_report['system_status'].upper()}")
        print(f"Active Components: {len(initial_report['components'])}")
        print(f"Monitoring Active: {initial_report['monitoring_active']}")
        
        # Start continuous monitoring
        print("\n🔄 Starting continuous production monitoring...")
        await monitor.start_continuous_monitoring(interval_seconds=30)
        
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
        await monitor.stop_monitoring()
    except Exception as e:
        print(f"\n❌ Monitoring system error: {e}")
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())