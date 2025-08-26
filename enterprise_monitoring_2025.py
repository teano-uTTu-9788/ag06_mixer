#!/usr/bin/env python3
"""
Enterprise Monitoring System 2025
Google SRE + Netflix + Amazon observability patterns
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import aiohttp
import sqlite3
from pathlib import Path

# Google SRE: SLI/SLO definitions
@dataclass
class SLI:
    """Service Level Indicator"""
    name: str
    description: str
    query: str
    good_threshold: float
    total_threshold: float
    window: str = "5m"

@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    description: str
    target: float  # e.g., 99.9 for 99.9% availability
    window: str = "30d"
    slis: List[SLI] = None

@dataclass
class ErrorBudget:
    """Error budget calculation"""
    slo_target: float
    total_requests: int
    failed_requests: int
    budget_consumed: float
    budget_remaining: float
    burn_rate: float
    alert_threshold: float = 0.1  # Alert when 10% budget consumed

# Netflix: Chaos Engineering patterns
class ChaosMonkey:
    """Lightweight chaos engineering for resilience testing"""
    
    def __init__(self):
        self.experiments = []
        self.enabled = False
    
    async def random_latency_injection(self, service_name: str, latency_ms: int):
        """Inject random latency to test timeout handling"""
        if not self.enabled:
            return
            
        await asyncio.sleep(latency_ms / 1000.0)
        logging.info(f"Chaos: Injected {latency_ms}ms latency to {service_name}")
    
    async def simulate_service_failure(self, service_name: str, failure_rate: float):
        """Simulate service failures"""
        import random
        if not self.enabled or random.random() > failure_rate:
            return False
            
        logging.warning(f"Chaos: Simulating failure for {service_name}")
        return True

# Amazon: CloudWatch-style metrics collection
class MetricsCollector:
    """CloudWatch-inspired metrics collection"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics = {}
        
    def put_metric(self, namespace: str, metric_name: str, value: float, 
                  unit: str = "Count", dimensions: Dict[str, str] = None):
        """Put custom metric (CloudWatch API style)"""
        key = f"{namespace}.{metric_name}"
        timestamp = datetime.now(timezone.utc)
        
        metric_point = {
            "timestamp": timestamp.isoformat(),
            "value": value,
            "unit": unit,
            "dimensions": dimensions or {}
        }
        
        self.metrics[key].append(metric_point)
    
    def get_metric_statistics(self, namespace: str, metric_name: str, 
                            start_time: datetime, end_time: datetime,
                            period: int = 300, statistic: str = "Average"):
        """Get metric statistics"""
        key = f"{namespace}.{metric_name}"
        data_points = []
        
        for point in self.metrics[key]:
            point_time = datetime.fromisoformat(point["timestamp"])
            if start_time <= point_time <= end_time:
                data_points.append(point["value"])
        
        if not data_points:
            return None
            
        if statistic == "Average":
            return sum(data_points) / len(data_points)
        elif statistic == "Sum":
            return sum(data_points)
        elif statistic == "Maximum":
            return max(data_points)
        elif statistic == "Minimum":
            return min(data_points)
        
        return None

# Google: Structured alerting with escalation
class AlertManager:
    """Google-style alerting with escalation policies"""
    
    def __init__(self):
        self.alerts = []
        self.escalation_policies = {
            "critical": {"escalate_after": 300, "max_escalations": 3},
            "warning": {"escalate_after": 900, "max_escalations": 2},
            "info": {"escalate_after": None, "max_escalations": 1}
        }
    
    async def fire_alert(self, severity: str, title: str, description: str, 
                        labels: Dict[str, str] = None):
        """Fire an alert with escalation"""
        alert = {
            "id": f"alert_{int(time.time() * 1000)}",
            "severity": severity,
            "title": title,
            "description": description,
            "labels": labels or {},
            "fired_at": datetime.now(timezone.utc).isoformat(),
            "status": "firing",
            "escalation_level": 0
        }
        
        self.alerts.append(alert)
        logging.warning(f"Alert fired: {severity} - {title}")
        
        # Simulate escalation (in production, integrate with PagerDuty/OpsGenie)
        policy = self.escalation_policies.get(severity, {})
        if policy.get("escalate_after"):
            await asyncio.sleep(1)  # Simulate escalation delay
            alert["escalation_level"] += 1
            logging.error(f"Alert escalated: {title} (level {alert['escalation_level']})")

class EnterpriseMonitoring:
    """Comprehensive monitoring system following big tech patterns"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.chaos_monkey = ChaosMonkey()
        self.start_time = time.time()
        
        # Initialize database
        self.db_path = Path("monitoring.db")
        self._init_database()
        
        # Define SLOs
        self.slos = [
            SLO(
                name="API Availability",
                description="99.9% of API requests should succeed",
                target=99.9,
                slis=[
                    SLI(
                        name="success_rate",
                        description="Ratio of successful requests",
                        query="sum(rate(api_requests_total{status=~'2..'}[5m])) / sum(rate(api_requests_total[5m]))",
                        good_threshold=0.999,
                        total_threshold=1.0
                    )
                ]
            ),
            SLO(
                name="API Latency",
                description="95% of requests should complete under 1s",
                target=95.0,
                slis=[
                    SLI(
                        name="latency_p95",
                        description="95th percentile request latency",
                        query="histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
                        good_threshold=1.0,
                        total_threshold=5.0
                    )
                ]
            )
        ]
        
        self.error_budgets = {}
        self._calculate_error_budgets()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp TEXT,
                    namespace TEXT,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    dimensions TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    severity TEXT,
                    title TEXT,
                    description TEXT,
                    labels TEXT,
                    fired_at TEXT,
                    resolved_at TEXT,
                    status TEXT
                )
            """)
    
    def _calculate_error_budgets(self):
        """Calculate error budgets for all SLOs"""
        for slo in self.slos:
            # Simulate metrics for demonstration
            total_requests = 10000
            failed_requests = int(total_requests * (100 - slo.target) / 100)
            
            budget_consumed = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
            budget_remaining = max(0, (100 - slo.target) - budget_consumed)
            burn_rate = budget_consumed / 30  # Per day over 30-day window
            
            self.error_budgets[slo.name] = ErrorBudget(
                slo_target=slo.target,
                total_requests=total_requests,
                failed_requests=failed_requests,
                budget_consumed=budget_consumed,
                budget_remaining=budget_remaining,
                burn_rate=burn_rate
            )
    
    async def collect_api_metrics(self, api_url: str):
        """Collect metrics from API endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Health check
                start_time = time.time()
                async with session.get(f"{api_url}/health") as response:
                    latency = time.time() - start_time
                    
                    self.metrics_collector.put_metric(
                        "API", "health_check_latency", latency * 1000, "Milliseconds"
                    )
                    
                    if response.status == 200:
                        self.metrics_collector.put_metric("API", "health_check_success", 1)
                        logging.info(f"Health check successful: {latency*1000:.2f}ms")
                    else:
                        self.metrics_collector.put_metric("API", "health_check_failure", 1)
                        await self.alert_manager.fire_alert(
                            "critical",
                            "API Health Check Failed",
                            f"Health endpoint returned {response.status}",
                            {"endpoint": "/health", "status": str(response.status)}
                        )
                
                # Metrics endpoint
                try:
                    async with session.get(f"{api_url}/metrics") as response:
                        if response.status == 200:
                            metrics_data = await response.text()
                            self._parse_prometheus_metrics(metrics_data)
                except Exception as e:
                    logging.warning(f"Failed to collect Prometheus metrics: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to collect API metrics: {e}")
            await self.alert_manager.fire_alert(
                "critical",
                "API Unreachable",
                f"Failed to connect to API: {str(e)}",
                {"api_url": api_url}
            )
    
    def _parse_prometheus_metrics(self, metrics_text: str):
        """Parse Prometheus metrics format"""
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
                
            try:
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name = parts[0]
                    metric_value = float(parts[1])
                    
                    # Extract namespace
                    namespace = "Prometheus"
                    if '_' in metric_name:
                        namespace = metric_name.split('_')[0].upper()
                    
                    self.metrics_collector.put_metric(
                        namespace, metric_name, metric_value, "Count"
                    )
            except (ValueError, IndexError):
                continue
    
    async def monitor_error_budgets(self):
        """Monitor error budgets and alert on burn rate"""
        for slo_name, budget in self.error_budgets.items():
            if budget.burn_rate > budget.alert_threshold:
                await self.alert_manager.fire_alert(
                    "warning",
                    f"High Error Budget Burn Rate: {slo_name}",
                    f"Burning {budget.burn_rate:.2f}% of error budget per day",
                    {"slo": slo_name, "burn_rate": str(budget.burn_rate)}
                )
            
            if budget.budget_remaining < 10:  # Less than 10% remaining
                await self.alert_manager.fire_alert(
                    "critical",
                    f"Error Budget Exhausted: {slo_name}",
                    f"Only {budget.budget_remaining:.1f}% error budget remaining",
                    {"slo": slo_name, "remaining": str(budget.budget_remaining)}
                )
    
    async def run_chaos_experiments(self):
        """Run chaos engineering experiments"""
        if not self.chaos_monkey.enabled:
            return
            
        # Random latency injection
        await self.chaos_monkey.random_latency_injection("api", 500)
        
        # Simulate occasional failures
        if await self.chaos_monkey.simulate_service_failure("database", 0.01):
            await self.alert_manager.fire_alert(
                "info",
                "Chaos Experiment: Service Failure",
                "Chaos monkey simulated database failure",
                {"experiment": "service_failure", "service": "database"}
            )
    
    async def generate_sre_report(self) -> Dict[str, Any]:
        """Generate SRE-style reliability report"""
        uptime = time.time() - self.start_time
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_period": f"{uptime:.0f} seconds",
            "slo_compliance": {},
            "error_budgets": {},
            "alert_summary": {
                "total_alerts": len(self.alert_manager.alerts),
                "by_severity": defaultdict(int)
            },
            "key_metrics": {
                "uptime_seconds": uptime,
                "total_metrics_collected": sum(len(m) for m in self.metrics_collector.metrics.values()),
                "chaos_experiments_run": len(self.chaos_monkey.experiments)
            }
        }
        
        # SLO compliance
        for slo in self.slos:
            budget = self.error_budgets.get(slo.name)
            if budget:
                compliance_percent = 100 - (budget.failed_requests / budget.total_requests * 100)
                report["slo_compliance"][slo.name] = {
                    "target": slo.target,
                    "actual": compliance_percent,
                    "compliant": compliance_percent >= slo.target
                }
                
                report["error_budgets"][slo.name] = asdict(budget)
        
        # Alert summary
        for alert in self.alert_manager.alerts:
            report["alert_summary"]["by_severity"][alert["severity"]] += 1
        
        return report
    
    async def start_monitoring(self, api_url: str = "http://localhost:8090", interval: int = 60):
        """Start the monitoring loop"""
        logging.info("Starting enterprise monitoring system")
        
        try:
            while True:
                # Collect metrics
                await self.collect_api_metrics(api_url)
                
                # Check error budgets
                await self.monitor_error_budgets()
                
                # Run chaos experiments (if enabled)
                await self.run_chaos_experiments()
                
                # Generate periodic report
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    report = await self.generate_sre_report()
                    logging.info("SRE Report generated", extra={"report": report})
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
            await self.alert_manager.fire_alert(
                "critical",
                "Monitoring System Failure",
                f"Monitoring system encountered error: {str(e)}",
                {"component": "monitoring_loop"}
            )

async def main():
    """Main monitoring entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitoring = EnterpriseMonitoring()
    
    # Enable chaos engineering for testing
    monitoring.chaos_monkey.enabled = False  # Set to True for chaos testing
    
    # Start monitoring
    await monitoring.start_monitoring(
        api_url="http://localhost:8090",
        interval=30  # Check every 30 seconds
    )

if __name__ == "__main__":
    asyncio.run(main())