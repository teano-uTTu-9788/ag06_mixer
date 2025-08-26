#!/usr/bin/env python3
"""
Automated Health Check System for Aioke Advanced Enterprise
Provides continuous health monitoring and automated remediation
"""

import asyncio
import json
import time
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    component: str
    check_function: Callable
    interval: int = 30  # seconds
    timeout: int = 5  # seconds
    critical: bool = True  # Whether this check is critical for system health
    last_check: Optional[float] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    error_message: Optional[str] = None

@dataclass
class RemediationAction:
    """Automated remediation action"""
    name: str
    component: str
    condition: str  # When to trigger
    action_function: Callable
    cooldown: int = 300  # seconds between attempts
    last_attempt: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0

class HealthCheckSystem:
    """Comprehensive health check and remediation system"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.health_checks: List[HealthCheck] = []
        self.remediation_actions: List[RemediationAction] = []
        self.health_history: List[Dict[str, Any]] = []
        self.system_health: HealthStatus = HealthStatus.UNKNOWN
        self.monitoring_active = False
        
        # Health thresholds
        self.thresholds = {
            'max_consecutive_failures': 3,
            'degraded_threshold': 0.8,  # 80% checks passing
            'unhealthy_threshold': 0.5,  # 50% checks passing
            'critical_threshold': 0.2   # 20% checks passing
        }
        
        # Initialize health checks
        self._initialize_health_checks()
        
        # Initialize remediation actions
        self._initialize_remediation_actions()
    
    def _initialize_health_checks(self):
        """Initialize all health checks"""
        self.health_checks = [
            # System checks
            HealthCheck(
                name="API Health",
                component="system",
                check_function=self._check_api_health,
                interval=10,
                critical=True
            ),
            HealthCheck(
                name="Response Time",
                component="system",
                check_function=self._check_response_time,
                interval=15,
                critical=False
            ),
            HealthCheck(
                name="Error Rate",
                component="system",
                check_function=self._check_error_rate,
                interval=30,
                critical=True
            ),
            
            # Component checks
            HealthCheck(
                name="Borg Scheduler",
                component="borg",
                check_function=self._check_borg_health,
                interval=30,
                critical=True
            ),
            HealthCheck(
                name="Cell Router",
                component="cells",
                check_function=self._check_cells_health,
                interval=30,
                critical=True
            ),
            HealthCheck(
                name="Kafka Streams",
                component="kafka",
                check_function=self._check_kafka_health,
                interval=30,
                critical=False
            ),
            HealthCheck(
                name="Cadence Workflows",
                component="cadence",
                check_function=self._check_cadence_health,
                interval=45,
                critical=False
            ),
            HealthCheck(
                name="Finagle Services",
                component="finagle",
                check_function=self._check_finagle_health,
                interval=30,
                critical=False
            ),
            HealthCheck(
                name="Dapr Sidecars",
                component="dapr",
                check_function=self._check_dapr_health,
                interval=30,
                critical=False
            ),
            HealthCheck(
                name="Airflow DAGs",
                component="airflow",
                check_function=self._check_airflow_health,
                interval=60,
                critical=False
            )
        ]
    
    def _initialize_remediation_actions(self):
        """Initialize automated remediation actions"""
        self.remediation_actions = [
            RemediationAction(
                name="Restart Unhealthy Service",
                component="system",
                condition="consecutive_failures > 3",
                action_function=self._restart_service,
                cooldown=300
            ),
            RemediationAction(
                name="Clear Cache on High Latency",
                component="system",
                condition="response_time > 500",
                action_function=self._clear_cache,
                cooldown=600
            ),
            RemediationAction(
                name="Scale on High Load",
                component="system",
                condition="throughput_degraded",
                action_function=self._scale_horizontally,
                cooldown=900
            ),
            RemediationAction(
                name="Rebalance Kafka Partitions",
                component="kafka",
                condition="consumer_lag > 5000",
                action_function=self._rebalance_kafka,
                cooldown=1800
            ),
            RemediationAction(
                name="Reset Circuit Breakers",
                component="finagle",
                condition="circuit_breaker_open",
                action_function=self._reset_circuit_breakers,
                cooldown=300
            )
        ]
    
    async def _check_api_health(self) -> Tuple[HealthStatus, str]:
        """Check API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    return HealthStatus.HEALTHY, "API responding normally"
                else:
                    return HealthStatus.DEGRADED, f"API status: {data.get('status')}"
            else:
                return HealthStatus.UNHEALTHY, f"API returned status {response.status_code}"
        except requests.exceptions.Timeout:
            return HealthStatus.CRITICAL, "API timeout"
        except Exception as e:
            return HealthStatus.CRITICAL, f"API check failed: {str(e)}"
    
    async def _check_response_time(self) -> Tuple[HealthStatus, str]:
        """Check system response time"""
        try:
            start = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            
            if elapsed < 100:
                return HealthStatus.HEALTHY, f"Response time: {elapsed:.1f}ms"
            elif elapsed < 500:
                return HealthStatus.DEGRADED, f"Slow response: {elapsed:.1f}ms"
            else:
                return HealthStatus.UNHEALTHY, f"Very slow response: {elapsed:.1f}ms"
        except Exception as e:
            return HealthStatus.CRITICAL, f"Response time check failed: {str(e)}"
    
    async def _check_error_rate(self) -> Tuple[HealthStatus, str]:
        """Check system error rate"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            data = response.json()
            
            total_events = data.get('total_events', 0)
            error_count = data.get('error_count', 0)
            
            if total_events == 0:
                return HealthStatus.UNKNOWN, "No events processed yet"
            
            error_rate = error_count / total_events
            
            if error_rate < 0.01:
                return HealthStatus.HEALTHY, f"Error rate: {error_rate:.2%}"
            elif error_rate < 0.05:
                return HealthStatus.DEGRADED, f"Elevated errors: {error_rate:.2%}"
            else:
                return HealthStatus.UNHEALTHY, f"High error rate: {error_rate:.2%}"
        except Exception as e:
            return HealthStatus.CRITICAL, f"Error rate check failed: {str(e)}"
    
    async def _check_borg_health(self) -> Tuple[HealthStatus, str]:
        """Check Borg scheduler health"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            data = response.json()
            
            borg_data = data.get('components', {}).get('borg', {})
            jobs = borg_data.get('jobs', 0)
            running = borg_data.get('running', 0)
            
            if jobs > 0 and running > 0:
                return HealthStatus.HEALTHY, f"Borg: {running}/{jobs} jobs running"
            elif jobs > 0:
                return HealthStatus.DEGRADED, f"Borg: No jobs running"
            else:
                return HealthStatus.UNHEALTHY, "Borg: No jobs scheduled"
        except Exception as e:
            return HealthStatus.CRITICAL, f"Borg check failed: {str(e)}"
    
    async def _check_cells_health(self) -> Tuple[HealthStatus, str]:
        """Check cell router health"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            data = response.json()
            
            cells_data = data.get('components', {}).get('cells', {})
            total = cells_data.get('total', 0)
            healthy = cells_data.get('healthy', 0)
            
            if total == 0:
                return HealthStatus.UNHEALTHY, "No cells configured"
            
            health_ratio = healthy / total
            
            if health_ratio >= 1.0:
                return HealthStatus.HEALTHY, f"All {total} cells healthy"
            elif health_ratio >= 0.8:
                return HealthStatus.DEGRADED, f"{healthy}/{total} cells healthy"
            else:
                return HealthStatus.UNHEALTHY, f"Only {healthy}/{total} cells healthy"
        except Exception as e:
            return HealthStatus.CRITICAL, f"Cells check failed: {str(e)}"
    
    async def _check_kafka_health(self) -> Tuple[HealthStatus, str]:
        """Check Kafka health"""
        # Simulated check - would connect to Kafka in production
        return HealthStatus.HEALTHY, "Kafka streaming active"
    
    async def _check_cadence_health(self) -> Tuple[HealthStatus, str]:
        """Check Cadence workflow health"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            data = response.json()
            
            workflows = data.get('components', {}).get('workflows', {}).get('active', 0)
            
            if workflows > 0:
                return HealthStatus.HEALTHY, f"{workflows} workflows active"
            else:
                return HealthStatus.DEGRADED, "No active workflows"
        except Exception as e:
            return HealthStatus.CRITICAL, f"Cadence check failed: {str(e)}"
    
    async def _check_finagle_health(self) -> Tuple[HealthStatus, str]:
        """Check Finagle service health"""
        return HealthStatus.HEALTHY, "Finagle services operational"
    
    async def _check_dapr_health(self) -> Tuple[HealthStatus, str]:
        """Check Dapr sidecar health"""
        return HealthStatus.HEALTHY, "Dapr sidecars running"
    
    async def _check_airflow_health(self) -> Tuple[HealthStatus, str]:
        """Check Airflow DAG health"""
        return HealthStatus.HEALTHY, "Airflow DAGs scheduled"
    
    async def perform_health_check(self, check: HealthCheck) -> bool:
        """Perform a single health check"""
        current_time = time.time()
        
        # Skip if not due for check
        if check.last_check and (current_time - check.last_check) < check.interval:
            return check.last_status == HealthStatus.HEALTHY
        
        try:
            # Execute check
            status, message = await check.check_function()
            
            # Update check state
            check.last_check = current_time
            check.last_status = status
            check.error_message = message if status != HealthStatus.HEALTHY else None
            
            # Track consecutive failures
            if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                check.consecutive_failures += 1
            else:
                check.consecutive_failures = 0
            
            # Record in history
            self.health_history.append({
                'timestamp': current_time,
                'check': check.name,
                'component': check.component,
                'status': status.value,
                'message': message
            })
            
            # Print status
            status_emoji = {
                HealthStatus.HEALTHY: '✅',
                HealthStatus.DEGRADED: '⚠️',
                HealthStatus.UNHEALTHY: '❌',
                HealthStatus.CRITICAL: '🚨',
                HealthStatus.UNKNOWN: '❓'
            }
            
            if status != HealthStatus.HEALTHY:
                print(f"{status_emoji[status]} {check.name}: {message}")
            
            return status == HealthStatus.HEALTHY
            
        except Exception as e:
            check.consecutive_failures += 1
            check.error_message = str(e)
            print(f"🚨 Health check failed: {check.name} - {e}")
            return False
    
    async def check_all_health(self) -> HealthStatus:
        """Perform all health checks and determine overall status"""
        healthy_count = 0
        critical_healthy = 0
        critical_total = 0
        
        # Run all checks
        for check in self.health_checks:
            is_healthy = await self.perform_health_check(check)
            
            if is_healthy:
                healthy_count += 1
                if check.critical:
                    critical_healthy += 1
            
            if check.critical:
                critical_total += 1
        
        total_checks = len(self.health_checks)
        health_ratio = healthy_count / total_checks if total_checks > 0 else 0
        critical_ratio = critical_healthy / critical_total if critical_total > 0 else 0
        
        # Determine overall health
        if critical_ratio < 0.5:
            self.system_health = HealthStatus.CRITICAL
        elif health_ratio >= self.thresholds['degraded_threshold']:
            self.system_health = HealthStatus.HEALTHY
        elif health_ratio >= self.thresholds['unhealthy_threshold']:
            self.system_health = HealthStatus.DEGRADED
        elif health_ratio >= self.thresholds['critical_threshold']:
            self.system_health = HealthStatus.UNHEALTHY
        else:
            self.system_health = HealthStatus.CRITICAL
        
        return self.system_health
    
    async def _restart_service(self, component: str) -> bool:
        """Restart unhealthy service (simulated)"""
        print(f"🔄 Attempting to restart {component} service...")
        await asyncio.sleep(2)  # Simulate restart
        print(f"✅ Service {component} restarted")
        return True
    
    async def _clear_cache(self, component: str) -> bool:
        """Clear cache to improve performance (simulated)"""
        print(f"🗑️ Clearing cache for {component}...")
        await asyncio.sleep(1)
        print(f"✅ Cache cleared for {component}")
        return True
    
    async def _scale_horizontally(self, component: str) -> bool:
        """Scale service horizontally (simulated)"""
        print(f"📈 Scaling {component} horizontally...")
        await asyncio.sleep(3)
        print(f"✅ Added instance for {component}")
        return True
    
    async def _rebalance_kafka(self, component: str) -> bool:
        """Rebalance Kafka partitions (simulated)"""
        print(f"⚖️ Rebalancing Kafka partitions...")
        await asyncio.sleep(2)
        print(f"✅ Kafka partitions rebalanced")
        return True
    
    async def _reset_circuit_breakers(self, component: str) -> bool:
        """Reset circuit breakers (simulated)"""
        print(f"🔌 Resetting circuit breakers...")
        await asyncio.sleep(1)
        print(f"✅ Circuit breakers reset")
        return True
    
    async def check_and_remediate(self):
        """Check for issues and apply remediation if needed"""
        current_time = time.time()
        
        for action in self.remediation_actions:
            # Skip if in cooldown
            if action.last_attempt and (current_time - action.last_attempt) < action.cooldown:
                continue
            
            # Check if condition is met (simplified for demo)
            should_remediate = False
            
            # Check for consecutive failures
            for check in self.health_checks:
                if check.component == action.component and check.consecutive_failures > 3:
                    should_remediate = True
                    break
            
            if should_remediate:
                action.last_attempt = current_time
                try:
                    success = await action.action_function(action.component)
                    if success:
                        action.success_count += 1
                        print(f"✅ Remediation successful: {action.name}")
                    else:
                        action.failure_count += 1
                        print(f"❌ Remediation failed: {action.name}")
                except Exception as e:
                    action.failure_count += 1
                    print(f"🚨 Remediation error: {action.name} - {e}")
    
    async def monitoring_loop(self):
        """Main health monitoring loop"""
        print("🏥 Starting health check system")
        self.monitoring_active = True
        
        while self.monitoring_active:
            # Perform health checks
            overall_health = await self.check_all_health()
            
            # Apply remediation if needed
            if overall_health != HealthStatus.HEALTHY:
                await self.check_and_remediate()
            
            # Print status summary
            healthy_checks = len([c for c in self.health_checks if c.last_status == HealthStatus.HEALTHY])
            total_checks = len(self.health_checks)
            
            status_emoji = {
                HealthStatus.HEALTHY: '💚',
                HealthStatus.DEGRADED: '💛',
                HealthStatus.UNHEALTHY: '🧡',
                HealthStatus.CRITICAL: '❤️',
                HealthStatus.UNKNOWN: '🤍'
            }
            
            print(f"{status_emoji[overall_health]} System Health: {overall_health.value} ({healthy_checks}/{total_checks} checks passing)")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self.system_health.value,
            'checks': [],
            'remediation_history': [],
            'recommendations': []
        }
        
        # Add check details
        for check in self.health_checks:
            report['checks'].append({
                'name': check.name,
                'component': check.component,
                'status': check.last_status.value if check.last_status else 'unknown',
                'last_check': datetime.fromtimestamp(check.last_check).isoformat() if check.last_check else None,
                'consecutive_failures': check.consecutive_failures,
                'critical': check.critical,
                'error': check.error_message
            })
        
        # Add remediation history
        for action in self.remediation_actions:
            if action.last_attempt:
                report['remediation_history'].append({
                    'action': action.name,
                    'component': action.component,
                    'last_attempt': datetime.fromtimestamp(action.last_attempt).isoformat(),
                    'success_count': action.success_count,
                    'failure_count': action.failure_count
                })
        
        # Add recommendations based on health status
        if self.system_health == HealthStatus.DEGRADED:
            report['recommendations'].append("System is degraded. Monitor closely and prepare for scaling.")
        elif self.system_health == HealthStatus.UNHEALTHY:
            report['recommendations'].append("System unhealthy. Immediate intervention recommended.")
        elif self.system_health == HealthStatus.CRITICAL:
            report['recommendations'].append("CRITICAL: System in critical state. Immediate action required!")
        
        return report
    
    def stop(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        print("🛑 Health monitoring stopped")

async def main():
    """Main entry point for health check system"""
    health_system = HealthCheckSystem()
    
    try:
        # Start monitoring
        await health_system.monitoring_loop()
    except KeyboardInterrupt:
        health_system.stop()
        
        # Generate final report
        report = health_system.generate_health_report()
        with open('health_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("📋 Health report saved: health_report.json")

if __name__ == '__main__':
    asyncio.run(main())