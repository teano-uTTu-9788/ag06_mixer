"""
Netflix-style Chaos Engineering Platform
Following Chaos Monkey, Chaos Kong, and Chaos Gorilla patterns
"""

import asyncio
import random
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
import traceback

class ChaosExperiment(Enum):
    """Types of chaos experiments"""
    # Instance failures
    CHAOS_MONKEY = "chaos_monkey"  # Random instance termination
    
    # Network failures  
    LATENCY_MONKEY = "latency_monkey"  # Add network latency
    CONFORMITY_MONKEY = "conformity_monkey"  # Kill non-conforming instances
    
    # Regional failures
    CHAOS_KONG = "chaos_kong"  # Simulate region failure
    
    # Zone failures
    CHAOS_GORILLA = "chaos_gorilla"  # Simulate availability zone failure
    
    # Time failures
    DOCTOR_MONKEY = "doctor_monkey"  # Health check failures
    JANITOR_MONKEY = "janitor_monkey"  # Clean up unused resources
    
    # Security failures
    SECURITY_MONKEY = "security_monkey"  # Security violations
    
    # Data failures
    CHAOS_DB = "chaos_db"  # Database failures
    CHAOS_CACHE = "chaos_cache"  # Cache failures
    
    # Application failures
    EXCEPTION_MONKEY = "exception_monkey"  # Inject exceptions
    MEMORY_MONKEY = "memory_monkey"  # Memory leaks
    CPU_MONKEY = "cpu_monkey"  # CPU spikes

class BlastRadius(Enum):
    """Scope of chaos experiment"""
    SINGLE_INSTANCE = "instance"
    SERVICE = "service"
    AVAILABILITY_ZONE = "az"
    REGION = "region"
    GLOBAL = "global"

class SafetyLevel(Enum):
    """Safety constraints for chaos"""
    DRY_RUN = "dry_run"  # No actual failures
    LOW = "low"  # Minor failures only
    MEDIUM = "medium"  # Moderate failures
    HIGH = "high"  # Significant failures
    EXTREME = "extreme"  # Maximum chaos

@dataclass
class ChaosTarget:
    """Target for chaos experiment"""
    target_id: str
    target_type: str  # instance, service, zone, region
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_score: float = 100.0
    critical: bool = False  # Don't target critical services
    
@dataclass
class ChaosResult:
    """Result of chaos experiment"""
    experiment_id: str
    experiment_type: ChaosExperiment
    target: ChaosTarget
    start_time: float
    end_time: float
    success: bool
    impact_score: float  # 0-100
    rollback_performed: bool = False
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ChaosSchedule:
    """Schedule for chaos experiments"""
    enabled: bool = True
    start_hour: int = 9  # Business hours only
    end_hour: int = 17
    days_of_week: List[int] = field(default_factory=lambda: [1,2,3,4,5])  # Weekdays
    probability: float = 0.1  # 10% chance per check
    min_interval_minutes: int = 30  # Minimum time between experiments

class ChaosMonkey:
    """Netflix Chaos Monkey - random instance termination"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MEDIUM):
        self.safety_level = safety_level
        self.terminated_instances: List[str] = []
        
    async def execute(self, targets: List[ChaosTarget]) -> ChaosResult:
        """Terminate random instance"""
        
        # Filter targets based on safety
        safe_targets = self._filter_by_safety(targets)
        
        if not safe_targets:
            return ChaosResult(
                experiment_id=f"cm_{int(time.time())}",
                experiment_type=ChaosExperiment.CHAOS_MONKEY,
                target=ChaosTarget("none", "none"),
                start_time=time.time(),
                end_time=time.time(),
                success=False,
                impact_score=0,
                error="No safe targets available"
            )
        
        # Select random target
        target = random.choice(safe_targets)
        
        start_time = time.time()
        
        # Simulate termination
        if self.safety_level != SafetyLevel.DRY_RUN:
            await self._terminate_instance(target)
            self.terminated_instances.append(target.target_id)
        
        end_time = time.time()
        
        return ChaosResult(
            experiment_id=f"cm_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_MONKEY,
            target=target,
            start_time=start_time,
            end_time=end_time,
            success=True,
            impact_score=self._calculate_impact(target),
            metrics={
                "instances_terminated": 1,
                "recovery_time": random.uniform(30, 120)  # Simulated
            }
        )
    
    def _filter_by_safety(self, targets: List[ChaosTarget]) -> List[ChaosTarget]:
        """Filter targets based on safety level"""
        
        if self.safety_level == SafetyLevel.LOW:
            # Only non-critical with high health
            return [t for t in targets if not t.critical and t.health_score > 80]
        elif self.safety_level == SafetyLevel.MEDIUM:
            # Non-critical targets
            return [t for t in targets if not t.critical]
        elif self.safety_level == SafetyLevel.HIGH:
            # Most targets except critical with low health
            return [t for t in targets if not (t.critical and t.health_score < 50)]
        elif self.safety_level == SafetyLevel.EXTREME:
            # All targets
            return targets
        else:  # DRY_RUN
            return targets
    
    async def _terminate_instance(self, target: ChaosTarget):
        """Simulate instance termination"""
        await asyncio.sleep(0.1)  # Simulate termination delay
        target.health_score = 0
    
    def _calculate_impact(self, target: ChaosTarget) -> float:
        """Calculate impact score"""
        base_impact = 20.0
        
        if target.critical:
            base_impact *= 3
        
        if target.health_score < 50:
            base_impact *= 1.5
        
        return min(100, base_impact)

class LatencyMonkey:
    """Inject network latency"""
    
    def __init__(self, min_latency_ms: int = 100, max_latency_ms: int = 1000):
        self.min_latency = min_latency_ms
        self.max_latency = max_latency_ms
        self.active_delays: Dict[str, float] = {}
    
    async def execute(self, targets: List[ChaosTarget]) -> ChaosResult:
        """Add latency to network calls"""
        
        target = random.choice(targets)
        latency = random.uniform(self.min_latency, self.max_latency)
        
        start_time = time.time()
        
        # Inject latency
        self.active_delays[target.target_id] = latency
        
        # Simulate latency period
        await asyncio.sleep(random.uniform(5, 30))  # Active for 5-30 seconds
        
        # Remove latency
        if target.target_id in self.active_delays:
            del self.active_delays[target.target_id]
        
        end_time = time.time()
        
        return ChaosResult(
            experiment_id=f"lm_{int(start_time)}",
            experiment_type=ChaosExperiment.LATENCY_MONKEY,
            target=target,
            start_time=start_time,
            end_time=end_time,
            success=True,
            impact_score=min(100, latency / 10),  # 1000ms = impact 100
            metrics={
                "latency_ms": latency,
                "duration_seconds": end_time - start_time,
                "requests_affected": random.randint(100, 1000)
            }
        )

class ChaosKong:
    """Simulate entire region failure"""
    
    def __init__(self):
        self.failed_regions: Set[str] = set()
    
    async def execute(self, regions: List[str]) -> ChaosResult:
        """Fail entire region"""
        
        if not regions:
            return ChaosResult(
                experiment_id=f"ck_{int(time.time())}",
                experiment_type=ChaosExperiment.CHAOS_KONG,
                target=ChaosTarget("none", "region"),
                start_time=time.time(),
                end_time=time.time(),
                success=False,
                impact_score=0,
                error="No regions available"
            )
        
        region = random.choice(regions)
        start_time = time.time()
        
        # Simulate region failure
        self.failed_regions.add(region)
        
        # Let it fail for a period
        await asyncio.sleep(random.uniform(60, 300))  # 1-5 minutes
        
        # Recover region
        self.failed_regions.discard(region)
        
        end_time = time.time()
        
        return ChaosResult(
            experiment_id=f"ck_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_KONG,
            target=ChaosTarget(region, "region"),
            start_time=start_time,
            end_time=end_time,
            success=True,
            impact_score=80,  # High impact
            metrics={
                "region": region,
                "downtime_seconds": end_time - start_time,
                "services_affected": random.randint(10, 50)
            }
        )

class ChaosDB:
    """Database chaos experiments"""
    
    async def execute(self, db_targets: List[ChaosTarget]) -> ChaosResult:
        """Inject database failures"""
        
        experiments = [
            self._connection_failure,
            self._slow_queries,
            self._replication_lag,
            self._deadlock
        ]
        
        experiment = random.choice(experiments)
        target = random.choice(db_targets) if db_targets else ChaosTarget("db", "database")
        
        return await experiment(target)
    
    async def _connection_failure(self, target: ChaosTarget) -> ChaosResult:
        """Simulate connection failures"""
        start_time = time.time()
        
        await asyncio.sleep(random.uniform(1, 10))
        
        return ChaosResult(
            experiment_id=f"db_conn_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_DB,
            target=target,
            start_time=start_time,
            end_time=time.time(),
            success=True,
            impact_score=60,
            metrics={
                "failure_type": "connection_timeout",
                "connections_failed": random.randint(10, 100)
            }
        )
    
    async def _slow_queries(self, target: ChaosTarget) -> ChaosResult:
        """Simulate slow queries"""
        start_time = time.time()
        
        query_time = random.uniform(5, 30)  # 5-30 second queries
        await asyncio.sleep(query_time)
        
        return ChaosResult(
            experiment_id=f"db_slow_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_DB,
            target=target,
            start_time=start_time,
            end_time=time.time(),
            success=True,
            impact_score=40,
            metrics={
                "failure_type": "slow_queries",
                "query_time_seconds": query_time,
                "queries_affected": random.randint(50, 500)
            }
        )
    
    async def _replication_lag(self, target: ChaosTarget) -> ChaosResult:
        """Simulate replication lag"""
        start_time = time.time()
        
        lag_seconds = random.uniform(1, 60)
        await asyncio.sleep(10)
        
        return ChaosResult(
            experiment_id=f"db_lag_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_DB,
            target=target,
            start_time=start_time,
            end_time=time.time(),
            success=True,
            impact_score=35,
            metrics={
                "failure_type": "replication_lag",
                "lag_seconds": lag_seconds
            }
        )
    
    async def _deadlock(self, target: ChaosTarget) -> ChaosResult:
        """Simulate deadlock"""
        start_time = time.time()
        
        await asyncio.sleep(random.uniform(0.5, 3))
        
        return ChaosResult(
            experiment_id=f"db_deadlock_{int(start_time)}",
            experiment_type=ChaosExperiment.CHAOS_DB,
            target=target,
            start_time=start_time,
            end_time=time.time(),
            success=True,
            impact_score=70,
            metrics={
                "failure_type": "deadlock",
                "transactions_rolled_back": random.randint(1, 10)
            }
        )

class ChaosOrchestrator:
    """Main chaos engineering orchestrator"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MEDIUM):
        self.safety_level = safety_level
        self.schedule = ChaosSchedule()
        
        # Initialize chaos experiments
        self.chaos_monkey = ChaosMonkey(safety_level)
        self.latency_monkey = LatencyMonkey()
        self.chaos_kong = ChaosKong()
        self.chaos_db = ChaosDB()
        
        # Tracking
        self.experiment_history: List[ChaosResult] = []
        self.active_experiments: Set[str] = set()
        
        # Gameday mode
        self.gameday_mode = False
        
        # Monitoring
        self.monitors: List[Callable] = []
    
    async def run_experiment(self, experiment_type: ChaosExperiment,
                            targets: List[ChaosTarget]) -> ChaosResult:
        """Run specific chaos experiment"""
        
        if not self._should_run():
            return ChaosResult(
                experiment_id="skip",
                experiment_type=experiment_type,
                target=ChaosTarget("none", "none"),
                start_time=time.time(),
                end_time=time.time(),
                success=False,
                impact_score=0,
                error="Outside schedule window or disabled"
            )
        
        # Check safety
        if not await self._safety_check(experiment_type, targets):
            return ChaosResult(
                experiment_id="safety_block",
                experiment_type=experiment_type,
                target=ChaosTarget("none", "none"),
                start_time=time.time(),
                end_time=time.time(),
                success=False,
                impact_score=0,
                error="Failed safety check"
            )
        
        # Execute experiment
        result = None
        
        try:
            if experiment_type == ChaosExperiment.CHAOS_MONKEY:
                result = await self.chaos_monkey.execute(targets)
            elif experiment_type == ChaosExperiment.LATENCY_MONKEY:
                result = await self.latency_monkey.execute(targets)
            elif experiment_type == ChaosExperiment.CHAOS_KONG:
                regions = [t.target_id for t in targets if t.target_type == "region"]
                result = await self.chaos_kong.execute(regions)
            elif experiment_type == ChaosExperiment.CHAOS_DB:
                db_targets = [t for t in targets if "db" in t.target_type]
                result = await self.chaos_db.execute(db_targets)
            else:
                result = await self._generic_experiment(experiment_type, targets)
        
        except Exception as e:
            result = ChaosResult(
                experiment_id=f"error_{int(time.time())}",
                experiment_type=experiment_type,
                target=targets[0] if targets else ChaosTarget("none", "none"),
                start_time=time.time(),
                end_time=time.time(),
                success=False,
                impact_score=0,
                error=str(e)
            )
        
        # Record result
        if result:
            self.experiment_history.append(result)
            
            # Notify monitors
            await self._notify_monitors(result)
            
            # Auto-rollback if impact too high
            if result.impact_score > 90 and self.safety_level != SafetyLevel.EXTREME:
                await self._rollback(result)
        
        return result
    
    def _should_run(self) -> bool:
        """Check if experiments should run"""
        
        if not self.schedule.enabled:
            return False
        
        if self.gameday_mode:
            return True  # Always run in gameday mode
        
        now = datetime.now()
        
        # Check day of week
        if now.weekday() not in self.schedule.days_of_week:
            return False
        
        # Check time of day
        if not (self.schedule.start_hour <= now.hour < self.schedule.end_hour):
            return False
        
        # Check probability
        return random.random() < self.schedule.probability
    
    async def _safety_check(self, experiment_type: ChaosExperiment,
                           targets: List[ChaosTarget]) -> bool:
        """Perform safety checks"""
        
        # Don't run multiple experiments simultaneously
        if len(self.active_experiments) > 0 and self.safety_level != SafetyLevel.EXTREME:
            return False
        
        # Check recent failures
        recent_failures = sum(1 for r in self.experiment_history[-10:]
                            if not r.success)
        if recent_failures > 3:
            return False
        
        # Check critical targets
        critical_targets = sum(1 for t in targets if t.critical)
        if critical_targets > 0 and self.safety_level == SafetyLevel.LOW:
            return False
        
        return True
    
    async def _generic_experiment(self, experiment_type: ChaosExperiment,
                                 targets: List[ChaosTarget]) -> ChaosResult:
        """Generic chaos experiment"""
        
        target = random.choice(targets) if targets else ChaosTarget("generic", "unknown")
        start_time = time.time()
        
        # Simulate experiment
        await asyncio.sleep(random.uniform(1, 10))
        
        return ChaosResult(
            experiment_id=f"generic_{int(start_time)}",
            experiment_type=experiment_type,
            target=target,
            start_time=start_time,
            end_time=time.time(),
            success=True,
            impact_score=random.uniform(10, 50),
            metrics={"type": "generic"}
        )
    
    async def _rollback(self, result: ChaosResult):
        """Rollback experiment if needed"""
        
        result.rollback_performed = True
        
        # Implement rollback logic based on experiment type
        if result.experiment_type == ChaosExperiment.CHAOS_MONKEY:
            # Restart terminated instances
            pass
        elif result.experiment_type == ChaosExperiment.LATENCY_MONKEY:
            # Remove latency
            self.latency_monkey.active_delays.clear()
    
    async def _notify_monitors(self, result: ChaosResult):
        """Notify monitoring systems"""
        
        for monitor in self.monitors:
            try:
                await monitor(result)
            except Exception:
                pass  # Don't let monitor failures stop experiments
    
    def get_report(self) -> Dict[str, Any]:
        """Get chaos engineering report"""
        
        total = len(self.experiment_history)
        successful = sum(1 for r in self.experiment_history if r.success)
        
        avg_impact = (sum(r.impact_score for r in self.experiment_history) / total
                     if total > 0 else 0)
        
        by_type = {}
        for result in self.experiment_history:
            exp_type = result.experiment_type.value
            if exp_type not in by_type:
                by_type[exp_type] = {"count": 0, "success": 0}
            by_type[exp_type]["count"] += 1
            if result.success:
                by_type[exp_type]["success"] += 1
        
        return {
            "total_experiments": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_impact": avg_impact,
            "experiments_by_type": by_type,
            "safety_level": self.safety_level.value,
            "schedule": {
                "enabled": self.schedule.enabled,
                "probability": self.schedule.probability
            }
        }

# Example usage
async def main():
    """Demonstrate chaos engineering"""
    
    print("🐒 Netflix-style Chaos Engineering Platform")
    print("=" * 60)
    
    # Create chaos orchestrator
    chaos = ChaosOrchestrator(SafetyLevel.MEDIUM)
    
    # Create test targets
    targets = [
        ChaosTarget("instance-1", "instance", health_score=95),
        ChaosTarget("instance-2", "instance", health_score=80),
        ChaosTarget("instance-3", "instance", health_score=70, critical=True),
        ChaosTarget("database-1", "database", health_score=90),
        ChaosTarget("us-west-1", "region", health_score=85),
    ]
    
    print("\n🎯 Available Targets:")
    for target in targets:
        critical = " [CRITICAL]" if target.critical else ""
        print(f"  - {target.target_id}: {target.target_type} (health={target.health_score}){critical}")
    
    # Run experiments
    print("\n🧪 Running Chaos Experiments:")
    print("-" * 40)
    
    # Chaos Monkey
    result = await chaos.run_experiment(ChaosExperiment.CHAOS_MONKEY, targets)
    print(f"\nChaos Monkey:")
    print(f"  Target: {result.target.target_id}")
    print(f"  Success: {result.success}")
    print(f"  Impact: {result.impact_score:.1f}/100")
    
    # Latency Monkey
    result = await chaos.run_experiment(ChaosExperiment.LATENCY_MONKEY, targets)
    print(f"\nLatency Monkey:")
    print(f"  Target: {result.target.target_id}")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Latency: {result.metrics.get('latency_ms', 0):.0f}ms")
    
    # Chaos DB
    result = await chaos.run_experiment(ChaosExperiment.CHAOS_DB, targets)
    print(f"\nChaos DB:")
    print(f"  Target: {result.target.target_id}")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Failure Type: {result.metrics.get('failure_type', 'unknown')}")
    
    # Get report
    report = chaos.get_report()
    
    print("\n📊 Chaos Engineering Report:")
    print("-" * 40)
    print(f"Total Experiments: {report['total_experiments']}")
    print(f"Success Rate: {report['success_rate']*100:.1f}%")
    print(f"Average Impact: {report['average_impact']:.1f}/100")
    print(f"Safety Level: {report['safety_level']}")
    
    print("\n✅ Chaos engineering platform operational!")
    
    return chaos

if __name__ == "__main__":
    asyncio.run(main())