"""
Enterprise A/B Testing Platform with Statistical Significance
Following practices from Google, Facebook, Netflix, and Optimizely
"""

import asyncio
import json
import time
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import hashlib
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class ExperimentType(Enum):
    """Types of experiments"""
    AB_TEST = "ab_test"  # Standard A/B
    MULTI_VARIANT = "multi_variant"  # A/B/C/D...
    BANDIT = "bandit"  # Multi-armed bandit
    FACTORIAL = "factorial"  # Multiple factors
    SEQUENTIAL = "sequential"  # Sequential testing

class AllocationMethod(Enum):
    """Traffic allocation methods"""
    RANDOM = "random"
    DETERMINISTIC = "deterministic"  # Hash-based
    STRATIFIED = "stratified"  # By user segments
    ADAPTIVE = "adaptive"  # Thompson sampling
    EPSILON_GREEDY = "epsilon_greedy"

class MetricType(Enum):
    """Types of metrics"""
    CONVERSION = "conversion"  # Binary
    REVENUE = "revenue"  # Continuous
    ENGAGEMENT = "engagement"  # Time-based
    RETENTION = "retention"  # Cohort-based
    CUSTOM = "custom"

@dataclass
class Variant:
    """Experiment variant"""
    variant_id: str
    name: str
    description: str
    allocation_percent: float
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

@dataclass
class Metric:
    """Experiment metric"""
    metric_id: str
    name: str
    metric_type: MetricType
    is_primary: bool = False
    minimum_detectable_effect: float = 0.05  # 5% MDE
    variance: Optional[float] = None

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[Variant]
    metrics: List[Metric]
    start_date: datetime
    end_date: Optional[datetime]
    minimum_sample_size: int
    confidence_level: float = 0.95
    power: float = 0.80
    allocation_method: AllocationMethod = AllocationMethod.DETERMINISTIC
    segments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Assignment:
    """User assignment to variant"""
    user_id: str
    experiment_id: str
    variant_id: str
    timestamp: float
    segment: Optional[str] = None
    forced: bool = False  # QA override

@dataclass
class Event:
    """Tracked event"""
    event_id: str
    user_id: str
    experiment_id: str
    variant_id: str
    metric_id: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExperimentPlatform:
    """Main A/B testing platform"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.assignments: Dict[str, Assignment] = {}  # user_id:experiment_id -> Assignment
        self.events: List[Event] = []
        self.analyzer = StatisticalAnalyzer()
        self.allocator = TrafficAllocator()
        self.monitor = ExperimentMonitor()
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new experiment"""
        # Validate configuration
        self._validate_config(config)
        
        # Calculate sample size if not provided
        if config.minimum_sample_size == 0:
            config.minimum_sample_size = self._calculate_sample_size(config)
        
        self.experiments[config.experiment_id] = config
        
        return config.experiment_id
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""
        # Check variant allocations sum to 100%
        total_allocation = sum(v.allocation_percent for v in config.variants)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError(f"Variant allocations must sum to 100%, got {total_allocation}")
        
        # Ensure control variant exists
        control_count = sum(1 for v in config.variants if v.is_control)
        if control_count != 1:
            raise ValueError(f"Exactly one control variant required, got {control_count}")
        
        # Check primary metric exists
        primary_count = sum(1 for m in config.metrics if m.is_primary)
        if primary_count != 1:
            raise ValueError(f"Exactly one primary metric required, got {primary_count}")
    
    def _calculate_sample_size(self, config: ExperimentConfig) -> int:
        """Calculate minimum sample size for statistical power"""
        primary_metric = next(m for m in config.metrics if m.is_primary)
        
        # Using standard sample size formula
        # n = 2 * (Z_alpha + Z_beta)^2 * variance / MDE^2
        
        z_alpha = stats.norm.ppf((1 + config.confidence_level) / 2)
        z_beta = stats.norm.ppf(config.power)
        
        # Estimate variance based on metric type
        if primary_metric.variance:
            variance = primary_metric.variance
        elif primary_metric.metric_type == MetricType.CONVERSION:
            # Assume p=0.5 for maximum variance
            variance = 0.5 * (1 - 0.5)
        else:
            variance = 1.0  # Default
        
        mde = primary_metric.minimum_detectable_effect
        
        sample_size_per_variant = math.ceil(
            2 * ((z_alpha + z_beta) ** 2) * variance / (mde ** 2)
        )
        
        return sample_size_per_variant * len(config.variants)
    
    async def get_assignment(self, user_id: str, experiment_id: str,
                            segment: Optional[str] = None) -> Assignment:
        """Get or create user assignment"""
        
        assignment_key = f"{user_id}:{experiment_id}"
        
        # Check existing assignment
        if assignment_key in self.assignments:
            return self.assignments[assignment_key]
        
        # Get experiment
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Check if experiment is active
        now = datetime.now()
        if now < config.start_date or (config.end_date and now > config.end_date):
            # Return control for inactive experiments
            control_variant = next(v for v in config.variants if v.is_control)
            assignment = Assignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=control_variant.variant_id,
                timestamp=time.time(),
                segment=segment
            )
        else:
            # Allocate variant
            variant = await self.allocator.allocate(user_id, config, segment)
            
            assignment = Assignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=variant.variant_id,
                timestamp=time.time(),
                segment=segment
            )
        
        self.assignments[assignment_key] = assignment
        
        # Track assignment
        await self.monitor.track_assignment(assignment)
        
        return assignment
    
    async def track_event(self, user_id: str, experiment_id: str,
                         metric_id: str, value: float = 1.0,
                         metadata: Optional[Dict] = None):
        """Track conversion/metric event"""
        
        # Get user's assignment
        assignment_key = f"{user_id}:{experiment_id}"
        
        if assignment_key not in self.assignments:
            # User not in experiment
            return
        
        assignment = self.assignments[assignment_key]
        
        event = Event(
            event_id=f"{user_id}:{metric_id}:{time.time()}",
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=assignment.variant_id,
            metric_id=metric_id,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Track in monitor
        await self.monitor.track_event(event)
        
        # Check for early stopping
        if await self._should_stop_early(experiment_id):
            await self.stop_experiment(experiment_id)
    
    async def _should_stop_early(self, experiment_id: str) -> bool:
        """Check if experiment should stop early"""
        
        config = self.experiments[experiment_id]
        
        # Get current results
        results = await self.analyzer.analyze(experiment_id, self.events)
        
        # Check for statistical significance
        if results.get("significant", False):
            # Check if we have enough samples
            total_samples = results.get("total_samples", 0)
            if total_samples >= config.minimum_sample_size * 0.5:  # 50% threshold
                return True
        
        # Check for futility (unlikely to reach significance)
        if results.get("futility", False):
            return True
        
        return False
    
    async def stop_experiment(self, experiment_id: str):
        """Stop experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].end_date = datetime.now()
    
    async def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Filter events for this experiment
        experiment_events = [e for e in self.events if e.experiment_id == experiment_id]
        
        # Analyze results
        results = await self.analyzer.analyze(experiment_id, experiment_events)
        
        # Add experiment metadata
        results["experiment"] = {
            "id": experiment_id,
            "name": config.name,
            "type": config.experiment_type.value,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat() if config.end_date else None,
            "variants": len(config.variants),
            "minimum_sample_size": config.minimum_sample_size
        }
        
        return results

class TrafficAllocator:
    """Traffic allocation engine"""
    
    async def allocate(self, user_id: str, config: ExperimentConfig,
                       segment: Optional[str] = None) -> Variant:
        """Allocate user to variant"""
        
        if config.allocation_method == AllocationMethod.RANDOM:
            return self._random_allocation(config)
        elif config.allocation_method == AllocationMethod.DETERMINISTIC:
            return self._deterministic_allocation(user_id, config)
        elif config.allocation_method == AllocationMethod.STRATIFIED:
            return self._stratified_allocation(user_id, config, segment)
        elif config.allocation_method == AllocationMethod.ADAPTIVE:
            return await self._adaptive_allocation(user_id, config)
        else:
            return self._deterministic_allocation(user_id, config)
    
    def _random_allocation(self, config: ExperimentConfig) -> Variant:
        """Random allocation"""
        rand = random.uniform(0, 100)
        cumulative = 0
        
        for variant in config.variants:
            cumulative += variant.allocation_percent
            if rand <= cumulative:
                return variant
        
        return config.variants[-1]
    
    def _deterministic_allocation(self, user_id: str, config: ExperimentConfig) -> Variant:
        """Deterministic hash-based allocation"""
        # Hash user_id with experiment_id for consistency
        hash_input = f"{user_id}:{config.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to 0-100 range
        bucket = (hash_value % 10000) / 100
        
        cumulative = 0
        for variant in config.variants:
            cumulative += variant.allocation_percent
            if bucket <= cumulative:
                return variant
        
        return config.variants[-1]
    
    def _stratified_allocation(self, user_id: str, config: ExperimentConfig,
                              segment: Optional[str]) -> Variant:
        """Stratified allocation by segment"""
        
        # If segment specified and in config, use segment-specific allocation
        if segment and segment in config.segments:
            # Use segment in hash for different allocation
            hash_input = f"{user_id}:{config.experiment_id}:{segment}"
        else:
            hash_input = f"{user_id}:{config.experiment_id}"
        
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100
        
        cumulative = 0
        for variant in config.variants:
            cumulative += variant.allocation_percent
            if bucket <= cumulative:
                return variant
        
        return config.variants[-1]
    
    async def _adaptive_allocation(self, user_id: str, config: ExperimentConfig) -> Variant:
        """Adaptive allocation using Thompson sampling"""
        
        # Get current performance for each variant
        # In production, would use Bayesian updating
        
        # For now, use deterministic as fallback
        return self._deterministic_allocation(user_id, config)

class StatisticalAnalyzer:
    """Statistical analysis engine"""
    
    async def analyze(self, experiment_id: str, events: List[Event]) -> Dict[str, Any]:
        """Analyze experiment results"""
        
        # Group events by variant and metric
        variant_metrics = defaultdict(lambda: defaultdict(list))
        
        for event in events:
            variant_metrics[event.variant_id][event.metric_id].append(event.value)
        
        results = {
            "variants": {},
            "metrics": {},
            "significant": False,
            "winner": None,
            "confidence": 0,
            "total_samples": len(events)
        }
        
        # Calculate statistics for each variant
        for variant_id, metrics in variant_metrics.items():
            variant_stats = {}
            
            for metric_id, values in metrics.items():
                if values:
                    variant_stats[metric_id] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                        "sum": sum(values),
                        "conversion_rate": sum(1 for v in values if v > 0) / len(values)
                    }
            
            results["variants"][variant_id] = variant_stats
        
        # Perform statistical tests
        if len(variant_metrics) >= 2:
            # Get control and treatment
            variants = list(variant_metrics.keys())
            
            if len(variants) == 2:
                # Simple A/B test
                control = variants[0]
                treatment = variants[1]
                
                # For each metric, perform test
                for metric_id in set().union(*[set(m.keys()) for m in variant_metrics.values()]):
                    control_values = variant_metrics[control].get(metric_id, [])
                    treatment_values = variant_metrics[treatment].get(metric_id, [])
                    
                    if control_values and treatment_values:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)
                        effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0
                        
                        # Calculate confidence interval
                        ci = self._calculate_confidence_interval(
                            control_values, treatment_values, confidence=0.95
                        )
                        
                        results["metrics"][metric_id] = {
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "effect_size": effect_size,
                            "confidence_interval": ci,
                            "relative_change": (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values) if np.mean(control_values) > 0 else 0
                        }
                        
                        # Check for overall significance
                        if p_value < 0.05:
                            results["significant"] = True
                            results["confidence"] = 1 - p_value
                            
                            # Determine winner
                            if np.mean(treatment_values) > np.mean(control_values):
                                results["winner"] = treatment
                            else:
                                results["winner"] = control
        
        # Check for futility
        results["futility"] = await self._check_futility(results)
        
        return results
    
    def _calculate_confidence_interval(self, control: List[float], treatment: List[float],
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for difference"""
        
        diff = np.mean(treatment) - np.mean(control)
        
        # Standard error
        se = np.sqrt(np.var(control)/len(control) + np.var(treatment)/len(treatment))
        
        # Critical value
        z = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence interval
        ci_lower = diff - z * se
        ci_upper = diff + z * se
        
        return (ci_lower, ci_upper)
    
    async def _check_futility(self, results: Dict) -> bool:
        """Check if experiment is futile to continue"""
        
        # Simple futility check: if confidence interval includes 0 with large sample
        for metric_results in results.get("metrics", {}).values():
            ci = metric_results.get("confidence_interval", (0, 0))
            
            # If CI is very wide or includes 0 with large sample, might be futile
            if results["total_samples"] > 1000:
                if ci[0] < 0 < ci[1]:
                    ci_width = ci[1] - ci[0]
                    if ci_width > 0.1:  # Wide CI
                        return True
        
        return False

class ExperimentMonitor:
    """Monitoring and alerting for experiments"""
    
    def __init__(self):
        self.assignment_counts: Dict[str, int] = defaultdict(int)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.alerts: List[Dict] = []
    
    async def track_assignment(self, assignment: Assignment):
        """Track assignment"""
        key = f"{assignment.experiment_id}:{assignment.variant_id}"
        self.assignment_counts[key] += 1
        
        # Check for imbalance
        await self._check_allocation_imbalance(assignment.experiment_id)
    
    async def track_event(self, event: Event):
        """Track event"""
        key = f"{event.experiment_id}:{event.variant_id}:{event.metric_id}"
        self.event_counts[key] += 1
    
    async def _check_allocation_imbalance(self, experiment_id: str):
        """Check for allocation imbalance (SRM - Sample Ratio Mismatch)"""
        
        # Get all assignments for experiment
        experiment_assignments = {
            k: v for k, v in self.assignment_counts.items()
            if k.startswith(f"{experiment_id}:")
        }
        
        if len(experiment_assignments) >= 2:
            counts = list(experiment_assignments.values())
            total = sum(counts)
            
            if total > 100:  # Need enough samples
                # Check if ratios match expected
                # Using chi-square test for SRM detection
                expected_ratio = 1 / len(counts)  # Assuming equal allocation
                expected_counts = [total * expected_ratio] * len(counts)
                
                chi2, p_value = stats.chisquare(counts, expected_counts)
                
                if p_value < 0.001:  # Strong evidence of SRM
                    self.alerts.append({
                        "type": "srm",
                        "experiment_id": experiment_id,
                        "p_value": p_value,
                        "severity": "high",
                        "message": f"Sample ratio mismatch detected (p={p_value:.4f})"
                    })

# Multi-Armed Bandit implementation
class MultiArmedBandit:
    """Multi-armed bandit for adaptive experiments"""
    
    def __init__(self, n_arms: int, algorithm: str = "thompson"):
        self.n_arms = n_arms
        self.algorithm = algorithm
        
        # Thompson sampling parameters
        self.alpha = np.ones(n_arms)  # Successes
        self.beta = np.ones(n_arms)   # Failures
        
        # UCB parameters
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self) -> int:
        """Select arm to play"""
        
        if self.algorithm == "thompson":
            # Thompson sampling
            samples = [np.random.beta(self.alpha[i], self.beta[i]) 
                      for i in range(self.n_arms)]
            return np.argmax(samples)
        
        elif self.algorithm == "ucb":
            # Upper Confidence Bound
            total_counts = np.sum(self.counts)
            
            if total_counts < self.n_arms:
                # Play each arm once first
                return int(total_counts)
            
            ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / self.counts)
            return np.argmax(ucb_values)
        
        else:
            # Random selection
            return np.random.randint(self.n_arms)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics"""
        
        if self.algorithm == "thompson":
            # Update Beta distribution parameters
            if reward > 0:
                self.alpha[arm] += reward
            else:
                self.beta[arm] += 1
        
        elif self.algorithm == "ucb":
            # Update counts and values
            self.counts[arm] += 1
            n = self.counts[arm]
            value = self.values[arm]
            self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# Example usage
async def main():
    """Demonstrate A/B testing platform"""
    
    print("🧪 Enterprise A/B Testing Platform")
    print("=" * 60)
    
    # Initialize platform
    platform = ExperimentPlatform()
    
    # Create experiment
    config = ExperimentConfig(
        experiment_id="exp_001",
        name="New Audio Processing Algorithm",
        description="Test new ML-based audio enhancement",
        experiment_type=ExperimentType.AB_TEST,
        variants=[
            Variant("control", "Current Algorithm", "Existing processing", 50, is_control=True),
            Variant("treatment", "ML Algorithm", "New ML-based processing", 50)
        ],
        metrics=[
            Metric("audio_quality", "Audio Quality Score", MetricType.ENGAGEMENT, is_primary=True),
            Metric("processing_time", "Processing Time", MetricType.ENGAGEMENT),
            Metric("user_satisfaction", "User Satisfaction", MetricType.CONVERSION)
        ],
        start_date=datetime.now(),
        end_date=None,
        minimum_sample_size=1000,
        confidence_level=0.95,
        power=0.80
    )
    
    exp_id = platform.create_experiment(config)
    print(f"\n✅ Created Experiment: {config.name}")
    print(f"  Minimum Sample Size: {config.minimum_sample_size}")
    
    # Simulate user assignments and events
    print("\n🎲 Simulating User Traffic:")
    print("-" * 40)
    
    users = [f"user_{i}" for i in range(200)]
    
    for user_id in users:
        # Get assignment
        assignment = await platform.get_assignment(user_id, exp_id)
        
        # Simulate events based on variant
        if assignment.variant_id == "treatment":
            # Treatment performs better
            quality_score = np.random.normal(0.85, 0.1)
            satisfied = random.random() < 0.75
        else:
            # Control baseline
            quality_score = np.random.normal(0.80, 0.1)
            satisfied = random.random() < 0.65
        
        # Track events
        await platform.track_event(user_id, exp_id, "audio_quality", quality_score)
        await platform.track_event(user_id, exp_id, "user_satisfaction", 1.0 if satisfied else 0.0)
    
    # Get results
    results = await platform.get_results(exp_id)
    
    print(f"\n📊 Experiment Results:")
    print("-" * 40)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Statistically Significant: {results['significant']}")
    
    if results['significant']:
        print(f"Winner: {results['winner']}")
        print(f"Confidence: {results['confidence']*100:.1f}%")
    
    # Show variant performance
    print("\n📈 Variant Performance:")
    for variant_id, stats in results["variants"].items():
        print(f"\n{variant_id}:")
        for metric_id, metric_stats in stats.items():
            print(f"  {metric_id}: {metric_stats['mean']:.3f} (n={metric_stats['count']})")
    
    # Show metric analysis
    if results.get("metrics"):
        print("\n🔬 Statistical Analysis:")
        for metric_id, analysis in results["metrics"].items():
            print(f"\n{metric_id}:")
            print(f"  P-value: {analysis['p_value']:.4f}")
            print(f"  Effect Size: {analysis['effect_size']:.3f}")
            print(f"  Relative Change: {analysis['relative_change']*100:.1f}%")
    
    print("\n✅ A/B testing platform operational!")
    
    return platform

if __name__ == "__main__":
    asyncio.run(main())