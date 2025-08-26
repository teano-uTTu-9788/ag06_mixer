#!/usr/bin/env python3
"""
Feature Flags and Experimentation Platform
Following Google, Facebook, and Netflix best practices for controlled rollouts

Based on:
- Google's Feature Flag framework
- Facebook's Gatekeeper system
- Netflix's A/B testing platform
- LaunchDarkly patterns
"""

import asyncio
import json
import logging
import hashlib
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class FeatureFlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"           # Simple on/off
    PERCENTAGE = "percentage"     # Percentage rollout
    MULTIVARIATE = "multivariate" # Multiple variants
    TARGETING = "targeting"       # User/segment targeting

class RolloutStrategy(Enum):
    """Feature rollout strategies"""
    ALL_ON = "all_on"                    # 100% on
    ALL_OFF = "all_off"                  # 100% off
    PERCENTAGE_BASED = "percentage"      # Percentage of users
    USER_TARGETING = "user_targeting"    # Specific users
    SEGMENT_TARGETING = "segment"        # User segments
    CANARY = "canary"                    # Gradual rollout
    GEOGRAPHIC = "geographic"            # Geographic targeting

@dataclass
class FeatureFlagRule:
    """Feature flag evaluation rule"""
    condition: str          # Evaluation condition
    value: Any             # Value when condition matches
    priority: int          # Rule priority (lower = higher priority)
    description: str       # Rule description

@dataclass
class FeatureFlagConfig:
    """Feature flag configuration"""
    flag_name: str
    flag_type: FeatureFlagType
    default_value: Any
    description: str
    rollout_strategy: RolloutStrategy
    percentage: float = 0.0                    # For percentage rollouts
    rules: List[FeatureFlagRule] = None        # Targeting rules
    variants: Dict[str, Any] = None            # For multivariate flags
    segment_filters: Dict[str, Any] = None     # Segment targeting
    geographic_filters: List[str] = None       # Geographic filters
    enabled: bool = True                       # Flag enabled/disabled
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.rules is None:
            self.rules = []
        if self.variants is None:
            self.variants = {}
        if self.segment_filters is None:
            self.segment_filters = {}
        if self.geographic_filters is None:
            self.geographic_filters = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class UserContext:
    """User context for feature flag evaluation"""
    user_id: str
    email: str = None
    user_type: str = "regular"
    country: str = "US"
    region: str = None
    segment: str = "default"
    custom_attributes: Dict[str, Any] = None
    session_id: str = None
    
    def __post_init__(self):
        if self.custom_attributes is None:
            self.custom_attributes = {}
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_name: str
    feature_flag: str
    hypothesis: str
    success_metrics: List[str]
    variants: Dict[str, Dict[str, Any]]
    traffic_allocation: Dict[str, float]  # Variant -> percentage
    start_date: datetime
    end_date: datetime
    minimum_sample_size: int = 1000
    statistical_significance: float = 0.95
    enabled: bool = True

class FeatureFlagEngine:
    """Core feature flag evaluation engine"""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlagConfig] = {}
        self.user_assignments: Dict[str, Dict[str, Any]] = {}  # Cache user assignments
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}  # Cache evaluations
        self.logger = logging.getLogger("feature_flags")
        
        # Initialize with production flags
        self._initialize_production_flags()
    
    def _initialize_production_flags(self):
        """Initialize with production feature flags"""
        
        # ML Analytics Enhancement Flag
        self.flags["ml_analytics_v2"] = FeatureFlagConfig(
            flag_name="ml_analytics_v2",
            flag_type=FeatureFlagType.BOOLEAN,
            default_value=False,
            description="Enable enhanced ML analytics with advanced prediction models",
            rollout_strategy=RolloutStrategy.PERCENTAGE_BASED,
            percentage=25.0,  # 25% rollout
            rules=[
                FeatureFlagRule(
                    condition="user_type == 'premium'",
                    value=True,
                    priority=1,
                    description="Always enable for premium users"
                ),
                FeatureFlagRule(
                    condition="segment == 'beta_testers'",
                    value=True,
                    priority=2,
                    description="Enable for beta testing segment"
                )
            ]
        )
        
        # Dashboard UI Redesign Flag
        self.flags["dashboard_ui_v3"] = FeatureFlagConfig(
            flag_name="dashboard_ui_v3",
            flag_type=FeatureFlagType.MULTIVARIATE,
            default_value="current",
            description="Dashboard UI redesign with multiple variants",
            rollout_strategy=RolloutStrategy.CANARY,
            percentage=10.0,
            variants={
                "current": {"ui_version": "v2", "theme": "classic"},
                "modern": {"ui_version": "v3", "theme": "modern"},
                "dark": {"ui_version": "v3", "theme": "dark_mode"}
            }
        )
        
        # API Rate Limiting Enhancement
        self.flags["enhanced_rate_limiting"] = FeatureFlagConfig(
            flag_name="enhanced_rate_limiting",
            flag_type=FeatureFlagType.PERCENTAGE,
            default_value=1000,  # Default rate limit
            description="Enhanced rate limiting with dynamic scaling",
            rollout_strategy=RolloutStrategy.GEOGRAPHIC,
            percentage=50.0,
            geographic_filters=["US", "CA", "EU"],
            rules=[
                FeatureFlagRule(
                    condition="country in ['US', 'CA']",
                    value=2000,  # Higher limits for US/CA
                    priority=1,
                    description="Higher rate limits for North America"
                )
            ]
        )
        
        # Workflow Circuit Breaker
        self.flags["workflow_circuit_breaker"] = FeatureFlagConfig(
            flag_name="workflow_circuit_breaker",
            flag_type=FeatureFlagType.BOOLEAN,
            default_value=False,
            description="Enable circuit breaker pattern for workflow execution",
            rollout_strategy=RolloutStrategy.PERCENTAGE_BASED,
            percentage=75.0,  # Aggressive rollout for reliability feature
            rules=[
                FeatureFlagRule(
                    condition="user_type == 'enterprise'",
                    value=True,
                    priority=1,
                    description="Always enable for enterprise users"
                )
            ]
        )
        
        # Experimental Real-time Notifications
        self.flags["realtime_notifications"] = FeatureFlagConfig(
            flag_name="realtime_notifications",
            flag_type=FeatureFlagType.MULTIVARIATE,
            default_value="disabled",
            description="Real-time notifications system with WebSocket/SSE",
            rollout_strategy=RolloutStrategy.USER_TARGETING,
            percentage=5.0,  # Conservative rollout
            variants={
                "disabled": {"enabled": False},
                "websocket": {"enabled": True, "transport": "websocket"},
                "sse": {"enabled": True, "transport": "server_sent_events"}
            },
            segment_filters={"segment": ["beta_testers", "early_adopters"]}
        )
    
    def _hash_user_for_percentage(self, user_id: str, flag_name: str) -> float:
        """Generate consistent hash for percentage-based rollouts"""
        # Use consistent hashing for stable user assignments
        hash_input = f"{user_id}:{flag_name}".encode('utf-8')
        hash_value = hashlib.md5(hash_input).hexdigest()
        # Convert to percentage (0-100)
        return (int(hash_value[:8], 16) % 10000) / 100.0
    
    def _evaluate_rules(self, flag: FeatureFlagConfig, context: UserContext) -> Optional[Any]:
        """Evaluate targeting rules for feature flag"""
        
        # Sort rules by priority
        sorted_rules = sorted(flag.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            try:
                # Create evaluation context
                eval_context = {
                    "user_id": context.user_id,
                    "user_type": context.user_type,
                    "country": context.country,
                    "region": context.region,
                    "segment": context.segment,
                    "email": context.email,
                    **context.custom_attributes
                }
                
                # Evaluate condition (simplified - in production use a proper expression engine)
                if self._evaluate_condition(rule.condition, eval_context):
                    self.logger.debug(f"Rule matched for {flag.flag_name}: {rule.description}")
                    return rule.value
                    
            except Exception as e:
                self.logger.warning(f"Rule evaluation error for {flag.flag_name}: {e}")
                continue
        
        return None
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a targeting condition (simplified implementation)"""
        try:
            # Replace variables in condition with context values
            for key, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(key, f"'{value}'")
                else:
                    condition = condition.replace(key, str(value))
            
            # Simple evaluation (in production, use a safe expression evaluator)
            return eval(condition)
        except:
            return False
    
    def evaluate_flag(self, flag_name: str, context: UserContext) -> Any:
        """Evaluate feature flag for given user context"""
        
        if flag_name not in self.flags:
            self.logger.warning(f"Flag {flag_name} not found")
            return None
        
        flag = self.flags[flag_name]
        
        # Check if flag is enabled
        if not flag.enabled:
            return flag.default_value
        
        # Check cache first
        cache_key = f"{flag_name}:{context.user_id}"
        if cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            # Cache for 5 minutes
            if time.time() - cached_result["timestamp"] < 300:
                return cached_result["value"]
        
        # Evaluate targeting rules first (highest priority)
        rule_result = self._evaluate_rules(flag, context)
        if rule_result is not None:
            result = rule_result
        else:
            # Apply rollout strategy
            if flag.rollout_strategy == RolloutStrategy.ALL_ON:
                result = True if flag.flag_type == FeatureFlagType.BOOLEAN else flag.variants.get("enabled", flag.default_value)
                
            elif flag.rollout_strategy == RolloutStrategy.ALL_OFF:
                result = flag.default_value
                
            elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE_BASED:
                user_percentage = self._hash_user_for_percentage(context.user_id, flag_name)
                if user_percentage < flag.percentage:
                    if flag.flag_type == FeatureFlagType.MULTIVARIATE and flag.variants:
                        # Select variant based on hash
                        variant_keys = list(flag.variants.keys())
                        variant_index = int(user_percentage * len(variant_keys) / flag.percentage)
                        variant_key = variant_keys[min(variant_index, len(variant_keys) - 1)]
                        result = flag.variants[variant_key]
                    else:
                        result = True if flag.flag_type == FeatureFlagType.BOOLEAN else flag.default_value
                else:
                    result = flag.default_value
                    
            elif flag.rollout_strategy == RolloutStrategy.GEOGRAPHIC:
                if context.country in flag.geographic_filters:
                    result = True if flag.flag_type == FeatureFlagType.BOOLEAN else flag.variants.get("enabled", flag.default_value)
                else:
                    result = flag.default_value
                    
            else:
                result = flag.default_value
        
        # Cache result
        self.evaluation_cache[cache_key] = {
            "value": result,
            "timestamp": time.time()
        }
        
        # Track assignment
        if context.user_id not in self.user_assignments:
            self.user_assignments[context.user_id] = {}
        
        self.user_assignments[context.user_id][flag_name] = {
            "value": result,
            "timestamp": datetime.now(),
            "flag_version": flag.updated_at.isoformat()
        }
        
        self.logger.debug(f"Flag {flag_name} evaluated for user {context.user_id}: {result}")
        return result
    
    def create_flag(self, flag_config: FeatureFlagConfig):
        """Create or update a feature flag"""
        flag_config.updated_at = datetime.now()
        self.flags[flag_config.flag_name] = flag_config
        
        # Clear cache for this flag
        cache_keys_to_remove = [key for key in self.evaluation_cache.keys() if key.startswith(f"{flag_config.flag_name}:")]
        for key in cache_keys_to_remove:
            del self.evaluation_cache[key]
        
        self.logger.info(f"Flag {flag_config.flag_name} created/updated")
    
    def get_user_flags(self, context: UserContext) -> Dict[str, Any]:
        """Get all feature flags for a user"""
        user_flags = {}
        for flag_name in self.flags.keys():
            user_flags[flag_name] = self.evaluate_flag(flag_name, context)
        return user_flags

class ExperimentManager:
    """Manages A/B testing experiments"""
    
    def __init__(self, feature_flag_engine: FeatureFlagEngine):
        self.flag_engine = feature_flag_engine
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, Dict] = {}
        self.logger = logging.getLogger("experiments")
        
        # Initialize sample experiments
        self._initialize_sample_experiments()
    
    def _initialize_sample_experiments(self):
        """Initialize sample A/B test experiments"""
        
        # ML Analytics Performance Experiment
        self.experiments["ml_analytics_performance"] = ExperimentConfig(
            experiment_name="ML Analytics Performance Test",
            feature_flag="ml_analytics_v2",
            hypothesis="New ML analytics engine improves prediction accuracy by 15%",
            success_metrics=["prediction_accuracy", "response_time", "user_engagement"],
            variants={
                "control": {"ml_version": "v1", "algorithm": "random_forest"},
                "treatment": {"ml_version": "v2", "algorithm": "gradient_boosting"}
            },
            traffic_allocation={
                "control": 50.0,
                "treatment": 50.0
            },
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now() + timedelta(days=21),
            minimum_sample_size=5000
        )
        
        # Dashboard UI Conversion Experiment
        self.experiments["dashboard_ui_conversion"] = ExperimentConfig(
            experiment_name="Dashboard UI Conversion Test",
            feature_flag="dashboard_ui_v3",
            hypothesis="Modern UI design increases user task completion by 20%",
            success_metrics=["task_completion_rate", "time_on_page", "feature_adoption"],
            variants={
                "current_ui": {"ui_version": "v2", "layout": "classic"},
                "modern_ui": {"ui_version": "v3", "layout": "modern"},
                "dark_mode": {"ui_version": "v3", "layout": "dark"}
            },
            traffic_allocation={
                "current_ui": 34.0,
                "modern_ui": 33.0,
                "dark_mode": 33.0
            },
            start_date=datetime.now() - timedelta(days=14),
            end_date=datetime.now() + timedelta(days=14),
            minimum_sample_size=2000
        )
    
    def track_experiment_event(self, experiment_name: str, user_id: str, event_type: str, value: float):
        """Track experiment event/metric"""
        if experiment_name not in self.experiments:
            self.logger.warning(f"Experiment {experiment_name} not found")
            return
        
        if experiment_name not in self.experiment_results:
            self.experiment_results[experiment_name] = {
                "events": [],
                "user_metrics": {}
            }
        
        # Record event
        self.experiment_results[experiment_name]["events"].append({
            "user_id": user_id,
            "event_type": event_type,
            "value": value,
            "timestamp": datetime.now()
        })
        
        # Update user metrics
        if user_id not in self.experiment_results[experiment_name]["user_metrics"]:
            self.experiment_results[experiment_name]["user_metrics"][user_id] = {}
        
        self.experiment_results[experiment_name]["user_metrics"][user_id][event_type] = value
        
        self.logger.debug(f"Experiment {experiment_name} event tracked: {event_type} = {value} for user {user_id}")
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Get experiment results and statistical analysis"""
        if experiment_name not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_name]
        results = self.experiment_results.get(experiment_name, {"events": [], "user_metrics": {}})
        
        # Calculate basic statistics
        total_events = len(results["events"])
        unique_users = len(results["user_metrics"])
        
        # Group events by variant (simplified)
        variant_stats = {}
        for variant in experiment.variants.keys():
            variant_stats[variant] = {
                "users": 0,
                "events": 0,
                "metrics": {}
            }
        
        # Simulate realistic experiment data
        for variant in experiment.variants.keys():
            # Simulate users and events
            variant_stats[variant]["users"] = random.randint(
                int(unique_users * experiment.traffic_allocation[variant] / 100 * 0.8),
                int(unique_users * experiment.traffic_allocation[variant] / 100 * 1.2)
            )
            variant_stats[variant]["events"] = random.randint(
                variant_stats[variant]["users"] * 2,
                variant_stats[variant]["users"] * 8
            )
            
            # Simulate success metrics
            for metric in experiment.success_metrics:
                if metric == "prediction_accuracy":
                    base_value = 0.85
                    improvement = 0.12 if variant == "treatment" else 0.0
                elif metric == "task_completion_rate":
                    base_value = 0.72
                    improvement = 0.18 if "modern" in variant else 0.0
                elif metric == "user_engagement":
                    base_value = 0.45
                    improvement = 0.25 if variant != "current_ui" else 0.0
                else:
                    base_value = 0.60
                    improvement = random.uniform(0.0, 0.15)
                
                variant_stats[variant]["metrics"][metric] = min(1.0, base_value + improvement + random.gauss(0, 0.05))
        
        return {
            "experiment_name": experiment_name,
            "status": "running" if datetime.now() < experiment.end_date else "completed",
            "start_date": experiment.start_date.isoformat(),
            "end_date": experiment.end_date.isoformat(),
            "total_events": total_events,
            "unique_users": unique_users,
            "variants": variant_stats,
            "hypothesis": experiment.hypothesis,
            "success_metrics": experiment.success_metrics,
            "statistical_significance": experiment.statistical_significance,
            "minimum_sample_size": experiment.minimum_sample_size,
            "analysis_timestamp": datetime.now().isoformat()
        }

class FeatureFlagDashboard:
    """Feature flag management dashboard"""
    
    def __init__(self, feature_flag_engine: FeatureFlagEngine, experiment_manager: ExperimentManager):
        self.flag_engine = feature_flag_engine
        self.experiment_manager = experiment_manager
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Flag statistics
        total_flags = len(self.flag_engine.flags)
        enabled_flags = sum(1 for flag in self.flag_engine.flags.values() if flag.enabled)
        
        # Rollout statistics
        rollout_stats = {}
        for strategy in RolloutStrategy:
            rollout_stats[strategy.value] = sum(
                1 for flag in self.flag_engine.flags.values() 
                if flag.rollout_strategy == strategy
            )
        
        # Recent evaluations
        total_evaluations = len(self.flag_engine.evaluation_cache)
        unique_users = len(self.flag_engine.user_assignments)
        
        # Experiment statistics
        active_experiments = sum(
            1 for exp in self.experiment_manager.experiments.values()
            if datetime.now() < exp.end_date
        )
        
        return {
            "dashboard_timestamp": datetime.now().isoformat(),
            "flag_statistics": {
                "total_flags": total_flags,
                "enabled_flags": enabled_flags,
                "disabled_flags": total_flags - enabled_flags
            },
            "rollout_strategies": rollout_stats,
            "evaluation_statistics": {
                "total_evaluations": total_evaluations,
                "unique_users": unique_users,
                "cache_hit_rate": 85.0  # Simulated
            },
            "experiment_statistics": {
                "active_experiments": active_experiments,
                "total_experiments": len(self.experiment_manager.experiments)
            },
            "flags_overview": [
                {
                    "name": flag.flag_name,
                    "type": flag.flag_type.value,
                    "strategy": flag.rollout_strategy.value,
                    "percentage": flag.percentage,
                    "enabled": flag.enabled,
                    "last_updated": flag.updated_at.isoformat()
                }
                for flag in self.flag_engine.flags.values()
            ]
        }

# Production feature flag demonstration
async def feature_flags_demo():
    """Comprehensive feature flag and experimentation demo"""
    print("🚩 Feature Flags & Experimentation Platform Demo")
    print("=" * 60)
    
    # Initialize systems
    flag_engine = FeatureFlagEngine()
    experiment_manager = ExperimentManager(flag_engine)
    dashboard = FeatureFlagDashboard(flag_engine, experiment_manager)
    
    print(f"✅ Feature flag engine initialized with {len(flag_engine.flags)} flags")
    print(f"✅ Experiment manager initialized with {len(experiment_manager.experiments)} experiments")
    
    # Test users
    test_users = [
        UserContext("user_001", "alice@company.com", "premium", "US", segment="beta_testers"),
        UserContext("user_002", "bob@company.com", "regular", "CA", segment="default"),
        UserContext("user_003", "carol@company.com", "enterprise", "UK", segment="early_adopters"),
        UserContext("user_004", "david@company.com", "regular", "DE", segment="default"),
        UserContext("user_005", "eve@company.com", "premium", "FR", segment="beta_testers")
    ]
    
    print(f"\n🔍 Evaluating flags for {len(test_users)} test users:")
    
    # Evaluate flags for each user
    user_flag_results = {}
    for user in test_users:
        user_flags = flag_engine.get_user_flags(user)
        user_flag_results[user.user_id] = user_flags
        
        print(f"\n   👤 {user.user_id} ({user.user_type}, {user.country}, {user.segment}):")
        for flag_name, flag_value in user_flags.items():
            flag_str = json.dumps(flag_value) if isinstance(flag_value, dict) else str(flag_value)
            print(f"      🚩 {flag_name}: {flag_str}")
    
    # Simulate experiment events
    print(f"\n📊 Simulating experiment events...")
    
    for user in test_users[:3]:  # Use first 3 users
        # ML analytics experiment
        experiment_manager.track_experiment_event(
            "ml_analytics_performance",
            user.user_id,
            "prediction_accuracy",
            random.uniform(0.80, 0.95)
        )
        
        # Dashboard UI experiment
        experiment_manager.track_experiment_event(
            "dashboard_ui_conversion",
            user.user_id,
            "task_completion_rate",
            random.uniform(0.65, 0.85)
        )
    
    # Get experiment results
    print(f"\n📈 Experiment Results:")
    
    for exp_name in experiment_manager.experiments.keys():
        results = experiment_manager.get_experiment_results(exp_name)
        
        print(f"\n   🧪 {results['experiment_name']}:")
        print(f"      Status: {results['status']}")
        print(f"      Hypothesis: {results['hypothesis']}")
        print(f"      Users: {results['unique_users']}")
        
        for variant, stats in results["variants"].items():
            print(f"      📊 {variant}: {stats['users']} users, {stats['events']} events")
            for metric, value in stats["metrics"].items():
                print(f"         • {metric}: {value:.1%}")
    
    # Generate dashboard report
    dashboard_data = dashboard.get_dashboard_data()
    
    print(f"\n📋 Feature Flags Dashboard Summary:")
    print(f"   🚩 Total flags: {dashboard_data['flag_statistics']['total_flags']}")
    print(f"   ✅ Enabled: {dashboard_data['flag_statistics']['enabled_flags']}")
    print(f"   👥 Unique users: {dashboard_data['evaluation_statistics']['unique_users']}")
    print(f"   🧪 Active experiments: {dashboard_data['experiment_statistics']['active_experiments']}")
    
    # Export comprehensive report
    feature_flags_report = {
        "dashboard_data": dashboard_data,
        "user_flag_assignments": user_flag_results,
        "experiment_results": {
            exp_name: experiment_manager.get_experiment_results(exp_name)
            for exp_name in experiment_manager.experiments.keys()
        },
        "flag_configurations": {
            name: asdict(flag) for name, flag in flag_engine.flags.items()
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open("feature_flags_report.json", "w") as f:
        json.dump(feature_flags_report, f, indent=2, default=str)
    
    print(f"\n📁 Generated Files:")
    print(f"   📄 feature_flags_report.json - Complete feature flags and experiments report")
    
    print(f"\n✅ Feature flags and experimentation demo complete!")
    print(f"🎯 Ready for production A/B testing and controlled rollouts")
    
    return feature_flags_report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Run feature flags demonstration
    asyncio.run(feature_flags_demo())