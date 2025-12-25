#!/usr/bin/env python3
"""
Performance Optimization Monitoring System
Following Google SRE practices for production performance monitoring and optimization
"""

import asyncio
import json
import time
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import statistics
from collections import deque
import threading
import gc

# Import production components
from integrated_workflow_system import IntegratedWorkflowSystem
from specialized_workflow_agent import SpecializedWorkflowAgent
from production_logging_audit_system import ProductionLoggingSystem, LogLevel, AuditEventType

class PerformanceMetricType(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    GARBAGE_COLLECTION = "garbage_collection"
    DATABASE_LATENCY = "database_latency"
    WORKFLOW_DURATION = "workflow_duration"
    AGENT_EFFICIENCY = "agent_efficiency"

class OptimizationStrategy(Enum):
    SCALE_UP = "scale_up"
    SCALE_OUT = "scale_out"
    CACHE_OPTIMIZATION = "cache_optimization"
    RESOURCE_TUNING = "resource_tuning"
    ALGORITHM_IMPROVEMENT = "algorithm_improvement"
    CONCURRENCY_ADJUSTMENT = "concurrency_adjustment"
    MEMORY_OPTIMIZATION = "memory_optimization"
    DATABASE_OPTIMIZATION = "database_optimization"

@dataclass
class PerformanceMetric:
    metric_id: str
    timestamp: datetime
    metric_type: PerformanceMetricType
    value: float
    unit: str
    component: str
    tags: Dict[str, str]
    threshold_warning: float
    threshold_critical: float
    is_anomaly: bool = False
    percentile_rank: float = 0.0

@dataclass
class PerformanceBaseline:
    component: str
    metric_type: PerformanceMetricType
    baseline_value: float
    standard_deviation: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    seasonal_patterns: Dict[str, float]

@dataclass
class OptimizationRecommendation:
    recommendation_id: str
    timestamp: datetime
    component: str
    strategy: OptimizationStrategy
    description: str
    expected_improvement: float
    confidence_score: float
    implementation_effort: str  # LOW, MEDIUM, HIGH
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    metrics_impacted: List[PerformanceMetricType]
    estimated_cost: float
    roi_estimate: float

class PerformanceOptimizationMonitor:
    """Production performance monitoring with Google SRE best practices"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.metrics_buffer: deque = deque(maxlen=self.config['monitoring']['max_metrics_buffer'])
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.recommendations: List[OptimizationRecommendation] = []
        self.system = None
        self.agents: Dict[str, SpecializedWorkflowAgent] = {}
        self.logging_system = ProductionLoggingSystem()
        
        # Performance tracking data structures
        self.response_times: Dict[str, deque] = {}
        self.throughput_counters: Dict[str, int] = {}
        self.error_counters: Dict[str, int] = {}
        self.monitoring_active = True
        
        # Thread-safe metric collection
        self.metrics_lock = threading.Lock()
        
        # Create monitoring directory
        self.monitoring_path = Path(self.config['monitoring']['data_path'])
        self.monitoring_path.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for performance monitoring"""
        return {
            'monitoring': {
                'data_path': './performance_monitoring',
                'collection_interval_seconds': 10,
                'max_metrics_buffer': 10000,
                'anomaly_detection_enabled': True,
                'baseline_update_interval_hours': 24,
                'percentile_calculations': [50, 95, 99],
                'alert_thresholds': {
                    'cpu_usage_warning': 70.0,
                    'cpu_usage_critical': 90.0,
                    'memory_usage_warning': 80.0,
                    'memory_usage_critical': 95.0,
                    'response_time_warning': 5000.0,  # 5 seconds
                    'response_time_critical': 10000.0,  # 10 seconds
                    'error_rate_warning': 1.0,  # 1%
                    'error_rate_critical': 5.0   # 5%
                }
            },
            'optimization': {
                'ml_enabled': True,
                'recommendation_confidence_threshold': 0.7,
                'auto_optimization_enabled': False,  # Require manual approval
                'cost_benefit_analysis': True,
                'roi_threshold': 1.5,  # Minimum 1.5x ROI for recommendations
                'implementation_simulation': True
            },
            'baseline': {
                'minimum_samples': 100,
                'confidence_level': 0.95,
                'seasonal_analysis': True,
                'outlier_rejection_enabled': True,
                'outlier_threshold_std_devs': 3.0
            },
            'alerts': {
                'enabled': True,
                'notification_methods': ['log', 'audit'],
                'escalation_enabled': True,
                'suppression_minutes': 15,  # Prevent alert spam
            }
        }
    
    async def initialize(self):
        """Initialize performance monitoring system"""
        print("🔧 Initializing Performance Optimization Monitor...")
        
        # Initialize core system
        self.system = IntegratedWorkflowSystem()
        
        # Initialize production agents
        agent_configs = [
            ("performance_agent", "Performance metrics collection agent"),
            ("optimization_agent", "Performance optimization analysis agent"),
            ("baseline_agent", "Performance baseline management agent")
        ]
        
        for agent_id, description in agent_configs:
            agent = SpecializedWorkflowAgent(agent_id)
            await agent.initialize()
            self.agents[agent_id] = agent
            print(f"✅ {agent_id} initialized: {description}")
        
        # Load existing baselines
        await self._load_performance_baselines()
        
        # Start background monitoring
        asyncio.create_task(self._continuous_monitoring_loop())
        
        print("✅ Performance Optimization Monitor initialized")
        return True
    
    async def collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive performance metrics"""
        timestamp = datetime.utcnow()
        metrics = []
        
        try:
            # System-level metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # CPU metrics
            metrics.append(PerformanceMetric(
                metric_id=f"cpu_{int(time.time())}",
                timestamp=timestamp,
                metric_type=PerformanceMetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                component="system",
                tags={"host": "localhost"},
                threshold_warning=self.config['monitoring']['alert_thresholds']['cpu_usage_warning'],
                threshold_critical=self.config['monitoring']['alert_thresholds']['cpu_usage_critical']
            ))
            
            # Memory metrics
            metrics.append(PerformanceMetric(
                metric_id=f"memory_{int(time.time())}",
                timestamp=timestamp,
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                value=memory.percent,
                unit="percent",
                component="system",
                tags={"host": "localhost"},
                threshold_warning=self.config['monitoring']['alert_thresholds']['memory_usage_warning'],
                threshold_critical=self.config['monitoring']['alert_thresholds']['memory_usage_critical']
            ))
            
            # Disk I/O metrics
            if disk_io:
                metrics.append(PerformanceMetric(
                    metric_id=f"disk_read_{int(time.time())}",
                    timestamp=timestamp,
                    metric_type=PerformanceMetricType.DISK_IO,
                    value=disk_io.read_bytes,
                    unit="bytes",
                    component="system",
                    tags={"operation": "read", "host": "localhost"},
                    threshold_warning=1e9,  # 1GB
                    threshold_critical=5e9   # 5GB
                ))
            
            # Network I/O metrics
            if network_io:
                metrics.append(PerformanceMetric(
                    metric_id=f"network_sent_{int(time.time())}",
                    timestamp=timestamp,
                    metric_type=PerformanceMetricType.NETWORK_IO,
                    value=network_io.bytes_sent,
                    unit="bytes",
                    component="system",
                    tags={"direction": "sent", "host": "localhost"},
                    threshold_warning=1e8,  # 100MB
                    threshold_critical=1e9   # 1GB
                ))
            
            # Application-specific metrics
            if self.system:
                # Workflow performance metrics
                workflow_metrics = await self._collect_workflow_metrics()
                metrics.extend(workflow_metrics)
            
            # Agent performance metrics
            for agent_id, agent in self.agents.items():
                agent_metrics = await self._collect_agent_metrics(agent_id, agent)
                metrics.extend(agent_metrics)
            
            # Garbage collection metrics
            gc_stats = gc.get_stats()
            if gc_stats:
                for i, gen_stats in enumerate(gc_stats):
                    metrics.append(PerformanceMetric(
                        metric_id=f"gc_gen{i}_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.GARBAGE_COLLECTION,
                        value=gen_stats['collections'],
                        unit="count",
                        component="python_runtime",
                        tags={"generation": str(i)},
                        threshold_warning=100,
                        threshold_critical=1000
                    ))
            
        except Exception as e:
            print(f"❌ Error collecting metrics: {e}")
        
        # Store metrics in buffer
        with self.metrics_lock:
            self.metrics_buffer.extend(metrics)
        
        # Detect anomalies
        if self.config['monitoring']['anomaly_detection_enabled']:
            await self._detect_anomalies(metrics)
        
        # Generate alerts if needed
        await self._check_alert_thresholds(metrics)
        
        return metrics
    
    async def _collect_workflow_metrics(self) -> List[PerformanceMetric]:
        """Collect workflow-specific performance metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Get system health (this provides performance data)
            health_data = await self.system.get_system_health()
            
            if 'performance' in health_data:
                perf_data = health_data['performance']
                
                # Response time metric
                if 'avg_response_time_ms' in perf_data:
                    metrics.append(PerformanceMetric(
                        metric_id=f"workflow_response_time_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.RESPONSE_TIME,
                        value=perf_data['avg_response_time_ms'],
                        unit="milliseconds",
                        component="workflow_system",
                        tags={"type": "average"},
                        threshold_warning=self.config['monitoring']['alert_thresholds']['response_time_warning'],
                        threshold_critical=self.config['monitoring']['alert_thresholds']['response_time_critical']
                    ))
                
                # Throughput metric
                if 'throughput_per_hour' in perf_data:
                    metrics.append(PerformanceMetric(
                        metric_id=f"workflow_throughput_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.THROUGHPUT,
                        value=perf_data['throughput_per_hour'],
                        unit="workflows/hour",
                        component="workflow_system",
                        tags={"type": "hourly"},
                        threshold_warning=1.0,
                        threshold_critical=0.1
                    ))
                
                # Error rate metric
                if 'error_rate_percent' in perf_data:
                    metrics.append(PerformanceMetric(
                        metric_id=f"workflow_error_rate_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.ERROR_RATE,
                        value=perf_data['error_rate_percent'],
                        unit="percent",
                        component="workflow_system",
                        tags={"type": "error_rate"},
                        threshold_warning=self.config['monitoring']['alert_thresholds']['error_rate_warning'],
                        threshold_critical=self.config['monitoring']['alert_thresholds']['error_rate_critical']
                    ))
        
        except Exception as e:
            print(f"❌ Error collecting workflow metrics: {e}")
        
        return metrics
    
    async def _collect_agent_metrics(self, agent_id: str, agent: SpecializedWorkflowAgent) -> List[PerformanceMetric]:
        """Collect agent-specific performance metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Get agent status
            agent_status = await agent.get_agent_status()
            
            if 'performance' in agent_status:
                perf_data = agent_status['performance']
                
                # Agent efficiency metric
                if 'success_rate_percent' in perf_data:
                    metrics.append(PerformanceMetric(
                        metric_id=f"agent_efficiency_{agent_id}_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.AGENT_EFFICIENCY,
                        value=perf_data['success_rate_percent'],
                        unit="percent",
                        component=f"agent_{agent_id}",
                        tags={"agent_id": agent_id, "metric": "success_rate"},
                        threshold_warning=95.0,
                        threshold_critical=80.0
                    ))
                
                # Queue length metric
                if 'queue_size' in perf_data:
                    metrics.append(PerformanceMetric(
                        metric_id=f"agent_queue_{agent_id}_{int(time.time())}",
                        timestamp=timestamp,
                        metric_type=PerformanceMetricType.QUEUE_LENGTH,
                        value=perf_data['queue_size'],
                        unit="count",
                        component=f"agent_{agent_id}",
                        tags={"agent_id": agent_id, "metric": "queue_length"},
                        threshold_warning=50.0,
                        threshold_critical=100.0
                    ))
        
        except Exception as e:
            print(f"❌ Error collecting metrics for agent {agent_id}: {e}")
        
        return metrics
    
    async def _detect_anomalies(self, metrics: List[PerformanceMetric]):
        """Detect performance anomalies using statistical methods"""
        for metric in metrics:
            baseline_key = f"{metric.component}_{metric.metric_type.value}"
            
            if baseline_key in self.baselines:
                baseline = self.baselines[baseline_key]
                
                # Calculate z-score
                if baseline.standard_deviation > 0:
                    z_score = abs(metric.value - baseline.baseline_value) / baseline.standard_deviation
                    
                    # Mark as anomaly if beyond threshold
                    if z_score > self.config['baseline']['outlier_threshold_std_devs']:
                        metric.is_anomaly = True
                        
                        # Log anomaly
                        self.logging_system.log(
                            LogLevel.WARNING,
                            f"Performance anomaly detected: {metric.component}.{metric.metric_type.value} = {metric.value} (z-score: {z_score:.2f})",
                            "performance_monitor",
                            metadata={
                                "metric_id": metric.metric_id,
                                "baseline_value": baseline.baseline_value,
                                "z_score": z_score,
                                "threshold": self.config['baseline']['outlier_threshold_std_devs']
                            },
                            tags=["anomaly", "performance"]
                        )
                        
                        # Audit the anomaly
                        self.logging_system.audit(
                            AuditEventType.SYSTEM_EVENT,
                            "performance_anomaly_detected",
                            f"{metric.component}_{metric.metric_type.value}",
                            metadata={
                                "metric_value": metric.value,
                                "baseline_value": baseline.baseline_value,
                                "z_score": z_score
                            }
                        )
    
    async def _check_alert_thresholds(self, metrics: List[PerformanceMetric]):
        """Check metrics against alert thresholds"""
        for metric in metrics:
            alert_triggered = False
            alert_level = None
            
            if metric.value >= metric.threshold_critical:
                alert_triggered = True
                alert_level = "CRITICAL"
            elif metric.value >= metric.threshold_warning:
                alert_triggered = True
                alert_level = "WARNING"
            
            if alert_triggered:
                # Log alert
                self.logging_system.log(
                    LogLevel.CRITICAL if alert_level == "CRITICAL" else LogLevel.WARNING,
                    f"Performance alert: {metric.component}.{metric.metric_type.value} = {metric.value} {metric.unit} ({alert_level})",
                    "performance_monitor",
                    metadata={
                        "metric_id": metric.metric_id,
                        "threshold_warning": metric.threshold_warning,
                        "threshold_critical": metric.threshold_critical,
                        "alert_level": alert_level
                    },
                    tags=["alert", "performance", alert_level.lower()]
                )
                
                # Audit the alert
                self.logging_system.audit(
                    AuditEventType.SYSTEM_EVENT,
                    "performance_alert_triggered",
                    f"{metric.component}_{metric.metric_type.value}",
                    outcome="SUCCESS",
                    severity=LogLevel.CRITICAL if alert_level == "CRITICAL" else LogLevel.WARNING,
                    metadata={
                        "metric_value": metric.value,
                        "alert_level": alert_level,
                        "threshold": metric.threshold_critical if alert_level == "CRITICAL" else metric.threshold_warning
                    }
                )
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate ML-driven optimization recommendations"""
        recommendations = []
        timestamp = datetime.utcnow()
        
        if not self.config['optimization']['ml_enabled']:
            return recommendations
        
        try:
            # Analyze recent metrics for optimization opportunities
            recent_metrics = list(self.metrics_buffer)[-1000:] if self.metrics_buffer else []
            
            if len(recent_metrics) < 10:
                return recommendations
            
            # Group metrics by component and type
            metrics_by_component = {}
            for metric in recent_metrics:
                key = f"{metric.component}_{metric.metric_type.value}"
                if key not in metrics_by_component:
                    metrics_by_component[key] = []
                metrics_by_component[key].append(metric)
            
            # Analyze each metric group for optimization opportunities
            for key, metric_group in metrics_by_component.items():
                if len(metric_group) < 5:
                    continue
                
                component, metric_type_str = key.split('_', 1)
                metric_type = PerformanceMetricType(metric_type_str)
                
                # Calculate statistics
                values = [m.value for m in metric_group]
                avg_value = statistics.mean(values)
                max_value = max(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                # Generate recommendations based on metric type and values
                recommendation = await self._analyze_metric_for_optimization(
                    component, metric_type, avg_value, max_value, std_dev, metric_group
                )
                
                if recommendation:
                    recommendations.append(recommendation)
        
        except Exception as e:
            print(f"❌ Error generating optimization recommendations: {e}")
        
        # Filter recommendations by confidence threshold
        filtered_recommendations = [
            rec for rec in recommendations
            if rec.confidence_score >= self.config['optimization']['recommendation_confidence_threshold']
        ]
        
        # Store recommendations
        self.recommendations.extend(filtered_recommendations)
        
        # Log recommendations
        for rec in filtered_recommendations:
            self.logging_system.log(
                LogLevel.INFO,
                f"Optimization recommendation generated: {rec.strategy.value} for {rec.component} - {rec.description}",
                "optimization_engine",
                metadata={
                    "recommendation_id": rec.recommendation_id,
                    "expected_improvement": rec.expected_improvement,
                    "confidence_score": rec.confidence_score,
                    "priority": rec.priority
                },
                tags=["optimization", "recommendation"]
            )
        
        return filtered_recommendations
    
    async def _analyze_metric_for_optimization(self, component: str, metric_type: PerformanceMetricType,
                                            avg_value: float, max_value: float, std_dev: float,
                                            metric_group: List[PerformanceMetric]) -> Optional[OptimizationRecommendation]:
        """Analyze specific metric for optimization opportunities"""
        
        # Get baseline for comparison
        baseline_key = f"{component}_{metric_type.value}"
        baseline = self.baselines.get(baseline_key)
        
        if not baseline:
            return None
        
        recommendation_id = f"opt_{component}_{metric_type.value}_{int(time.time())}"
        
        # CPU Usage optimization
        if metric_type == PerformanceMetricType.CPU_USAGE:
            if avg_value > 80.0:
                return OptimizationRecommendation(
                    recommendation_id=recommendation_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    strategy=OptimizationStrategy.SCALE_UP,
                    description=f"High CPU usage detected ({avg_value:.1f}%). Consider scaling up CPU resources or optimizing algorithms.",
                    expected_improvement=25.0,
                    confidence_score=0.85,
                    implementation_effort="MEDIUM",
                    priority="HIGH",
                    metrics_impacted=[PerformanceMetricType.CPU_USAGE, PerformanceMetricType.RESPONSE_TIME],
                    estimated_cost=500.0,
                    roi_estimate=2.5
                )
        
        # Memory Usage optimization
        elif metric_type == PerformanceMetricType.MEMORY_USAGE:
            if avg_value > 85.0:
                return OptimizationRecommendation(
                    recommendation_id=recommendation_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                    description=f"High memory usage detected ({avg_value:.1f}%). Implement memory optimization or increase available memory.",
                    expected_improvement=30.0,
                    confidence_score=0.9,
                    implementation_effort="HIGH",
                    priority="CRITICAL",
                    metrics_impacted=[PerformanceMetricType.MEMORY_USAGE, PerformanceMetricType.GARBAGE_COLLECTION],
                    estimated_cost=300.0,
                    roi_estimate=3.0
                )
        
        # Response Time optimization
        elif metric_type == PerformanceMetricType.RESPONSE_TIME:
            if avg_value > baseline.baseline_value * 1.5:  # 50% slower than baseline
                return OptimizationRecommendation(
                    recommendation_id=recommendation_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                    description=f"Response time degradation detected ({avg_value:.0f}ms vs {baseline.baseline_value:.0f}ms baseline). Implement caching or optimize database queries.",
                    expected_improvement=40.0,
                    confidence_score=0.8,
                    implementation_effort="MEDIUM",
                    priority="HIGH",
                    metrics_impacted=[PerformanceMetricType.RESPONSE_TIME, PerformanceMetricType.THROUGHPUT],
                    estimated_cost=200.0,
                    roi_estimate=4.0
                )
        
        # Throughput optimization
        elif metric_type == PerformanceMetricType.THROUGHPUT:
            if avg_value < baseline.baseline_value * 0.8:  # 20% lower than baseline
                return OptimizationRecommendation(
                    recommendation_id=recommendation_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    strategy=OptimizationStrategy.CONCURRENCY_ADJUSTMENT,
                    description=f"Throughput decline detected ({avg_value:.1f} vs {baseline.baseline_value:.1f} baseline). Adjust concurrency settings or optimize bottlenecks.",
                    expected_improvement=35.0,
                    confidence_score=0.75,
                    implementation_effort="MEDIUM",
                    priority="MEDIUM",
                    metrics_impacted=[PerformanceMetricType.THROUGHPUT, PerformanceMetricType.QUEUE_LENGTH],
                    estimated_cost=100.0,
                    roi_estimate=2.0
                )
        
        # Error Rate optimization
        elif metric_type == PerformanceMetricType.ERROR_RATE:
            if avg_value > 2.0:  # More than 2% error rate
                return OptimizationRecommendation(
                    recommendation_id=recommendation_id,
                    timestamp=datetime.utcnow(),
                    component=component,
                    strategy=OptimizationStrategy.ALGORITHM_IMPROVEMENT,
                    description=f"High error rate detected ({avg_value:.1f}%). Improve error handling, input validation, or algorithm robustness.",
                    expected_improvement=60.0,
                    confidence_score=0.9,
                    implementation_effort="HIGH",
                    priority="CRITICAL",
                    metrics_impacted=[PerformanceMetricType.ERROR_RATE, PerformanceMetricType.THROUGHPUT],
                    estimated_cost=800.0,
                    roi_estimate=5.0
                )
        
        return None
    
    async def update_performance_baselines(self):
        """Update performance baselines using recent data"""
        if len(self.metrics_buffer) < self.config['baseline']['minimum_samples']:
            return
        
        # Group metrics by component and type
        metrics_by_key = {}
        for metric in self.metrics_buffer:
            key = f"{metric.component}_{metric.metric_type.value}"
            if key not in metrics_by_key:
                metrics_by_key[key] = []
            metrics_by_key[key].append(metric)
        
        # Update baselines
        for key, metric_group in metrics_by_key.items():
            if len(metric_group) < self.config['baseline']['minimum_samples']:
                continue
            
            # key format is component_metric_type
            # We need to find the split point.
            # Since component can have underscores, we should iterate over PerformanceMetricType values to find the match.
            metric_type = None
            component = None
            for mt in PerformanceMetricType:
                if key.endswith(mt.value):
                     metric_type = mt
                     component = key[:-len(mt.value)-1] # -1 for the underscore
                     break
            
            if not metric_type:
                 # Fallback for unexpected keys
                 try:
                    component, metric_type_str = key.split('_', 1)
                    metric_type = PerformanceMetricType(metric_type_str)
                 except ValueError:
                    continue

            # Filter outliers if enabled
            values = [m.value for m in metric_group]
            if self.config['baseline']['outlier_rejection_enabled']:
                values = self._remove_outliers(values)
            
            if len(values) < self.config['baseline']['minimum_samples']:
                continue
            
            # Calculate baseline statistics
            baseline_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            # Calculate confidence interval
            confidence_level = self.config['baseline']['confidence_level']
            margin_of_error = 1.96 * (std_dev / np.sqrt(len(values)))  # 95% confidence
            confidence_interval = (baseline_value - margin_of_error, baseline_value + margin_of_error)
            
            # Calculate seasonal patterns
            seasonal_patterns = {}
            if self.config['baseline'].get('seasonal_analysis', False):
                seasonal_patterns = self._calculate_seasonal_patterns(metric_group)

            # Update baseline
            self.baselines[key] = PerformanceBaseline(
                component=component,
                metric_type=metric_type,
                baseline_value=baseline_value,
                standard_deviation=std_dev,
                sample_size=len(values),
                confidence_interval=confidence_interval,
                last_updated=datetime.utcnow(),
                seasonal_patterns=seasonal_patterns
            )
        
        # Save baselines to disk
        await self._save_performance_baselines()
        
        # Log baseline updates
        self.logging_system.log(
            LogLevel.INFO,
            f"Performance baselines updated: {len(self.baselines)} baselines",
            "baseline_manager",
            metadata={
                "baselines_count": len(self.baselines),
                "update_timestamp": datetime.utcnow().isoformat()
            },
            tags=["baseline", "update"]
        )
    
    def _calculate_seasonal_patterns(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """
        Calculate seasonal patterns from metrics.
        Currently implements hourly seasonality (0-23).
        """
        hourly_values: Dict[int, List[float]] = {}

        for metric in metrics:
            hour = metric.timestamp.hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(metric.value)

        seasonal_patterns: Dict[str, float] = {}

        for hour, values in hourly_values.items():
            if values:
                avg_val = statistics.mean(values)
                seasonal_patterns[f"hour_{hour}"] = avg_val

        return seasonal_patterns

    def _remove_outliers(self, values: List[float]) -> List[float]:
        """Remove outliers using IQR method"""
        if len(values) < 4:
            return values
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [v for v in values if lower_bound <= v <= upper_bound]
    
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                await self.collect_performance_metrics()
                
                # Generate recommendations periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    await self.generate_optimization_recommendations()
                
                # Update baselines periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    await self.update_performance_baselines()
                
                # Wait for next collection
                await asyncio.sleep(self.config['monitoring']['collection_interval_seconds'])
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(self.config['monitoring']['collection_interval_seconds'])
    
    async def _save_performance_baselines(self):
        """Save performance baselines to disk"""
        baselines_file = self.monitoring_path / "performance_baselines.json"
        baselines_data = {}
        
        for key, baseline in self.baselines.items():
            baseline_dict = asdict(baseline)
            baseline_dict['last_updated'] = baseline_dict['last_updated'].isoformat()
            baseline_dict['metric_type'] = baseline_dict['metric_type'].value
            baselines_data[key] = baseline_dict
        
        with open(baselines_file, 'w') as f:
            json.dump(baselines_data, f, indent=2, default=str)
    
    async def _load_performance_baselines(self):
        """Load performance baselines from disk"""
        baselines_file = self.monitoring_path / "performance_baselines.json"
        
        if baselines_file.exists():
            try:
                with open(baselines_file, 'r') as f:
                    baselines_data = json.load(f)
                
                for key, baseline_dict in baselines_data.items():
                    baseline_dict['last_updated'] = datetime.fromisoformat(baseline_dict['last_updated'])
                    baseline_dict['metric_type'] = PerformanceMetricType(baseline_dict['metric_type'])
                    self.baselines[key] = PerformanceBaseline(**baseline_dict)
                
                print(f"📋 Loaded {len(self.baselines)} performance baselines")
                
            except Exception as e:
                print(f"❌ Failed to load baselines: {e}")
    
    async def get_performance_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive performance monitoring status"""
        
        # Calculate statistics
        total_metrics = len(self.metrics_buffer)
        total_baselines = len(self.baselines)
        total_recommendations = len(self.recommendations)
        
        active_recommendations = len([r for r in self.recommendations if r.priority in ["HIGH", "CRITICAL"]])
        
        # Recent metrics analysis
        recent_metrics = list(self.metrics_buffer)[-100:] if self.metrics_buffer else []
        anomaly_count = len([m for m in recent_metrics if m.is_anomaly])
        
        # Component analysis
        components = set(m.component for m in recent_metrics)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'metrics': {
                'total_collected': total_metrics,
                'recent_anomalies': anomaly_count,
                'monitored_components': len(components),
                'collection_interval_seconds': self.config['monitoring']['collection_interval_seconds']
            },
            'baselines': {
                'total_baselines': total_baselines,
                'baseline_coverage': list(self.baselines.keys()),
                'minimum_samples': self.config['baseline']['minimum_samples']
            },
            'optimization': {
                'total_recommendations': total_recommendations,
                'active_high_priority': active_recommendations,
                'ml_enabled': self.config['optimization']['ml_enabled'],
                'confidence_threshold': self.config['optimization']['recommendation_confidence_threshold']
            },
            'performance_summary': {
                'components_monitored': list(components),
                'anomaly_detection': self.config['monitoring']['anomaly_detection_enabled'],
                'alert_thresholds': self.config['monitoring']['alert_thresholds']
            }
        }

async def main():
    """Main performance optimization monitoring entry point"""
    performance_monitor = PerformanceOptimizationMonitor()
    
    try:
        # Initialize monitoring system
        await performance_monitor.initialize()
        
        print("\n🔄 Starting performance monitoring cycle...")
        
        # Collect initial metrics
        initial_metrics = await performance_monitor.collect_performance_metrics()
        print(f"📊 Collected {len(initial_metrics)} initial metrics")
        
        # Wait for some data collection
        await asyncio.sleep(30)
        
        # Generate optimization recommendations
        recommendations = await performance_monitor.generate_optimization_recommendations()
        print(f"💡 Generated {len(recommendations)} optimization recommendations")
        
        for rec in recommendations[:3]:  # Show first 3 recommendations
            print(f"   • {rec.strategy.value}: {rec.description[:80]}... (Priority: {rec.priority})")
        
        # Get system status
        status = await performance_monitor.get_performance_monitoring_status()
        print(f"\n📊 Performance Monitoring Status:")
        print(f"   Metrics collected: {status['metrics']['total_collected']}")
        print(f"   Baselines: {status['baselines']['total_baselines']}")
        print(f"   Recommendations: {status['optimization']['total_recommendations']}")
        print(f"   High priority actions: {status['optimization']['active_high_priority']}")
        print(f"   Monitored components: {status['metrics']['monitored_components']}")
        
        print("\n🏆 PERFORMANCE OPTIMIZATION MONITORING OPERATIONAL")
        
        # Continue monitoring for a short while in demo
        print("\n🔄 Monitoring for 60 seconds...")
        await asyncio.sleep(60)
        
        performance_monitor.monitoring_active = False
        print("⏹️ Performance monitoring demo complete")
        
    except Exception as e:
        print(f"\n❌ Performance monitoring error: {e}")
        performance_monitor.monitoring_active = False

if __name__ == "__main__":
    asyncio.run(main())