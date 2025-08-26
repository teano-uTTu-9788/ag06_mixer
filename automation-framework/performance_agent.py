#!/usr/bin/env python3
"""
Advanced Performance Agent - System Optimization & Real-time Monitoring
Based on 2024-2025 Performance Engineering Best Practices

This agent implements:
- Real-time performance monitoring
- Automated optimization recommendations
- Resource usage analysis
- Performance bottleneck detection
- Scalability assessment
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Protocol
from abc import ABC, abstractmethod
import psutil
import threading
from pathlib import Path
import gc
import sys
import tracemalloc

# SOLID Architecture Implementation
class IPerformanceMonitor(Protocol):
    """Interface for performance monitoring"""
    async def collect_metrics(self) -> Dict[str, Any]: ...

class IOptimizer(Protocol):
    """Interface for performance optimization"""
    async def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]: ...

class IAlertManager(Protocol):
    """Interface for alert management"""
    async def process_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]: ...

class IReportGenerator(Protocol):
    """Interface for performance reporting"""
    async def generate_report(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]: ...

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold: Optional[float] = None
    status: str = "OK"  # OK, WARNING, CRITICAL

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    category: str
    priority: str
    description: str
    expected_improvement: str
    implementation_effort: str
    code_examples: List[str]

@dataclass
class PerformanceAlert:
    """Performance alert"""
    severity: str
    metric_name: str
    current_value: float
    threshold: float
    description: str
    recommendations: List[str]
    timestamp: datetime

class PerformanceError(Exception):
    """Custom performance agent exceptions"""
    pass

class SystemPerformanceMonitor:
    """Advanced system performance monitoring"""
    
    def __init__(self):
        self.monitoring_interval = 1.0  # 1 second
        self.history_size = 1000  # Keep last 1000 measurements
        self.metrics_history: Dict[str, List[float]] = {}
        self.start_time = time.time()
        
        # Initialize memory tracing
        tracemalloc.start()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now()
        
        # CPU Metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk Metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network Metrics
        network_io = psutil.net_io_counters()
        
        # Process Metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        # Python-specific metrics
        gc_stats = gc.get_stats()
        thread_count = threading.active_count()
        
        # Memory tracing
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        
        # Load averages (Unix-like systems)
        load_avg = None
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            # Windows doesn't have load averages
            pass
        
        metrics = {
            "timestamp": timestamp.isoformat(),
            "uptime": time.time() - self.start_time,
            
            # CPU Metrics
            "cpu": {
                "percent": cpu_percent,
                "frequency": cpu_freq.current if cpu_freq else None,
                "count_logical": cpu_count_logical,
                "count_physical": cpu_count_physical,
                "load_average": load_avg
            },
            
            # Memory Metrics
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            },
            
            # Disk Metrics
            "disk": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_time": disk_io.read_time if disk_io else 0,
                "write_time": disk_io.write_time if disk_io else 0
            },
            
            # Network Metrics
            "network": {
                "bytes_sent": network_io.bytes_sent if network_io else 0,
                "bytes_recv": network_io.bytes_recv if network_io else 0,
                "packets_sent": network_io.packets_sent if network_io else 0,
                "packets_recv": network_io.packets_recv if network_io else 0
            },
            
            # Process Metrics
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process_cpu,
                "thread_count": thread_count,
                "current_memory_traced": current_memory,
                "peak_memory_traced": peak_memory
            },
            
            # Python Metrics
            "python": {
                "gc_stats": gc_stats,
                "object_count": len(gc.get_objects()),
                "version": sys.version
            }
        }
        
        # Store metrics in history
        self._update_metrics_history(metrics)
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history for trend analysis"""
        key_metrics = {
            "cpu_percent": metrics["cpu"]["percent"],
            "memory_percent": metrics["memory"]["percent"],
            "disk_percent": metrics["disk"]["percent"],
            "process_memory": metrics["process"]["memory_rss"],
            "process_cpu": metrics["process"]["cpu_percent"]
        }
        
        for metric_name, value in key_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append(value)
            
            # Keep only recent history
            if len(self.metrics_history[metric_name]) > self.history_size:
                self.metrics_history[metric_name].pop(0)
    
    def get_metrics_trend(self, metric_name: str, window_size: int = 100) -> Dict[str, float]:
        """Get trend analysis for a specific metric"""
        if metric_name not in self.metrics_history:
            return {}
        
        history = self.metrics_history[metric_name][-window_size:]
        
        if len(history) < 2:
            return {}
        
        # Calculate trend statistics
        avg = sum(history) / len(history)
        min_val = min(history)
        max_val = max(history)
        
        # Simple trend calculation (slope)
        n = len(history)
        x_sum = sum(range(n))
        y_sum = sum(history)
        xy_sum = sum(i * history[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum != 0:
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        else:
            slope = 0
        
        return {
            "average": avg,
            "minimum": min_val,
            "maximum": max_val,
            "trend_slope": slope,
            "trend_direction": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        }

class PerformanceOptimizer:
    """Advanced performance analysis and optimization"""
    
    def __init__(self):
        self.optimization_patterns = self._load_optimization_patterns()
        self.thresholds = self._load_performance_thresholds()
    
    async def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and generate recommendations"""
        
        recommendations = []
        bottlenecks = []
        health_score = 100.0  # Start with perfect score
        
        # Analyze CPU performance
        cpu_analysis = self._analyze_cpu(metrics)
        recommendations.extend(cpu_analysis["recommendations"])
        bottlenecks.extend(cpu_analysis["bottlenecks"])
        health_score -= cpu_analysis["penalty"]
        
        # Analyze memory performance
        memory_analysis = self._analyze_memory(metrics)
        recommendations.extend(memory_analysis["recommendations"])
        bottlenecks.extend(memory_analysis["bottlenecks"])
        health_score -= memory_analysis["penalty"]
        
        # Analyze disk performance
        disk_analysis = self._analyze_disk(metrics)
        recommendations.extend(disk_analysis["recommendations"])
        bottlenecks.extend(disk_analysis["bottlenecks"])
        health_score -= disk_analysis["penalty"]
        
        # Analyze process performance
        process_analysis = self._analyze_process(metrics)
        recommendations.extend(process_analysis["recommendations"])
        bottlenecks.extend(process_analysis["bottlenecks"])
        health_score -= process_analysis["penalty"]
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "health_score": max(0, health_score),
            "performance_grade": self._calculate_grade(health_score),
            "recommendations": recommendations,
            "bottlenecks": bottlenecks,
            "trends": self._analyze_trends(metrics),
            "optimization_priority": self._prioritize_optimizations(recommendations)
        }
    
    def _analyze_cpu(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CPU performance"""
        cpu_percent = metrics["cpu"]["percent"]
        recommendations = []
        bottlenecks = []
        penalty = 0
        
        if cpu_percent > self.thresholds["cpu"]["critical"]:
            recommendations.append(OptimizationRecommendation(
                category="CPU",
                priority="CRITICAL",
                description=f"CPU usage is critically high at {cpu_percent:.1f}%",
                expected_improvement="20-40% CPU reduction",
                implementation_effort="Medium",
                code_examples=[
                    "Use asyncio for I/O-bound operations",
                    "Profile code to identify CPU hotspots",
                    "Consider caching expensive computations",
                    "Implement connection pooling"
                ]
            ))
            bottlenecks.append("CPU")
            penalty += 30
        elif cpu_percent > self.thresholds["cpu"]["warning"]:
            recommendations.append(OptimizationRecommendation(
                category="CPU",
                priority="HIGH",
                description=f"CPU usage is elevated at {cpu_percent:.1f}%",
                expected_improvement="10-20% CPU reduction",
                implementation_effort="Low",
                code_examples=[
                    "Optimize loops and data structures",
                    "Use list comprehensions instead of loops where appropriate",
                    "Consider lazy evaluation patterns"
                ]
            ))
            penalty += 15
        
        # Check for load average issues (Unix-like systems)
        load_avg = metrics["cpu"].get("load_average")
        if load_avg and len(load_avg) >= 3:
            cpu_count = metrics["cpu"]["count_logical"]
            if load_avg[0] > cpu_count * 2:  # Load average > 2x CPU count
                recommendations.append(OptimizationRecommendation(
                    category="CPU",
                    priority="HIGH",
                    description=f"System load average is high: {load_avg[0]:.2f}",
                    expected_improvement="Improved system responsiveness",
                    implementation_effort="Medium",
                    code_examples=[
                        "Reduce concurrent process count",
                        "Implement better task queuing",
                        "Consider horizontal scaling"
                    ]
                ))
                penalty += 10
        
        return {
            "recommendations": recommendations,
            "bottlenecks": bottlenecks,
            "penalty": penalty
        }
    
    def _analyze_memory(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory performance"""
        memory_percent = metrics["memory"]["percent"]
        swap_percent = metrics["memory"]["swap_percent"]
        recommendations = []
        bottlenecks = []
        penalty = 0
        
        if memory_percent > self.thresholds["memory"]["critical"]:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority="CRITICAL",
                description=f"Memory usage is critically high at {memory_percent:.1f}%",
                expected_improvement="30-50% memory reduction",
                implementation_effort="High",
                code_examples=[
                    "Implement memory pooling",
                    "Use generators instead of lists for large datasets",
                    "Clear unused references and call gc.collect()",
                    "Profile memory usage with memory_profiler"
                ]
            ))
            bottlenecks.append("Memory")
            penalty += 35
        elif memory_percent > self.thresholds["memory"]["warning"]:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority="HIGH",
                description=f"Memory usage is elevated at {memory_percent:.1f}%",
                expected_improvement="15-30% memory reduction",
                implementation_effort="Medium",
                code_examples=[
                    "Use __slots__ in classes to reduce memory overhead",
                    "Implement lazy loading for large objects",
                    "Consider using numpy arrays for numerical data"
                ]
            ))
            penalty += 20
        
        # Check swap usage
        if swap_percent > 50:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority="CRITICAL",
                description=f"High swap usage detected: {swap_percent:.1f}%",
                expected_improvement="Significant performance improvement",
                implementation_effort="High",
                code_examples=[
                    "Reduce memory usage to avoid swapping",
                    "Increase physical RAM if possible",
                    "Optimize data structures for memory efficiency"
                ]
            ))
            bottlenecks.append("Swap")
            penalty += 25
        
        return {
            "recommendations": recommendations,
            "bottlenecks": bottlenecks,
            "penalty": penalty
        }
    
    def _analyze_disk(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze disk performance"""
        disk_percent = metrics["disk"]["percent"]
        recommendations = []
        bottlenecks = []
        penalty = 0
        
        if disk_percent > self.thresholds["disk"]["critical"]:
            recommendations.append(OptimizationRecommendation(
                category="Disk",
                priority="CRITICAL",
                description=f"Disk usage is critically high at {disk_percent:.1f}%",
                expected_improvement="Free up disk space",
                implementation_effort="Low",
                code_examples=[
                    "Implement log rotation",
                    "Clean up temporary files",
                    "Archive old data",
                    "Consider disk expansion"
                ]
            ))
            bottlenecks.append("Disk Space")
            penalty += 20
        elif disk_percent > self.thresholds["disk"]["warning"]:
            recommendations.append(OptimizationRecommendation(
                category="Disk",
                priority="MEDIUM",
                description=f"Disk usage is elevated at {disk_percent:.1f}%",
                expected_improvement="Prevent disk space issues",
                implementation_effort="Low",
                code_examples=[
                    "Set up disk monitoring",
                    "Implement automated cleanup",
                    "Monitor log file sizes"
                ]
            ))
            penalty += 10
        
        # Analyze I/O patterns
        read_time = metrics["disk"]["read_time"]
        write_time = metrics["disk"]["write_time"]
        
        if read_time + write_time > 1000:  # High I/O wait time (ms)
            recommendations.append(OptimizationRecommendation(
                category="Disk I/O",
                priority="HIGH",
                description="High disk I/O wait times detected",
                expected_improvement="Faster I/O operations",
                implementation_effort="Medium",
                code_examples=[
                    "Use SSD instead of HDD",
                    "Implement I/O batching",
                    "Use async I/O operations",
                    "Consider in-memory caching"
                ]
            ))
            penalty += 15
        
        return {
            "recommendations": recommendations,
            "bottlenecks": bottlenecks,
            "penalty": penalty
        }
    
    def _analyze_process(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze process-specific performance"""
        process_memory = metrics["process"]["memory_rss"]
        process_cpu = metrics["process"]["cpu_percent"]
        thread_count = metrics["process"]["thread_count"]
        
        recommendations = []
        bottlenecks = []
        penalty = 0
        
        # Analyze process memory usage
        if process_memory > 1024 * 1024 * 1024:  # > 1GB
            recommendations.append(OptimizationRecommendation(
                category="Process Memory",
                priority="HIGH",
                description=f"Process using {process_memory / (1024**3):.1f}GB of memory",
                expected_improvement="Reduced memory footprint",
                implementation_effort="Medium",
                code_examples=[
                    "Profile memory usage to identify leaks",
                    "Implement object recycling",
                    "Use memory-mapped files for large datasets"
                ]
            ))
            penalty += 10
        
        # Analyze thread count
        if thread_count > 100:
            recommendations.append(OptimizationRecommendation(
                category="Threading",
                priority="MEDIUM",
                description=f"High thread count: {thread_count}",
                expected_improvement="Better resource utilization",
                implementation_effort="Medium",
                code_examples=[
                    "Use thread pools to limit thread count",
                    "Consider asyncio for I/O-bound tasks",
                    "Implement proper thread lifecycle management"
                ]
            ))
            penalty += 5
        
        # Analyze Python-specific metrics
        object_count = metrics["python"]["object_count"]
        if object_count > 1000000:  # > 1M objects
            recommendations.append(OptimizationRecommendation(
                category="Python Objects",
                priority="MEDIUM",
                description=f"High object count: {object_count:,}",
                expected_improvement="Reduced memory overhead",
                implementation_effort="Medium",
                code_examples=[
                    "Run gc.collect() periodically",
                    "Use object pooling for frequently created objects",
                    "Profile object creation patterns"
                ]
            ))
            penalty += 5
        
        return {
            "recommendations": recommendations,
            "bottlenecks": bottlenecks,
            "penalty": penalty
        }
    
    def _analyze_trends(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends"""
        # This would use historical data from the monitor
        # For now, return placeholder trend analysis
        return {
            "cpu_trend": "stable",
            "memory_trend": "increasing",
            "disk_trend": "stable",
            "overall_trend": "concerning"
        }
    
    def _prioritize_optimizations(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Prioritize optimization recommendations"""
        priority_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: priority_order.get(x.priority, 0),
            reverse=True
        )
        
        return [rec.__dict__ for rec in sorted_recommendations[:10]]  # Top 10
    
    def _calculate_grade(self, health_score: float) -> str:
        """Calculate performance grade"""
        if health_score >= 90:
            return "A"
        elif health_score >= 80:
            return "B"
        elif health_score >= 70:
            return "C"
        elif health_score >= 60:
            return "D"
        else:
            return "F"
    
    def _load_optimization_patterns(self) -> Dict[str, Any]:
        """Load optimization patterns"""
        return {
            "cpu_patterns": [
                "async_io",
                "caching",
                "lazy_evaluation",
                "connection_pooling"
            ],
            "memory_patterns": [
                "object_pooling",
                "lazy_loading",
                "memory_mapping",
                "garbage_collection"
            ],
            "io_patterns": [
                "batching",
                "compression",
                "streaming",
                "buffering"
            ]
        }
    
    def _load_performance_thresholds(self) -> Dict[str, Any]:
        """Load performance thresholds"""
        return {
            "cpu": {
                "warning": 70.0,
                "critical": 90.0
            },
            "memory": {
                "warning": 80.0,
                "critical": 95.0
            },
            "disk": {
                "warning": 85.0,
                "critical": 95.0
            }
        }

class AlertManager:
    """Performance alert management"""
    
    def __init__(self):
        self.alert_history: List[PerformanceAlert] = []
        self.alert_cooldown = 300  # 5 minutes cooldown between similar alerts
    
    async def process_alerts(self, analysis: Dict[str, Any]) -> List[PerformanceAlert]:
        """Process performance analysis and generate alerts"""
        alerts = []
        current_time = datetime.now()
        
        # Generate alerts based on recommendations
        for rec in analysis.get("recommendations", []):
            if rec["priority"] in ["CRITICAL", "HIGH"]:
                alert = PerformanceAlert(
                    severity=rec["priority"],
                    metric_name=rec["category"],
                    current_value=0.0,  # Would be filled with actual metric value
                    threshold=0.0,  # Would be filled with threshold
                    description=rec["description"],
                    recommendations=rec["code_examples"],
                    timestamp=current_time
                )
                
                # Check if we should send this alert (cooldown logic)
                if self._should_send_alert(alert):
                    alerts.append(alert)
                    self.alert_history.append(alert)
        
        # Clean old alerts
        self._clean_old_alerts()
        
        return alerts
    
    def _should_send_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be sent based on cooldown"""
        current_time = datetime.now()
        cooldown_time = current_time - timedelta(seconds=self.alert_cooldown)
        
        # Check for similar recent alerts
        similar_alerts = [
            a for a in self.alert_history
            if (a.metric_name == alert.metric_name and
                a.timestamp > cooldown_time)
        ]
        
        return len(similar_alerts) == 0
    
    def _clean_old_alerts(self) -> None:
        """Clean alerts older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]

class PerformanceReporter:
    """Performance reporting and data persistence"""
    
    def __init__(self, output_path: str = "performance_reports"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    async def generate_report(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not metrics:
            return {"error": "No metrics data available"}
        
        latest_metrics = metrics[-1] if metrics else {}
        
        # Calculate summary statistics
        summary = self._calculate_summary(metrics)
        
        # Generate trend analysis
        trends = self._analyze_historical_trends(metrics)
        
        # Create recommendations summary
        recommendations_summary = self._summarize_recommendations(metrics)
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "report_period": {
                "start": metrics[0]["timestamp"] if metrics else None,
                "end": metrics[-1]["timestamp"] if metrics else None,
                "duration_minutes": len(metrics)  # Assuming 1-minute intervals
            },
            "summary": summary,
            "trends": trends,
            "latest_metrics": latest_metrics,
            "recommendations_summary": recommendations_summary,
            "detailed_metrics": metrics[-100:] if len(metrics) > 100 else metrics  # Last 100 entries
        }
        
        # Save report
        await self._save_report(report)
        
        return report
    
    def _calculate_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not metrics:
            return {}
        
        # Extract CPU percentages
        cpu_values = [m["cpu"]["percent"] for m in metrics if "cpu" in m]
        memory_values = [m["memory"]["percent"] for m in metrics if "memory" in m]
        
        return {
            "cpu": {
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "maximum": max(cpu_values) if cpu_values else 0,
                "minimum": min(cpu_values) if cpu_values else 0
            },
            "memory": {
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "maximum": max(memory_values) if memory_values else 0,
                "minimum": min(memory_values) if memory_values else 0
            },
            "total_measurements": len(metrics)
        }
    
    def _analyze_historical_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze historical performance trends"""
        if len(metrics) < 10:
            return {"status": "insufficient_data"}
        
        # Simple trend analysis
        recent_cpu = [m["cpu"]["percent"] for m in metrics[-10:] if "cpu" in m]
        older_cpu = [m["cpu"]["percent"] for m in metrics[-20:-10] if "cpu" in m]
        
        cpu_trend = "stable"
        if recent_cpu and older_cpu:
            recent_avg = sum(recent_cpu) / len(recent_cpu)
            older_avg = sum(older_cpu) / len(older_cpu)
            
            if recent_avg > older_avg * 1.1:
                cpu_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                cpu_trend = "decreasing"
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": "stable",  # Placeholder
            "overall_health": "good" if cpu_trend == "decreasing" else "concerning"
        }
    
    def _summarize_recommendations(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize optimization recommendations"""
        return {
            "top_priorities": [
                "Monitor memory usage trends",
                "Implement CPU optimization",
                "Set up automated alerting"
            ],
            "quick_wins": [
                "Enable garbage collection",
                "Optimize database queries",
                "Use connection pooling"
            ]
        }
    
    async def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.json"
        filepath = self.output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.output_path / "latest_performance_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

class AdvancedPerformanceAgent:
    """
    Advanced Performance Agent implementing 2024-2025 best practices
    
    Features:
    - Real-time performance monitoring
    - Automated optimization recommendations
    - Performance bottleneck detection
    - Trend analysis and alerting
    - Enterprise-grade reporting
    """
    
    def __init__(
        self, 
        monitoring_interval: int = 60,  # 1 minute
        output_path: str = "performance_reports"
    ):
        self.monitoring_interval = monitoring_interval
        self.output_path = output_path
        self.is_running = False
        self.metrics_buffer: List[Dict[str, Any]] = []
        
        # Dependency injection following SOLID principles
        self.monitor: IPerformanceMonitor = SystemPerformanceMonitor()
        self.optimizer: IOptimizer = PerformanceOptimizer()
        self.alert_manager: IAlertManager = AlertManager()
        self.reporter: IReportGenerator = PerformanceReporter(output_path)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the performance agent"""
        self.logger.info("Starting Advanced Performance Agent")
        self._check_system_resources()
        
        self.is_running = True
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"Performance Agent started - monitoring every {self.monitoring_interval}s")
    
    async def stop(self) -> None:
        """Stop the performance agent"""
        self.logger.info("Stopping Advanced Performance Agent")
        self.is_running = False
        
        # Generate final report
        if self.metrics_buffer:
            await self.reporter.generate_report(self.metrics_buffer)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._perform_monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(10)  # Wait 10 seconds before retry
    
    async def _perform_monitoring_cycle(self) -> None:
        """Perform one complete monitoring cycle"""
        try:
            # Step 1: Collect metrics
            metrics = await self.monitor.collect_metrics()
            self.metrics_buffer.append(metrics)
            
            # Keep buffer size manageable (last 1440 entries = 24 hours at 1-minute intervals)
            if len(self.metrics_buffer) > 1440:
                self.metrics_buffer.pop(0)
            
            # Step 2: Analyze performance
            analysis = await self.optimizer.analyze_performance(metrics)
            
            # Step 3: Process alerts
            alerts = await self.alert_manager.process_alerts(analysis)
            
            # Log performance summary
            health_score = analysis.get("health_score", 0)
            grade = analysis.get("performance_grade", "F")
            
            self.logger.info(f"Performance check - Score: {health_score:.1f}, Grade: {grade}")
            
            if alerts:
                self.logger.warning(f"Generated {len(alerts)} performance alerts")
                for alert in alerts:
                    self.logger.warning(f"ALERT: {alert.description}")
            
            # Step 4: Generate periodic reports (every hour)
            if len(self.metrics_buffer) % 60 == 0:  # Every 60 cycles (1 hour)
                await self.reporter.generate_report(self.metrics_buffer)
                self.logger.info("Generated hourly performance report")
                
        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {e}")
            raise
    
    async def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current performance metrics"""
        if self.metrics_buffer:
            return self.metrics_buffer[-1]
        return None
    
    async def get_performance_analysis(self) -> Optional[Dict[str, Any]]:
        """Get current performance analysis"""
        if self.metrics_buffer:
            return await self.optimizer.analyze_performance(self.metrics_buffer[-1])
        return None
    
    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest performance report"""
        try:
            latest_path = Path(self.output_path) / "latest_performance_report.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load latest report: {e}")
            return None
    
    async def force_report_generation(self) -> Dict[str, Any]:
        """Force generation of performance report"""
        if self.metrics_buffer:
            return await self.reporter.generate_report(self.metrics_buffer)
        return {"error": "No metrics data available"}
    
    def _check_system_resources(self) -> None:
        """Check system resources before starting"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 95:
            raise PerformanceError(f"CPU usage too high to start monitoring: {cpu_percent}%")
        if memory_percent > 95:
            raise PerformanceError(f"Memory usage too high to start monitoring: {memory_percent}%")
        
        self.logger.info(f"System resources OK - CPU: {cpu_percent}%, Memory: {memory_percent}%")

# Factory pattern for agent creation
class PerformanceAgentFactory:
    """Factory for creating performance agents with different configurations"""
    
    @staticmethod
    def create_standard_agent() -> AdvancedPerformanceAgent:
        """Create standard performance agent"""
        return AdvancedPerformanceAgent(monitoring_interval=60)
    
    @staticmethod
    def create_high_frequency_agent() -> AdvancedPerformanceAgent:
        """Create high-frequency monitoring agent"""
        return AdvancedPerformanceAgent(monitoring_interval=10)  # 10 seconds
    
    @staticmethod
    def create_enterprise_agent() -> AdvancedPerformanceAgent:
        """Create enterprise-grade performance agent"""
        return AdvancedPerformanceAgent(
            monitoring_interval=30,  # 30 seconds
            output_path="enterprise_performance_reports"
        )

async def main():
    """Main function for running the performance agent"""
    try:
        # Create and start the agent
        agent = PerformanceAgentFactory.create_standard_agent()
        await agent.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\nShutting down Performance Agent...")
        await agent.stop()
    except Exception as e:
        print(f"Performance Agent error: {e}")

if __name__ == "__main__":
    asyncio.run(main())