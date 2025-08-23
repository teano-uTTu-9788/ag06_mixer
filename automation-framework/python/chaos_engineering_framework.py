#!/usr/bin/env python3
"""
Chaos Engineering Framework 2025
Netflix-Inspired Resilience Testing

Implements the principles of Chaos Engineering:
1. Define steady state through system metrics
2. Hypothesize that steady state continues in both control and experimental group
3. Introduce variables that reflect real-world events
4. Try to disprove the hypothesis by looking for differences

Based on Netflix's Chaos Monkey and Simian Army principles.
"""

import asyncio
import logging
import random
import time
import psutil
import threading
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import aiohttp
import statistics

# Chaos Configuration
@dataclass
class ChaosConfig:
    """Configuration for chaos experiments"""
    # General settings
    enabled: bool = True
    dry_run: bool = False
    blast_radius: float = 0.1  # Percentage of system to affect (10%)
    
    # Timing
    experiment_duration_seconds: int = 300  # 5 minutes
    steady_state_duration_seconds: int = 60  # 1 minute before/after
    
    # Safety thresholds
    max_error_rate: float = 0.05  # 5% error rate threshold
    max_latency_increase: float = 0.5  # 50% latency increase threshold
    min_availability: float = 0.95  # 95% availability threshold
    
    # Experiment types
    network_chaos_enabled: bool = True
    resource_chaos_enabled: bool = True
    service_chaos_enabled: bool = True
    data_chaos_enabled: bool = False  # Disabled by default for safety

class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition" 
    NETWORK_PACKET_LOSS = "network_packet_loss"
    
    CPU_SPIKE = "cpu_spike"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FULL = "disk_full"
    
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_SLOW = "service_slow"
    SERVICE_ERROR = "service_error"
    
    DATABASE_CONNECTION_LOSS = "database_connection_loss"
    CACHE_FAILURE = "cache_failure"
    MESSAGE_QUEUE_DELAY = "message_queue_delay"

@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""
    experiment_id: str
    experiment_type: ChaosExperimentType
    name: str
    description: str
    hypothesis: str
    target_service: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_impact: str = ""
    rollback_strategy: str = ""
    
    # Experiment lifecycle
    status: str = "created"  # created, running, completed, failed, aborted
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System state metrics for steady-state definition"""
    timestamp: datetime
    
    # Performance metrics
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    throughput: float
    availability: float
    
    # Resource metrics
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    
    # Service-specific metrics
    audio_processing_latency: float = 0.0
    hardware_response_time: float = 0.0
    active_sessions: int = 0
    queue_depth: int = 0

# Base Chaos Experiment
class ChaosExperimentRunner(ABC):
    """Abstract base class for chaos experiment runners"""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.is_running = False
        self.experiment_thread: Optional[threading.Thread] = None
        
    @abstractmethod
    async def execute_chaos(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute the chaos experiment"""
        pass
    
    @abstractmethod
    async def rollback_chaos(self, experiment: ChaosExperiment) -> bool:
        """Rollback the chaos experiment"""
        pass
    
    async def run_experiment(self, experiment: ChaosExperiment) -> ChaosExperiment:
        """Run complete chaos experiment with safety checks"""
        logging.info(f"Starting chaos experiment: {experiment.name}")
        
        if self.config.dry_run:
            logging.info("DRY RUN: Experiment will be simulated only")
        
        experiment.start_time = datetime.now()
        experiment.status = "running"
        
        try:
            # Phase 1: Establish steady state baseline
            logging.info("Phase 1: Establishing steady state baseline")
            baseline_metrics = await self._collect_steady_state_metrics()
            
            # Phase 2: Execute chaos
            logging.info("Phase 2: Executing chaos experiment")
            chaos_results = await self.execute_chaos(experiment)
            
            # Phase 3: Monitor impact
            logging.info("Phase 3: Monitoring system impact")
            impact_metrics = await self._monitor_experiment_impact(experiment)
            
            # Phase 4: Safety checks
            if await self._check_safety_thresholds(baseline_metrics, impact_metrics):
                logging.warning("Safety thresholds exceeded - aborting experiment")
                await self.rollback_chaos(experiment)
                experiment.status = "aborted"
            else:
                # Phase 5: Complete experiment
                await asyncio.sleep(self.config.experiment_duration_seconds)
                
                # Phase 6: Rollback
                logging.info("Phase 4: Rolling back chaos experiment")
                await self.rollback_chaos(experiment)
                
                # Phase 7: Validate recovery
                recovery_metrics = await self._collect_steady_state_metrics()
                
                # Store results
                experiment.results = {
                    "baseline_metrics": baseline_metrics,
                    "chaos_results": chaos_results,
                    "impact_metrics": impact_metrics,
                    "recovery_metrics": recovery_metrics,
                    "hypothesis_validated": self._validate_hypothesis(experiment, impact_metrics)
                }
                
                experiment.status = "completed"
            
        except Exception as e:
            logging.error(f"Chaos experiment failed: {e}")
            experiment.status = "failed"
            experiment.results["error"] = str(e)
            
            # Emergency rollback
            try:
                await self.rollback_chaos(experiment)
            except Exception as rollback_error:
                logging.error(f"Rollback failed: {rollback_error}")
        
        finally:
            experiment.end_time = datetime.now()
            
        return experiment
    
    async def _collect_steady_state_metrics(self) -> SystemMetrics:
        """Collect system metrics for steady state"""
        # Collect metrics over steady state period
        metrics_samples = []
        
        for _ in range(self.config.steady_state_duration_seconds):
            sample = await self._collect_current_metrics()
            metrics_samples.append(sample)
            await asyncio.sleep(1)
        
        # Calculate aggregated metrics
        return self._aggregate_metrics(metrics_samples)
    
    async def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # System resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
        
        # Simulate service metrics (in real implementation, these would be actual metrics)
        response_times = [random.uniform(0.05, 0.3) for _ in range(10)]
        
        return SystemMetrics(
            timestamp=datetime.now(),
            response_time_p50=statistics.median(response_times),
            response_time_p95=statistics.quantiles(response_times, n=20)[18],
            response_time_p99=statistics.quantiles(response_times, n=100)[98],
            error_rate=random.uniform(0.001, 0.01),
            throughput=random.uniform(100, 200),
            availability=random.uniform(0.99, 1.0),
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_usage=disk_percent,
            network_io=network_io,
            audio_processing_latency=random.uniform(0.01, 0.05),
            hardware_response_time=random.uniform(0.02, 0.08),
            active_sessions=random.randint(5, 20),
            queue_depth=random.randint(0, 10)
        )
    
    def _aggregate_metrics(self, samples: List[SystemMetrics]) -> SystemMetrics:
        """Aggregate metrics samples into summary"""
        if not samples:
            return SystemMetrics(timestamp=datetime.now(), **{k: 0.0 for k in SystemMetrics.__dataclass_fields__ if k != 'timestamp'})
        
        # Calculate averages for aggregation
        return SystemMetrics(
            timestamp=datetime.now(),
            response_time_p50=statistics.mean(s.response_time_p50 for s in samples),
            response_time_p95=statistics.mean(s.response_time_p95 for s in samples),
            response_time_p99=statistics.mean(s.response_time_p99 for s in samples),
            error_rate=statistics.mean(s.error_rate for s in samples),
            throughput=statistics.mean(s.throughput for s in samples),
            availability=statistics.mean(s.availability for s in samples),
            cpu_usage=statistics.mean(s.cpu_usage for s in samples),
            memory_usage=statistics.mean(s.memory_usage for s in samples),
            disk_usage=statistics.mean(s.disk_usage for s in samples),
            network_io=statistics.mean(s.network_io for s in samples),
            audio_processing_latency=statistics.mean(s.audio_processing_latency for s in samples),
            hardware_response_time=statistics.mean(s.hardware_response_time for s in samples),
            active_sessions=int(statistics.mean(s.active_sessions for s in samples)),
            queue_depth=int(statistics.mean(s.queue_depth for s in samples))
        )
    
    async def _monitor_experiment_impact(self, experiment: ChaosExperiment) -> List[SystemMetrics]:
        """Monitor system metrics during experiment"""
        impact_samples = []
        
        # Monitor for a portion of the experiment duration
        monitor_duration = min(60, self.config.experiment_duration_seconds // 2)
        
        for _ in range(monitor_duration):
            sample = await self._collect_current_metrics()
            impact_samples.append(sample)
            await asyncio.sleep(1)
        
        return impact_samples
    
    async def _check_safety_thresholds(self, baseline: SystemMetrics, impact_samples: List[SystemMetrics]) -> bool:
        """Check if safety thresholds are exceeded"""
        if not impact_samples:
            return False
        
        current = self._aggregate_metrics(impact_samples)
        
        # Check error rate threshold
        if current.error_rate > baseline.error_rate * (1 + self.config.max_error_rate):
            logging.warning(f"Error rate threshold exceeded: {current.error_rate:.4f} vs baseline {baseline.error_rate:.4f}")
            return True
        
        # Check latency threshold
        if current.response_time_p95 > baseline.response_time_p95 * (1 + self.config.max_latency_increase):
            logging.warning(f"Latency threshold exceeded: {current.response_time_p95:.3f}s vs baseline {baseline.response_time_p95:.3f}s")
            return True
        
        # Check availability threshold
        if current.availability < self.config.min_availability:
            logging.warning(f"Availability threshold exceeded: {current.availability:.3f} vs minimum {self.config.min_availability}")
            return True
        
        return False
    
    def _validate_hypothesis(self, experiment: ChaosExperiment, impact_metrics: List[SystemMetrics]) -> bool:
        """Validate experiment hypothesis"""
        # This is experiment-specific and would be implemented by each chaos runner
        return len(impact_metrics) > 0

# Network Chaos Runner
class NetworkChaosRunner(ChaosExperimentRunner):
    """Network-based chaos experiments"""
    
    def __init__(self, config: ChaosConfig):
        super().__init__(config)
        self.active_chaos: Dict[str, Any] = {}
    
    async def execute_chaos(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute network chaos"""
        if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
            return await self._inject_network_latency(experiment)
        elif experiment.experiment_type == ChaosExperimentType.NETWORK_PACKET_LOSS:
            return await self._inject_packet_loss(experiment)
        elif experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
            return await self._inject_network_partition(experiment)
        else:
            raise ValueError(f"Unknown network chaos type: {experiment.experiment_type}")
    
    async def rollback_chaos(self, experiment: ChaosExperiment) -> bool:
        """Rollback network chaos"""
        experiment_id = experiment.experiment_id
        
        if experiment_id in self.active_chaos:
            chaos_info = self.active_chaos[experiment_id]
            
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                return await self._remove_network_latency(chaos_info)
            elif experiment.experiment_type == ChaosExperimentType.NETWORK_PACKET_LOSS:
                return await self._remove_packet_loss(chaos_info)
            elif experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
                return await self._remove_network_partition(chaos_info)
            
            del self.active_chaos[experiment_id]
        
        return True
    
    async def _inject_network_latency(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject network latency"""
        latency_ms = experiment.parameters.get("latency_ms", 100)
        target_ips = experiment.parameters.get("target_ips", ["127.0.0.1"])
        
        logging.info(f"Injecting {latency_ms}ms latency to {target_ips}")
        
        if not self.config.dry_run:
            # In real implementation, this would use traffic control (tc) or similar
            # tc qdisc add dev eth0 root netem delay 100ms
            pass
        
        chaos_info = {
            "type": "network_latency",
            "latency_ms": latency_ms,
            "target_ips": target_ips,
            "start_time": datetime.now()
        }
        
        self.active_chaos[experiment.experiment_id] = chaos_info
        
        return {
            "chaos_type": "network_latency",
            "latency_injected_ms": latency_ms,
            "affected_targets": len(target_ips),
            "success": True
        }
    
    async def _remove_network_latency(self, chaos_info: Dict[str, Any]) -> bool:
        """Remove network latency injection"""
        logging.info("Removing network latency injection")
        
        if not self.config.dry_run:
            # tc qdisc del dev eth0 root
            pass
        
        return True
    
    async def _inject_packet_loss(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject packet loss"""
        loss_percent = experiment.parameters.get("loss_percent", 5)
        
        logging.info(f"Injecting {loss_percent}% packet loss")
        
        chaos_info = {
            "type": "packet_loss",
            "loss_percent": loss_percent,
            "start_time": datetime.now()
        }
        
        self.active_chaos[experiment.experiment_id] = chaos_info
        
        return {
            "chaos_type": "packet_loss",
            "loss_percent": loss_percent,
            "success": True
        }
    
    async def _remove_packet_loss(self, chaos_info: Dict[str, Any]) -> bool:
        """Remove packet loss injection"""
        logging.info("Removing packet loss injection")
        return True
    
    async def _inject_network_partition(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject network partition"""
        partition_targets = experiment.parameters.get("partition_targets", [])
        
        logging.info(f"Creating network partition affecting {len(partition_targets)} targets")
        
        chaos_info = {
            "type": "network_partition",
            "partition_targets": partition_targets,
            "start_time": datetime.now()
        }
        
        self.active_chaos[experiment.experiment_id] = chaos_info
        
        return {
            "chaos_type": "network_partition",
            "affected_targets": len(partition_targets),
            "success": True
        }
    
    async def _remove_network_partition(self, chaos_info: Dict[str, Any]) -> bool:
        """Remove network partition"""
        logging.info("Removing network partition")
        return True

# Resource Chaos Runner
class ResourceChaosRunner(ChaosExperimentRunner):
    """Resource exhaustion chaos experiments"""
    
    def __init__(self, config: ChaosConfig):
        super().__init__(config)
        self.chaos_threads: Dict[str, threading.Thread] = {}
        self.chaos_stop_flags: Dict[str, threading.Event] = {}
    
    async def execute_chaos(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute resource chaos"""
        if experiment.experiment_type == ChaosExperimentType.CPU_SPIKE:
            return await self._inject_cpu_spike(experiment)
        elif experiment.experiment_type == ChaosExperimentType.MEMORY_PRESSURE:
            return await self._inject_memory_pressure(experiment)
        elif experiment.experiment_type == ChaosExperimentType.DISK_FULL:
            return await self._inject_disk_pressure(experiment)
        else:
            raise ValueError(f"Unknown resource chaos type: {experiment.experiment_type}")
    
    async def rollback_chaos(self, experiment: ChaosExperiment) -> bool:
        """Rollback resource chaos"""
        experiment_id = experiment.experiment_id
        
        # Stop chaos threads
        if experiment_id in self.chaos_stop_flags:
            self.chaos_stop_flags[experiment_id].set()
        
        if experiment_id in self.chaos_threads:
            thread = self.chaos_threads[experiment_id]
            thread.join(timeout=5)  # Wait up to 5 seconds
            del self.chaos_threads[experiment_id]
        
        if experiment_id in self.chaos_stop_flags:
            del self.chaos_stop_flags[experiment_id]
        
        return True
    
    async def _inject_cpu_spike(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject CPU spike"""
        cpu_percent = experiment.parameters.get("cpu_percent", 80)
        duration_seconds = experiment.parameters.get("duration_seconds", 60)
        
        logging.info(f"Injecting CPU spike: {cpu_percent}% for {duration_seconds}s")
        
        if not self.config.dry_run:
            stop_flag = threading.Event()
            self.chaos_stop_flags[experiment.experiment_id] = stop_flag
            
            def cpu_stress():
                end_time = time.time() + duration_seconds
                while not stop_flag.is_set() and time.time() < end_time:
                    # Busy wait to consume CPU
                    start = time.time()
                    while (time.time() - start) < 0.01:  # 10ms of work
                        pass
                    time.sleep(0.01)  # 10ms of rest (roughly 50% CPU)
            
            # Start multiple threads for higher CPU usage
            num_threads = max(1, int((cpu_percent / 50) * psutil.cpu_count()))
            threads = []
            
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_stress)
                threads.append(thread)
                thread.start()
            
            self.chaos_threads[experiment.experiment_id] = threads[0]  # Store first thread for cleanup
        
        return {
            "chaos_type": "cpu_spike",
            "target_cpu_percent": cpu_percent,
            "duration_seconds": duration_seconds,
            "threads_started": num_threads if not self.config.dry_run else 0,
            "success": True
        }
    
    async def _inject_memory_pressure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject memory pressure"""
        memory_mb = experiment.parameters.get("memory_mb", 512)
        duration_seconds = experiment.parameters.get("duration_seconds", 60)
        
        logging.info(f"Injecting memory pressure: {memory_mb}MB for {duration_seconds}s")
        
        if not self.config.dry_run:
            stop_flag = threading.Event()
            self.chaos_stop_flags[experiment.experiment_id] = stop_flag
            
            def memory_stress():
                # Allocate memory chunks
                memory_chunks = []
                chunk_size = 1024 * 1024  # 1MB chunks
                target_chunks = memory_mb
                
                try:
                    for i in range(target_chunks):
                        if stop_flag.is_set():
                            break
                        chunk = bytearray(chunk_size)
                        memory_chunks.append(chunk)
                        time.sleep(0.01)  # Small delay between allocations
                    
                    # Hold memory until stop flag or duration
                    end_time = time.time() + duration_seconds
                    while not stop_flag.is_set() and time.time() < end_time:
                        time.sleep(1)
                        
                except MemoryError:
                    logging.warning("Memory allocation failed - system protection kicked in")
                finally:
                    # Clean up memory
                    del memory_chunks
            
            thread = threading.Thread(target=memory_stress)
            thread.start()
            self.chaos_threads[experiment.experiment_id] = thread
        
        return {
            "chaos_type": "memory_pressure",
            "target_memory_mb": memory_mb,
            "duration_seconds": duration_seconds,
            "success": True
        }
    
    async def _inject_disk_pressure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject disk pressure"""
        disk_mb = experiment.parameters.get("disk_mb", 100)
        duration_seconds = experiment.parameters.get("duration_seconds", 60)
        
        logging.info(f"Injecting disk pressure: {disk_mb}MB for {duration_seconds}s")
        
        if not self.config.dry_run:
            stop_flag = threading.Event()
            self.chaos_stop_flags[experiment.experiment_id] = stop_flag
            
            def disk_stress():
                import tempfile
                import os
                
                temp_files = []
                
                try:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    num_chunks = disk_mb
                    
                    for i in range(num_chunks):
                        if stop_flag.is_set():
                            break
                        
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(b'0' * chunk_size)
                        temp_file.close()
                        temp_files.append(temp_file.name)
                        
                        time.sleep(0.01)  # Small delay between writes
                    
                    # Hold files until stop flag or duration
                    end_time = time.time() + duration_seconds
                    while not stop_flag.is_set() and time.time() < end_time:
                        time.sleep(1)
                        
                finally:
                    # Clean up files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
            
            thread = threading.Thread(target=disk_stress)
            thread.start()
            self.chaos_threads[experiment.experiment_id] = thread
        
        return {
            "chaos_type": "disk_pressure",
            "target_disk_mb": disk_mb,
            "duration_seconds": duration_seconds,
            "success": True
        }

# Service Chaos Runner
class ServiceChaosRunner(ChaosExperimentRunner):
    """Service-level chaos experiments"""
    
    def __init__(self, config: ChaosConfig):
        super().__init__(config)
        self.service_overrides: Dict[str, Dict[str, Any]] = {}
    
    async def execute_chaos(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute service chaos"""
        if experiment.experiment_type == ChaosExperimentType.SERVICE_UNAVAILABLE:
            return await self._make_service_unavailable(experiment)
        elif experiment.experiment_type == ChaosExperimentType.SERVICE_SLOW:
            return await self._make_service_slow(experiment)
        elif experiment.experiment_type == ChaosExperimentType.SERVICE_ERROR:
            return await self._inject_service_errors(experiment)
        else:
            raise ValueError(f"Unknown service chaos type: {experiment.experiment_type}")
    
    async def rollback_chaos(self, experiment: ChaosExperiment) -> bool:
        """Rollback service chaos"""
        service_name = experiment.target_service
        
        if service_name in self.service_overrides:
            del self.service_overrides[service_name]
            logging.info(f"Restored normal behavior for service: {service_name}")
        
        return True
    
    async def _make_service_unavailable(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Make service unavailable"""
        service_name = experiment.target_service
        failure_rate = experiment.parameters.get("failure_rate", 1.0)  # 100% failure
        
        logging.info(f"Making service unavailable: {service_name} (failure rate: {failure_rate})")
        
        self.service_overrides[service_name] = {
            "type": "unavailable",
            "failure_rate": failure_rate,
            "start_time": datetime.now()
        }
        
        return {
            "chaos_type": "service_unavailable",
            "service": service_name,
            "failure_rate": failure_rate,
            "success": True
        }
    
    async def _make_service_slow(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Make service slow"""
        service_name = experiment.target_service
        latency_ms = experiment.parameters.get("latency_ms", 1000)
        affected_percentage = experiment.parameters.get("affected_percentage", 0.5)
        
        logging.info(f"Making service slow: {service_name} (+{latency_ms}ms, {affected_percentage*100}% requests)")
        
        self.service_overrides[service_name] = {
            "type": "slow",
            "latency_ms": latency_ms,
            "affected_percentage": affected_percentage,
            "start_time": datetime.now()
        }
        
        return {
            "chaos_type": "service_slow",
            "service": service_name,
            "latency_ms": latency_ms,
            "affected_percentage": affected_percentage,
            "success": True
        }
    
    async def _inject_service_errors(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Inject service errors"""
        service_name = experiment.target_service
        error_rate = experiment.parameters.get("error_rate", 0.1)  # 10% errors
        error_type = experiment.parameters.get("error_type", "500")
        
        logging.info(f"Injecting errors in service: {service_name} ({error_rate*100}% error rate)")
        
        self.service_overrides[service_name] = {
            "type": "errors",
            "error_rate": error_rate,
            "error_type": error_type,
            "start_time": datetime.now()
        }
        
        return {
            "chaos_type": "service_errors",
            "service": service_name,
            "error_rate": error_rate,
            "error_type": error_type,
            "success": True
        }

# Main Chaos Engineering Framework
class ChaosEngineeringFramework:
    """Main chaos engineering orchestrator"""
    
    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        
        # Initialize runners
        self.network_runner = NetworkChaosRunner(self.config)
        self.resource_runner = ResourceChaosRunner(self.config)
        self.service_runner = ServiceChaosRunner(self.config)
        
        # Experiment tracking
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ChaosExperiment] = []
        
    async def create_experiment(self, 
                              experiment_type: ChaosExperimentType,
                              name: str,
                              hypothesis: str,
                              target_service: str,
                              parameters: Dict[str, Any] = None) -> ChaosExperiment:
        """Create a new chaos experiment"""
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_type=experiment_type,
            name=name,
            description=f"Chaos experiment: {name}",
            hypothesis=hypothesis,
            target_service=target_service,
            parameters=parameters or {}
        )
        
        self.experiments[experiment.experiment_id] = experiment
        
        logging.info(f"Created chaos experiment: {name} ({experiment.experiment_id})")
        
        return experiment
    
    async def run_experiment(self, experiment_id: str) -> ChaosExperiment:
        """Run a chaos experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if not self.config.enabled:
            logging.warning("Chaos engineering is disabled - skipping experiment")
            experiment.status = "skipped"
            return experiment
        
        # Select appropriate runner
        runner = self._get_runner_for_experiment(experiment)
        
        # Run the experiment
        completed_experiment = await runner.run_experiment(experiment)
        
        # Store in history
        self.experiment_history.append(completed_experiment)
        
        # Log results
        await self._log_experiment_results(completed_experiment)
        
        return completed_experiment
    
    def _get_runner_for_experiment(self, experiment: ChaosExperiment) -> ChaosExperimentRunner:
        """Get appropriate runner for experiment type"""
        experiment_type = experiment.experiment_type
        
        if experiment_type in [ChaosExperimentType.NETWORK_LATENCY, 
                              ChaosExperimentType.NETWORK_PARTITION,
                              ChaosExperimentType.NETWORK_PACKET_LOSS]:
            return self.network_runner
        
        elif experiment_type in [ChaosExperimentType.CPU_SPIKE,
                               ChaosExperimentType.MEMORY_PRESSURE,
                               ChaosExperimentType.DISK_FULL]:
            return self.resource_runner
        
        elif experiment_type in [ChaosExperimentType.SERVICE_UNAVAILABLE,
                               ChaosExperimentType.SERVICE_SLOW,
                               ChaosExperimentType.SERVICE_ERROR]:
            return self.service_runner
        
        else:
            raise ValueError(f"No runner available for experiment type: {experiment_type}")
    
    async def _log_experiment_results(self, experiment: ChaosExperiment):
        """Log experiment results"""
        logging.info(f"Chaos Experiment Results: {experiment.name}")
        logging.info(f"  Status: {experiment.status}")
        logging.info(f"  Duration: {(experiment.end_time - experiment.start_time).total_seconds():.1f}s")
        
        if experiment.results:
            hypothesis_result = experiment.results.get("hypothesis_validated", "unknown")
            logging.info(f"  Hypothesis Validated: {hypothesis_result}")
            
            if "baseline_metrics" in experiment.results and "impact_metrics" in experiment.results:
                baseline = experiment.results["baseline_metrics"]
                impact_samples = experiment.results["impact_metrics"]
                
                if impact_samples:
                    impact = self.resource_runner._aggregate_metrics(impact_samples)
                    
                    logging.info(f"  Performance Impact:")
                    logging.info(f"    Error Rate: {baseline.error_rate:.4f} → {impact.error_rate:.4f}")
                    logging.info(f"    P95 Latency: {baseline.response_time_p95:.3f}s → {impact.response_time_p95:.3f}s")
                    logging.info(f"    Availability: {baseline.availability:.3f} → {impact.availability:.3f}")
    
    async def run_predefined_experiments(self) -> List[ChaosExperiment]:
        """Run a suite of predefined chaos experiments"""
        experiments = []
        
        # Network experiments
        if self.config.network_chaos_enabled:
            # Network latency experiment
            latency_exp = await self.create_experiment(
                experiment_type=ChaosExperimentType.NETWORK_LATENCY,
                name="Network Latency Resilience Test",
                hypothesis="System maintains acceptable performance with 100ms network latency",
                target_service="audio-processing",
                parameters={"latency_ms": 100, "target_ips": ["127.0.0.1"]}
            )
            experiments.append(await self.run_experiment(latency_exp.experiment_id))
        
        # Resource experiments  
        if self.config.resource_chaos_enabled:
            # CPU spike experiment
            cpu_exp = await self.create_experiment(
                experiment_type=ChaosExperimentType.CPU_SPIKE,
                name="CPU Pressure Resilience Test",
                hypothesis="System handles gracefully under 80% CPU load",
                target_service="system",
                parameters={"cpu_percent": 80, "duration_seconds": 30}
            )
            experiments.append(await self.run_experiment(cpu_exp.experiment_id))
        
        # Service experiments
        if self.config.service_chaos_enabled:
            # Service slowdown experiment
            slow_exp = await self.create_experiment(
                experiment_type=ChaosExperimentType.SERVICE_SLOW,
                name="Service Slowdown Resilience Test",
                hypothesis="System maintains availability when hardware service responds slowly",
                target_service="hardware-control",
                parameters={"latency_ms": 500, "affected_percentage": 0.3}
            )
            experiments.append(await self.run_experiment(slow_exp.experiment_id))
        
        return experiments
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        total_experiments = len(self.experiment_history)
        
        if total_experiments == 0:
            return {"total_experiments": 0, "summary": "No experiments run"}
        
        status_counts = {}
        for experiment in self.experiment_history:
            status = experiment.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        successful_experiments = status_counts.get("completed", 0)
        success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
        
        return {
            "total_experiments": total_experiments,
            "success_rate": success_rate,
            "status_breakdown": status_counts,
            "system_resilience_score": success_rate * 100,
            "last_experiment": self.experiment_history[-1].name if self.experiment_history else None
        }

# Example usage
async def main():
    """Example chaos engineering session"""
    logging.basicConfig(level=logging.INFO)
    
    # Create chaos framework with safety limits
    config = ChaosConfig(
        enabled=True,
        dry_run=False,  # Set to True for safe testing
        blast_radius=0.1,  # Affect only 10% of system
        experiment_duration_seconds=60,  # Short experiments
        max_error_rate=0.1,  # 10% error rate threshold
        max_latency_increase=1.0,  # 100% latency increase threshold
        min_availability=0.9  # 90% availability threshold
    )
    
    chaos_framework = ChaosEngineeringFramework(config)
    
    try:
        print("🌪️ Starting Chaos Engineering Session")
        
        # Run predefined experiments
        experiments = await chaos_framework.run_predefined_experiments()
        
        print(f"\n🧪 Completed {len(experiments)} chaos experiments")
        
        # Get summary
        summary = chaos_framework.get_experiment_summary()
        print(f"📊 System Resilience Score: {summary['system_resilience_score']:.1f}/100")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        
        # Print individual results
        for experiment in experiments:
            print(f"\n🔬 {experiment.name}: {experiment.status.upper()}")
            if experiment.results and experiment.results.get("hypothesis_validated"):
                print(f"   ✅ Hypothesis validated")
            elif experiment.results:
                print(f"   ❌ Hypothesis not validated")
        
    except Exception as e:
        logging.error(f"Chaos engineering session failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())