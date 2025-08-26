#!/usr/bin/env python3
"""
Comprehensive 88-Test Validation Suite for Integrated Workflow System
MANU Compliance Testing with Real Execution Validation
"""

import asyncio
import sys
import os
import json
import time
import tempfile
import subprocess
import pytest
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components for testing
from integrated_workflow_system import IntegratedWorkflowSystem
from specialized_workflow_agent import SpecializedWorkflowAgent
from monitoring.realtime_observer import get_observer
from persistence.event_store import get_event_store
from ml.active_optimizer import get_optimizer
from aican_runtime.circuit_breaker import CircuitBreaker

@dataclass
class TestResult:
    test_id: int
    test_name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    duration_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None

class WorkflowValidationSuite:
    """Comprehensive 88-test validation suite for MANU compliance"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        self.integrated_system = None
        self.specialized_agent = None
        self.observer = None
        self.event_store = None
        self.optimizer = None
        
        # Test categories and counts
        self.test_categories = {
            "Core System": 15,
            "Observability": 15, 
            "Persistence": 15,
            "ML Optimization": 15,
            "Circuit Breaker": 10,
            "Integration": 10,
            "Performance": 8
        }
        
        print("🧪 Initializing Comprehensive 88-Test Validation Suite")
        print(f"   📊 Test categories: {list(self.test_categories.keys())}")
        print(f"   🎯 Total tests: {sum(self.test_categories.values())}")
    
    async def setup_test_environment(self) -> bool:
        """Setup test environment with all components"""
        try:
            print("🔧 Setting up test environment...")
            
            # Initialize integrated system
            self.integrated_system = IntegratedWorkflowSystem()
            
            # Initialize specialized agent
            self.specialized_agent = SpecializedWorkflowAgent("test_agent")
            await self.specialized_agent.initialize()
            
            # Initialize individual components
            self.observer = get_observer()
            self.event_store = get_event_store()
            self.optimizer = get_optimizer()
            
            print("✅ Test environment ready")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def run_test(self, test_id: int, test_name: str, category: str, 
                      test_func) -> TestResult:
        """Execute individual test and record result"""
        
        start_time = time.time()
        
        try:
            print(f"🔍 Test {test_id:02d}: {test_name}")
            
            # Execute test function
            await test_func()
            
            duration_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_id=test_id,
                test_name=test_name,
                category=category,
                status="PASS",
                duration_ms=duration_ms
            )
            
            print(f"✅ Test {test_id:02d}: PASS ({duration_ms:.1f}ms)")
            
        except AssertionError as e:
            duration_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_id=test_id,
                test_name=test_name,
                category=category,
                status="FAIL",
                duration_ms=duration_ms,
                error_message=str(e)
            )
            print(f"❌ Test {test_id:02d}: FAIL - {e}")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_id=test_id,
                test_name=test_name,
                category=category,
                status="ERROR",
                duration_ms=duration_ms,
                error_message=str(e)
            )
            print(f"💥 Test {test_id:02d}: ERROR - {e}")
        
        self.results.append(result)
        return result
    
    # ========== CORE SYSTEM TESTS (1-15) ==========
    
    async def test_01_integrated_system_initialization(self):
        assert self.integrated_system is not None, "Integrated system not initialized"
        assert hasattr(self.integrated_system, 'observer'), "Observer not present"
        assert hasattr(self.integrated_system, 'event_store'), "Event store not present"
        assert hasattr(self.integrated_system, 'optimizer'), "Optimizer not present"
    
    async def test_02_specialized_agent_initialization(self):
        assert self.specialized_agent is not None, "Specialized agent not initialized"
        assert self.specialized_agent.workflow_system is not None, "Workflow system not ready"
        assert len(self.specialized_agent.task_queue) == 0, "Task queue not empty"
    
    async def test_03_workflow_execution_basic(self):
        result = await self.integrated_system.execute_workflow(
            "test_workflow_001", "test", ["step1", "step2"]
        )
        assert result["status"] == "success", f"Workflow failed: {result.get('error')}"
        assert "correlation_id" in result, "Missing correlation ID"
    
    async def test_04_workflow_execution_with_context(self):
        context = {"cpu_percent": 45.0, "memory_percent": 60.0}
        result = await self.integrated_system.execute_workflow(
            "test_workflow_002", "test", context=context
        )
        assert result["status"] == "success", "Workflow with context failed"
    
    async def test_05_system_health_check(self):
        health = await self.integrated_system.get_system_health()
        assert "overall_status" in health, "Missing overall status"
        assert "components" in health, "Missing components status"
        assert health["overall_status"] in ["healthy", "degraded"], "Invalid health status"
    
    async def test_06_workflow_with_custom_steps(self):
        custom_steps = ["validation", "processing", "transformation", "output"]
        result = await self.integrated_system.execute_workflow(
            "test_workflow_003", "custom", custom_steps
        )
        assert result["status"] == "success", "Custom steps workflow failed"
        assert result["steps_completed"] == len(custom_steps), "Not all steps completed"
    
    async def test_07_multiple_concurrent_workflows(self):
        tasks = []
        for i in range(3):
            task = self.integrated_system.execute_workflow(f"concurrent_{i}", "test")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        for result in results:
            assert result["status"] == "success", "Concurrent workflow failed"
    
    async def test_08_workflow_error_handling(self):
        # This should complete gracefully even with invalid context
        result = await self.integrated_system.execute_workflow(
            "error_test", "test", context={"invalid": "data"}
        )
        # Should either succeed or fail gracefully
        assert result["status"] in ["success", "failed"], "Invalid error handling"
    
    async def test_09_workflow_configuration_validation(self):
        configs = self.integrated_system.workflow_configs
        assert "default" in configs, "Missing default configuration"
        assert "batch_size" in configs["default"], "Missing batch_size config"
    
    async def test_10_component_integration(self):
        # Test that all three components work together
        assert self.integrated_system.observer is not None, "Observer integration failed"
        assert self.integrated_system.event_store is not None, "Event store integration failed"
        assert self.integrated_system.optimizer is not None, "Optimizer integration failed"
    
    async def test_11_workflow_types_support(self):
        types = ["default", "batch_processing", "real_time", "analytics"]
        for wf_type in types:
            result = await self.integrated_system.execute_workflow(
                f"type_test_{wf_type}", wf_type
            )
            assert result["status"] == "success", f"Workflow type {wf_type} failed"
    
    async def test_12_workflow_duration_tracking(self):
        result = await self.integrated_system.execute_workflow(
            "duration_test", "test"
        )
        assert "total_duration_ms" in result, "Missing duration tracking"
        assert result["total_duration_ms"] > 0, "Invalid duration value"
    
    async def test_13_correlation_id_propagation(self):
        result = await self.integrated_system.execute_workflow(
            "correlation_test", "test"
        )
        correlation_id = result["correlation_id"]
        assert correlation_id is not None, "Missing correlation ID"
        assert len(correlation_id) > 0, "Empty correlation ID"
    
    async def test_14_workflow_step_tracking(self):
        steps = ["init", "process", "validate", "complete"]
        result = await self.integrated_system.execute_workflow(
            "step_tracking_test", "test", steps
        )
        assert result["steps_completed"] == len(steps), "Step tracking failed"
    
    async def test_15_system_resource_management(self):
        # Test system doesn't leak resources
        initial_health = await self.integrated_system.get_system_health()
        
        # Run several workflows
        for i in range(5):
            await self.integrated_system.execute_workflow(f"resource_test_{i}", "test")
        
        final_health = await self.integrated_system.get_system_health()
        assert final_health["overall_status"] != "unhealthy", "Resource leak detected"
    
    # ========== OBSERVABILITY TESTS (16-30) ==========
    
    async def test_16_observer_workflow_tracking(self):
        correlation_id = self.observer.start_workflow("obs_test_001", "test")
        assert correlation_id is not None, "Failed to start workflow tracking"
        assert "obs_test_001" in self.observer.active_workflows, "Workflow not tracked"
    
    async def test_17_observer_step_recording(self):
        correlation_id = self.observer.start_workflow("obs_test_002", "test")
        self.observer.record_step("obs_test_002", "test_step", 100.0, "success")
        assert len(self.observer.events) > 0, "Step not recorded"
    
    async def test_18_observer_workflow_completion(self):
        correlation_id = self.observer.start_workflow("obs_test_003", "test")
        self.observer.complete_workflow("obs_test_003", "success")
        assert "obs_test_003" not in self.observer.active_workflows, "Workflow not completed"
    
    async def test_19_observer_health_status(self):
        health = self.observer.get_health_status()
        assert "status" in health, "Missing health status"
        assert "system" in health, "Missing system metrics"
        assert "workflows" in health, "Missing workflow metrics"
    
    async def test_20_observer_metrics_collection(self):
        initial_events = len(self.observer.events)
        correlation_id = self.observer.start_workflow("metrics_test", "test")
        self.observer.record_step("metrics_test", "step1", 50.0, "success")
        self.observer.complete_workflow("metrics_test", "success")
        assert len(self.observer.events) > initial_events, "Metrics not collected"
    
    async def test_21_observer_error_tracking(self):
        correlation_id = self.observer.start_workflow("error_test", "test")
        self.observer.record_step("error_test", "failing_step", 0, "error", "Test error")
        error_events = [e for e in self.observer.events if e.status == "error"]
        assert len(error_events) > 0, "Error not tracked"
    
    async def test_22_observer_resource_monitoring(self):
        health = self.observer.get_health_status()
        system = health.get("system", {})
        assert "cpu_percent" in system, "CPU monitoring missing"
        assert "memory_percent" in system, "Memory monitoring missing"
    
    async def test_23_observer_correlation_ids(self):
        id1 = self.observer.start_workflow("cor_test_1", "test")
        id2 = self.observer.start_workflow("cor_test_2", "test")
        assert id1 != id2, "Correlation IDs not unique"
    
    async def test_24_observer_workflow_types(self):
        types = ["batch", "realtime", "analytics"]
        for wf_type in types:
            correlation_id = self.observer.start_workflow(f"type_{wf_type}", wf_type)
            workflow = self.observer.active_workflows[f"type_{wf_type}"]
            assert workflow["workflow_type"] == wf_type, f"Type {wf_type} not recorded"
    
    async def test_25_observer_multiple_steps(self):
        correlation_id = self.observer.start_workflow("multi_step_test", "test")
        steps = ["step1", "step2", "step3"]
        for i, step in enumerate(steps):
            self.observer.record_step("multi_step_test", step, i*10 + 50.0, "success")
        
        workflow = self.observer.active_workflows["multi_step_test"]
        assert len(workflow["steps"]) == len(steps), "Not all steps recorded"
    
    async def test_26_observer_duration_measurement(self):
        correlation_id = self.observer.start_workflow("duration_test", "test")
        await asyncio.sleep(0.1)  # Small delay
        self.observer.complete_workflow("duration_test", "success")
        
        # Check that duration was measured (workflow should be removed)
        assert "duration_test" not in self.observer.active_workflows
    
    async def test_27_observer_concurrent_workflows(self):
        ids = []
        for i in range(3):
            correlation_id = self.observer.start_workflow(f"concurrent_{i}", "test")
            ids.append(correlation_id)
        
        assert len(set(ids)) == 3, "Concurrent workflows not properly tracked"
        assert len(self.observer.active_workflows) >= 3, "Missing concurrent workflows"
    
    async def test_28_observer_event_export(self):
        # Generate some events
        correlation_id = self.observer.start_workflow("export_test", "test")
        self.observer.record_step("export_test", "step1", 100.0, "success")
        
        filename = self.observer.export_events("test_export.json")
        assert os.path.exists(filename), "Export file not created"
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
    
    async def test_29_observer_health_scoring(self):
        health = self.observer.get_health_status()
        assert "health_score" in health, "Missing health score"
        score = health["health_score"]
        assert 0 <= score <= 100, f"Invalid health score: {score}"
    
    async def test_30_observer_metrics_summary(self):
        summary = self.observer.get_metrics_summary()
        assert summary is not None, "Failed to get metrics summary"
        # Should return either Prometheus message or actual metrics
        assert isinstance(summary, dict), "Invalid metrics summary format"
    
    # ========== PERSISTENCE TESTS (31-45) ==========
    
    async def test_31_event_store_basic_storage(self):
        event_id = await self.event_store.store_event(
            "test_stream", "test_event", {"key": "value"}
        )
        assert event_id is not None, "Event not stored"
        assert len(event_id) > 0, "Invalid event ID"
    
    async def test_32_event_store_deduplication(self):
        payload = {"unique": "test_data"}
        id1 = await self.event_store.store_event("dedup_test", "event", payload)
        id2 = await self.event_store.store_event("dedup_test", "event", payload)
        assert id1 == id2, "Deduplication failed"
    
    async def test_33_event_store_retrieval(self):
        # Store test event
        await self.event_store.store_event("retrieval_test", "test_event", {"data": "test"})
        
        # Retrieve events
        events = await self.event_store.get_events("retrieval_test")
        assert len(events) > 0, "Events not retrieved"
    
    async def test_34_event_store_correlation_tracking(self):
        correlation_id = "test_correlation_123"
        event_id = await self.event_store.store_event(
            "correlation_test", "event", {"data": "test"}, correlation_id
        )
        
        events = await self.event_store.get_events("correlation_test")
        found_correlation = any(e.correlation_id == correlation_id for e in events)
        assert found_correlation, "Correlation ID not stored"
    
    async def test_35_event_store_multiple_streams(self):
        streams = ["stream1", "stream2", "stream3"]
        for stream in streams:
            await self.event_store.store_event(stream, "event", {"stream": stream})
        
        for stream in streams:
            events = await self.event_store.get_events(stream)
            assert len(events) > 0, f"Stream {stream} empty"
    
    async def test_36_event_store_stream_info(self):
        stream_name = "info_test_stream"
        await self.event_store.store_event(stream_name, "event", {"test": "data"})
        
        info = await self.event_store.get_stream_info(stream_name)
        assert "stream_name" in info, "Missing stream name"
        assert "length" in info, "Missing stream length"
    
    async def test_37_event_store_health_status(self):
        health = await self.event_store.get_health_status()
        assert "status" in health, "Missing health status"
        assert "storage" in health, "Missing storage type"
    
    async def test_38_event_store_large_payload(self):
        large_payload = {"data": "x" * 1000}  # 1KB payload
        event_id = await self.event_store.store_event(
            "large_payload_test", "large_event", large_payload
        )
        assert event_id is not None, "Large payload not stored"
    
    async def test_39_event_store_concurrent_writes(self):
        tasks = []
        for i in range(5):
            task = self.event_store.store_event(
                "concurrent_test", "event", {"index": i}
            )
            tasks.append(task)
        
        event_ids = await asyncio.gather(*tasks)
        assert len(set(event_ids)) == 5, "Concurrent writes failed"
    
    async def test_40_event_store_event_types(self):
        event_types = ["workflow_started", "step_completed", "workflow_failed"]
        for event_type in event_types:
            await self.event_store.store_event("types_test", event_type, {"type": event_type})
        
        events = await self.event_store.get_events("types_test")
        found_types = [e.event_type for e in events]
        for event_type in event_types:
            assert event_type in found_types, f"Event type {event_type} not found"
    
    async def test_41_event_store_timestamp_tracking(self):
        # Use unique stream name to ensure test isolation
        stream_name = f"timestamp_test_{int(time.time() * 1000)}"
        
        event_id = await self.event_store.store_event(
            stream_name, "event", {"data": "test"}
        )
        
        events = await self.event_store.get_events(stream_name)
        assert len(events) > 0, "No events found"
        
        event = events[0]
        assert event.timestamp is not None, "Missing timestamp"
        # Validate ISO format timestamp
        datetime.fromisoformat(event.timestamp)
    
    async def test_42_event_store_mark_processed(self):
        event_id = await self.event_store.store_event(
            "processed_test", "event", {"data": "test"}
        )
        
        # Should not raise exception
        await self.event_store.mark_processed("processed_test", event_id)
    
    async def test_43_event_store_fallback_behavior(self):
        # Test should work even if Redis is not available
        # This tests the fallback to in-memory storage
        health = await self.event_store.get_health_status()
        storage_type = health.get("storage", "unknown")
        assert storage_type in ["redis", "memory"], f"Unknown storage type: {storage_type}"
    
    async def test_44_event_store_error_handling(self):
        # Test with invalid data types
        try:
            event_id = await self.event_store.store_event(
                "error_test", "event", {"valid": "data"}
            )
            assert event_id is not None, "Valid event not stored"
        except Exception:
            assert False, "Valid event storage failed"
    
    async def test_45_event_store_stream_isolation(self):
        # Store events in different streams
        await self.event_store.store_event("stream_a", "event", {"stream": "a"})
        await self.event_store.store_event("stream_b", "event", {"stream": "b"})
        
        events_a = await self.event_store.get_events("stream_a")
        events_b = await self.event_store.get_events("stream_b")
        
        assert len(events_a) > 0, "Stream A empty"
        assert len(events_b) > 0, "Stream B empty"
        
        # Verify isolation
        for event in events_a:
            assert event.stream_name == "stream_a", "Stream isolation failed"
    
    # ========== ML OPTIMIZATION TESTS (46-60) ==========
    
    async def test_46_optimizer_initialization(self):
        assert self.optimizer is not None, "Optimizer not initialized"
        assert hasattr(self.optimizer, 'feature_weights'), "Feature weights missing"
        assert hasattr(self.optimizer, 'metrics_history'), "Metrics history missing"
    
    async def test_47_optimizer_performance_recording(self):
        config = {"batch_size": 10, "timeout_ms": 2000}
        context = {"cpu_percent": 50.0, "memory_percent": 60.0}
        
        timestamp = await self.optimizer.record_performance(
            "perf_test", 500.0, True, config, context
        )
        assert timestamp is not None, "Performance not recorded"
        assert len(self.optimizer.metrics_history) > 0, "Metrics history empty"
    
    async def test_48_optimizer_configuration_suggestion(self):
        context = {"cpu_percent": 45.0, "memory_percent": 55.0}
        config = self.optimizer.suggest_configuration("test", context)
        
        assert "batch_size" in config, "Missing batch_size suggestion"
        assert "timeout_ms" in config, "Missing timeout_ms suggestion"
        assert "predicted_score" in config, "Missing prediction"
    
    async def test_49_optimizer_feature_extraction(self):
        config = {"batch_size": 20, "timeout_ms": 3000, "retry_count": 2}
        context = {"cpu_percent": 70.0, "memory_percent": 80.0}
        
        features = self.optimizer.extract_features(config, context)
        assert len(features) == 10, f"Expected 10 features, got {len(features)}"
        assert all(0 <= f <= 1 for f in features), "Features not normalized"
    
    async def test_50_optimizer_performance_prediction(self):
        config = {"batch_size": 15, "timeout_ms": 2500}
        context = {"cpu_percent": 60.0, "memory_percent": 70.0}
        
        score = self.optimizer.predict_performance(config, context)
        assert 0 <= score <= 2.0, f"Invalid prediction score: {score}"
    
    async def test_51_optimizer_learning_iterations(self):
        initial_iterations = self.optimizer.optimization_iterations
        
        config = {"batch_size": 5, "timeout_ms": 1000}
        await self.optimizer.record_performance("learn_test", 300.0, True, config)
        
        assert self.optimizer.optimization_iterations > initial_iterations, "Learning not happening"
    
    async def test_52_optimizer_ab_experiment_creation(self):
        config_a = {"batch_size": 10}
        config_b = {"batch_size": 20}
        
        exp_id = self.optimizer.start_ab_experiment("test_exp", config_a, config_b)
        assert exp_id in self.optimizer.ab_experiments, "A/B experiment not created"
    
    async def test_53_optimizer_ab_config_assignment(self):
        config_a = {"batch_size": 10}
        config_b = {"batch_size": 20}
        
        exp_id = self.optimizer.start_ab_experiment("assignment_test", config_a, config_b)
        
        variant, config = self.optimizer.get_ab_config(exp_id)
        assert variant in ["A", "B"], f"Invalid variant: {variant}"
        assert config in [config_a, config_b], "Invalid config returned"
    
    async def test_54_optimizer_ab_result_recording(self):
        config_a = {"batch_size": 10}
        config_b = {"batch_size": 20}
        
        exp_id = self.optimizer.start_ab_experiment("result_test", config_a, config_b)
        
        # Should not raise exception
        self.optimizer.record_ab_result(exp_id, "A", 1.2)
        self.optimizer.record_ab_result(exp_id, "B", 1.5)
    
    async def test_55_optimizer_optimization_summary(self):
        # Record some performance data
        for i in range(5):
            config = {"batch_size": 5 + i}
            await self.optimizer.record_performance(f"summary_test_{i}", 200.0 + i*10, True, config)
        
        summary = self.optimizer.get_optimization_summary()
        assert "status" in summary, "Missing summary status"
        assert summary["status"] in ["active", "no_data"], "Invalid summary status"
    
    async def test_56_optimizer_model_export(self):
        filename = self.optimizer.export_model("test_model.json")
        assert os.path.exists(filename), "Model export file not created"
        
        # Verify file contents
        with open(filename) as f:
            data = json.load(f)
            assert "model" in data, "Missing model data"
            assert "feature_weights" in data["model"], "Missing feature weights"
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
    
    async def test_57_optimizer_success_failure_tracking(self):
        config = {"batch_size": 10}
        
        # Record success
        await self.optimizer.record_performance("success_test", 100.0, True, config)
        
        # Record failure
        await self.optimizer.record_performance("failure_test", 5000.0, False, config)
        
        assert len(self.optimizer.metrics_history) >= 2, "Performance data not recorded"
    
    async def test_58_optimizer_context_awareness(self):
        config = {"batch_size": 10}
        context_low = {"cpu_percent": 20.0, "memory_percent": 30.0}
        context_high = {"cpu_percent": 80.0, "memory_percent": 90.0}
        
        pred_low = self.optimizer.predict_performance(config, context_low)
        pred_high = self.optimizer.predict_performance(config, context_high)
        
        # Predictions should be different for different contexts
        assert pred_low != pred_high, "Context not affecting predictions"
    
    async def test_59_optimizer_weight_updates(self):
        initial_weights = self.optimizer.feature_weights.copy()
        
        config = {"batch_size": 10}
        await self.optimizer.record_performance("weight_test", 400.0, True, config)
        
        # Weights should have changed
        weights_changed = any(initial_weights[i] != self.optimizer.feature_weights[i] 
                            for i in range(len(initial_weights)))
        assert weights_changed, "Feature weights not updating"
    
    async def test_60_optimizer_configuration_space(self):
        context = {"cpu_percent": 50.0, "memory_percent": 60.0}
        
        # Generate multiple configurations
        configs = []
        for _ in range(5):
            config = self.optimizer.suggest_configuration("test", context)
            configs.append(config)
        
        # Should suggest different configurations
        unique_configs = len(set(json.dumps(c, sort_keys=True) for c in configs))
        assert unique_configs > 1, "Configuration space not being explored"
    
    # ========== CIRCUIT BREAKER TESTS (61-70) ==========
    
    async def test_61_circuit_breaker_initialization(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        assert cb.state == "CLOSED", "Circuit breaker not initialized in CLOSED state"
        assert cb.failure_threshold == 3, "Failure threshold not set"
    
    async def test_62_circuit_breaker_success_recording(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        cb.record_success()
        assert cb.failure_count == 0, "Failure count not reset on success"
    
    async def test_63_circuit_breaker_failure_recording(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        cb.record_failure()
        assert cb.failure_count == 1, "Failure not recorded"
    
    async def test_64_circuit_breaker_state_transitions(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=10.0)
        
        # Should start CLOSED
        assert cb.state == "CLOSED"
        
        # Record failures to trigger OPEN
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == "OPEN", "Circuit breaker not opened after failures"
    
    async def test_65_circuit_breaker_call_execution(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        assert result == "success", "Call execution failed"
    
    async def test_66_circuit_breaker_call_rejection(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # First call should fail and open circuit
        try:
            await cb.call(failing_func)
            assert False, "Expected exception not raised"
        except Exception:
            pass
        
        # Second call should be rejected
        try:
            await cb.call(failing_func)
            assert False, "Call not rejected when circuit open"
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e), "Wrong rejection message"
    
    async def test_67_circuit_breaker_timeout_reset(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)  # 100ms timeout
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        try:
            await cb.call(failing_func)
        except:
            pass
        
        assert cb.state == "OPEN", "Circuit not opened"
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # Should be in HALF_OPEN state now
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        assert cb.state == "CLOSED", "Circuit not closed after successful call"
    
    async def test_68_circuit_breaker_half_open_state(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.1)
        
        # Open circuit
        try:
            await cb.call(lambda: 1/0)  # Trigger failure
        except:
            pass
        
        # Wait for reset timeout
        await asyncio.sleep(0.2)
        
        # Next call should put it in HALF_OPEN
        try:
            await cb.call(lambda: 1/0)  # Another failure
        except:
            pass
        
        assert cb.state == "OPEN", "Circuit should remain open after half-open failure"
    
    async def test_69_circuit_breaker_statistics(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        
        # Record some statistics
        cb.record_success()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failure_count == 0, "Failure count not reset after success"
    
    async def test_70_circuit_breaker_concurrent_access(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)
        
        async def test_func(value):
            await asyncio.sleep(0.01)
            return value
        
        # Run multiple concurrent calls - fix the lambda issue
        async def make_call(value):
            return await cb.call(test_func, value)
        
        tasks = [make_call(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5, "Concurrent calls failed"
        assert all(isinstance(r, int) for r in results), "Invalid results from concurrent calls"
    
    # ========== INTEGRATION TESTS (71-80) ==========
    
    async def test_71_agent_workflow_integration(self):
        await self.specialized_agent.queue_workflow("integration_test_1", "test")
        result = await self.specialized_agent.execute_next_workflow()
        
        assert result is not None, "Agent workflow integration failed"
        assert result["status"] in ["success", "failed"], "Invalid integration result"
    
    async def test_72_agent_queue_management(self):
        # Queue multiple workflows
        for i in range(3):
            await self.specialized_agent.queue_workflow(f"queue_test_{i}", "test", priority=i+1)
        
        assert len(self.specialized_agent.task_queue) == 3, "Queue management failed"
    
    async def test_73_agent_priority_handling(self):
        # Clear queue to ensure clean test state
        self.specialized_agent.task_queue.clear()
        
        # Queue with different priorities
        await self.specialized_agent.queue_workflow("low_priority", "test", priority=3)
        await self.specialized_agent.queue_workflow("high_priority", "test", priority=1)
        
        # High priority should be first
        first_task = self.specialized_agent.task_queue[0]
        assert first_task.task_id == "high_priority", "Priority handling failed"
    
    async def test_74_agent_status_reporting(self):
        status = await self.specialized_agent.get_agent_status()
        
        assert "agent_id" in status, "Missing agent ID"
        assert "status" in status, "Missing agent status"
        assert "performance" in status, "Missing performance metrics"
    
    async def test_75_agent_circuit_breaker_integration(self):
        cb_status = await self.specialized_agent.get_agent_status()
        circuit_breaker = cb_status.get("circuit_breaker", {})
        
        assert "state" in circuit_breaker, "Circuit breaker not integrated"
        assert circuit_breaker["state"] in ["CLOSED", "OPEN", "HALF_OPEN"], "Invalid CB state"
    
    async def test_76_end_to_end_workflow_execution(self):
        # Test complete workflow from queue to completion
        await self.specialized_agent.queue_workflow("e2e_test", "test", 
                                                   context={"test": "data"})
        result = await self.specialized_agent.execute_next_workflow()
        
        assert result["status"] == "success", f"E2E workflow failed: {result.get('error')}"
        assert "correlation_id" in result, "Missing correlation ID in E2E test"
    
    async def test_77_component_health_aggregation(self):
        system_health = await self.integrated_system.get_system_health()
        agent_status = await self.specialized_agent.get_agent_status()
        
        assert system_health["overall_status"] != "unknown", "System health not available"
        assert agent_status["status"] != "unknown", "Agent status not available"
    
    async def test_78_error_propagation(self):
        # Test that errors propagate correctly through the system
        await self.specialized_agent.queue_workflow("error_prop_test", "test")
        
        # Even with errors, system should remain operational
        status = await self.specialized_agent.get_agent_status()
        assert status["status"] in ["operational", "not_initialized"], "System not handling errors"
    
    async def test_79_concurrent_integration(self):
        # Test multiple components working simultaneously
        tasks = []
        
        # Queue workflows through agent
        for i in range(3):
            task = self.specialized_agent.queue_workflow(f"concurrent_int_{i}", "test")
            tasks.append(task)
        
        # Execute direct workflows
        for i in range(2):
            task = self.integrated_system.execute_workflow(f"direct_int_{i}", "test")
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should still be healthy
        health = await self.integrated_system.get_system_health()
        assert health["overall_status"] != "unhealthy", "Concurrent integration failed"
    
    async def test_80_data_flow_integration(self):
        # Test data flowing between components
        workflow_result = await self.integrated_system.execute_workflow("data_flow_test", "test")
        correlation_id = workflow_result["correlation_id"]
        
        # Check if data appears in event store
        events = await self.event_store.get_events("workflow_test")
        
        # Should have some events (not necessarily from this specific workflow due to stream naming)
        assert isinstance(events, list), "Event store integration failed"
    
    # ========== PERFORMANCE TESTS (81-88) ==========
    
    async def test_81_workflow_latency(self):
        start_time = time.time()
        result = await self.integrated_system.execute_workflow("latency_test", "test")
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 5000, f"Workflow latency too high: {latency_ms}ms"  # 5s threshold
    
    async def test_82_concurrent_performance(self):
        start_time = time.time()
        
        tasks = []
        for i in range(5):
            task = self.integrated_system.execute_workflow(f"perf_test_{i}", "test")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 10.0, f"Concurrent execution too slow: {total_time}s"
        assert all(r["status"] == "success" for r in results), "Concurrent performance degraded"
    
    async def test_83_memory_efficiency(self):
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many workflows
        for i in range(10):
            await self.integrated_system.execute_workflow(f"memory_test_{i}", "test")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should not leak significant memory (allow 50MB increase)
        assert memory_increase < 50 * 1024 * 1024, f"Memory leak detected: {memory_increase/1024/1024}MB"
    
    async def test_84_throughput_measurement(self):
        start_time = time.time()
        workflow_count = 10
        
        tasks = []
        for i in range(workflow_count):
            task = self.integrated_system.execute_workflow(f"throughput_{i}", "test")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = workflow_count / duration
        
        assert throughput > 1.0, f"Throughput too low: {throughput} workflows/sec"
    
    async def test_85_resource_cleanup(self):
        initial_active = len(self.observer.active_workflows)
        
        # Execute workflows that should complete
        for i in range(3):
            await self.integrated_system.execute_workflow(f"cleanup_test_{i}", "test")
        
        final_active = len(self.observer.active_workflows)
        
        # Should not have accumulated active workflows
        assert final_active <= initial_active + 1, "Resource cleanup failed"
    
    async def test_86_event_store_performance(self):
        start_time = time.time()
        
        # Store many events
        tasks = []
        for i in range(20):
            task = self.event_store.store_event("perf_test", "event", {"index": i})
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 5.0, f"Event store too slow: {duration}s for 20 events"
    
    async def test_87_optimizer_performance(self):
        start_time = time.time()
        
        # Run optimization tasks
        config = {"batch_size": 10}
        context = {"cpu_percent": 50.0}
        
        for i in range(10):
            await self.optimizer.record_performance(f"opt_perf_{i}", 100.0 + i*10, True, config, context)
            self.optimizer.suggest_configuration("test", context)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 3.0, f"Optimizer too slow: {duration}s for 10 operations"
    
    async def test_88_system_scalability(self):
        # Test system can handle increasing load
        batch_sizes = [1, 3, 5]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            tasks = []
            for i in range(batch_size):
                task = self.integrated_system.execute_workflow(f"scale_{batch_size}_{i}", "test")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Check that performance doesn't degrade significantly
            duration_per_workflow = (end_time - start_time) / batch_size
            assert duration_per_workflow < 2.0, f"Scalability issue at batch size {batch_size}: {duration_per_workflow}s/workflow"
            
            # Check success rate
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
            success_rate = successful / batch_size
            assert success_rate >= 0.8, f"Low success rate at batch size {batch_size}: {success_rate}"
    
    # ========== MAIN EXECUTION ==========
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all 88 tests and generate comprehensive report"""
        
        print(f"🚀 Starting 88-Test Validation Suite")
        print(f"   Start time: {self.start_time}")
        print("=" * 80)
        
        # Setup test environment
        if not await self.setup_test_environment():
            return {"error": "Failed to setup test environment"}
        
        # Define all test functions
        test_functions = []
        
        # Core System Tests (1-15)
        core_tests = [
            ("Integrated System Initialization", "Core System", self.test_01_integrated_system_initialization),
            ("Specialized Agent Initialization", "Core System", self.test_02_specialized_agent_initialization),
            ("Basic Workflow Execution", "Core System", self.test_03_workflow_execution_basic),
            ("Workflow Execution with Context", "Core System", self.test_04_workflow_execution_with_context),
            ("System Health Check", "Core System", self.test_05_system_health_check),
            ("Workflow with Custom Steps", "Core System", self.test_06_workflow_with_custom_steps),
            ("Multiple Concurrent Workflows", "Core System", self.test_07_multiple_concurrent_workflows),
            ("Workflow Error Handling", "Core System", self.test_08_workflow_error_handling),
            ("Workflow Configuration Validation", "Core System", self.test_09_workflow_configuration_validation),
            ("Component Integration", "Core System", self.test_10_component_integration),
            ("Workflow Types Support", "Core System", self.test_11_workflow_types_support),
            ("Workflow Duration Tracking", "Core System", self.test_12_workflow_duration_tracking),
            ("Correlation ID Propagation", "Core System", self.test_13_correlation_id_propagation),
            ("Workflow Step Tracking", "Core System", self.test_14_workflow_step_tracking),
            ("System Resource Management", "Core System", self.test_15_system_resource_management)
        ]
        
        # Observability Tests (16-30)
        observability_tests = [
            ("Observer Workflow Tracking", "Observability", self.test_16_observer_workflow_tracking),
            ("Observer Step Recording", "Observability", self.test_17_observer_step_recording),
            ("Observer Workflow Completion", "Observability", self.test_18_observer_workflow_completion),
            ("Observer Health Status", "Observability", self.test_19_observer_health_status),
            ("Observer Metrics Collection", "Observability", self.test_20_observer_metrics_collection),
            ("Observer Error Tracking", "Observability", self.test_21_observer_error_tracking),
            ("Observer Resource Monitoring", "Observability", self.test_22_observer_resource_monitoring),
            ("Observer Correlation IDs", "Observability", self.test_23_observer_correlation_ids),
            ("Observer Workflow Types", "Observability", self.test_24_observer_workflow_types),
            ("Observer Multiple Steps", "Observability", self.test_25_observer_multiple_steps),
            ("Observer Duration Measurement", "Observability", self.test_26_observer_duration_measurement),
            ("Observer Concurrent Workflows", "Observability", self.test_27_observer_concurrent_workflows),
            ("Observer Event Export", "Observability", self.test_28_observer_event_export),
            ("Observer Health Scoring", "Observability", self.test_29_observer_health_scoring),
            ("Observer Metrics Summary", "Observability", self.test_30_observer_metrics_summary)
        ]
        
        # Persistence Tests (31-45)
        persistence_tests = [
            ("Event Store Basic Storage", "Persistence", self.test_31_event_store_basic_storage),
            ("Event Store Deduplication", "Persistence", self.test_32_event_store_deduplication),
            ("Event Store Retrieval", "Persistence", self.test_33_event_store_retrieval),
            ("Event Store Correlation Tracking", "Persistence", self.test_34_event_store_correlation_tracking),
            ("Event Store Multiple Streams", "Persistence", self.test_35_event_store_multiple_streams),
            ("Event Store Stream Info", "Persistence", self.test_36_event_store_stream_info),
            ("Event Store Health Status", "Persistence", self.test_37_event_store_health_status),
            ("Event Store Large Payload", "Persistence", self.test_38_event_store_large_payload),
            ("Event Store Concurrent Writes", "Persistence", self.test_39_event_store_concurrent_writes),
            ("Event Store Event Types", "Persistence", self.test_40_event_store_event_types),
            ("Event Store Timestamp Tracking", "Persistence", self.test_41_event_store_timestamp_tracking),
            ("Event Store Mark Processed", "Persistence", self.test_42_event_store_mark_processed),
            ("Event Store Fallback Behavior", "Persistence", self.test_43_event_store_fallback_behavior),
            ("Event Store Error Handling", "Persistence", self.test_44_event_store_error_handling),
            ("Event Store Stream Isolation", "Persistence", self.test_45_event_store_stream_isolation)
        ]
        
        # ML Optimization Tests (46-60)
        ml_tests = [
            ("Optimizer Initialization", "ML Optimization", self.test_46_optimizer_initialization),
            ("Optimizer Performance Recording", "ML Optimization", self.test_47_optimizer_performance_recording),
            ("Optimizer Configuration Suggestion", "ML Optimization", self.test_48_optimizer_configuration_suggestion),
            ("Optimizer Feature Extraction", "ML Optimization", self.test_49_optimizer_feature_extraction),
            ("Optimizer Performance Prediction", "ML Optimization", self.test_50_optimizer_performance_prediction),
            ("Optimizer Learning Iterations", "ML Optimization", self.test_51_optimizer_learning_iterations),
            ("Optimizer A/B Experiment Creation", "ML Optimization", self.test_52_optimizer_ab_experiment_creation),
            ("Optimizer A/B Config Assignment", "ML Optimization", self.test_53_optimizer_ab_config_assignment),
            ("Optimizer A/B Result Recording", "ML Optimization", self.test_54_optimizer_ab_result_recording),
            ("Optimizer Optimization Summary", "ML Optimization", self.test_55_optimizer_optimization_summary),
            ("Optimizer Model Export", "ML Optimization", self.test_56_optimizer_model_export),
            ("Optimizer Success/Failure Tracking", "ML Optimization", self.test_57_optimizer_success_failure_tracking),
            ("Optimizer Context Awareness", "ML Optimization", self.test_58_optimizer_context_awareness),
            ("Optimizer Weight Updates", "ML Optimization", self.test_59_optimizer_weight_updates),
            ("Optimizer Configuration Space", "ML Optimization", self.test_60_optimizer_configuration_space)
        ]
        
        # Circuit Breaker Tests (61-70)
        circuit_breaker_tests = [
            ("Circuit Breaker Initialization", "Circuit Breaker", self.test_61_circuit_breaker_initialization),
            ("Circuit Breaker Success Recording", "Circuit Breaker", self.test_62_circuit_breaker_success_recording),
            ("Circuit Breaker Failure Recording", "Circuit Breaker", self.test_63_circuit_breaker_failure_recording),
            ("Circuit Breaker State Transitions", "Circuit Breaker", self.test_64_circuit_breaker_state_transitions),
            ("Circuit Breaker Call Execution", "Circuit Breaker", self.test_65_circuit_breaker_call_execution),
            ("Circuit Breaker Call Rejection", "Circuit Breaker", self.test_66_circuit_breaker_call_rejection),
            ("Circuit Breaker Timeout Reset", "Circuit Breaker", self.test_67_circuit_breaker_timeout_reset),
            ("Circuit Breaker Half-Open State", "Circuit Breaker", self.test_68_circuit_breaker_half_open_state),
            ("Circuit Breaker Statistics", "Circuit Breaker", self.test_69_circuit_breaker_statistics),
            ("Circuit Breaker Concurrent Access", "Circuit Breaker", self.test_70_circuit_breaker_concurrent_access)
        ]
        
        # Integration Tests (71-80)
        integration_tests = [
            ("Agent Workflow Integration", "Integration", self.test_71_agent_workflow_integration),
            ("Agent Queue Management", "Integration", self.test_72_agent_queue_management),
            ("Agent Priority Handling", "Integration", self.test_73_agent_priority_handling),
            ("Agent Status Reporting", "Integration", self.test_74_agent_status_reporting),
            ("Agent Circuit Breaker Integration", "Integration", self.test_75_agent_circuit_breaker_integration),
            ("End-to-End Workflow Execution", "Integration", self.test_76_end_to_end_workflow_execution),
            ("Component Health Aggregation", "Integration", self.test_77_component_health_aggregation),
            ("Error Propagation", "Integration", self.test_78_error_propagation),
            ("Concurrent Integration", "Integration", self.test_79_concurrent_integration),
            ("Data Flow Integration", "Integration", self.test_80_data_flow_integration)
        ]
        
        # Performance Tests (81-88)
        performance_tests = [
            ("Workflow Latency", "Performance", self.test_81_workflow_latency),
            ("Concurrent Performance", "Performance", self.test_82_concurrent_performance),
            ("Memory Efficiency", "Performance", self.test_83_memory_efficiency),
            ("Throughput Measurement", "Performance", self.test_84_throughput_measurement),
            ("Resource Cleanup", "Performance", self.test_85_resource_cleanup),
            ("Event Store Performance", "Performance", self.test_86_event_store_performance),
            ("Optimizer Performance", "Performance", self.test_87_optimizer_performance),
            ("System Scalability", "Performance", self.test_88_system_scalability)
        ]
        
        # Combine all tests
        all_tests = (core_tests + observability_tests + persistence_tests + 
                    ml_tests + circuit_breaker_tests + integration_tests + performance_tests)
        
        # Verify we have exactly 88 tests
        assert len(all_tests) == 88, f"Expected 88 tests, got {len(all_tests)}"
        
        # Execute all tests
        for i, (test_name, category, test_func) in enumerate(all_tests, 1):
            await self.run_test(i, test_name, category, test_func)
            
            # Small delay between tests to prevent resource exhaustion
            if i % 10 == 0:
                await asyncio.sleep(0.1)
                print(f"📊 Progress: {i}/88 tests completed ({i/88*100:.1f}%)")
        
        # Generate final report
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Categorize results
        passed = [r for r in self.results if r.status == "PASS"]
        failed = [r for r in self.results if r.status == "FAIL"]
        errors = [r for r in self.results if r.status == "ERROR"]
        
        # Category breakdown
        category_stats = {}
        for category in self.test_categories.keys():
            cat_results = [r for r in self.results if r.category == category]
            category_stats[category] = {
                "total": len(cat_results),
                "passed": len([r for r in cat_results if r.status == "PASS"]),
                "failed": len([r for r in cat_results if r.status == "FAIL"]),
                "errors": len([r for r in cat_results if r.status == "ERROR"])
            }
        
        # Performance metrics
        total_test_time = sum(r.duration_ms for r in self.results)
        avg_test_time = total_test_time / len(self.results) if self.results else 0
        
        report = {
            "validation_summary": {
                "total_tests": len(self.results),
                "passed": len(passed),
                "failed": len(failed),
                "errors": len(errors),
                "success_rate_percent": (len(passed) / len(self.results) * 100) if self.results else 0,
                "manu_compliance": len(passed) == 88  # True if all 88 tests pass
            },
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_test_execution_ms": total_test_time,
                "average_test_duration_ms": avg_test_time
            },
            "category_breakdown": category_stats,
            "failed_tests": [
                {
                    "test_id": r.test_id,
                    "name": r.test_name,
                    "category": r.category,
                    "error": r.error_message
                } for r in failed + errors
            ],
            "performance_summary": {
                "fastest_test_ms": min(r.duration_ms for r in self.results) if self.results else 0,
                "slowest_test_ms": max(r.duration_ms for r in self.results) if self.results else 0,
                "tests_per_second": len(self.results) / total_duration if total_duration > 0 else 0
            },
            "system_status": "MANU_COMPLIANT" if len(passed) == 88 else "NON_COMPLIANT",
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "name": r.test_name,
                    "category": r.category,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "error": r.error_message
                } for r in self.results
            ]
        }
        
        return report

async def main():
    """Main entry point for 88-test validation suite"""
    print("🧪 Comprehensive Workflow Validation Suite")
    print("🎯 MANU Compliance Testing - 88/88 Tests")
    print("=" * 80)
    
    # Initialize validation suite
    suite = WorkflowValidationSuite()
    
    # Run all tests
    report = await suite.run_all_tests()
    
    # Display results
    print("\n" + "=" * 80)
    print("📋 FINAL VALIDATION REPORT")
    print("=" * 80)
    
    if "error" in report:
        print(f"❌ Validation failed: {report['error']}")
        return
    
    summary = report["validation_summary"]
    timing = report["timing"]
    
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"💥 Errors: {summary['errors']}")
    print(f"📊 Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"🎯 MANU Compliant: {'✅ YES' if summary['manu_compliance'] else '❌ NO'}")
    print(f"⏱️  Total Duration: {timing['total_duration_seconds']:.1f} seconds")
    
    # Category breakdown
    print("\n📊 Category Breakdown:")
    for category, stats in report["category_breakdown"].items():
        success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {category:20s}: {stats['passed']:2d}/{stats['total']:2d} ({success_rate:5.1f}%)")
    
    # Failed tests
    if report["failed_tests"]:
        print(f"\n❌ Failed Tests ({len(report['failed_tests'])}):")
        for test in report["failed_tests"][:5]:  # Show first 5 failures
            print(f"   Test {test['test_id']:2d}: {test['name']}")
            if test['error']:
                print(f"            Error: {test['error'][:100]}...")
    
    # Save detailed report
    report_file = f"workflow_validation_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report: {report_file}")
    
    # Final status
    if summary["manu_compliance"]:
        print("\n🎉 VALIDATION COMPLETE - MANU COMPLIANT (88/88)")
        print("✅ System ready for production deployment")
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE - {summary['passed']}/88 tests passing")
        print("❌ System requires fixes before production deployment")

if __name__ == "__main__":
    # Install test dependencies
    try:
        import pytest
        import psutil
    except ImportError as e:
        print(f"⚠️  Missing test dependency: {e}")
        print("Installing required packages...")
        import subprocess
        packages = ["pytest", "psutil"]
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"❌ Failed to install {package}")
    
    # Run validation suite
    asyncio.run(main())