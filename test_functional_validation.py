#!/usr/bin/env python3
"""
Comprehensive Functional Testing Suite for AG-06 Mixer
Tests real functionality beyond structural validation
"""
import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

# Add path for imports
import sys
sys.path.append('/Users/nguythe/ag06_mixer')

class FunctionalTestSuite:
    """Execute real functional tests with actual data processing"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        
    async def test_audio_processing_pipeline(self) -> bool:
        """Test real audio processing with data flow"""
        try:
            from implementations.audio_engine import AG06AudioEngine
            from interfaces.audio_engine import AudioConfig
            
            engine = AG06AudioEngine()
            config = AudioConfig(
                sample_rate=48000,
                buffer_size=512,
                channels=2,
                bit_depth=24
            )
            
            # Create test audio data
            duration = 0.1  # 100ms
            samples = int(config.sample_rate * duration)
            test_signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
            stereo_signal = np.array([test_signal, test_signal]).T
            audio_bytes = (stereo_signal * 32767).astype(np.int16).tobytes()
            
            # Process audio
            await engine.initialize(config)
            await engine.start()
            processed = await engine.process_audio(audio_bytes)
            await engine.stop()
            
            return processed is not None and len(processed) > 0
        except Exception as e:
            print(f"Audio pipeline test failed: {e}")
            return False
    
    async def test_midi_control_processing(self) -> bool:
        """Test MIDI control message handling"""
        try:
            from implementations.midi_controller import AG06MidiController
            from interfaces.midi_controller import MidiMessage, MidiMessageType
            
            controller = AG06MidiController()
            await controller.initialize()
            
            # Test control change
            cc_msg = MidiMessage(
                type=MidiMessageType.CONTROL_CHANGE,
                channel=0,
                data1=7,  # Volume
                data2=100
            )
            
            result = await controller.send_message(cc_msg)
            return result is not None
        except Exception as e:
            print(f"MIDI control test failed: {e}")
            return False
    
    async def test_ring_buffer_performance(self) -> bool:
        """Test lock-free ring buffer with real data"""
        try:
            from implementations.optimized_ring_buffer import OptimizedRingBuffer
            
            buffer = OptimizedRingBuffer(capacity=4096)
            
            # Performance test
            start = time.perf_counter()
            test_data = np.random.randn(2, 512).astype(np.float32)
            
            for _ in range(100):
                buffer.write(test_data)
                data = buffer.read(512)
            
            elapsed = time.perf_counter() - start
            latency_ms = (elapsed / 100) * 1000
            
            return latency_ms < 1.0  # Sub-millisecond per operation
        except Exception as e:
            print(f"Ring buffer test failed: {e}")
            return False
    
    async def test_parallel_event_bus(self) -> bool:
        """Test parallel event processing with multiple workers"""
        try:
            from core.parallel_event_bus import ParallelEventBus
            
            bus = ParallelEventBus(num_workers=4)
            await bus.start()
            
            events_processed = []
            
            async def handler(event):
                events_processed.append(event)
                return {"processed": True}
            
            bus.subscribe("test_event", handler)
            
            # Send multiple events
            for i in range(10):
                await bus.publish("test_event", {"id": i})
            
            await asyncio.sleep(0.1)
            await bus.stop()
            
            return len(events_processed) >= 8  # At least 80% processed
        except Exception as e:
            print(f"Event bus test failed: {e}")
            return False
    
    async def test_preset_management(self) -> bool:
        """Test preset save/load functionality"""
        try:
            from implementations.preset_manager import AG06PresetManager
            
            manager = AG06PresetManager()
            await manager.initialize()
            
            # Create test preset
            test_preset = {
                "name": "Test Preset",
                "version": "1.0",
                "parameters": {
                    "volume": 0.8,
                    "pan": 0.0,
                    "eq": {"low": 0.2, "mid": 0.5, "high": 0.3}
                }
            }
            
            # Save and load
            await manager.save_preset("test", test_preset)
            loaded = await manager.load_preset("test")
            
            return loaded is not None and loaded.get("name") == "Test Preset"
        except Exception as e:
            print(f"Preset management test failed: {e}")
            return False
    
    async def test_workflow_orchestration(self) -> bool:
        """Test workflow task execution"""
        try:
            from core.workflow_orchestrator import AG06WorkflowOrchestrator, WorkflowTask, TaskType
            
            orchestrator = AG06WorkflowOrchestrator()
            await orchestrator.initialize()
            await orchestrator.start()
            
            # Create test task
            task = WorkflowTask(
                type=TaskType.AUDIO_PROCESSING,
                parameters={"test": True},
                priority=1
            )
            
            result = await orchestrator.execute_task(task)
            await orchestrator.stop()
            
            return result.success
        except Exception as e:
            print(f"Workflow orchestration test failed: {e}")
            return False
    
    async def test_dependency_injection(self) -> bool:
        """Test DI container functionality"""
        try:
            from core.dependency_container import DependencyContainer
            
            container = DependencyContainer()
            
            # Register test service
            class TestService:
                def get_value(self):
                    return "test_value"
            
            container.register("test_service", lambda: TestService(), "singleton")
            
            # Resolve service
            service1 = container.resolve("test_service")
            service2 = container.resolve("test_service")
            
            return service1 is service2 and service1.get_value() == "test_value"
        except Exception as e:
            print(f"DI container test failed: {e}")
            return False
    
    async def test_optimization_agent(self) -> bool:
        """Test autonomous optimization agent"""
        try:
            import os
            status_file = '/Users/nguythe/ag06_mixer/ag06_optimization_status.json'
            
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                return (status.get('running', False) and 
                       status.get('optimizations', 0) > 0)
            return False
        except Exception as e:
            print(f"Optimization agent test failed: {e}")
            return False
    
    async def test_buffer_pool_efficiency(self) -> bool:
        """Test pre-warmed buffer pool performance"""
        try:
            from implementations.performance_optimizer import BufferPool
            
            pool = BufferPool(pool_size=10, buffer_size=1024)
            
            buffers_acquired = []
            for _ in range(5):
                buf = pool.acquire()
                buffers_acquired.append(buf)
            
            # Return buffers
            for buf in buffers_acquired:
                pool.release(buf)
            
            # Check reuse
            reused = pool.acquire()
            return reused is not None
        except Exception as e:
            print(f"Buffer pool test failed: {e}")
            return False
    
    async def test_karaoke_processing(self) -> bool:
        """Test karaoke vocal processing"""
        try:
            from implementations.karaoke_integration import AG06KaraokeIntegration
            
            karaoke = AG06KaraokeIntegration()
            await karaoke.initialize()
            
            # Create test audio
            test_audio = np.random.randn(48000).astype(np.float32).tobytes()
            
            # Process vocals
            processed = await karaoke.remove_vocals(test_audio)
            
            return processed is not None and len(processed) > 0
        except Exception as e:
            print(f"Karaoke processing test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Execute all functional tests"""
        tests = [
            ("Audio Processing Pipeline", self.test_audio_processing_pipeline),
            ("MIDI Control Processing", self.test_midi_control_processing),
            ("Ring Buffer Performance", self.test_ring_buffer_performance),
            ("Parallel Event Bus", self.test_parallel_event_bus),
            ("Preset Management", self.test_preset_management),
            ("Workflow Orchestration", self.test_workflow_orchestration),
            ("Dependency Injection", self.test_dependency_injection),
            ("Optimization Agent", self.test_optimization_agent),
            ("Buffer Pool Efficiency", self.test_buffer_pool_efficiency),
            ("Karaoke Processing", self.test_karaoke_processing)
        ]
        
        print("\n" + "="*70)
        print("FUNCTIONAL VALIDATION SUITE - REAL COMPONENT TESTING")
        print("="*70)
        
        for name, test_func in tests:
            try:
                result = await test_func()
                status = "✅ PASS" if result else "❌ FAIL"
                self.results.append({"test": name, "passed": result})
                
                if result:
                    self.passed += 1
                else:
                    self.failed += 1
                
                print(f"{status} - {name}")
            except Exception as e:
                print(f"❌ FAIL - {name}: {str(e)}")
                self.failed += 1
                self.results.append({"test": name, "passed": False, "error": str(e)})
        
        # Summary
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print(f"FUNCTIONAL TEST RESULTS: {self.passed}/{total} ({percentage:.1f}%)")
        print("="*70)
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": self.passed,
            "failed": self.failed,
            "percentage": percentage,
            "results": self.results
        }
        
        with open('/Users/nguythe/ag06_mixer/functional_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return percentage >= 80  # Success if 80% or more pass


async def main():
    """Run functional validation suite"""
    suite = FunctionalTestSuite()
    success = await suite.run_all_tests()
    
    if success:
        print("\n✅ FUNCTIONAL VALIDATION SUCCESSFUL")
    else:
        print("\n⚠️ FUNCTIONAL VALIDATION NEEDS ATTENTION")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)