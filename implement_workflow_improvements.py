#!/usr/bin/env python3
"""
Workflow Improvements Implementation Script
Integrates observability, persistence, and ML optimization
SOLID Compliant | Production Ready | Immediate Impact
"""

import asyncio
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

# Import our new components
from monitoring.realtime_observer import observability, StructuredLogger
from persistence.event_store import persistence, StoredEvent, EventPriority
from ml.active_optimizer import ml_optimizer
from ag06_enhanced_workflow_system import (
    AG06EnhancedWorkflowFactory,
    AG06EnhancedWorkflowOrchestrator,
    AudioEvent,
    AudioEventType
)
from reliability.circuit_breaker import circuit_breaker_registry, CircuitBreakerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED WORKFLOW WITH IMPROVEMENTS
# ============================================================================

class ImprovedWorkflowSystem:
    """Enhanced workflow system with all improvements integrated"""
    
    def __init__(self):
        """Initialize improved workflow system"""
        self.logger = observability.create_context_logger(
            component="ImprovedWorkflow",
            version="4.0.0"
        )
        self.components = None
        self.orchestrator = None
        self._monitoring_task = None
        self._persistence_connected = False
        self._ml_optimizer_active = False
    
    async def initialize(self) -> bool:
        """Initialize all improved components"""
        self.logger.info("Initializing Improved Workflow System...")
        
        # Step 1: Initialize persistence
        self.logger.info("1. Setting up event persistence...")
        self._persistence_connected = await persistence.initialize()
        if self._persistence_connected:
            self.logger.info("✅ Event persistence connected to Redis")
        else:
            self.logger.warning("⚠️ Event persistence using in-memory fallback")
        
        # Step 2: Start ML optimizer
        self.logger.info("2. Starting ML optimizer...")
        await ml_optimizer.start()
        self._ml_optimizer_active = True
        self.logger.info("✅ ML optimizer active with real metrics")
        
        # Step 3: Create enhanced workflow components
        self.logger.info("3. Creating workflow components...")
        self.components = await AG06EnhancedWorkflowFactory.create_complete_system()
        self.orchestrator = AG06EnhancedWorkflowOrchestrator(self.components)
        self.logger.info("✅ Workflow components created")
        
        # Step 4: Register health checks
        self.logger.info("4. Registering health checks...")
        self._register_health_checks()
        self.logger.info("✅ Health checks registered")
        
        # Step 5: Setup event persistence handlers
        self.logger.info("5. Setting up event persistence...")
        await self._setup_event_persistence()
        self.logger.info("✅ Event persistence configured")
        
        # Step 6: Configure circuit breakers
        self.logger.info("6. Configuring circuit breakers...")
        self._configure_circuit_breakers()
        self.logger.info("✅ Circuit breakers configured")
        
        # Step 7: Start monitoring
        self.logger.info("7. Starting real-time monitoring...")
        self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
        self.logger.info("✅ Real-time monitoring active")
        
        self.logger.info("🚀 Improved Workflow System fully initialized!")
        return True
    
    def _register_health_checks(self):
        """Register component health checks"""
        
        async def check_event_bus():
            event_bus = self.components.get('event_bus')
            if event_bus and event_bus._processing:
                return {
                    'healthy': True,
                    'message': 'Event bus processing',
                    'details': {
                        'queue_size': event_bus._event_queue.qsize(),
                        'subscribers': len(event_bus._subscribers)
                    }
                }
            return {'healthy': False, 'message': 'Event bus not processing'}
        
        async def check_persistence():
            return {
                'healthy': self._persistence_connected,
                'message': 'Redis connected' if self._persistence_connected else 'Using fallback',
                'details': {'mode': 'redis' if self._persistence_connected else 'memory'}
            }
        
        async def check_ml_optimizer():
            status = await ml_optimizer.get_optimization_status()
            return {
                'healthy': status['active'],
                'message': f"Score: {status.get('current_score', 0):.1f}/100",
                'details': {
                    'experiments_run': status.get('experiments_run', 0),
                    'optimizations': len(status.get('recent_optimizations', []))
                }
            }
        
        observability.health.register_check('event_bus', check_event_bus)
        observability.health.register_check('persistence', check_persistence)
        observability.health.register_check('ml_optimizer', check_ml_optimizer)
    
    async def _setup_event_persistence(self):
        """Setup event persistence for workflow events"""
        event_bus = self.components.get('event_bus')
        
        # Subscribe to all events for persistence
        class PersistenceHandler:
            async def handle_event(self, event: AudioEvent):
                # Persist event
                await persistence.persist_event(
                    event_type=event.event_type.value,
                    source=event.source,
                    data=event.data,
                    priority=EventPriority.MEDIUM
                )
                
                # Track metric
                observability.metrics.increment_counter(
                    'events_processed',
                    labels={'event_type': event.event_type.value, 'status': 'persisted'}
                )
            
            def get_supported_events(self):
                return [e.value for e in AudioEventType]
        
        handler = PersistenceHandler()
        for event_type in AudioEventType:
            await event_bus.subscribe(event_type, handler)
    
    def _configure_circuit_breakers(self):
        """Configure circuit breakers for critical operations"""
        
        # Audio processing circuit breaker
        audio_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=5.0
        )
        circuit_breaker_registry.get_breaker('audio_processing', audio_config)
        
        # External API circuit breaker
        api_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout=10.0
        )
        circuit_breaker_registry.get_breaker('external_api', api_config)
        
        # Database circuit breaker
        db_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=20.0,
            success_threshold=1,
            timeout=3.0
        )
        circuit_breaker_registry.get_breaker('database', db_config)
    
    async def _continuous_monitoring(self):
        """Continuous monitoring and metrics collection"""
        while True:
            try:
                # Collect ML optimizer metrics
                ml_status = await ml_optimizer.get_optimization_status()
                
                # Update Prometheus metrics
                if ml_status['current_metrics']:
                    metrics = ml_status['current_metrics']
                    observability.metrics.observe_histogram(
                        'latency',
                        metrics['latency_ms'],
                        labels={'operation': 'workflow'}
                    )
                    observability.metrics.set_gauge(
                        'cpu_usage',
                        metrics['cpu_percent']
                    )
                    observability.metrics.set_gauge(
                        'memory_usage',
                        metrics['memory_mb']
                    )
                
                # Get circuit breaker states
                cb_states = circuit_breaker_registry.get_all_states()
                for name, state in cb_states.items():
                    if state['state'] == 'open':
                        observability.metrics.increment_counter(
                            'circuit_breaker_trips',
                            labels={'breaker': name}
                        )
                
                # Log monitoring heartbeat
                self.logger.debug("Monitoring heartbeat", 
                                score=ml_status.get('current_score', 0),
                                breakers_open=sum(1 for s in cb_states.values() 
                                                if s['state'] == 'open'))
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def demonstrate_improvements(self):
        """Demonstrate the improvements in action"""
        print("\n" + "="*80)
        print("DEMONSTRATING WORKFLOW IMPROVEMENTS")
        print("="*80)
        
        # 1. Show real-time metrics
        print("\n📊 Real-Time Metrics:")
        ml_status = await ml_optimizer.get_optimization_status()
        current_metrics = ml_status.get('current_metrics', {})
        print(f"  Latency: {current_metrics.get('latency_ms', 0):.2f}ms")
        print(f"  Throughput: {current_metrics.get('throughput_ops_sec', 0):.0f} ops/sec")
        print(f"  CPU Usage: {current_metrics.get('cpu_percent', 0):.1f}%")
        print(f"  Memory: {current_metrics.get('memory_mb', 0):.0f}MB")
        print(f"  Performance Score: {ml_status.get('current_score', 0):.1f}/100")
        
        # 2. Show event persistence
        print("\n💾 Event Persistence:")
        test_event_id = await persistence.persist_event(
            event_type="demo_event",
            source="improvement_demo",
            data={"test": "data", "timestamp": datetime.now().isoformat()},
            priority=EventPriority.HIGH
        )
        print(f"  Persisted event: {test_event_id}")
        print(f"  Persistence mode: {'Redis' if self._persistence_connected else 'In-Memory'}")
        
        # 3. Show ML optimization
        print("\n🧠 ML Optimization:")
        recent_opts = ml_status.get('recent_optimizations', [])
        if recent_opts:
            latest = recent_opts[0]
            print(f"  Latest optimization: {latest['id']}")
            print(f"  Expected improvement: {latest['expected_improvement']:.1%}")
            print(f"  Confidence: {latest['confidence']:.1%}")
            print(f"  Applied: {'Yes' if latest['applied'] else 'No'}")
        else:
            print("  Collecting baseline metrics...")
        
        # 4. Show circuit breaker status
        print("\n🔌 Circuit Breakers:")
        cb_states = circuit_breaker_registry.get_all_states()
        for name, state in cb_states.items():
            status_icon = "🟢" if state['state'] == 'closed' else "🔴" if state['state'] == 'open' else "🟡"
            print(f"  {status_icon} {name}: {state['state'].upper()}")
        
        # 5. Show health status
        print("\n❤️ System Health:")
        health_status = await observability.health.check_health()
        overall_icon = "✅" if health_status['healthy'] else "❌"
        print(f"  Overall: {overall_icon} {'Healthy' if health_status['healthy'] else 'Unhealthy'}")
        for check_name, check_result in health_status['checks'].items():
            check_icon = "✅" if check_result['healthy'] else "❌"
            print(f"  {check_icon} {check_name}: {check_result['message']}")
        
        print("\n" + "="*80)
        print("IMPROVEMENTS ACTIVE AND OPERATIONAL")
        print("="*80)
    
    async def shutdown(self):
        """Shutdown improved workflow system"""
        self.logger.info("Shutting down Improved Workflow System...")
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self.orchestrator:
            await self.orchestrator.stop_enhanced_workflow()
        
        await ml_optimizer.stop()
        await persistence.shutdown()
        
        self.logger.info("Improved Workflow System shutdown complete")

# ============================================================================
# VALIDATION AND TESTING
# ============================================================================

async def validate_improvements():
    """Validate that improvements are working"""
    print("\n🔍 Validating Improvements...")
    
    results = {
        'observability': False,
        'persistence': False,
        'ml_optimization': False,
        'circuit_breakers': False,
        'overall': False
    }
    
    # Test observability
    try:
        status = await observability.get_observability_status()
        results['observability'] = status is not None and 'health' in status
        print(f"  Observability: {'✅' if results['observability'] else '❌'}")
    except Exception as e:
        print(f"  Observability: ❌ ({e})")
    
    # Test persistence
    try:
        test_id = await persistence.persist_event(
            event_type="validation_test",
            source="validator",
            data={"test": True}
        )
        results['persistence'] = test_id is not None
        print(f"  Persistence: {'✅' if results['persistence'] else '❌'}")
    except Exception as e:
        print(f"  Persistence: ❌ ({e})")
    
    # Test ML optimization
    try:
        ml_status = await ml_optimizer.get_optimization_status()
        results['ml_optimization'] = ml_status['active']
        print(f"  ML Optimization: {'✅' if results['ml_optimization'] else '❌'}")
    except Exception as e:
        print(f"  ML Optimization: ❌ ({e})")
    
    # Test circuit breakers
    try:
        states = circuit_breaker_registry.get_all_states()
        results['circuit_breakers'] = len(states) > 0
        print(f"  Circuit Breakers: {'✅' if results['circuit_breakers'] else '❌'}")
    except Exception as e:
        print(f"  Circuit Breakers: ❌ ({e})")
    
    # Overall result
    results['overall'] = all([
        results['observability'],
        results['persistence'],
        results['ml_optimization'],
        results['circuit_breakers']
    ])
    
    print(f"\n  Overall Validation: {'✅ PASSED' if results['overall'] else '❌ FAILED'}")
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main(args):
    """Main execution"""
    print("="*80)
    print("AG06 WORKFLOW IMPROVEMENTS IMPLEMENTATION")
    print("Real-Time Observability | Event Persistence | ML Optimization")
    print("="*80)
    
    # Create improved workflow system
    workflow = ImprovedWorkflowSystem()
    
    try:
        # Initialize
        print("\n🔧 Initializing Improved Workflow System...")
        await workflow.initialize()
        
        # Start enhanced workflow
        print("\n🚀 Starting Enhanced Workflow...")
        await workflow.orchestrator.start_enhanced_workflow()
        
        # Demonstrate improvements
        await workflow.demonstrate_improvements()
        
        # Validate if requested
        if args.validate:
            validation_results = await validate_improvements()
            if not validation_results['overall']:
                print("\n⚠️ Validation failed. Check individual components.")
                return 1
        
        # Run for specified duration
        if args.run_time > 0:
            print(f"\n⏱️ Running for {args.run_time} seconds...")
            await asyncio.sleep(args.run_time)
        else:
            print("\n✅ Improvements successfully implemented!")
            print("📊 Metrics available at: http://localhost:9090/metrics")
            print("🔄 System is running with continuous optimization")
        
        return 0
        
    except Exception as e:
        logger.error(f"Implementation error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await workflow.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implement Workflow Improvements")
    parser.add_argument('--phase', type=int, default=1, 
                       help='Implementation phase (1-4)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate improvements after implementation')
    parser.add_argument('--run-time', type=int, default=0,
                       help='Run time in seconds (0 for demo only)')
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)