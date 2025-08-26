#!/usr/bin/env python3
"""
Specialized Workflow Agent for Production Deployment
Enterprise-grade workflow orchestration with circuit breaker integration
"""

import asyncio
import sys
import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our integrated workflow system components
from integrated_workflow_system import IntegratedWorkflowSystem
from aican_runtime.circuit_breaker import CircuitBreaker

@dataclass
class WorkflowTask:
    task_id: str
    workflow_type: str
    priority: int = 1  # 1=highest, 5=lowest
    context: Dict[str, Any] = None
    steps: List[str] = None
    retry_count: int = 0
    max_retries: int = 3

class SpecializedWorkflowAgent:
    """Production-grade specialized workflow agent"""
    
    def __init__(self, agent_id: str = "workflow_agent_001"):
        self.agent_id = agent_id
        self.workflow_system = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30.0
        )
        self.task_queue: List[WorkflowTask] = []
        self.active_workflows: Dict[str, Any] = {}
        self.completed_workflows: List[str] = []
        self.failed_workflows: List[str] = []
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_duration": 0.0,
            "uptime_start": datetime.now()
        }
        
        print(f"🤖 Specialized Workflow Agent {self.agent_id} initialized")
        print(f"   ✅ Circuit breaker configured (threshold: 5 failures)")
        print(f"   ✅ Task queue ready")
        print(f"   ✅ Performance metrics enabled")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the agent"""
        logger = logging.getLogger(f"workflow_agent_{self.agent_id}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s | WORKFLOW-AGENT-{self.agent_id} | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the workflow system and components"""
        try:
            self.logger.info("Initializing integrated workflow system...")
            self.workflow_system = IntegratedWorkflowSystem()
            
            # Verify all components are healthy
            health = await self.workflow_system.get_system_health()
            if health["overall_status"] != "healthy":
                self.logger.warning(f"System health: {health['overall_status']}")
            
            self.logger.info("✅ Specialized workflow agent fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def queue_workflow(self, task_id: str, workflow_type: str, 
                           context: Dict[str, Any] = None, 
                           steps: List[str] = None,
                           priority: int = 1) -> bool:
        """Queue a workflow task for execution"""
        
        task = WorkflowTask(
            task_id=task_id,
            workflow_type=workflow_type,
            priority=priority,
            context=context or {},
            steps=steps
        )
        
        # Insert based on priority (lower number = higher priority)
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority < existing_task.priority:
                self.task_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task)
        
        self.logger.info(f"📋 Queued workflow {task_id} (priority: {priority}, queue size: {len(self.task_queue)})")
        return True
    
    async def execute_next_workflow(self) -> Optional[Dict[str, Any]]:
        """Execute the next workflow in the queue"""
        
        if not self.task_queue:
            return None
            
        if not self.workflow_system:
            self.logger.error("Workflow system not initialized")
            return None
        
        task = self.task_queue.pop(0)
        
        try:
            # Check circuit breaker before execution
            if self.circuit_breaker.state == "OPEN":
                self.logger.warning(f"Circuit breaker OPEN - queuing {task.task_id} for later")
                self.task_queue.append(task)  # Re-queue for later
                return {"status": "circuit_breaker_open", "task_id": task.task_id}
            
            self.logger.info(f"🚀 Executing workflow {task.task_id} (type: {task.workflow_type})")
            self.active_workflows[task.task_id] = {
                "start_time": datetime.now(),
                "task": task
            }
            
            # Execute workflow with circuit breaker protection
            try:
                result = await self.workflow_system.execute_workflow(
                    task.task_id,
                    task.workflow_type,
                    task.steps,
                    task.context
                )
                
                # Mark circuit breaker success
                self.circuit_breaker.record_success()
                
                # Update metrics
                self.metrics["total_workflows"] += 1
                if result["status"] == "success":
                    self.metrics["successful_workflows"] += 1
                    self.completed_workflows.append(task.task_id)
                    
                    duration = result.get("total_duration_ms", 0)
                    self._update_average_duration(duration)
                    
                    self.logger.info(f"✅ Workflow {task.task_id} completed successfully ({duration}ms)")
                else:
                    self.metrics["failed_workflows"] += 1
                    self.failed_workflows.append(task.task_id)
                    self.logger.error(f"❌ Workflow {task.task_id} failed: {result.get('error', 'Unknown')}")
                
                # Remove from active workflows
                if task.task_id in self.active_workflows:
                    del self.active_workflows[task.task_id]
                
                return result
                
            except Exception as e:
                # Record failure with circuit breaker
                self.circuit_breaker.record_failure()
                
                # Handle retry logic
                task.retry_count += 1
                if task.retry_count <= task.max_retries:
                    self.logger.warning(f"Retrying workflow {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    self.task_queue.insert(0, task)  # Re-queue at front
                    return {"status": "retrying", "task_id": task.task_id, "error": str(e)}
                else:
                    self.logger.error(f"❌ Workflow {task.task_id} failed after {task.max_retries} retries: {e}")
                    self.metrics["failed_workflows"] += 1
                    self.failed_workflows.append(task.task_id)
                    
                    if task.task_id in self.active_workflows:
                        del self.active_workflows[task.task_id]
                    
                    return {"status": "failed", "task_id": task.task_id, "error": str(e)}
                
        except Exception as e:
            self.logger.error(f"❌ Critical error executing workflow {task.task_id}: {e}")
            self.logger.error(traceback.format_exc())
            
            if task.task_id in self.active_workflows:
                del self.active_workflows[task.task_id]
            
            return {"status": "error", "task_id": task.task_id, "error": str(e)}
    
    def _update_average_duration(self, duration_ms: float):
        """Update rolling average duration"""
        current_avg = self.metrics["average_duration"]
        total_successful = self.metrics["successful_workflows"]
        
        if total_successful == 1:
            self.metrics["average_duration"] = duration_ms
        else:
            # Rolling average
            self.metrics["average_duration"] = ((current_avg * (total_successful - 1)) + duration_ms) / total_successful
    
    async def process_workflow_queue(self, max_concurrent: int = 3, 
                                   process_duration: int = 300) -> Dict[str, Any]:
        """Process workflow queue for specified duration"""
        
        self.logger.info(f"🔄 Starting workflow queue processing (max concurrent: {max_concurrent}, duration: {process_duration}s)")
        
        start_time = datetime.now()
        end_time = start_time.timestamp() + process_duration
        processed_count = 0
        
        while datetime.now().timestamp() < end_time:
            # Process up to max_concurrent workflows simultaneously
            active_count = len(self.active_workflows)
            
            if active_count < max_concurrent and self.task_queue:
                # Start new workflow
                result = await self.execute_next_workflow()
                if result:
                    processed_count += 1
            else:
                # Wait a bit before checking again
                await asyncio.sleep(1.0)
            
            # Check circuit breaker state
            if self.circuit_breaker.state == "OPEN":
                self.logger.warning("Circuit breaker is OPEN - waiting for reset...")
                await asyncio.sleep(5.0)
        
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "processing_duration_seconds": processing_duration,
            "workflows_processed": processed_count,
            "queue_remaining": len(self.task_queue),
            "active_workflows": len(self.active_workflows),
            "circuit_breaker_state": self.circuit_breaker.state,
            "metrics": self.metrics.copy()
        }
        
        self.logger.info(f"📊 Queue processing complete: {summary}")
        return summary
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        
        uptime = (datetime.now() - self.metrics["uptime_start"]).total_seconds()
        success_rate = 0.0
        if self.metrics["total_workflows"] > 0:
            success_rate = (self.metrics["successful_workflows"] / self.metrics["total_workflows"]) * 100
        
        # Get system health if available
        system_health = None
        if self.workflow_system:
            try:
                system_health = await self.workflow_system.get_system_health()
            except Exception:
                system_health = {"status": "unknown"}
        
        return {
            "agent_id": self.agent_id,
            "status": "operational" if self.workflow_system else "not_initialized",
            "uptime_seconds": uptime,
            "uptime_human": f"{uptime/3600:.1f} hours",
            "queue": {
                "pending_tasks": len(self.task_queue),
                "active_workflows": len(self.active_workflows),
                "completed_workflows": len(self.completed_workflows),
                "failed_workflows": len(self.failed_workflows)
            },
            "performance": {
                "total_workflows": self.metrics["total_workflows"],
                "success_rate_percent": success_rate,
                "average_duration_ms": self.metrics["average_duration"],
                "workflows_per_hour": (self.metrics["total_workflows"] / max(uptime/3600, 0.1))
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "last_failure_time": self.circuit_breaker.last_failure_time
            },
            "system_health": system_health
        }
    
    async def run_comprehensive_demo(self, num_workflows: int = 20) -> Dict[str, Any]:
        """Run comprehensive demonstration of agent capabilities"""
        
        self.logger.info(f"🎯 Starting comprehensive agent demo with {num_workflows} workflows")
        
        # Queue various workflow types with different priorities
        workflow_types = ["data_processing", "validation", "optimization", "reporting", "cleanup"]
        
        for i in range(num_workflows):
            workflow_type = workflow_types[i % len(workflow_types)]
            priority = (i % 3) + 1  # Vary priority 1-3
            
            context = {
                "demo_id": f"demo_{i:03d}",
                "batch_id": f"batch_{i//5:02d}",
                "cpu_percent": 30.0 + (i * 2),
                "memory_percent": 50.0 + (i * 1.5),
                "complexity_level": (i % 5) + 1
            }
            
            await self.queue_workflow(
                task_id=f"demo_workflow_{i:03d}",
                workflow_type=workflow_type,
                context=context,
                priority=priority
            )
        
        # Process the queued workflows
        processing_summary = await self.process_workflow_queue(
            max_concurrent=4,
            process_duration=60  # 1 minute processing
        )
        
        # Get final agent status
        final_status = await self.get_agent_status()
        
        demo_results = {
            "demo_summary": {
                "workflows_queued": num_workflows,
                "workflows_processed": processing_summary["workflows_processed"],
                "processing_time_seconds": processing_summary["processing_duration_seconds"],
                "success_rate_percent": final_status["performance"]["success_rate_percent"]
            },
            "processing_summary": processing_summary,
            "final_status": final_status
        }
        
        self.logger.info(f"🎉 Demo complete - Success rate: {final_status['performance']['success_rate_percent']:.1f}%")
        
        return demo_results

async def main():
    """Main entry point for specialized workflow agent"""
    print("🚀 Starting Specialized Workflow Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = SpecializedWorkflowAgent()
    
    if not await agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    # Run comprehensive demo
    demo_results = await agent.run_comprehensive_demo(15)
    
    print("\n" + "=" * 60)
    print("📋 Demo Results Summary:")
    print(f"   Workflows queued: {demo_results['demo_summary']['workflows_queued']}")
    print(f"   Workflows processed: {demo_results['demo_summary']['workflows_processed']}")
    print(f"   Success rate: {demo_results['demo_summary']['success_rate_percent']:.1f}%")
    print(f"   Processing time: {demo_results['demo_summary']['processing_time_seconds']:.1f}s")
    
    # Export results
    results_file = "specialized_workflow_agent_results.json"
    with open(results_file, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"📄 Full results: {results_file}")
    print("\n✅ Specialized Workflow Agent demo complete!")

if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import numpy as np
        import psutil
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Installing required packages...")
        import subprocess
        packages = ["numpy", "psutil"]
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"❌ Failed to install {package} - continuing with fallbacks")
    
    asyncio.run(main())