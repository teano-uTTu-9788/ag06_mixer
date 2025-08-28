#!/usr/bin/env python3
"""
Cloud-Native Advanced Patterns 2025
Latest practices from AWS, Microsoft Azure, Google Cloud Platform
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta
import random

# ============================================================================
# AWS LAMBDA POWERTOOLS - Observability & Serverless Best Practices
# ============================================================================

class LogLevel(Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LambdaContext:
    """AWS Lambda context simulation"""
    function_name: str
    function_version: str
    invoked_function_arn: str
    memory_limit_in_mb: int
    remaining_time_in_millis: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    log_group_name: str = field(default="")
    log_stream_name: str = field(default="")

class PowerToolsLogger:
    """AWS Lambda Powertools structured logger"""
    
    def __init__(self, service_name: str, log_level: LogLevel = LogLevel.INFO):
        self.service_name = service_name
        self.log_level = log_level
        self.correlation_id: Optional[str] = None
        self.cold_start = True
        self.structured_logs: List[Dict] = []
        
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking"""
        self.correlation_id = correlation_id
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Structured logging with context"""
        if self._should_log(level):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level.value,
                "service": self.service_name,
                "message": message,
                "cold_start": self.cold_start,
                **kwargs
            }
            
            if self.correlation_id:
                log_entry["correlation_id"] = self.correlation_id
                
            self.structured_logs.append(log_entry)
            
            # Simulate CloudWatch output
            print(json.dumps(log_entry))
        
        # First log after cold start
        if self.cold_start:
            self.cold_start = False
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level"""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return levels.index(level) >= levels.index(self.log_level)

class PowerToolsMetrics:
    """AWS Lambda Powertools metrics"""
    
    def __init__(self, namespace: str, service: str):
        self.namespace = namespace
        self.service = service
        self.metrics: List[Dict] = []
        self.metadata: Dict[str, str] = {}
    
    def add_metric(self, name: str, value: float, unit: str = "Count", **dimensions):
        """Add a metric"""
        metric = {
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.utcnow().isoformat(),
            "Dimensions": [
                {"Name": "Service", "Value": self.service}
            ] + [{"Name": k, "Value": v} for k, v in dimensions.items()]
        }
        self.metrics.append(metric)
    
    def add_metadata(self, key: str, value: str):
        """Add metadata for metrics context"""
        self.metadata[key] = value
    
    def publish_metrics(self) -> Dict:
        """Publish metrics to CloudWatch (simulated)"""
        cloudwatch_payload = {
            "Namespace": self.namespace,
            "MetricData": self.metrics,
            "Metadata": self.metadata
        }
        
        # Clear metrics after publishing
        self.metrics.clear()
        return cloudwatch_payload

class PowerToolsTracer:
    """AWS Lambda Powertools tracing with X-Ray"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.trace_id = self._generate_trace_id()
        self.segments: List[Dict] = []
        self.current_segment: Optional[Dict] = None
    
    def _generate_trace_id(self) -> str:
        """Generate X-Ray trace ID format"""
        timestamp = hex(int(time.time()))[2:]
        random_part = ''.join(random.choices('0123456789abcdef', k=24))
        return f"1-{timestamp}-{random_part}"
    
    def begin_segment(self, name: str, **annotations):
        """Begin a trace segment"""
        segment = {
            "id": str(uuid.uuid4()).replace('-', '')[:16],
            "name": name,
            "trace_id": self.trace_id,
            "start_time": time.time(),
            "service": {"name": self.service_name},
            "annotations": annotations,
            "metadata": {},
            "subsegments": []
        }
        self.segments.append(segment)
        self.current_segment = segment
        return segment
    
    def end_segment(self, segment: Dict):
        """End a trace segment"""
        segment["end_time"] = time.time()
        segment["duration"] = segment["end_time"] - segment["start_time"]
    
    def begin_subsegment(self, name: str, segment_type: str = "subsegment"):
        """Begin a subsegment for external calls, SQL, etc."""
        if not self.current_segment:
            raise ValueError("No active segment to add subsegment")
        
        subsegment = {
            "id": str(uuid.uuid4()).replace('-', '')[:16],
            "name": name,
            "start_time": time.time(),
            "type": segment_type,
            "namespace": "remote" if segment_type == "subsegment" else "aws"
        }
        
        self.current_segment["subsegments"].append(subsegment)
        return subsegment
    
    def end_subsegment(self, subsegment: Dict):
        """End a subsegment"""
        subsegment["end_time"] = time.time()
        subsegment["duration"] = subsegment["end_time"] - subsegment["start_time"]

class LambdaPowerTools:
    """AWS Lambda Powertools orchestrator"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = PowerToolsLogger(service_name)
        self.metrics = PowerToolsMetrics("MyApp", service_name)
        self.tracer = PowerToolsTracer(service_name)
    
    async def setup_structured_logging(self, config: Dict) -> Dict:
        """Setup structured logging configuration"""
        log_level = getattr(LogLevel, config.get("log_level", "INFO"))
        correlation_id_enabled = config.get("correlation_id", False)
        
        if correlation_id_enabled:
            self.logger.set_correlation_id(str(uuid.uuid4()))
        
        return {
            "logging_configured": True,
            "correlation_id_enabled": correlation_id_enabled,
            "log_level": config.get("log_level", "INFO")
        }
    
    async def collect_custom_metrics(self, config: Dict) -> Dict:
        """Collect custom metrics"""
        namespace = config.get("namespace", "MyApp")
        for metric_name, value in config.get("metrics", {}).items():
            self.metrics.add_metric(metric_name, value)
        
        return {
            "metrics_collected": True,
            "namespace": namespace,
            "metric_count": len(config.get("metrics", {}))
        }
    
    async def enable_xray_tracing(self, config: Dict) -> Dict:
        """Enable X-Ray tracing"""
        service_name = config.get("service_name", self.service_name)
        segment_name = f"{service_name}_segment"
        
        segment = self.tracer.begin_segment(segment_name)
        
        return {
            "tracing_enabled": True,
            "trace_id": self.tracer.trace_id,
            "service_name": service_name
        }
    
    async def process_events(self, events: List[Dict]) -> Dict:
        """Process events from various sources"""
        processed_count = 0
        errors = []
        
        for event in events:
            try:
                # Simulate event processing
                processed_count += 1
            except Exception as e:
                errors.append(str(e))
        
        return {
            "events_processed": processed_count,
            "errors": errors,
            "success_rate": processed_count / len(events) if events else 0
        }
    
    async def manage_parameters(self, config: Dict) -> Dict:
        """Manage SSM parameters"""
        action = config.get("action", "get")
        parameters = config.get("parameters", {})
        
        return {
            "parameters_managed": True,
            "action": action,
            "parameter_count": len(parameters)
        }
    
    async def manage_secrets(self, config: Dict) -> Dict:
        """Manage secrets"""
        action = config.get("action", "get")
        secrets = config.get("secrets", {})
        
        return {
            "secrets_managed": True,
            "action": action,
            "secret_count": len(secrets)
        }
    
    async def handle_api_request(self, request: Dict) -> Dict:
        """Handle API Gateway request"""
        method = request.get("httpMethod", "GET")
        path = request.get("path", "/")
        
        return {
            "request_handled": True,
            "method": method,
            "path": path,
            "status_code": 200
        }
    
    async def optimize_cold_start(self, config: Dict) -> Dict:
        """Optimize cold start performance"""
        optimizations = config.get("optimizations", [])
        
        return {
            "cold_start_optimized": True,
            "optimizations_applied": optimizations,
            "estimated_improvement": "30%"
        }
    
    def lambda_handler_decorator(self, func):
        """Decorator for Lambda handlers"""
        async def wrapper(event: Dict, context: LambdaContext):
            # Set up correlation ID
            correlation_id = event.get('correlation_id', str(uuid.uuid4()))
            self.logger.set_correlation_id(correlation_id)
            
            # Begin tracing
            segment = self.tracer.begin_segment("lambda_handler", 
                                                function_name=context.function_name,
                                                correlation_id=correlation_id)
            
            try:
                self.logger.log(LogLevel.INFO, "Lambda execution started", 
                               function_name=context.function_name,
                               request_id=context.request_id)
                
                # Add metrics
                self.metrics.add_metric("Invocations", 1, "Count")
                self.metrics.add_metric("ColdStart", 1 if self.logger.cold_start else 0, "Count")
                
                # Execute function
                start_time = time.time()
                result = await func(event, context)
                execution_time = (time.time() - start_time) * 1000
                
                # Record metrics
                self.metrics.add_metric("Duration", execution_time, "Milliseconds")
                self.metrics.add_metric("Success", 1, "Count")
                
                self.logger.log(LogLevel.INFO, "Lambda execution completed",
                               execution_time_ms=execution_time)
                
                return result
                
            except Exception as e:
                self.metrics.add_metric("Errors", 1, "Count")
                self.logger.log(LogLevel.ERROR, "Lambda execution failed",
                               error=str(e), error_type=type(e).__name__)
                raise
            finally:
                self.tracer.end_segment(segment)
                self.metrics.publish_metrics()
        
        return wrapper

# ============================================================================
# MICROSOFT AZURE DURABLE FUNCTIONS - Serverless Workflows
# ============================================================================

class OrchestrationStatus(Enum):
    """Durable function orchestration status"""
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    TERMINATED = "Terminated"

@dataclass
class DurableOrchestrationContext:
    """Azure Durable Functions orchestration context"""
    instance_id: str
    is_replaying: bool = False
    current_utc_datetime: datetime = field(default_factory=datetime.utcnow)
    history: List[Dict] = field(default_factory=list)
    
class ActivityFunction:
    """Azure Durable Functions activity"""
    
    def __init__(self, name: str, handler: callable):
        self.name = name
        self.handler = handler
        self.execution_history: List[Dict] = []
    
    async def execute(self, input_data: Any) -> Any:
        """Execute activity function"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            result = await self.handler(input_data)
            execution_time = time.time() - start_time
            
            execution_record = {
                "execution_id": execution_id,
                "input": input_data,
                "result": result,
                "execution_time": execution_time,
                "status": "Completed",
                "timestamp": datetime.utcnow().isoformat()
            }
            self.execution_history.append(execution_record)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_record = {
                "execution_id": execution_id,
                "input": input_data,
                "error": str(e),
                "execution_time": execution_time,
                "status": "Failed",
                "timestamp": datetime.utcnow().isoformat()
            }
            self.execution_history.append(execution_record)
            raise

class DurableOrchestrator:
    """Azure Durable Functions orchestrator"""
    
    def __init__(self):
        self.activity_functions: Dict[str, ActivityFunction] = {}
        self.orchestrations: Dict[str, Dict] = {}
        
        # Register sample activities
        self._register_sample_activities()
    
    def _register_sample_activities(self):
        """Register sample activity functions"""
        
        async def process_payment_activity(payment_data: Dict) -> Dict:
            """Process payment activity"""
            await asyncio.sleep(0.2)  # Simulate payment processing
            return {
                "payment_id": str(uuid.uuid4()),
                "status": "processed",
                "amount": payment_data.get("amount", 0)
            }
        
        async def send_email_activity(email_data: Dict) -> Dict:
            """Send email activity"""
            await asyncio.sleep(0.1)  # Simulate email sending
            return {
                "email_id": str(uuid.uuid4()),
                "status": "sent",
                "recipient": email_data.get("recipient", "")
            }
        
        async def update_inventory_activity(inventory_data: Dict) -> Dict:
            """Update inventory activity"""
            await asyncio.sleep(0.05)  # Simulate inventory update
            return {
                "inventory_id": str(uuid.uuid4()),
                "status": "updated",
                "quantity": inventory_data.get("quantity", 0)
            }
        
        self.activity_functions["ProcessPayment"] = ActivityFunction("ProcessPayment", process_payment_activity)
        self.activity_functions["SendEmail"] = ActivityFunction("SendEmail", send_email_activity)
        self.activity_functions["UpdateInventory"] = ActivityFunction("UpdateInventory", update_inventory_activity)
    
    async def start_orchestration(self, orchestrator_function: str, input_data: Any) -> str:
        """Start a durable orchestration"""
        instance_id = str(uuid.uuid4())
        
        orchestration = {
            "instance_id": instance_id,
            "orchestrator_function": orchestrator_function,
            "input": input_data,
            "status": OrchestrationStatus.RUNNING,
            "start_time": datetime.utcnow(),
            "history": [],
            "output": None
        }
        
        self.orchestrations[instance_id] = orchestration
        
        # Start orchestration execution
        asyncio.create_task(self._execute_orchestration(instance_id))
        
        return instance_id
    
    async def _execute_orchestration(self, instance_id: str):
        """Execute orchestration workflow"""
        orchestration = self.orchestrations[instance_id]
        
        try:
            # Create orchestration context
            context = DurableOrchestrationContext(instance_id=instance_id)
            
            # Execute orchestrator function
            if orchestration["orchestrator_function"] == "order_processing":
                result = await self._order_processing_orchestrator(context, orchestration["input"])
            else:
                result = {"error": "Unknown orchestrator function"}
            
            # Update orchestration
            orchestration["status"] = OrchestrationStatus.COMPLETED
            orchestration["output"] = result
            orchestration["end_time"] = datetime.utcnow()
            
        except Exception as e:
            orchestration["status"] = OrchestrationStatus.FAILED
            orchestration["output"] = {"error": str(e)}
            orchestration["end_time"] = datetime.utcnow()
    
    async def _order_processing_orchestrator(self, context: DurableOrchestrationContext, input_data: Dict) -> Dict:
        """Sample order processing orchestrator"""
        
        # Step 1: Process payment
        payment_result = await self._call_activity("ProcessPayment", {
            "amount": input_data.get("amount", 0),
            "payment_method": input_data.get("payment_method", "card")
        })
        
        context.history.append({
            "activity": "ProcessPayment",
            "result": payment_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Step 2: Update inventory
        inventory_result = await self._call_activity("UpdateInventory", {
            "product_id": input_data.get("product_id", ""),
            "quantity": input_data.get("quantity", 1)
        })
        
        context.history.append({
            "activity": "UpdateInventory", 
            "result": inventory_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Step 3: Send confirmation email
        email_result = await self._call_activity("SendEmail", {
            "recipient": input_data.get("customer_email", ""),
            "template": "order_confirmation",
            "data": {"payment_id": payment_result["payment_id"]}
        })
        
        context.history.append({
            "activity": "SendEmail",
            "result": email_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "order_id": str(uuid.uuid4()),
            "payment_id": payment_result["payment_id"],
            "inventory_id": inventory_result["inventory_id"],
            "email_id": email_result["email_id"],
            "status": "completed"
        }
    
    async def _call_activity(self, activity_name: str, input_data: Any) -> Any:
        """Call an activity function"""
        if activity_name not in self.activity_functions:
            raise ValueError(f"Activity function not found: {activity_name}")
        
        activity = self.activity_functions[activity_name]
        return await activity.execute(input_data)
    
    def get_orchestration_status(self, instance_id: str) -> Optional[Dict]:
        """Get orchestration status"""
        return self.orchestrations.get(instance_id)

# ============================================================================
# GOOGLE CLOUD RUN - Serverless Containers with Knative
# ============================================================================

class RevisionTrafficAllocation:
    """Traffic allocation for Cloud Run revisions"""
    
    def __init__(self):
        self.allocations: Dict[str, int] = {}  # revision -> percentage
        
    def set_traffic(self, revision: str, percentage: int):
        """Set traffic percentage for revision"""
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.allocations[revision] = percentage
    
    def normalize_traffic(self):
        """Normalize traffic to sum to 100%"""
        total = sum(self.allocations.values())
        if total == 0:
            return
        
        for revision in self.allocations:
            self.allocations[revision] = int((self.allocations[revision] / total) * 100)

@dataclass
class CloudRunRevision:
    """Cloud Run service revision"""
    name: str
    image: str
    cpu: str = "1"
    memory: str = "512Mi"
    min_instances: int = 0
    max_instances: int = 1000
    concurrency: int = 1000
    timeout: int = 300
    env_vars: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "READY"

class KnativeService:
    """Google Cloud Run service with Knative patterns"""
    
    def __init__(self, service_name: str, namespace: str = "default"):
        self.service_name = service_name
        self.namespace = namespace
        self.revisions: Dict[str, CloudRunRevision] = {}
        self.traffic_allocation = RevisionTrafficAllocation()
        self.request_metrics: List[Dict] = []
        self.auto_scaling_config = {
            "min_instances": 0,
            "max_instances": 100,
            "target_concurrency": 100,
            "scale_down_delay": 30  # seconds
        }
    
    def create_revision(self, revision_name: str, image: str, **config) -> CloudRunRevision:
        """Create a new revision"""
        revision = CloudRunRevision(
            name=revision_name,
            image=image,
            **config
        )
        
        self.revisions[revision_name] = revision
        
        # If first revision, allocate 100% traffic
        if len(self.revisions) == 1:
            self.traffic_allocation.set_traffic(revision_name, 100)
        
        return revision
    
    def update_traffic_allocation(self, allocations: Dict[str, int]):
        """Update traffic allocation across revisions"""
        for revision, percentage in allocations.items():
            if revision not in self.revisions:
                raise ValueError(f"Revision not found: {revision}")
            self.traffic_allocation.set_traffic(revision, percentage)
        
        self.traffic_allocation.normalize_traffic()
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming request with traffic routing"""
        
        # Route request based on traffic allocation
        selected_revision = self._route_request()
        
        if not selected_revision:
            return {"error": "No available revisions", "status": 503}
        
        # Process request
        start_time = time.time()
        
        try:
            # Simulate request processing
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
            response = {
                "status": 200,
                "revision": selected_revision.name,
                "service": self.service_name,
                "response": f"Processed by {selected_revision.name}",
                "processing_time": (time.time() - start_time) * 1000
            }
            
            # Record metrics
            self.request_metrics.append({
                "revision": selected_revision.name,
                "processing_time": response["processing_time"],
                "status": 200,
                "timestamp": time.time()
            })
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.request_metrics.append({
                "revision": selected_revision.name,
                "processing_time": (time.time() - start_time) * 1000,
                "status": 500,
                "error": str(e),
                "timestamp": time.time()
            })
            
            return {"error": str(e), "status": 500}
    
    def _route_request(self) -> Optional[CloudRunRevision]:
        """Route request based on traffic allocation"""
        if not self.traffic_allocation.allocations:
            return None
        
        # Simple weighted random selection
        random_num = random.randint(1, 100)
        cumulative = 0
        
        for revision_name, percentage in self.traffic_allocation.allocations.items():
            cumulative += percentage
            if random_num <= cumulative:
                return self.revisions.get(revision_name)
        
        # Fallback to first revision
        return list(self.revisions.values())[0] if self.revisions else None
    
    def get_metrics_summary(self) -> Dict:
        """Get service metrics summary"""
        if not self.request_metrics:
            return {"total_requests": 0}
        
        total_requests = len(self.request_metrics)
        avg_processing_time = sum(m["processing_time"] for m in self.request_metrics) / total_requests
        success_rate = len([m for m in self.request_metrics if m["status"] == 200]) / total_requests * 100
        
        # Per-revision metrics
        revision_metrics = defaultdict(lambda: {"requests": 0, "avg_time": 0, "success_rate": 0})
        for metric in self.request_metrics:
            rev = metric["revision"]
            revision_metrics[rev]["requests"] += 1
        
        for rev, metrics in revision_metrics.items():
            rev_requests = [m for m in self.request_metrics if m["revision"] == rev]
            metrics["avg_time"] = sum(m["processing_time"] for m in rev_requests) / len(rev_requests)
            metrics["success_rate"] = len([m for m in rev_requests if m["status"] == 200]) / len(rev_requests) * 100
        
        return {
            "total_requests": total_requests,
            "avg_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "revision_metrics": dict(revision_metrics),
            "traffic_allocation": self.traffic_allocation.allocations
        }

# ============================================================================
# CLOUD-NATIVE ORCHESTRATOR
# ============================================================================

class CloudNativeOrchestrator:
    """Orchestrates all cloud-native advanced patterns"""
    
    def __init__(self):
        self.lambda_powertools = LambdaPowerTools("demo-service")
        self.durable_orchestrator = DurableOrchestrator()
        self.knative_services: Dict[str, KnativeService] = {}
        self.metrics = defaultdict(int)
        
        # Initialize Cloud Run services
        self._init_knative_services()
    
    def _init_knative_services(self):
        """Initialize Knative services"""
        # Create sample service
        service = KnativeService("demo-api", "production")
        
        # Create revisions
        service.create_revision("demo-api-v1", "gcr.io/demo/api:v1", 
                               cpu="0.5", memory="256Mi")
        service.create_revision("demo-api-v2", "gcr.io/demo/api:v2",
                               cpu="1", memory="512Mi")
        
        # Set traffic allocation (90% v1, 10% v2 for canary)
        service.update_traffic_allocation({"demo-api-v1": 90, "demo-api-v2": 10})
        
        self.knative_services["demo-api"] = service
    
    async def demonstrate_lambda_powertools(self):
        """Demonstrate AWS Lambda Powertools"""
        print("\n⚡ AWS Lambda Powertools")
        print("-" * 50)
        
        # Create a sample Lambda function
        @self.lambda_powertools.lambda_handler_decorator
        async def sample_lambda(event: Dict, context: LambdaContext) -> Dict:
            """Sample Lambda function with Powertools"""
            
            # Use logger
            self.lambda_powertools.logger.log(LogLevel.INFO, "Processing request", 
                                             event_type=event.get("type", "unknown"))
            
            # Add custom metrics
            self.lambda_powertools.metrics.add_metric("CustomMetric", 42, "Count", 
                                                     Environment="production")
            
            # Add tracing subsegment
            subsegment = self.lambda_powertools.tracer.begin_subsegment("external-api-call")
            await asyncio.sleep(0.05)  # Simulate external call
            self.lambda_powertools.tracer.end_subsegment(subsegment)
            
            return {"status": "success", "processed": event.get("data", "")}
        
        # Execute Lambda
        test_context = LambdaContext(
            function_name="demo-function",
            function_version="1",
            invoked_function_arn="arn:aws:lambda:us-east-1:123456789:function:demo-function",
            memory_limit_in_mb=512,
            remaining_time_in_millis=30000
        )
        
        test_event = {"type": "test", "data": "sample data"}
        result = await sample_lambda(test_event, test_context)
        
        print(f"Lambda result: {result}")
        print(f"Structured logs: {len(self.lambda_powertools.logger.structured_logs)}")
        print(f"Metrics recorded: {len(self.lambda_powertools.metrics.metrics)}")
        print(f"Trace segments: {len(self.lambda_powertools.tracer.segments)}")
        print(f"Trace ID: {self.lambda_powertools.tracer.trace_id}")
        
        # Show sample log entry
        if self.lambda_powertools.logger.structured_logs:
            print("Sample log entry:")
            print(json.dumps(self.lambda_powertools.logger.structured_logs[-1], indent=2))
        
        self.metrics['lambda_invocations'] += 1
    
    async def demonstrate_azure_durable_functions(self):
        """Demonstrate Azure Durable Functions"""
        print("\n🔄 Azure Durable Functions")
        print("-" * 50)
        
        # Start orchestration
        order_data = {
            "customer_email": "customer@example.com",
            "product_id": "product-123",
            "quantity": 2,
            "amount": 99.99,
            "payment_method": "card"
        }
        
        instance_id = await self.durable_orchestrator.start_orchestration(
            "order_processing", order_data
        )
        
        print(f"Started orchestration: {instance_id}")
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Get status
        status = self.durable_orchestrator.get_orchestration_status(instance_id)
        if status:
            print(f"Status: {status['status'].value}")
            print(f"Activities executed: {len(status.get('history', []))}")
            
            if status.get('output'):
                print(f"Output: {json.dumps(status['output'], indent=2)}")
        
        # Show activity execution stats
        print(f"\nActivity Functions:")
        for name, activity in self.durable_orchestrator.activity_functions.items():
            executions = len(activity.execution_history)
            if executions > 0:
                avg_time = sum(e["execution_time"] for e in activity.execution_history) / executions
                print(f"  {name}: {executions} executions, avg {avg_time:.3f}s")
        
        self.metrics['durable_orchestrations'] += 1
    
    async def demonstrate_cloud_run_knative(self):
        """Demonstrate Google Cloud Run with Knative"""
        print("\n🚀 Google Cloud Run (Knative)")
        print("-" * 50)
        
        service = self.knative_services["demo-api"]
        
        # Send multiple requests to demonstrate traffic splitting
        print("Sending requests to demonstrate traffic splitting...")
        
        for i in range(10):
            request = {"request_id": i, "data": f"request-{i}"}
            response = await service.handle_request(request)
            if i < 3:  # Show first 3 responses
                print(f"Request {i}: {response['revision']} (status: {response['status']})")
        
        # Get metrics
        metrics = service.get_metrics_summary()
        print(f"\nService Metrics:")
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Average processing time: {metrics['avg_processing_time']:.2f}ms")
        print(f"  Success rate: {metrics['success_rate']:.1f}%")
        
        print(f"\nTraffic Allocation:")
        for revision, percentage in metrics['traffic_allocation'].items():
            print(f"  {revision}: {percentage}%")
        
        print(f"\nRevision Metrics:")
        for revision, rev_metrics in metrics['revision_metrics'].items():
            print(f"  {revision}: {rev_metrics['requests']} requests, "
                  f"{rev_metrics['avg_time']:.2f}ms avg, "
                  f"{rev_metrics['success_rate']:.1f}% success")
        
        # Demonstrate canary deployment
        print(f"\nCanary Deployment Demo:")
        print("Updating traffic to 50/50 split...")
        service.update_traffic_allocation({"demo-api-v1": 50, "demo-api-v2": 50})
        
        # Send more requests
        for i in range(5):
            request = {"request_id": i + 10, "data": f"canary-{i}"}
            await service.handle_request(request)
        
        updated_metrics = service.get_metrics_summary()
        print(f"Updated traffic allocation: {updated_metrics['traffic_allocation']}")
        
        self.metrics['knative_requests'] += updated_metrics['total_requests']
    
    def get_comprehensive_metrics(self) -> Dict:
        """Get comprehensive cloud-native metrics"""
        return {
            "cloud_native_2025": {
                "aws_lambda_powertools": {
                    "invocations": self.metrics['lambda_invocations'],
                    "structured_logs": len(self.lambda_powertools.logger.structured_logs),
                    "metrics_recorded": len(self.lambda_powertools.metrics.metrics),
                    "trace_segments": len(self.lambda_powertools.tracer.segments),
                    "cold_starts": sum(1 for log in self.lambda_powertools.logger.structured_logs 
                                     if log.get("cold_start", False))
                },
                "azure_durable_functions": {
                    "orchestrations": self.metrics['durable_orchestrations'],
                    "activity_functions": len(self.durable_orchestrator.activity_functions),
                    "total_orchestrations": len(self.durable_orchestrator.orchestrations),
                    "activity_executions": sum(
                        len(activity.execution_history) 
                        for activity in self.durable_orchestrator.activity_functions.values()
                    )
                },
                "google_cloud_run": {
                    "services": len(self.knative_services),
                    "total_requests": self.metrics['knative_requests'],
                    "revisions": sum(len(service.revisions) for service in self.knative_services.values()),
                    "average_processing_time": sum(
                        service.get_metrics_summary().get('avg_processing_time', 0)
                        for service in self.knative_services.values()
                    ) / len(self.knative_services) if self.knative_services else 0
                }
            }
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main demonstration of cloud-native patterns"""
    orchestrator = CloudNativeOrchestrator()
    
    print("☁️ CLOUD-NATIVE ADVANCED PATTERNS 2025")
    print("Latest practices from AWS, Microsoft Azure, Google Cloud Platform")
    print("=" * 70)
    
    # Demonstrate all patterns
    await orchestrator.demonstrate_lambda_powertools()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_azure_durable_functions()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_cloud_run_knative()
    
    # Show comprehensive metrics
    print("\n📊 COMPREHENSIVE CLOUD-NATIVE METRICS")
    print("=" * 70)
    metrics = orchestrator.get_comprehensive_metrics()
    print(json.dumps(metrics, indent=2))
    
    return orchestrator

if __name__ == "__main__":
    asyncio.run(main())