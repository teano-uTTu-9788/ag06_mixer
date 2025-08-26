# AG06 SDK - Enterprise Workflow Integration

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/ag06/ag06-sdk)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/ag06/ag06-sdk)

Enterprise-grade Python SDK for seamless integration with the AG06 production workflow system. Provides comprehensive access to ML analytics, real-time monitoring, workflow orchestration, and enterprise integrations.

## 🚀 Features

### Core Capabilities
- **ML-Powered Analytics** - Predictive insights, anomaly detection, performance forecasting
- **Real-Time Monitoring** - Live dashboards, metrics collection, alert management
- **Workflow Orchestration** - Multi-step process automation with circuit breaker protection
- **Enterprise Integration** - CI/CD pipelines, microservices, data pipelines, auto-scaling

### Advanced Features  
- **Async/Await Support** - Built for high-performance concurrent operations
- **Circuit Breaker Pattern** - Fault tolerance and resilience for production environments
- **Comprehensive Error Handling** - Detailed error reporting and retry logic
- **Flexible Configuration** - Environment-aware settings and authentication
- **Enterprise Logging** - Structured logging with correlation IDs

## 📦 Installation

### Basic Installation
```bash
pip install ag06-sdk
```

### Full Installation (with ML and Web dependencies)
```bash
pip install ag06-sdk[full]
```

### Development Installation
```bash
git clone https://github.com/ag06/ag06-sdk.git
cd ag06-sdk
pip install -e .[dev]
```

### Optional Dependencies
```bash
# ML analytics only
pip install ag06-sdk[ml]

# Web framework features only  
pip install ag06-sdk[web]

# Development tools
pip install ag06-sdk[dev]
```

## 🔧 Quick Start

### Basic Usage
```python
import asyncio
from ag06_sdk import AG06Client

async def main():
    async with AG06Client(base_url="http://localhost:8080") as client:
        # Health check
        health = await client.get_health()
        print(f"System Health: {'✅' if health.success else '❌'}")
        
        # Execute workflow
        result = await client.execute_workflow(
            "data_processing",
            {"input": "sample_data", "format": "json"}
        )
        print(f"Workflow: {'✅' if result.success else '❌'}")

asyncio.run(main())
```

### Specialized Clients
```python
from ag06_sdk import AG06Client, AnalyticsClient, WorkflowClient

async def advanced_usage():
    async with AG06Client() as client:
        # ML Analytics
        analytics = AnalyticsClient(client)
        insights = await analytics.comprehensive_analysis()
        print(f"Generated {len(insights)} ML insights")
        
        # Bulk Workflow Execution
        workflow_client = WorkflowClient(client)
        workflows = [
            {"workflow_type": "validation", "context": {"id": 1}},
            {"workflow_type": "processing", "context": {"id": 2}},
        ]
        results = await workflow_client.bulk_execute(workflows)
        print(f"Executed {len(results)} workflows")

asyncio.run(advanced_usage())
```

### Quick Functions
```python
from ag06_sdk import quick_health_check, quick_workflow_execute, quick_insights

# Quick health check
health = await quick_health_check("http://localhost:8080")
print(f"Healthy: {health['healthy']}")

# Quick workflow execution
result = await quick_workflow_execute("demo_workflow", {"test": True})
print(f"Success: {result['success']}")

# Quick ML insights
insights = await quick_insights("http://localhost:8080")
print(f"Insights: {list(insights.keys())}")
```

## 🏢 Enterprise Integration Examples

### 1. CI/CD Pipeline Integration
```python
from ag06_sdk.integration_examples import CICDIntegration

async def deployment_pipeline():
    cicd = CICDIntegration("http://localhost:8080")
    
    # Pre-deployment validation
    validation = await cicd.pre_deployment_validation()
    if not validation["passed"]:
        raise Exception("Pre-deployment validation failed")
    
    # Deploy application
    # ... your deployment code ...
    
    # Post-deployment monitoring
    monitor_result = await cicd.post_deployment_monitoring("deploy_123")
    print(f"Deployment monitoring: {monitor_result['status']}")

asyncio.run(deployment_pipeline())
```

### 2. Enterprise Monitoring
```python
from ag06_sdk.integration_examples import EnterpriseMonitor

async def continuous_monitoring():
    monitor = EnterpriseMonitor(
        "http://localhost:8080",
        {"check_interval": 300}  # 5 minutes
    )
    
    # Start continuous monitoring (runs indefinitely)
    await monitor.continuous_monitoring()

asyncio.run(continuous_monitoring())
```

### 3. Microservices Orchestration
```python
from ag06_sdk.integration_examples import MicroservicesOrchestrator

async def deploy_microservices():
    services_config = [
        {"name": "api-gateway", "image": "nginx:latest", "replicas": 2},
        {"name": "user-service", "image": "app:v1.0.0", "replicas": 3},
        {"name": "data-service", "image": "postgres:13", "replicas": 1}
    ]
    
    orchestrator = MicroservicesOrchestrator("http://localhost:8080", services_config)
    
    # Deploy service mesh
    deployment = await orchestrator.deploy_service_mesh()
    print(f"Service mesh deployment: {'✅' if deployment['overall_success'] else '❌'}")
    
    # Health check all services
    health = await orchestrator.health_check_all_services()
    print(f"All services healthy: {'✅' if health['overall_healthy'] else '❌'}")

asyncio.run(deploy_microservices())
```

### 4. Auto-Scaling Based on ML Predictions
```python
from ag06_sdk.integration_examples import AutoScalingManager

async def intelligent_scaling():
    scaler = AutoScalingManager("http://localhost:8080")
    
    # Make intelligent scaling decision
    decision = await scaler.intelligent_scaling_decision()
    print(f"Scaling action: {decision['action_taken']['type']}")
    print(f"Reason: {decision['action_taken']['reason']}")

asyncio.run(intelligent_scaling())
```

### 5. Data Pipeline Orchestration
```python
from ag06_sdk.integration_examples import DataPipelineIntegration

async def data_processing():
    pipeline = DataPipelineIntegration("http://localhost:8080")
    
    config = {
        "pipeline_id": "daily_etl_001",
        "data_source": "production_db",
        "destination": "data_warehouse",
        "transformations": ["clean", "validate", "enrich", "aggregate"],
        "quality_rules": ["no_nulls", "valid_dates", "consistent_formats"]
    }
    
    result = await pipeline.orchestrate_data_workflow(config)
    print(f"Pipeline success: {'✅' if result['success'] else '❌'}")
    
    for step_name, step_result in result['steps'].items():
        status = "✅" if step_result['success'] else "❌"
        print(f"  {step_name}: {status}")

asyncio.run(data_processing())
```

## 📊 API Reference

### AG06Client
Main client for AG06 system integration.

#### Methods
- `get_health()` - System health check
- `get_system_status()` - Comprehensive system status
- `execute_workflow(workflow_type, context, priority)` - Execute workflow
- `list_workflows(status, limit)` - List workflows with filtering
- `get_ml_insights(insight_types, time_horizon)` - Get ML predictive insights
- `get_metrics(metric_names, start_time, end_time)` - Retrieve system metrics

### AnalyticsClient
Specialized client for ML analytics operations.

#### Methods
- `comprehensive_analysis()` - Full system analysis
- `anomaly_monitor(callback, interval, threshold)` - Continuous anomaly monitoring

### WorkflowClient  
Specialized client for workflow operations.

#### Methods
- `bulk_execute(workflows, max_concurrent)` - Execute multiple workflows
- `wait_for_completion(workflow_id, timeout, poll_interval)` - Wait for workflow completion

## ⚙️ Configuration

### Environment Variables
```bash
export AG06_API_URL="http://localhost:8080"
export AG06_API_KEY="your_api_key_here" 
export AG06_TIMEOUT=30
export AG06_RETRY_ATTEMPTS=3
export AG06_LOG_LEVEL="INFO"
```

### Client Configuration
```python
from ag06_sdk import AG06Client, LogLevel

client = AG06Client(
    base_url="http://localhost:8080",
    api_key="your_api_key",
    timeout=30,
    retry_attempts=3,
    log_level=LogLevel.INFO
)
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ag06_sdk

# Run specific test file
pytest tests/test_ag06_client.py

# Run integration tests
pytest tests/integration/
```

### Test Configuration
```python
import pytest
from ag06_sdk import AG06Client

@pytest.fixture
async def ag06_client():
    client = AG06Client(base_url="http://localhost:8080")
    yield client
    await client.close()

async def test_health_check(ag06_client):
    health = await ag06_client.get_health()
    assert health.success is True
```

## 📈 Monitoring and Observability

### Built-in Logging
```python
import logging
from ag06_sdk import AG06Client, LogLevel

# Enable debug logging
client = AG06Client(log_level=LogLevel.DEBUG)

# Custom logger configuration
logger = logging.getLogger("ag06_sdk")
logger.setLevel(logging.INFO)
```

### Metrics Collection
```python
# Get system metrics
metrics = await client.get_metrics([
    "cpu_percent",
    "memory_percent", 
    "active_workflows",
    "completed_workflows"
])

if metrics.success:
    data = metrics.data
    print(f"CPU: {data['cpu_percent']}%")
    print(f"Memory: {data['memory_percent']}%")
    print(f"Active Workflows: {data['active_workflows']}")
```

### Error Handling
```python
from ag06_sdk import AG06Client, APIError, ConnectionError

async def robust_workflow():
    try:
        async with AG06Client() as client:
            result = await client.execute_workflow("my_workflow")
            
    except ConnectionError as e:
        print(f"Connection failed: {e}")
        # Implement retry logic or fallback
        
    except APIError as e:
        print(f"API error: {e}")
        # Handle API-specific errors
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle unexpected errors
```

## 🤝 Contributing

### Development Setup
```bash
git clone https://github.com/ag06/ag06-sdk.git
cd ag06-sdk

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black ag06_sdk.py integration_examples.py

# Type checking  
mypy ag06_sdk.py
```

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://docs.ag06.com/sdk](https://docs.ag06.com/sdk)
- **GitHub Issues**: [https://github.com/ag06/ag06-sdk/issues](https://github.com/ag06/ag06-sdk/issues)
- **Email**: support@ag06.com
- **Enterprise Support**: enterprise@ag06.com

## 🗺️ Roadmap

### v1.1.0 (Next Release)
- [ ] GraphQL client support
- [ ] Enhanced error recovery mechanisms
- [ ] Performance optimization tools
- [ ] Advanced caching strategies

### v1.2.0 (Future)
- [ ] Real-time event streaming
- [ ] Advanced ML model deployment
- [ ] Multi-cluster support
- [ ] Enhanced security features

### v2.0.0 (Long-term)
- [ ] Kubernetes native integration
- [ ] Multi-cloud deployment support  
- [ ] Advanced AI agent orchestration
- [ ] Enterprise SSO integration

---

**Built with ❤️ by the AG06 Team**

For enterprise customers: Contact enterprise@ag06.com for dedicated support, custom integrations, and SLA agreements.