# Usage Examples - Enhanced Workflow System

## Example 1: Basic Code Analysis Workflow

```python
import asyncio
from advanced_workflow_orchestrator import AdvancedWorkflowOrchestrator
from specialized_agent_factory import SpecializedAgentFactory

async def analyze_codebase():
    # Initialize components
    orchestrator = AdvancedWorkflowOrchestrator()
    factory = SpecializedAgentFactory()
    
    # Create specialized agent
    code_agent = await factory.create_agent('code_analyst', 'codebase_analyzer')
    orchestrator.register_agent('code_agent', code_agent, 'specialist')
    
    # Create workflow
    workflow_nodes = [
        {
            'id': 'analyze_solid',
            'name': 'SOLID Compliance Analysis',
            'agent_type': 'specialist',
            'config': {'action': 'solid_audit', 'target': './src'}
        }
    ]
    
    context = orchestrator.create_workflow('code_analysis', workflow_nodes)
    result = await orchestrator.execute_workflow('code_analysis')
    
    print(f"Analysis completed: {result['state']}")
    return result

# Run the analysis
asyncio.run(analyze_codebase())
```

## Example 2: Research-Driven Optimization

```python
from research_implementation_pipeline import ResearchImplementationPipeline

async def discover_and_implement():
    pipeline = ResearchImplementationPipeline()
    
    # Run discovery cycle
    cycle_report = await pipeline.run_discovery_cycle()
    
    print(f"Discovered {cycle_report['findings_discovered']} research findings")
    print(f"Implemented {cycle_report['successful_implementations']} patterns")
    
    return cycle_report

# Run the pipeline
asyncio.run(discover_and_implement())
```

## Example 3: Multi-Agent Workflow Optimization

```python
async def optimize_workflow():
    orchestrator = AdvancedWorkflowOrchestrator()
    factory = SpecializedAgentFactory()
    
    # Create multiple specialized agents
    code_agent = await factory.create_agent('code_analyst', 'code_specialist')
    workflow_agent = await factory.create_agent('workflow_optimizer', 'workflow_specialist')
    
    # Register agents
    orchestrator.register_agent('code_agent', code_agent, 'specialist')
    orchestrator.register_agent('workflow_agent', workflow_agent, 'specialist')
    
    # Create complex workflow with dependencies
    workflow_nodes = [
        {
            'id': 'code_analysis',
            'name': 'Comprehensive Code Analysis',
            'agent_type': 'specialist',
            'config': {'action': 'comprehensive_analysis'}
        },
        {
            'id': 'workflow_optimization',
            'name': 'Workflow Performance Optimization',
            'agent_type': 'specialist',
            'dependencies': ['code_analysis'],
            'config': {'action': 'full_optimization'}
        }
    ]
    
    context = orchestrator.create_workflow('optimization_workflow', workflow_nodes)
    result = await orchestrator.execute_workflow('optimization_workflow')
    
    return result

# Run optimization
asyncio.run(optimize_workflow())
```

## Example 4: Continuous Research Integration

```python
async def continuous_improvement():
    pipeline = ResearchImplementationPipeline()
    
    # Set up continuous pipeline (in production, would run indefinitely)
    config = {
        'discovery_interval_hours': 24,
        'max_implementations': 3,
        'min_priority_threshold': 0.8
    }
    
    pipeline = ResearchImplementationPipeline(config)
    
    # Run single cycle for demo
    report = await pipeline.run_discovery_cycle()
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"Pipeline success rate: {stats.get('success_rate', 0):.1f}%")
    
    return report

# Run continuous improvement
asyncio.run(continuous_improvement())
```

## Best Practices

1. **Error Handling**: Always wrap orchestrator calls in try-catch blocks
2. **Resource Management**: Use circuit breakers for fault tolerance
3. **Monitoring**: Regularly check metrics and performance
4. **Configuration**: Customize agent configurations for specific needs
5. **Logging**: Enable comprehensive logging for debugging
