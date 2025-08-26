#!/usr/bin/env python3
"""
AG06 Integration Examples
Real-world integration patterns and use cases for the AG06 SDK
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from ag06_sdk import AG06Client, WorkflowClient, AnalyticsClient, quick_health_check

# Example 1: Enterprise Monitoring Dashboard Integration
class EnterpriseMonitor:
    """Integration example for enterprise monitoring systems"""
    
    def __init__(self, ag06_url: str, monitoring_config: Dict[str, Any]):
        self.ag06_url = ag06_url
        self.config = monitoring_config
        self.alerts_sent = []
        
    async def continuous_monitoring(self):
        """Continuous system monitoring with alerting"""
        print("🔍 Starting Enterprise Monitoring Integration")
        
        async with AG06Client(base_url=self.ag06_url) as client:
            analytics = AnalyticsClient(client)
            
            while True:
                try:
                    # Get comprehensive analysis
                    analysis = await analytics.comprehensive_analysis()
                    
                    # Check for anomalies
                    if 'anomalies' in analysis and analysis['anomalies'].success:
                        anomaly_data = analysis['anomalies'].data
                        if anomaly_data.get('is_anomaly', False):
                            await self._send_alert("ANOMALY_DETECTED", anomaly_data)
                    
                    # Check system health
                    health = await client.get_health()
                    if not health.success:
                        await self._send_alert("SYSTEM_UNHEALTHY", health.data)
                    
                    # Check resource optimization opportunities
                    if 'optimization' in analysis and analysis['optimization'].success:
                        opt_data = analysis['optimization'].data
                        if opt_data.get('cost_savings_percent', 0) > 10:
                            await self._send_alert("OPTIMIZATION_OPPORTUNITY", opt_data)
                    
                    await asyncio.sleep(self.config.get('check_interval', 300))  # 5 minutes
                    
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """Send alert to enterprise monitoring system"""
        alert = {
            "type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "severity": self._determine_severity(alert_type, data),
            "data": data,
            "source": "AG06_SDK"
        }
        
        self.alerts_sent.append(alert)
        print(f"🚨 Alert: {alert_type} - {alert['severity']}")
        
        # In real implementation, this would send to:
        # - PagerDuty, OpsGenie, or similar
        # - Slack/Teams notifications
        # - Email alerts
        # - SIEM systems
    
    def _determine_severity(self, alert_type: str, data: Dict) -> str:
        """Determine alert severity based on type and data"""
        severity_map = {
            "ANOMALY_DETECTED": "HIGH",
            "SYSTEM_UNHEALTHY": "CRITICAL", 
            "OPTIMIZATION_OPPORTUNITY": "LOW"
        }
        return severity_map.get(alert_type, "MEDIUM")


# Example 2: CI/CD Pipeline Integration
class CICDIntegration:
    """Integration with CI/CD pipelines for automated testing"""
    
    def __init__(self, ag06_url: str):
        self.ag06_url = ag06_url
        
    async def pre_deployment_validation(self) -> Dict[str, Any]:
        """Run pre-deployment validation checks"""
        print("🔧 Running Pre-Deployment Validation")
        
        validation_results = {
            "passed": False,
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        async with AG06Client(base_url=self.ag06_url) as client:
            # 1. Health check
            health = await client.get_health()
            validation_results["checks"]["health"] = {
                "passed": health.success,
                "data": health.data
            }
            
            # 2. System status validation
            status = await client.get_system_status()
            validation_results["checks"]["system_status"] = {
                "passed": status.success,
                "data": status.data
            }
            
            # 3. Run validation workflow
            workflow_result = await client.execute_workflow(
                "pre_deployment_validation",
                {"validation_type": "full", "environment": "production"}
            )
            validation_results["checks"]["validation_workflow"] = {
                "passed": workflow_result.success,
                "data": workflow_result.data
            }
            
            # 4. Performance baseline check
            forecast = await client.get_performance_forecast()
            performance_ok = True
            if forecast.success and forecast.data:
                predicted_load = forecast.data.get('predicted_cpu_percent', 0)
                if predicted_load > 80:  # High load predicted
                    performance_ok = False
            
            validation_results["checks"]["performance_forecast"] = {
                "passed": performance_ok,
                "predicted_load": forecast.data if forecast.success else None
            }
        
        # Determine overall pass/fail
        all_checks_passed = all(
            check["passed"] for check in validation_results["checks"].values()
        )
        validation_results["passed"] = all_checks_passed
        
        print(f"✅ Validation: {'PASSED' if all_checks_passed else 'FAILED'}")
        return validation_results
    
    async def post_deployment_monitoring(self, deployment_id: str):
        """Monitor system after deployment"""
        print(f"📊 Post-Deployment Monitoring for {deployment_id}")
        
        monitoring_duration = 300  # 5 minutes
        start_time = datetime.now()
        
        async with AG06Client(base_url=self.ag06_url) as client:
            analytics = AnalyticsClient(client)
            
            while (datetime.now() - start_time).seconds < monitoring_duration:
                # Monitor for anomalies during deployment
                analysis = await analytics.comprehensive_analysis()
                
                if 'anomalies' in analysis and analysis['anomalies'].success:
                    anomaly_data = analysis['anomalies'].data
                    if anomaly_data.get('is_anomaly', False):
                        print(f"⚠️ Post-deployment anomaly detected: {anomaly_data}")
                        return {"status": "ANOMALY_DETECTED", "data": anomaly_data}
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        print("✅ Post-deployment monitoring completed successfully")
        return {"status": "SUCCESS", "monitoring_duration": monitoring_duration}


# Example 3: Data Pipeline Integration
class DataPipelineIntegration:
    """Integration with data processing pipelines"""
    
    def __init__(self, ag06_url: str):
        self.ag06_url = ag06_url
        
    async def orchestrate_data_workflow(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multi-step data processing workflow"""
        print("📊 Orchestrating Data Pipeline")
        
        pipeline_results = {
            "pipeline_id": pipeline_config.get("pipeline_id"),
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "success": False
        }
        
        async with AG06Client(base_url=self.ag06_url) as client:
            workflow_client = WorkflowClient(client)
            
            # Step 1: Data Extraction
            extract_result = await client.execute_workflow(
                "data_extraction",
                {
                    "source": pipeline_config.get("data_source"),
                    "format": pipeline_config.get("data_format", "json"),
                    "batch_size": pipeline_config.get("batch_size", 1000)
                }
            )
            pipeline_results["steps"]["extraction"] = {
                "success": extract_result.success,
                "data": extract_result.data
            }
            
            if not extract_result.success:
                return pipeline_results
            
            # Step 2: Data Transformation
            transform_result = await client.execute_workflow(
                "data_transformation",
                {
                    "transformation_rules": pipeline_config.get("transformations", []),
                    "quality_checks": True,
                    "validation_enabled": True
                }
            )
            pipeline_results["steps"]["transformation"] = {
                "success": transform_result.success,
                "data": transform_result.data
            }
            
            if not transform_result.success:
                return pipeline_results
            
            # Step 3: Data Loading
            load_result = await client.execute_workflow(
                "data_loading",
                {
                    "destination": pipeline_config.get("destination"),
                    "load_strategy": pipeline_config.get("load_strategy", "incremental"),
                    "backup_enabled": True
                }
            )
            pipeline_results["steps"]["loading"] = {
                "success": load_result.success,
                "data": load_result.data
            }
            
            # Step 4: Quality Validation
            validation_result = await client.execute_workflow(
                "data_quality_validation",
                {
                    "validation_rules": pipeline_config.get("quality_rules", []),
                    "threshold": 95  # 95% quality threshold
                }
            )
            pipeline_results["steps"]["validation"] = {
                "success": validation_result.success,
                "data": validation_result.data
            }
        
        # Determine overall success
        all_steps_success = all(
            step["success"] for step in pipeline_results["steps"].values()
        )
        pipeline_results["success"] = all_steps_success
        pipeline_results["end_time"] = datetime.now().isoformat()
        
        print(f"✅ Data Pipeline: {'SUCCESS' if all_steps_success else 'FAILED'}")
        return pipeline_results


# Example 4: Microservices Integration
class MicroservicesOrchestrator:
    """Integration for microservices orchestration"""
    
    def __init__(self, ag06_url: str, services_config: List[Dict[str, Any]]):
        self.ag06_url = ag06_url
        self.services = services_config
        
    async def deploy_service_mesh(self) -> Dict[str, Any]:
        """Deploy and configure service mesh"""
        print("🕸️ Deploying Service Mesh")
        
        deployment_results = {
            "deployment_id": f"mesh_{int(datetime.now().timestamp())}",
            "start_time": datetime.now().isoformat(),
            "services": {},
            "overall_success": False
        }
        
        async with AG06Client(base_url=self.ag06_url) as client:
            workflow_client = WorkflowClient(client)
            
            # Prepare service deployment configurations
            service_workflows = []
            for service in self.services:
                service_workflows.append({
                    "workflow_type": "microservice_deployment",
                    "context": {
                        "service_name": service["name"],
                        "image": service["image"],
                        "replicas": service.get("replicas", 2),
                        "resources": service.get("resources", {}),
                        "health_check": service.get("health_check", "/health"),
                        "environment": service.get("environment", {})
                    },
                    "priority": service.get("priority", 2)
                })
            
            # Execute deployments concurrently
            results = await workflow_client.bulk_execute(
                service_workflows,
                max_concurrent=3
            )
            
            # Process results
            for service, result in zip(self.services, results):
                if hasattr(result, 'success'):  # Valid APIResponse
                    deployment_results["services"][service["name"]] = {
                        "success": result.success,
                        "data": result.data,
                        "error": result.error
                    }
                else:  # Exception occurred
                    deployment_results["services"][service["name"]] = {
                        "success": False,
                        "error": str(result)
                    }
            
            # Service mesh configuration
            mesh_config_result = await client.execute_workflow(
                "service_mesh_configuration",
                {
                    "services": [s["name"] for s in self.services],
                    "load_balancer": "round_robin",
                    "circuit_breaker_enabled": True,
                    "observability_enabled": True
                }
            )
            deployment_results["mesh_configuration"] = {
                "success": mesh_config_result.success,
                "data": mesh_config_result.data
            }
        
        # Determine overall success
        services_success = all(
            svc["success"] for svc in deployment_results["services"].values()
        )
        mesh_success = deployment_results["mesh_configuration"]["success"]
        deployment_results["overall_success"] = services_success and mesh_success
        
        deployment_results["end_time"] = datetime.now().isoformat()
        
        print(f"✅ Service Mesh: {'SUCCESS' if deployment_results['overall_success'] else 'FAILED'}")
        return deployment_results
    
    async def health_check_all_services(self) -> Dict[str, Any]:
        """Health check all deployed services"""
        print("🏥 Health Checking All Services")
        
        health_results = {
            "check_time": datetime.now().isoformat(),
            "services": {},
            "overall_healthy": False
        }
        
        async with AG06Client(base_url=self.ag06_url) as client:
            for service in self.services:
                service_health = await client.execute_workflow(
                    "service_health_check",
                    {"service_name": service["name"]}
                )
                
                health_results["services"][service["name"]] = {
                    "healthy": service_health.success,
                    "data": service_health.data,
                    "response_time_ms": service_health.data.get("response_time_ms") if service_health.data else None
                }
        
        # Determine overall health
        all_healthy = all(
            svc["healthy"] for svc in health_results["services"].values()
        )
        health_results["overall_healthy"] = all_healthy
        
        print(f"🏥 Services Health: {'✅ HEALTHY' if all_healthy else '❌ UNHEALTHY'}")
        return health_results


# Example 5: Auto-scaling Integration
class AutoScalingManager:
    """Integration for auto-scaling based on ML predictions"""
    
    def __init__(self, ag06_url: str):
        self.ag06_url = ag06_url
        
    async def intelligent_scaling_decision(self) -> Dict[str, Any]:
        """Make intelligent scaling decisions based on ML insights"""
        print("⚖️ Making Intelligent Scaling Decision")
        
        scaling_decision = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": {},
            "action_taken": None,
            "confidence": 0.0
        }
        
        async with AG06Client(base_url=self.ag06_url) as client:
            analytics = AnalyticsClient(client)
            
            # Get comprehensive analysis
            analysis = await analytics.comprehensive_analysis()
            
            # Analyze performance forecast
            if 'forecast' in analysis and analysis['forecast'].success:
                forecast_data = analysis['forecast'].data
                predicted_load = forecast_data.get('predicted_cpu_percent', 0)
                predicted_memory = forecast_data.get('predicted_memory_percent', 0)
                
                scaling_decision["recommendations"]["cpu_forecast"] = {
                    "predicted_load": predicted_load,
                    "recommendation": self._get_cpu_recommendation(predicted_load)
                }
                
                scaling_decision["recommendations"]["memory_forecast"] = {
                    "predicted_usage": predicted_memory,
                    "recommendation": self._get_memory_recommendation(predicted_memory)
                }
            
            # Analyze resource optimization
            if 'optimization' in analysis and analysis['optimization'].success:
                opt_data = analysis['optimization'].data
                scaling_decision["recommendations"]["optimization"] = opt_data
            
            # Get current system metrics
            metrics = await client.get_metrics(
                metric_names=["cpu_percent", "memory_percent", "active_workflows"]
            )
            if metrics.success:
                scaling_decision["current_metrics"] = metrics.data
            
            # Make scaling decision
            action = self._determine_scaling_action(scaling_decision["recommendations"])
            scaling_decision["action_taken"] = action
            
            # Execute scaling if needed
            if action["type"] != "no_action":
                scaling_result = await client.execute_workflow(
                    "auto_scaling",
                    {
                        "action": action["type"],
                        "scale_factor": action.get("scale_factor", 1.0),
                        "resource_type": action.get("resource_type", "cpu"),
                        "justification": action.get("reason")
                    }
                )
                scaling_decision["execution_result"] = {
                    "success": scaling_result.success,
                    "data": scaling_result.data
                }
        
        print(f"⚖️ Scaling Decision: {scaling_decision['action_taken']['type']}")
        return scaling_decision
    
    def _get_cpu_recommendation(self, predicted_load: float) -> Dict[str, Any]:
        """Get CPU scaling recommendation"""
        if predicted_load > 80:
            return {"action": "scale_up", "reason": "High CPU load predicted"}
        elif predicted_load < 20:
            return {"action": "scale_down", "reason": "Low CPU utilization predicted"}
        else:
            return {"action": "maintain", "reason": "CPU load within optimal range"}
    
    def _get_memory_recommendation(self, predicted_memory: float) -> Dict[str, Any]:
        """Get memory scaling recommendation"""
        if predicted_memory > 85:
            return {"action": "scale_up", "reason": "High memory usage predicted"}
        elif predicted_memory < 30:
            return {"action": "scale_down", "reason": "Low memory utilization predicted"}
        else:
            return {"action": "maintain", "reason": "Memory usage within optimal range"}
    
    def _determine_scaling_action(self, recommendations: Dict) -> Dict[str, Any]:
        """Determine final scaling action from recommendations"""
        # Simple logic - in production this would be more sophisticated
        cpu_rec = recommendations.get("cpu_forecast", {}).get("recommendation", {})
        memory_rec = recommendations.get("memory_forecast", {}).get("recommendation", {})
        
        if cpu_rec.get("action") == "scale_up" or memory_rec.get("action") == "scale_up":
            return {
                "type": "scale_up",
                "scale_factor": 1.5,
                "resource_type": "both",
                "reason": "High resource utilization predicted"
            }
        elif cpu_rec.get("action") == "scale_down" and memory_rec.get("action") == "scale_down":
            return {
                "type": "scale_down", 
                "scale_factor": 0.7,
                "resource_type": "both",
                "reason": "Low resource utilization predicted"
            }
        else:
            return {
                "type": "no_action",
                "reason": "Resource utilization within optimal range"
            }


# Demonstration runner
async def run_integration_examples():
    """Run all integration examples"""
    print("🚀 AG06 Integration Examples")
    print("=" * 60)
    
    ag06_url = "http://localhost:8080"
    
    # Quick health check first
    health = await quick_health_check(ag06_url)
    if not health["healthy"]:
        print("❌ AG06 system not healthy - skipping examples")
        return
    
    print("✅ AG06 system healthy - proceeding with examples\n")
    
    # Example 1: Enterprise Monitoring (short demo)
    print("1. Enterprise Monitoring Integration:")
    monitor = EnterpriseMonitor(ag06_url, {"check_interval": 10})
    # Run monitoring for 30 seconds
    monitoring_task = asyncio.create_task(monitor.continuous_monitoring())
    await asyncio.sleep(30)
    monitoring_task.cancel()
    print(f"   Alerts sent: {len(monitor.alerts_sent)}\n")
    
    # Example 2: CI/CD Integration
    print("2. CI/CD Pipeline Integration:")
    cicd = CICDIntegration(ag06_url)
    validation_result = await cicd.pre_deployment_validation()
    print(f"   Validation: {'✅ PASSED' if validation_result['passed'] else '❌ FAILED'}\n")
    
    # Example 3: Data Pipeline Integration
    print("3. Data Pipeline Integration:")
    data_pipeline = DataPipelineIntegration(ag06_url)
    pipeline_config = {
        "pipeline_id": "demo_pipeline_001",
        "data_source": "demo_source",
        "destination": "demo_warehouse",
        "transformations": ["clean", "validate", "enrich"],
        "quality_rules": ["no_nulls", "valid_format"]
    }
    pipeline_result = await data_pipeline.orchestrate_data_workflow(pipeline_config)
    print(f"   Pipeline: {'✅ SUCCESS' if pipeline_result['success'] else '❌ FAILED'}\n")
    
    # Example 4: Microservices Integration
    print("4. Microservices Integration:")
    services_config = [
        {"name": "api-gateway", "image": "nginx:latest", "replicas": 2},
        {"name": "user-service", "image": "app:latest", "replicas": 3},
        {"name": "data-service", "image": "db:latest", "replicas": 1}
    ]
    microservices = MicroservicesOrchestrator(ag06_url, services_config)
    deployment_result = await microservices.deploy_service_mesh()
    print(f"   Deployment: {'✅ SUCCESS' if deployment_result['overall_success'] else '❌ FAILED'}\n")
    
    # Example 5: Auto-scaling Integration
    print("5. Auto-scaling Integration:")
    auto_scaler = AutoScalingManager(ag06_url)
    scaling_decision = await auto_scaler.intelligent_scaling_decision()
    print(f"   Scaling Action: {scaling_decision['action_taken']['type']}\n")
    
    print("✅ All Integration Examples Completed!")


if __name__ == "__main__":
    # Run integration examples
    try:
        asyncio.run(run_integration_examples())
    except KeyboardInterrupt:
        print("\n⏹️ Examples stopped by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()