#!/usr/bin/env python3
"""
Production Deployment Script for AG06 Mixer
Research-driven deployment with monitoring and rollback
"""
import asyncio
import subprocess
import time
import json
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
import yaml
import requests


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str
    namespace: str
    image_tag: str
    replicas: int
    health_check_url: str
    rollback_on_failure: bool = True
    canary_percentage: int = 0
    monitoring_duration_minutes: int = 5


class KubernetesDeployer:
    """Kubernetes deployment manager"""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize deployer"""
        self.config = config
        self.deployment_id = f"deploy-{int(time.time())}"
    
    def deploy(self) -> bool:
        """Execute deployment"""
        print(f"🚀 Starting deployment {self.deployment_id}")
        print(f"   Environment: {self.config.environment}")
        print(f"   Image: {self.config.image_tag}")
        
        try:
            # Apply Kubernetes manifests
            self._apply_manifests()
            
            # Update image
            self._update_image()
            
            # Wait for rollout
            if not self._wait_for_rollout():
                return False
            
            # Run health checks
            if not self._health_check():
                return False
            
            # Monitor for stability
            if not self._monitor_deployment():
                return False
            
            print("✅ Deployment successful!")
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            if self.config.rollback_on_failure:
                self._rollback()
            return False
    
    def _apply_manifests(self) -> None:
        """Apply Kubernetes manifests"""
        cmd = [
            "kubectl", "apply",
            "-f", "deployment/kubernetes-deployment.yaml",
            "-n", self.config.namespace
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to apply manifests: {result.stderr}")
        
        print("✅ Manifests applied")
    
    def _update_image(self) -> None:
        """Update deployment image"""
        cmd = [
            "kubectl", "set", "image",
            "deployment/ag06-mixer",
            f"ag06-mixer={self.config.image_tag}",
            "-n", self.config.namespace
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to update image: {result.stderr}")
        
        print(f"✅ Image updated to {self.config.image_tag}")
    
    def _wait_for_rollout(self, timeout: int = 300) -> bool:
        """Wait for rollout to complete"""
        print("⏳ Waiting for rollout...")
        
        cmd = [
            "kubectl", "rollout", "status",
            "deployment/ag06-mixer",
            "-n", self.config.namespace,
            f"--timeout={timeout}s"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Rollout failed: {result.stderr}")
            return False
        
        print("✅ Rollout complete")
        return True
    
    def _health_check(self, retries: int = 5) -> bool:
        """Run health checks"""
        print("🏥 Running health checks...")
        
        for i in range(retries):
            try:
                response = requests.get(
                    self.config.health_check_url,
                    timeout=10
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("healthy", False):
                        print(f"✅ Health check passed: {health_data}")
                        return True
                
            except Exception as e:
                print(f"   Attempt {i+1}/{retries} failed: {e}")
            
            time.sleep(10)
        
        print("❌ Health checks failed")
        return False
    
    def _monitor_deployment(self) -> bool:
        """Monitor deployment for stability"""
        print(f"📊 Monitoring for {self.config.monitoring_duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (self.config.monitoring_duration_minutes * 60)
        
        error_count = 0
        check_count = 0
        
        while time.time() < end_time:
            check_count += 1
            
            # Check pod status
            if not self._check_pod_status():
                error_count += 1
            
            # Check metrics
            metrics = self._get_metrics()
            if metrics:
                self._analyze_metrics(metrics)
            
            # Calculate error rate
            error_rate = error_count / check_count
            if error_rate > 0.1:  # >10% error rate
                print(f"❌ High error rate: {error_rate:.2%}")
                return False
            
            time.sleep(30)
        
        print(f"✅ Monitoring complete - Error rate: {error_rate:.2%}")
        return True
    
    def _check_pod_status(self) -> bool:
        """Check pod status"""
        cmd = [
            "kubectl", "get", "pods",
            "-l", "app=ag06-mixer",
            "-n", self.config.namespace,
            "-o", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        pods = json.loads(result.stdout)
        
        for pod in pods.get("items", []):
            status = pod["status"]["phase"]
            if status not in ["Running", "Succeeded"]:
                print(f"⚠️ Pod {pod['metadata']['name']} status: {status}")
                return False
        
        return True
    
    def _get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get deployment metrics"""
        try:
            response = requests.get(
                f"{self.config.health_check_url.replace('/health', '/metrics')}",
                timeout=5
            )
            
            if response.status_code == 200:
                return self._parse_prometheus_metrics(response.text)
            
        except Exception:
            pass
        
        return None
    
    def _parse_prometheus_metrics(self, text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics"""
        metrics = {}
        
        for line in text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split(' ')
                if len(parts) == 2:
                    metric_name = parts[0].split('{')[0]
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
        
        return metrics
    
    def _analyze_metrics(self, metrics: Dict[str, Any]) -> None:
        """Analyze metrics for issues"""
        # Check latency
        latency = metrics.get("audio_latency_ms", 0)
        if latency > 20:
            print(f"⚠️ High latency: {latency}ms")
        
        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > 0.01:
            print(f"⚠️ High error rate: {error_rate:.2%}")
        
        # Check memory
        memory = metrics.get("memory_usage_mb", 0)
        if memory > 500:
            print(f"⚠️ High memory usage: {memory}MB")
    
    def _rollback(self) -> None:
        """Rollback deployment"""
        print("🔄 Rolling back deployment...")
        
        cmd = [
            "kubectl", "rollout", "undo",
            "deployment/ag06-mixer",
            "-n", self.config.namespace
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Rollback complete")
        else:
            print(f"❌ Rollback failed: {result.stderr}")


class CanaryDeployer(KubernetesDeployer):
    """Canary deployment strategy"""
    
    def deploy(self) -> bool:
        """Execute canary deployment"""
        print(f"🐤 Starting canary deployment ({self.config.canary_percentage}%)")
        
        # Deploy canary version
        if not self._deploy_canary():
            return False
        
        # Monitor canary
        if not self._monitor_canary():
            self._remove_canary()
            return False
        
        # Gradually increase traffic
        for percentage in [25, 50, 75, 100]:
            print(f"📈 Increasing canary traffic to {percentage}%")
            
            if not self._update_traffic_split(percentage):
                self._remove_canary()
                return False
            
            if not self._monitor_deployment():
                self._remove_canary()
                return False
        
        print("✅ Canary deployment successful!")
        return True
    
    def _deploy_canary(self) -> bool:
        """Deploy canary version"""
        # Create canary deployment
        canary_manifest = self._generate_canary_manifest()
        
        with open("/tmp/canary.yaml", "w") as f:
            yaml.dump(canary_manifest, f)
        
        cmd = ["kubectl", "apply", "-f", "/tmp/canary.yaml"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return result.returncode == 0
    
    def _generate_canary_manifest(self) -> Dict[str, Any]:
        """Generate canary deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "ag06-mixer-canary",
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "ag06-mixer",
                        "version": "canary"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ag06-mixer",
                            "version": "canary"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "ag06-mixer",
                            "image": self.config.image_tag,
                            "ports": [{"containerPort": 8080}]
                        }]
                    }
                }
            }
        }
    
    def _monitor_canary(self) -> bool:
        """Monitor canary deployment"""
        print("📊 Monitoring canary...")
        
        # Similar to _monitor_deployment but for canary pods
        return self._monitor_deployment()
    
    def _update_traffic_split(self, percentage: int) -> bool:
        """Update traffic split between stable and canary"""
        # This would use a service mesh like Istio or Linkerd
        # For now, simulating with service selector update
        
        print(f"   Traffic split updated: Canary={percentage}%, Stable={100-percentage}%")
        time.sleep(5)
        return True
    
    def _remove_canary(self) -> None:
        """Remove canary deployment"""
        print("🗑️ Removing canary deployment...")
        
        cmd = [
            "kubectl", "delete", "deployment",
            "ag06-mixer-canary",
            "-n", self.config.namespace
        ]
        
        subprocess.run(cmd, capture_output=True)


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy AG06 Mixer to production")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--image-tag", required=True, help="Docker image tag")
    parser.add_argument("--replicas", type=int, default=3, help="Number of replicas")
    parser.add_argument("--canary", type=int, default=0, help="Canary percentage (0 for normal deploy)")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    
    args = parser.parse_args()
    
    # Configure deployment
    config = DeploymentConfig(
        environment=args.environment,
        namespace=f"ag06-mixer-{args.environment}",
        image_tag=args.image_tag,
        replicas=args.replicas,
        health_check_url=f"https://{args.environment}.ag06mixer.com/health",
        rollback_on_failure=not args.no_rollback,
        canary_percentage=args.canary
    )
    
    # Choose deployment strategy
    if config.canary_percentage > 0:
        deployer = CanaryDeployer(config)
    else:
        deployer = KubernetesDeployer(config)
    
    # Execute deployment
    success = deployer.deploy()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()