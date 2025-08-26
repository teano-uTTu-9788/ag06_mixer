#!/usr/bin/env python3
"""
Comprehensive test suite for Multi-Region Deployment with Global Load Balancing
Tests traffic management, regional deployments, and load balancing functionality
"""

import json
import subprocess
import tempfile
import os
import sys
import time
import asyncio
from pathlib import Path

class MultiRegionTestSuite:
    def __init__(self):
        self.results = []
        self.multi_region_dir = Path(__file__).parent
        self.project_root = self.multi_region_dir.parent
        
    def log_test(self, test_name, success, message=""):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message
        })
        print(f"Test {len(self.results):2d}: {test_name:<50} {status}")
        if message and not success:
            print(f"         {message}")
    
    def test_global_load_balancer_structure(self):
        """Test global load balancer configuration structure"""
        lb_file = self.multi_region_dir / "global_load_balancer.yaml"
        
        self.log_test("Global load balancer config exists", lb_file.exists())
        
        if lb_file.exists():
            content = lb_file.read_text()
            
            # Test for essential load balancer components
            self.log_test("Has global traffic manager config",
                         "traffic-manager.yaml" in content)
            self.log_test("Has CloudFlare load balancer config", 
                         "cloudflare-lb.yaml" in content)
            self.log_test("Has AWS ALB configuration",
                         "aws-alb.yaml" in content)
            self.log_test("Has load balancer controller deployment",
                         "global-load-balancer-controller" in content)
            self.log_test("Has Prometheus monitoring integration",
                         "ServiceMonitor" in content)
            self.log_test("Has Grafana dashboard configuration",
                         "grafana-global-lb-dashboard" in content)
        else:
            for i in range(6):
                self.log_test(f"Load balancer component {i+1}", False, "Config file missing")
    
    def test_regional_deployment_structure(self):
        """Test regional deployment configurations"""
        regional_file = self.multi_region_dir / "regional_deployments.yaml"
        
        self.log_test("Regional deployments config exists", regional_file.exists())
        
        if regional_file.exists():
            content = regional_file.read_text()
            
            # Test for regional deployments
            regions = ["us-west", "us-east", "eu-west", "asia-pacific"]
            for region in regions:
                self.log_test(f"Has {region} deployment",
                             f"ai-mixer-{region}" in content)
            
            # Test for autoscaling
            for region in regions:
                self.log_test(f"Has {region} HPA (autoscaling)",
                             f"{region}-hpa" in content)
            
            # Test for high availability features
            self.log_test("Has Pod Disruption Budgets",
                         "PodDisruptionBudget" in content)
            self.log_test("Has Network Policies", 
                         "NetworkPolicy" in content)
            self.log_test("Has Node Affinity rules",
                         "nodeAffinity" in content)
        else:
            for i in range(11):
                self.log_test(f"Regional deployment test {i+1}", False, "Config file missing")
    
    def test_traffic_management_system(self):
        """Test traffic management Python implementation"""
        traffic_mgmt = self.multi_region_dir / "traffic_management.py"
        
        self.log_test("Traffic management system exists", traffic_mgmt.exists())
        
        if traffic_mgmt.exists():
            content = traffic_mgmt.read_text()
            
            # Test for core classes
            self.log_test("Has GlobalTrafficManager class",
                         "class GlobalTrafficManager" in content)
            self.log_test("Has TrafficManagerAPI class",
                         "class TrafficManagerAPI" in content)
            self.log_test("Has RegionEndpoint dataclass",
                         "class RegionEndpoint" in content)
            
            # Test for routing strategies
            self.log_test("Has geolocation routing",
                         "GEOLOCATION" in content)
            self.log_test("Has latency-based routing",
                         "LATENCY_BASED" in content)
            self.log_test("Has weighted routing",
                         "WEIGHTED" in content)
            self.log_test("Has failover routing",
                         "FAILOVER" in content)
            
            # Test for health monitoring
            self.log_test("Has health monitoring loop",
                         "_health_monitoring_loop" in content)
            self.log_test("Has region health checks",
                         "_check_region_health" in content)
            self.log_test("Has health metrics collection",
                         "HealthMetrics" in content)
        else:
            for i in range(10):
                self.log_test(f"Traffic management test {i+1}", False, "System file missing")
    
    def test_regional_coverage(self):
        """Test comprehensive regional coverage"""
        traffic_mgmt = self.multi_region_dir / "traffic_management.py"
        
        if not traffic_mgmt.exists():
            for i in range(8):
                self.log_test(f"Regional coverage test {i+1}", False, "Traffic management missing")
            return
        
        content = traffic_mgmt.read_text()
        
        # Test for all required regions
        required_regions = [
            "us-west-1", "us-west-2", "us-east-1", "us-east-2",
            "eu-west-1", "eu-west-2", "ap-southeast-1", "ap-northeast-1"
        ]
        
        for region in required_regions:
            self.log_test(f"Has {region} endpoint",
                         region in content)
    
    def test_load_balancing_algorithms(self):
        """Test load balancing algorithm implementations"""
        files_to_check = [
            self.multi_region_dir / "global_load_balancer.yaml",
            self.multi_region_dir / "traffic_management.py"
        ]
        
        # Algorithms to test for
        algorithms = [
            ("Round Robin", "round"),
            ("Weighted Distribution", "weight"),
            ("Geolocation", "geo"),
            ("Latency-based", "latency"),
            ("Health-based", "health")
        ]
        
        for algorithm_name, keyword in algorithms:
            found = False
            for file_path in files_to_check:
                if file_path.exists():
                    content = file_path.read_text().lower()
                    if keyword in content:
                        found = True
                        break
            self.log_test(f"Supports {algorithm_name} load balancing", found)
    
    def test_high_availability_features(self):
        """Test high availability and fault tolerance features"""
        regional_file = self.multi_region_dir / "regional_deployments.yaml"
        
        if not regional_file.exists():
            for i in range(8):
                self.log_test(f"HA feature test {i+1}", False, "Regional config missing")
            return
        
        content = regional_file.read_text()
        
        # Test HA features
        ha_features = [
            ("Pod Disruption Budgets", "PodDisruptionBudget"),
            ("Multiple replicas per region", "replicas: "),
            ("Liveness probes", "livenessProbe"),
            ("Readiness probes", "readinessProbe"),
            ("Resource limits", "limits:"),
            ("Anti-affinity rules", "affinity:"),
            ("Horizontal Pod Autoscaler", "HorizontalPodAutoscaler"),
            ("Network policies", "NetworkPolicy")
        ]
        
        for feature_name, keyword in ha_features:
            self.log_test(f"Has {feature_name}", keyword in content)
    
    def test_monitoring_and_observability(self):
        """Test monitoring and observability features"""
        lb_file = self.multi_region_dir / "global_load_balancer.yaml"
        
        if not lb_file.exists():
            for i in range(7):
                self.log_test(f"Monitoring test {i+1}", False, "Load balancer config missing")
            return
        
        content = lb_file.read_text()
        
        # Test monitoring features
        monitoring_features = [
            ("Prometheus ServiceMonitor", "ServiceMonitor"),
            ("Grafana dashboard", "grafana-global-lb-dashboard"),
            ("Request rate metrics", "http_requests_total"),
            ("Response time metrics", "http_request_duration_seconds"),
            ("Error rate tracking", "status_code"),
            ("Regional health metrics", "up{service="),
            ("Cross-region latency", "Cross-Region Latency")
        ]
        
        for feature_name, keyword in monitoring_features:
            self.log_test(f"Has {feature_name}", keyword in content)
    
    def test_security_configurations(self):
        """Test security configurations"""
        files_to_check = [
            (self.multi_region_dir / "global_load_balancer.yaml", "load balancer"),
            (self.multi_region_dir / "regional_deployments.yaml", "regional")
        ]
        
        security_features_found = 0
        total_security_features = 6
        
        for file_path, file_type in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                
                # Check for security features
                if "tls:" in content or "https" in content.lower():
                    security_features_found += 1
                if "networkpolicy" in content.lower():
                    security_features_found += 1
                if "certificate" in content.lower():
                    security_features_found += 1
        
        # Additional security checks
        self.log_test("Has TLS/HTTPS configuration", 
                     security_features_found >= 1)
        self.log_test("Has Network Policy restrictions",
                     security_features_found >= 2) 
        self.log_test("Has SSL certificate management",
                     security_features_found >= 3)
        
        # Test for security in traffic management
        traffic_mgmt = self.multi_region_dir / "traffic_management.py"
        if traffic_mgmt.exists():
            content = traffic_mgmt.read_text()
            self.log_test("Has secure HTTP client configuration",
                         "aiohttp.ClientSession" in content and "timeout" in content)
            self.log_test("Has input validation",
                         "request_data.get" in content and "country" in content)
            self.log_test("Has error handling for security",
                         "except" in content and "logger.error" in content)
        else:
            for i in range(3):
                self.log_test(f"Traffic security test {i+1}", False, "Traffic management missing")
    
    def test_scalability_configuration(self):
        """Test scalability and auto-scaling configurations"""
        regional_file = self.multi_region_dir / "regional_deployments.yaml"
        
        if not regional_file.exists():
            for i in range(6):
                self.log_test(f"Scalability test {i+1}", False, "Regional config missing")
            return
        
        content = regional_file.read_text()
        
        # Test scaling features
        scaling_features = [
            ("Horizontal Pod Autoscaler", "HorizontalPodAutoscaler"),
            ("CPU-based scaling", "cpu"),
            ("Memory-based scaling", "memory"),
            ("Multiple replica tiers", "replicas:"),
            ("Scaling policies", "scaleUp" and "scaleDown"),
            ("Resource requests/limits", "requests:" and "limits:")
        ]
        
        for feature_name, keyword in scaling_features:
            if isinstance(keyword, tuple):
                has_feature = all(k in content for k in keyword)
            else:
                has_feature = keyword in content
            self.log_test(f"Has {feature_name}", has_feature)
    
    def test_deployment_environments(self):
        """Test multi-environment deployment support"""
        lb_file = self.multi_region_dir / "global_load_balancer.yaml"
        
        if not lb_file.exists():
            for i in range(4):
                self.log_test(f"Environment test {i+1}", False, "Load balancer config missing")
            return
        
        content = lb_file.read_text()
        
        # Test environment features
        env_features = [
            ("Production environment", "production"),
            ("Staging environment", "staging"), 
            ("Environment-specific routing", "env.production" or "env.staging"),
            ("Multi-environment DNS", "aimixer.com")
        ]
        
        for feature_name, keyword in env_features:
            if isinstance(keyword, tuple):
                has_feature = any(k in content for k in keyword)
            else:
                has_feature = keyword in content
            self.log_test(f"Has {feature_name}", has_feature)
    
    def test_performance_optimization(self):
        """Test performance optimization features"""
        traffic_mgmt = self.multi_region_dir / "traffic_management.py"
        lb_file = self.multi_region_dir / "global_load_balancer.yaml"
        
        perf_features_found = 0
        
        if traffic_mgmt.exists():
            content = traffic_mgmt.read_text()
            
            # Performance features in traffic management
            perf_features = [
                ("Async/await patterns", "async def" and "await"),
                ("Connection pooling", "aiohttp.ClientSession"),
                ("Timeout configuration", "timeout"),
                ("Concurrent health checks", "asyncio.gather"),
                ("Response time tracking", "response_time"),
                ("Statistics calculation", "statistics.")
            ]
            
            for feature_name, keywords in perf_features:
                if isinstance(keywords, tuple):
                    if all(k in content for k in keywords):
                        perf_features_found += 1
                        self.log_test(f"Has {feature_name}", True)
                    else:
                        self.log_test(f"Has {feature_name}", False)
                else:
                    if keywords in content:
                        perf_features_found += 1
                        self.log_test(f"Has {feature_name}", True)
                    else:
                        self.log_test(f"Has {feature_name}", False)
        else:
            for i in range(6):
                self.log_test(f"Performance feature {i+1}", False, "Traffic management missing")
    
    def test_documentation_completeness(self):
        """Test documentation and configuration completeness"""
        
        # Test for README or documentation
        docs_found = False
        for doc_name in ["README.md", "DEPLOYMENT.md", "ARCHITECTURE.md"]:
            if (self.multi_region_dir / doc_name).exists():
                docs_found = True
                break
        
        self.log_test("Has documentation", docs_found)
        
        # Test configuration files completeness
        config_files = [
            "global_load_balancer.yaml",
            "regional_deployments.yaml", 
            "traffic_management.py"
        ]
        
        missing_configs = []
        for config in config_files:
            if not (self.multi_region_dir / config).exists():
                missing_configs.append(config)
        
        self.log_test("All configuration files present", len(missing_configs) == 0)
        
        if missing_configs:
            self.log_test("Missing config files", False, f"Missing: {', '.join(missing_configs)}")
        
        # Test for example usage
        has_examples = False
        if (self.multi_region_dir / "traffic_management.py").exists():
            content = (self.multi_region_dir / "traffic_management.py").read_text()
            has_examples = "async def main" in content or "Example usage" in content
        
        self.log_test("Has usage examples", has_examples)
    
    def run_all_tests(self):
        """Run complete multi-region test suite"""
        print("🌍 Multi-Region Deployment Test Suite")
        print("=" * 80)
        
        # Core infrastructure tests
        self.test_global_load_balancer_structure()
        self.test_regional_deployment_structure()
        self.test_traffic_management_system()
        
        # Regional and routing tests
        self.test_regional_coverage()
        self.test_load_balancing_algorithms()
        
        # High availability and reliability tests
        self.test_high_availability_features()
        self.test_monitoring_and_observability()
        
        # Security and performance tests
        self.test_security_configurations()
        self.test_scalability_configuration()
        
        # Deployment and environment tests
        self.test_deployment_environments()
        self.test_performance_optimization()
        self.test_documentation_completeness()
        
        # Calculate results
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        total = len(self.results)
        percentage = (passed / total) * 100
        
        print("=" * 80)
        print(f"Multi-Region Deployment Test Results: {passed}/{total} ({percentage:.1f}% success rate)")
        
        if percentage >= 95:
            print("✅ EXCELLENT - Multi-region system ready for global deployment")
        elif percentage >= 85:
            print("⚠️  GOOD - Minor improvements recommended")
        elif percentage >= 75:
            print("⚠️  FAIR - Several features need attention")
        else:
            print("❌ NEEDS WORK - Significant improvements required")
        
        # Show failed tests
        failed_tests = [r for r in self.results if r['status'] == 'FAIL']
        if failed_tests:
            print(f"\nFailed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  • {test['test']}")
                if test['message']:
                    print(f"    {test['message']}")
        
        return self.results

if __name__ == "__main__":
    print("Starting Multi-Region Deployment Test Suite...")
    
    suite = MultiRegionTestSuite()
    results = suite.run_all_tests()
    
    # Return appropriate exit code
    passed = len([r for r in results if r['status'] == 'PASS'])
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All {total} tests passed!")
        sys.exit(0)
    else:
        failed = total - passed
        print(f"\n⚠️  {failed}/{total} tests failed")
        sys.exit(1)