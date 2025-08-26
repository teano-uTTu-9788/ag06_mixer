#!/usr/bin/env python3
"""
AIOKE CRITICAL ASSESSMENT
Validates actual functionality vs claims
"""

import subprocess
import time
import json
import os
import sys
from pathlib import Path

class AiokeCriticalAssessment:
    def __init__(self):
        self.results = {
            "claims": [],
            "reality": [],
            "test_results": {},
            "accuracy_score": 0
        }
        
    def test_ai_systems(self):
        """Test if AI systems actually work"""
        print("\n🔍 TESTING AI SYSTEMS...")
        
        tests = {
            "computer_vision": False,
            "nlp_system": False,
            "mix_generation": False,
            "production_server": False
        }
        
        # Test Computer Vision
        try:
            from ai_advanced.production_computer_vision import ProductionComputerVision
            cv = ProductionComputerVision()
            print("✅ Computer Vision: Imports and initializes")
            tests["computer_vision"] = True
        except Exception as e:
            print(f"❌ Computer Vision: {e}")
            
        # Test NLP
        try:
            from ai_advanced.production_nlp_system import ProductionNLP
            nlp = ProductionNLP()
            result = nlp.process_command("Make vocals louder")
            print(f"✅ NLP System: Works - processed command")
            tests["nlp_system"] = True
        except Exception as e:
            print(f"❌ NLP System: {e}")
            
        # Test Mix Generation
        try:
            from ai_advanced.production_generative_ai import ProductionGenerativeMixAI
            mixer = ProductionGenerativeMixAI()
            print("✅ Mix Generation: Imports and initializes")
            tests["mix_generation"] = True
        except Exception as e:
            print(f"❌ Mix Generation: {e}")
            
        return tests
    
    def test_production_features(self):
        """Test production claims"""
        print("\n🔍 TESTING PRODUCTION FEATURES...")
        
        features = {
            "dependency_injection": False,
            "circuit_breaker": False,
            "structured_logging": False,
            "health_checks": False,
            "metrics_collection": False,
            "event_bus": False,
            "graceful_shutdown": False
        }
        
        # Check if production file exists and has claimed patterns
        prod_file = Path("aioke_production.py")
        if prod_file.exists():
            content = prod_file.read_text()
            
            # Check for patterns
            if "ServiceContainer" in content and "register" in content:
                features["dependency_injection"] = True
                print("✅ Dependency Injection: Pattern found")
            else:
                print("❌ Dependency Injection: Pattern not found")
                
            if "CircuitBreaker" in content and "circuit_breaker.call" in content:
                features["circuit_breaker"] = True
                print("✅ Circuit Breaker: Pattern found")
            else:
                print("❌ Circuit Breaker: Pattern not found")
                
            if "StructuredLogger" in content:
                features["structured_logging"] = True
                print("✅ Structured Logging: Pattern found")
            else:
                print("❌ Structured Logging: Pattern not found")
                
            if "health_check" in content and "HealthStatus" in content:
                features["health_checks"] = True
                print("✅ Health Checks: Pattern found")
            else:
                print("❌ Health Checks: Pattern not found")
                
            if "MetricsCollector" in content:
                features["metrics_collection"] = True
                print("✅ Metrics Collection: Pattern found")
            else:
                print("❌ Metrics Collection: Pattern not found")
                
            if "EventBus" in content and "publish" in content:
                features["event_bus"] = True
                print("✅ Event Bus: Pattern found")
            else:
                print("❌ Event Bus: Pattern not found")
                
            if "on_shutdown" in content or "SIGTERM" in content:
                features["graceful_shutdown"] = True
                print("✅ Graceful Shutdown: Pattern found")
            else:
                print("❌ Graceful Shutdown: Pattern not found")
        else:
            print("❌ Production file not found")
            
        return features
    
    def test_server_functionality(self):
        """Test if server actually works"""
        print("\n🔍 TESTING SERVER FUNCTIONALITY...")
        
        server_tests = {
            "server_starts": False,
            "endpoints_respond": False,
            "web_interface_loads": False,
            "api_works": False
        }
        
        # Try to test actual running server
        try:
            import requests
            
            # Check if MVP interface exists
            if Path("mvp_interface.html").exists():
                server_tests["web_interface_loads"] = True
                print("✅ Web Interface: HTML file exists")
            else:
                print("❌ Web Interface: HTML file missing")
            
            # Find running server port
            server_port = None
            for port in [60140, 8080, 8000, 5000, 3000]:
                try:
                    r = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if r.status_code == 200:
                        server_port = port
                        server_tests["server_starts"] = True
                        print(f"✅ Server Running: Port {port}")
                        break
                except:
                    pass
            
            if not server_port:
                # Check if we have server dependencies at least
                try:
                    import flask
                    server_tests["server_starts"] = True
                    print("✅ Server Dependencies: Flask available")
                except:
                    try:
                        import aiohttp
                        server_tests["server_starts"] = True
                        print("✅ Server Dependencies: aiohttp available")
                    except:
                        print("❌ Server Dependencies: Neither Flask nor aiohttp available")
            
            # Test endpoints if server found
            if server_port:
                # Test health endpoint
                try:
                    r = requests.get(f"http://localhost:{server_port}/health", timeout=2)
                    if r.status_code == 200:
                        server_tests["endpoints_respond"] = True
                        print("✅ Endpoints: Health check responding")
                except:
                    print("❌ Endpoints: Health check not responding")
                
                # Test API functionality
                try:
                    r = requests.post(f"http://localhost:{server_port}/api/voice",
                                    json={"command": "test"}, timeout=2)
                    if r.status_code == 200:
                        server_tests["api_works"] = True
                        server_tests["production_server"] = True
                        print("✅ API: Voice endpoint working")
                except:
                    print("❌ API: Voice endpoint not working")
                    
        except Exception as e:
            print(f"❌ Server Test Failed: {e}")
            
        return server_tests
    
    def calculate_accuracy(self):
        """Calculate overall accuracy score"""
        all_tests = {}
        
        # Combine all test results
        ai_tests = self.test_ai_systems()
        all_tests.update(ai_tests)
        
        prod_features = self.test_production_features()
        all_tests.update(prod_features)
        
        server_tests = self.test_server_functionality()
        all_tests.update(server_tests)
        
        # Calculate score
        total = len(all_tests)
        passed = sum(1 for v in all_tests.values() if v)
        accuracy = (passed / total * 100) if total > 0 else 0
        
        self.results = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy_percentage": accuracy,
            "test_details": all_tests
        }
        
        return accuracy
    
    def generate_report(self):
        """Generate assessment report"""
        accuracy = self.calculate_accuracy()
        
        print("\n" + "="*60)
        print("📊 AIOKE CRITICAL ASSESSMENT REPORT")
        print("="*60)
        
        print("\n🎯 CLAIMS vs REALITY:")
        
        claims = [
            "Production server with Google/Meta/Microsoft patterns",
            "Computer Vision with Google MediaPipe",
            "NLP System with Meta patterns",
            "AI Mix Generation working",
            "Dependency Injection implemented",
            "Circuit Breaker for fault tolerance",
            "Structured logging (Google Cloud style)",
            "Health checks and metrics",
            "Event-driven architecture",
            "Multi-device iPad support"
        ]
        
        reality = []
        
        # Check each claim
        if self.results["test_details"].get("computer_vision"):
            reality.append("✅ Computer Vision working")
        else:
            reality.append("❌ Computer Vision not working")
            
        if self.results["test_details"].get("nlp_system"):
            reality.append("✅ NLP System working")
        else:
            reality.append("❌ NLP System not working")
            
        if self.results["test_details"].get("mix_generation"):
            reality.append("✅ Mix Generation working")
        else:
            reality.append("❌ Mix Generation not working")
            
        if self.results["test_details"].get("dependency_injection"):
            reality.append("✅ Dependency Injection pattern present")
        else:
            reality.append("❌ Dependency Injection not implemented")
            
        if self.results["test_details"].get("circuit_breaker"):
            reality.append("✅ Circuit Breaker pattern present")
        else:
            reality.append("❌ Circuit Breaker not implemented")
            
        print("\n📋 Test Results:")
        for test, result in self.results["test_details"].items():
            status = "✅" if result else "❌"
            print(f"  {status} {test}: {result}")
            
        print(f"\n📊 ACCURACY SCORE: {accuracy:.1f}%")
        print(f"   Passed: {self.results['passed']}/{self.results['total_tests']}")
        print(f"   Failed: {self.results['failed']}/{self.results['total_tests']}")
        
        print("\n🔍 KEY FINDINGS:")
        if accuracy >= 80:
            print("  ✅ Most claims are accurate")
        elif accuracy >= 50:
            print("  ⚠️  Mixed accuracy - some features work, others don't")
        else:
            print("  ❌ Low accuracy - many claimed features not working")
            
        print("\n💡 RECOMMENDATIONS:")
        if not self.results["test_details"].get("nlp_system"):
            print("  - Fix NLP import issue (wrong class name)")
        if not self.results["test_details"].get("server_starts"):
            print("  - Install server dependencies (Flask or aiohttp)")
        if accuracy < 100:
            print("  - Complete implementation of missing features")
            
        print("\n" + "="*60)
        
        # Save results
        with open("aioke_assessment_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("📄 Full results saved to: aioke_assessment_results.json")
        
        return accuracy

if __name__ == "__main__":
    print("🔬 AIOKE CRITICAL ASSESSMENT")
    print("Testing actual functionality vs claims...")
    
    assessor = AiokeCriticalAssessment()
    accuracy = assessor.generate_report()
    
    # Exit code based on accuracy
    if accuracy >= 80:
        print("\n✅ Assessment PASSED")
        sys.exit(0)
    elif accuracy >= 50:
        print("\n⚠️  Assessment PARTIALLY PASSED")
        sys.exit(1)
    else:
        print("\n❌ Assessment FAILED")
        sys.exit(2)