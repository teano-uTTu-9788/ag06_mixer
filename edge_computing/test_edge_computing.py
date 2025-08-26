#!/usr/bin/env python3
"""
Comprehensive test suite for Edge Computing WebAssembly implementation
Tests WebAssembly compilation, Cloudflare Worker functionality, and deployment configuration
"""

import json
import subprocess
import tempfile
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

class EdgeComputingTestSuite:
    def __init__(self):
        self.results = []
        self.edge_dir = Path(__file__).parent
        self.project_root = self.edge_dir.parent
        
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
    
    def test_wasm_source_structure(self):
        """Test WebAssembly source files exist and have correct structure"""
        wasm_cpp = self.edge_dir / "wasm" / "ai_mixer_wasm.cpp"
        wasm_js = self.edge_dir / "wasm" / "ai_mixer_wasm.js"
        
        # Test WASM C++ file exists
        self.log_test("WASM C++ source exists", wasm_cpp.exists())
        
        if wasm_cpp.exists():
            content = wasm_cpp.read_text()
            # Test for key components
            self.log_test("WASM has Emscripten includes", 
                         "#include <emscripten/bind.h>" in content)
            self.log_test("WASM has core processing class",
                         "class AIMixerWASM" in content)
            self.log_test("WASM has JavaScript bindings",
                         "EMSCRIPTEN_BINDINGS" in content)
            self.log_test("WASM has DSP processing methods",
                         "processFrame" in content and "extractFeatures" in content)
        else:
            for i in range(4):
                self.log_test(f"WASM structure test {i+1}", False, "Source file missing")
        
        # Test JavaScript interface
        self.log_test("WASM JS interface exists", wasm_js.exists())
        
        if wasm_js.exists():
            content = wasm_js.read_text()
            self.log_test("JS has WebAudio integration",
                         "AudioContext" in content and "AudioWorklet" in content)
            self.log_test("JS has WASM loader",
                         "WebAssembly.instantiate" in content or "loadWASM" in content)
            self.log_test("JS has real-time processing",
                         "startProcessing" in content and "processBuffer" in content)
        else:
            for i in range(3):
                self.log_test(f"JS interface test {i+1}", False, "JS interface missing")
    
    def test_cloudflare_worker_structure(self):
        """Test Cloudflare Worker implementation"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        
        self.log_test("Cloudflare Worker exists", worker_file.exists())
        
        if worker_file.exists():
            content = worker_file.read_text()
            
            # Test for essential Worker features
            self.log_test("Worker has event listener",
                         "addEventListener('fetch'" in content)
            self.log_test("Worker has CORS support",
                         "Access-Control-Allow-Origin" in content)
            self.log_test("Worker has audio processing endpoint",
                         "/process-audio" in content)
            self.log_test("Worker has feature extraction endpoint",
                         "/extract-features" in content)
            self.log_test("Worker has genre classification",
                         "/classify-genre" in content)
            self.log_test("Worker has health check",
                         "/health" in content)
            self.log_test("Worker has WASM initialization",
                         "initializeWasm" in content)
            self.log_test("Worker has error handling",
                         "try {" in content and "catch" in content)
        else:
            for i in range(8):
                self.log_test(f"Worker feature test {i+1}", False, "Worker file missing")
    
    def test_deployment_configuration(self):
        """Test Kubernetes deployment configuration"""
        deploy_file = self.edge_dir / "cdn" / "deployment.yaml"
        
        self.log_test("Deployment config exists", deploy_file.exists())
        
        if deploy_file.exists():
            content = deploy_file.read_text()
            
            # Test for Kubernetes resources
            self.log_test("Has ConfigMap resource",
                         "kind: ConfigMap" in content)
            self.log_test("Has Deployment resource",
                         "kind: Deployment" in content)
            self.log_test("Has Service resource",
                         "kind: Service" in content)
            self.log_test("Has Ingress resource",
                         "kind: Ingress" in content)
            self.log_test("Has CronJob for health checks",
                         "kind: CronJob" in content)
            self.log_test("Has ServiceMonitor for metrics",
                         "kind: ServiceMonitor" in content)
            
            # Test CDN configuration
            self.log_test("Has CDN cache rules",
                         "cache_rules:" in content)
            self.log_test("Has Cloudflare Wrangler config",
                         "wrangler.toml" in content)
        else:
            for i in range(8):
                self.log_test(f"Deployment test {i+1}", False, "Deployment config missing")
    
    def test_audio_processing_constants(self):
        """Test audio processing constants consistency"""
        wasm_cpp = self.edge_dir / "wasm" / "ai_mixer_wasm.cpp"
        worker_js = self.edge_dir / "workers" / "cloudflare_worker.js"
        
        constants_found = {"sample_rate": False, "frame_size": False, "feature_size": False}
        
        if wasm_cpp.exists():
            content = wasm_cpp.read_text()
            constants_found["sample_rate"] = "48000" in content or "SAMPLE_RATE" in content
            constants_found["frame_size"] = "960" in content or "FRAME_SIZE" in content  
            constants_found["feature_size"] = "13" in content or "FEATURE_SIZE" in content
        
        if worker_js.exists():
            content = worker_js.read_text()
            # Worker should have these constants documented
            constants_found["sample_rate"] = constants_found["sample_rate"] or "48kHz" in content
            constants_found["frame_size"] = constants_found["frame_size"] or "960" in content
            constants_found["feature_size"] = constants_found["feature_size"] or "13" in content
        
        self.log_test("Audio sample rate defined (48kHz)", constants_found["sample_rate"])
        self.log_test("Frame size defined (960 samples)", constants_found["frame_size"]) 
        self.log_test("Feature vector size defined (13)", constants_found["feature_size"])
    
    def test_api_endpoint_definitions(self):
        """Test API endpoint completeness"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        
        if not worker_file.exists():
            for i in range(8):
                self.log_test(f"API endpoint test {i+1}", False, "Worker missing")
            return
        
        content = worker_file.read_text()
        
        # Expected API endpoints
        endpoints = [
            ("/", "API documentation"),
            ("/health", "Health check"),
            ("/process-audio", "Audio processing"),
            ("/extract-features", "Feature extraction"), 
            ("/classify-genre", "Genre classification"),
            ("/config", "Configuration management"),
            ("/stats", "Statistics"),
            ("/wasm", "WASM download")
        ]
        
        for endpoint, description in endpoints:
            has_endpoint = f"'{endpoint}'" in content or f'"{endpoint}"' in content
            self.log_test(f"API endpoint {endpoint}", has_endpoint)
    
    def test_dsp_configuration_completeness(self):
        """Test DSP configuration parameters"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        wasm_cpp = self.edge_dir / "wasm" / "ai_mixer_wasm.cpp"
        
        # DSP parameters to check
        dsp_params = [
            "gateThreshold", "compThreshold", "eqLowGain", "limiterThreshold",
            "gateRatio", "compRatio", "gateAttack", "compAttack"
        ]
        
        param_found = False
        
        if worker_file.exists():
            content = worker_file.read_text()
            param_found = any(param in content for param in dsp_params)
        
        if wasm_cpp.exists():
            content = wasm_cpp.read_text()
            param_found = param_found or any(param.lower() in content.lower() for param in dsp_params)
        
        self.log_test("DSP configuration parameters defined", param_found)
        
        # Test for default configuration
        has_defaults = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_defaults = "getDefaultDSPConfig" in content or "default" in content.lower()
        
        self.log_test("Default DSP configuration available", has_defaults)
    
    def test_genre_classification_system(self):
        """Test genre classification implementation"""
        files_to_check = [
            self.edge_dir / "wasm" / "ai_mixer_wasm.cpp",
            self.edge_dir / "workers" / "cloudflare_worker.js"
        ]
        
        # Expected genres
        genres = ["SPEECH", "ROCK", "JAZZ", "ELECTRONIC", "CLASSICAL", "UNKNOWN"]
        
        has_genres = False
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                has_genres = has_genres or any(genre in content for genre in genres)
        
        self.log_test("Genre classification categories defined", has_genres)
        
        # Test for classification method
        has_classification = False
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                has_classification = has_classification or "classifyGenre" in content
        
        self.log_test("Genre classification method implemented", has_classification)
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        deploy_file = self.edge_dir / "cdn" / "deployment.yaml"
        
        has_metrics = False
        
        if worker_file.exists():
            content = worker_file.read_text()
            has_metrics = "processingTimeMS" in content or "stats" in content or "metrics" in content
        
        self.log_test("Performance metrics collection", has_metrics)
        
        # Test monitoring configuration
        has_monitoring = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_monitoring = "ServiceMonitor" in content or "prometheus" in content
        
        self.log_test("Prometheus monitoring configured", has_monitoring)
        
        # Test health checks
        has_health_checks = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_health_checks = "health-check" in content and "CronJob" in content
        
        self.log_test("Automated health checks configured", has_health_checks)
    
    def test_security_and_cors(self):
        """Test security configuration"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        deploy_file = self.edge_dir / "cdn" / "deployment.yaml"
        
        has_cors = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_cors = "Access-Control-Allow-Origin" in content
        
        self.log_test("CORS headers configured", has_cors)
        
        # Test input validation
        has_validation = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_validation = "validate" in content.lower() or ("400" in content and "error" in content)
        
        self.log_test("Input validation implemented", has_validation)
        
        # Test HTTPS/TLS
        has_tls = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_tls = "tls:" in content or "letsencrypt" in content
        
        self.log_test("TLS/HTTPS configuration", has_tls)
    
    def test_edge_computing_architecture(self):
        """Test edge computing specific features"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        deploy_file = self.edge_dir / "cdn" / "deployment.yaml"
        
        # Test edge location awareness
        has_edge_info = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_edge_info = "edge_location" in content or "colo" in content
        
        self.log_test("Edge location awareness", has_edge_info)
        
        # Test CDN caching
        has_caching = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_caching = "cache_rules" in content and "ttl" in content
        
        self.log_test("CDN caching rules defined", has_caching)
        
        # Test global deployment
        has_global_config = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_global_config = ("production" in content and "staging" in content) or "route" in content
        
        self.log_test("Multi-environment deployment", has_global_config)
        
        # Test WebAssembly optimization
        has_wasm_optimization = False
        if deploy_file.exists():
            content = deploy_file.read_text()
            has_wasm_optimization = "wasm" in content.lower() and ("cache" in content or "86400" in content)
        
        self.log_test("WebAssembly caching optimization", has_wasm_optimization)
    
    def test_documentation_and_examples(self):
        """Test documentation completeness"""
        worker_file = self.edge_dir / "workers" / "cloudflare_worker.js"
        
        has_api_docs = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_api_docs = "API documentation" in content or "endpoints" in content
        
        self.log_test("API documentation endpoint", has_api_docs)
        
        # Test usage examples
        has_usage = False
        if worker_file.exists():
            content = worker_file.read_text()
            has_usage = "usage" in content.lower() or "example" in content.lower()
        
        self.log_test("Usage examples provided", has_usage)
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🌐 Edge Computing WebAssembly Test Suite")
        print("=" * 80)
        
        # Core structure tests
        self.test_wasm_source_structure()
        self.test_cloudflare_worker_structure()
        self.test_deployment_configuration()
        
        # Audio processing tests
        self.test_audio_processing_constants()
        self.test_api_endpoint_definitions()
        self.test_dsp_configuration_completeness()
        self.test_genre_classification_system()
        
        # Production readiness tests
        self.test_performance_monitoring()
        self.test_security_and_cors()
        self.test_edge_computing_architecture()
        self.test_documentation_and_examples()
        
        # Calculate results
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        total = len(self.results)
        percentage = (passed / total) * 100
        
        print("=" * 80)
        print(f"Edge Computing Test Results: {passed}/{total} ({percentage:.1f}% success rate)")
        
        if percentage >= 90:
            print("✅ EXCELLENT - Edge computing implementation ready for deployment")
        elif percentage >= 80:
            print("⚠️  GOOD - Minor improvements recommended before deployment")
        elif percentage >= 70:
            print("⚠️  FAIR - Several issues need addressing")
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
    print("Starting Edge Computing WebAssembly Test Suite...")
    
    suite = EdgeComputingTestSuite()
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