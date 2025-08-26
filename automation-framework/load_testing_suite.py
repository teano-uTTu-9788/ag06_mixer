#!/usr/bin/env python3
"""
Load Testing Suite for Aioke Advanced Enterprise System
Validates system can handle 100x current load (target: 1000 events/sec)
"""

import asyncio
import json
import time
import statistics
import concurrent.futures
import threading
from typing import Dict, List, Any
from dataclasses import dataclass, field
import requests
from datetime import datetime

@dataclass
class LoadTestResult:
    """Result of a single load test"""
    scenario: str
    duration: float
    requests_sent: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_50: float
    percentile_95: float
    percentile_99: float
    throughput: float
    error_rate: float
    
class LoadTester:
    """Comprehensive load testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        self.response_times: List[float] = []
        self.errors: List[str] = []
        self.lock = threading.Lock()
        
    def reset_metrics(self):
        """Reset metrics for new test"""
        self.response_times = []
        self.errors = []
        
    def make_request(self, endpoint: str, method: str = "GET", data: Any = None) -> tuple[bool, float]:
        """Make a single HTTP request and return success status and response time"""
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            elif method == "POST":
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            with self.lock:
                self.response_times.append(response_time)
            
            return response.status_code == 200, response_time
            
        except Exception as e:
            with self.lock:
                self.errors.append(str(e))
            return False, 0
    
    def concurrent_requests(self, endpoint: str, num_requests: int, num_threads: int = 10):
        """Send concurrent requests using thread pool"""
        success_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for _ in range(num_requests):
                future = executor.submit(self.make_request, endpoint)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                success, _ = future.result()
                if success:
                    success_count += 1
        
        return success_count
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics from response times"""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        
        return {
            'average': statistics.mean(sorted_times),
            'min': min(sorted_times),
            'max': max(sorted_times),
            'median': statistics.median(sorted_times),
            'stddev': statistics.stdev(sorted_times) if len(sorted_times) > 1 else 0,
            'percentile_50': sorted_times[int(len(sorted_times) * 0.5)],
            'percentile_95': sorted_times[int(len(sorted_times) * 0.95)],
            'percentile_99': sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) >= 100 else sorted_times[-1]
        }
    
    def run_load_test(self, scenario: str, num_requests: int, num_threads: int, endpoint: str = "/health"):
        """Run a load test scenario"""
        print(f"\n🚀 Running: {scenario}")
        print(f"   Requests: {num_requests}, Threads: {num_threads}")
        
        self.reset_metrics()
        
        start_time = time.time()
        successful_requests = self.concurrent_requests(endpoint, num_requests, num_threads)
        duration = time.time() - start_time
        
        stats = self.calculate_statistics()
        
        result = LoadTestResult(
            scenario=scenario,
            duration=duration,
            requests_sent=num_requests,
            successful_requests=successful_requests,
            failed_requests=num_requests - successful_requests,
            average_response_time=stats.get('average', 0),
            min_response_time=stats.get('min', 0),
            max_response_time=stats.get('max', 0),
            percentile_50=stats.get('percentile_50', 0),
            percentile_95=stats.get('percentile_95', 0),
            percentile_99=stats.get('percentile_99', 0),
            throughput=num_requests / duration if duration > 0 else 0,
            error_rate=(num_requests - successful_requests) / num_requests if num_requests > 0 else 0
        )
        
        self.results.append(result)
        
        # Print summary
        print(f"   ✅ Success: {successful_requests}/{num_requests} ({(successful_requests/num_requests)*100:.1f}%)")
        print(f"   ⚡ Throughput: {result.throughput:.1f} req/s")
        print(f"   ⏱️  Response: {result.average_response_time:.1f}ms avg, {result.percentile_95:.1f}ms p95")
        
        return result
    
    def run_stress_test(self):
        """Run progressive stress test to find breaking point"""
        print("\n🔥 STRESS TEST - Finding System Breaking Point")
        
        stages = [
            (10, 1, "Baseline"),
            (100, 10, "Light Load"),
            (500, 20, "Moderate Load"),
            (1000, 50, "Heavy Load"),
            (2000, 100, "Extreme Load"),
            (5000, 200, "Breaking Point")
        ]
        
        for requests, threads, name in stages:
            result = self.run_load_test(
                f"Stress Test - {name}",
                requests,
                threads
            )
            
            # Stop if error rate is too high
            if result.error_rate > 0.1:
                print(f"   ⚠️  System showing stress at {result.throughput:.1f} req/s")
                if result.error_rate > 0.5:
                    print(f"   🛑 Breaking point reached at {requests} requests")
                    break
            
            # Brief pause between stages
            time.sleep(2)
    
    def run_spike_test(self):
        """Test sudden traffic spikes"""
        print("\n⚡ SPIKE TEST - Sudden Traffic Burst")
        
        # Normal load
        self.run_load_test("Pre-spike baseline", 100, 10)
        
        # Sudden spike
        self.run_load_test("Traffic spike", 2000, 100)
        
        # Return to normal
        self.run_load_test("Post-spike recovery", 100, 10)
    
    def run_endurance_test(self, duration_seconds: int = 60):
        """Test sustained load over time"""
        print(f"\n⏳ ENDURANCE TEST - {duration_seconds} seconds sustained load")
        
        requests_per_second = 100
        interval = 0.01  # 10ms between requests
        
        self.reset_metrics()
        start_time = time.time()
        requests_sent = 0
        successful_requests = 0
        
        while time.time() - start_time < duration_seconds:
            success, _ = self.make_request("/health")
            requests_sent += 1
            if success:
                successful_requests += 1
            
            time.sleep(interval)
            
            # Print progress every 10 seconds
            if requests_sent % 1000 == 0:
                elapsed = time.time() - start_time
                current_throughput = requests_sent / elapsed
                print(f"   Progress: {elapsed:.0f}s, {requests_sent} requests, {current_throughput:.1f} req/s")
        
        duration = time.time() - start_time
        stats = self.calculate_statistics()
        
        result = LoadTestResult(
            scenario="Endurance Test",
            duration=duration,
            requests_sent=requests_sent,
            successful_requests=successful_requests,
            failed_requests=requests_sent - successful_requests,
            average_response_time=stats.get('average', 0),
            min_response_time=stats.get('min', 0),
            max_response_time=stats.get('max', 0),
            percentile_50=stats.get('percentile_50', 0),
            percentile_95=stats.get('percentile_95', 0),
            percentile_99=stats.get('percentile_99', 0),
            throughput=requests_sent / duration,
            error_rate=(requests_sent - successful_requests) / requests_sent if requests_sent > 0 else 0
        )
        
        self.results.append(result)
        
        print(f"   ✅ Completed: {successful_requests}/{requests_sent} successful")
        print(f"   ⚡ Average throughput: {result.throughput:.1f} req/s")
        
        return result
    
    def run_pattern_tests(self):
        """Test each enterprise pattern endpoint"""
        print("\n🏢 PATTERN-SPECIFIC LOAD TESTS")
        
        patterns = [
            ("/health", "System Health"),
            ("/metrics", "Metrics Endpoint"),
            ("/status", "Status Endpoint")
        ]
        
        for endpoint, name in patterns:
            self.run_load_test(
                f"Pattern Test - {name}",
                500,
                20,
                endpoint
            )
    
    def run_concurrent_user_simulation(self, num_users: int = 100):
        """Simulate concurrent users with realistic behavior"""
        print(f"\n👥 USER SIMULATION - {num_users} concurrent users")
        
        def user_behavior():
            """Simulate a single user session"""
            # User arrives and checks health
            self.make_request("/health")
            time.sleep(0.5)
            
            # User checks metrics
            self.make_request("/metrics")
            time.sleep(1)
            
            # User checks status
            self.make_request("/status")
            time.sleep(0.5)
            
            # User makes multiple health checks
            for _ in range(5):
                self.make_request("/health")
                time.sleep(0.2)
        
        self.reset_metrics()
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(user_behavior) for _ in range(num_users)]
            concurrent.futures.wait(futures)
        
        duration = time.time() - start_time
        stats = self.calculate_statistics()
        
        print(f"   ✅ Simulated {num_users} users over {duration:.1f}s")
        print(f"   ⏱️  Response times: {stats.get('average', 0):.1f}ms avg, {stats.get('percentile_95', 0):.1f}ms p95")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'scenarios': [],
            'summary': {
                'total_requests': sum(r.requests_sent for r in self.results),
                'total_successful': sum(r.successful_requests for r in self.results),
                'average_throughput': statistics.mean([r.throughput for r in self.results]) if self.results else 0,
                'average_response_time': statistics.mean([r.average_response_time for r in self.results]) if self.results else 0,
                'max_throughput': max([r.throughput for r in self.results]) if self.results else 0,
                'min_response_time': min([r.min_response_time for r in self.results]) if self.results else 0
            },
            'recommendations': []
        }
        
        # Add scenario details
        for result in self.results:
            report['scenarios'].append({
                'name': result.scenario,
                'requests': result.requests_sent,
                'success_rate': (result.successful_requests / result.requests_sent * 100) if result.requests_sent > 0 else 0,
                'throughput': result.throughput,
                'response_times': {
                    'average': result.average_response_time,
                    'min': result.min_response_time,
                    'max': result.max_response_time,
                    'p50': result.percentile_50,
                    'p95': result.percentile_95,
                    'p99': result.percentile_99
                },
                'error_rate': result.error_rate * 100
            })
        
        # Generate recommendations
        max_throughput = report['summary']['max_throughput']
        
        if max_throughput < 100:
            report['recommendations'].append("⚠️ System struggling with moderate load. Consider horizontal scaling.")
        elif max_throughput < 500:
            report['recommendations'].append("📈 System handles moderate load well. Optimize for higher throughput.")
        elif max_throughput < 1000:
            report['recommendations'].append("✅ Good performance. Consider caching for further improvements.")
        else:
            report['recommendations'].append("🚀 Excellent performance! System ready for production scale.")
        
        # Check response times
        avg_response = report['summary']['average_response_time']
        if avg_response > 1000:
            report['recommendations'].append("⏱️ High response times detected. Investigate bottlenecks.")
        elif avg_response > 500:
            report['recommendations'].append("⏱️ Response times could be improved. Consider optimization.")
        
        return report

def main():
    """Run comprehensive load testing suite"""
    print("=" * 60)
    print("🚀 AIOKE ADVANCED ENTERPRISE - LOAD TESTING SUITE")
    print("=" * 60)
    
    tester = LoadTester()
    
    # Verify server is running
    try:
        response = requests.get(f"{tester.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the production server first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please run: python start_production_server.py")
        return
    
    # Run test scenarios
    print("\n📊 Starting Load Test Suite...")
    
    # 1. Basic load tests
    tester.run_load_test("Baseline Test", 100, 10)
    tester.run_load_test("Moderate Load", 500, 20)
    tester.run_load_test("Heavy Load", 1000, 50)
    
    # 2. Pattern-specific tests
    tester.run_pattern_tests()
    
    # 3. Stress test
    tester.run_stress_test()
    
    # 4. Spike test
    tester.run_spike_test()
    
    # 5. User simulation
    tester.run_concurrent_user_simulation(100)
    
    # 6. Endurance test (shorter for demo)
    tester.run_endurance_test(30)
    
    # Generate report
    report = tester.generate_report()
    
    # Save report
    with open('load_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Total Requests: {report['summary']['total_requests']:,}")
    print(f"Successful Requests: {report['summary']['total_successful']:,}")
    print(f"Max Throughput: {report['summary']['max_throughput']:.1f} req/s")
    print(f"Average Response Time: {report['summary']['average_response_time']:.1f}ms")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print("\n✅ Load testing complete! Report saved to load_test_report.json")

if __name__ == '__main__':
    main()