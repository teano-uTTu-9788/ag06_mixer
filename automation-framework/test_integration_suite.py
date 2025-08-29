#!/usr/bin/env python3
"""
Integration Test Suite for Terminal Automation Framework
Tests real command execution and module interactions
Following Google/Meta testing best practices
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Add framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terminal_automation_framework import (
    TerminalAutomationOrchestrator,
    HomebrewModule,
    GitModule,
    DockerModule,
    TestingModule
)


class IntegrationTestSuite:
    """Real-world integration tests with actual command execution."""
    
    def __init__(self):
        self.results = []
        self.orchestrator = None
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
        
        self.results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
    
    async def test_real_homebrew_operations(self):
        """Test actual Homebrew operations."""
        try:
            # Check if Homebrew is installed
            result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
            if result.returncode != 0:
                self.log_result("Homebrew Operations", False, "Homebrew not installed")
                return
            
            # Execute real Homebrew list command
            workflow_result = await self.orchestrator.execute_workflow('homebrew-test', [
                {'module': 'homebrew', 'task': 'list', 'context': {}}
            ])
            
            passed = workflow_result['tasks_succeeded'] == 1
            self.log_result("Homebrew Operations", passed)
            
        except Exception as e:
            self.log_result("Homebrew Operations", False, str(e))
    
    async def test_real_git_operations(self):
        """Test actual Git operations."""
        try:
            # Create a temporary Git repository
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                
                # Initialize Git repo
                subprocess.run(['git', 'init'], check=True, capture_output=True)
                subprocess.run(['git', 'config', 'user.name', 'Test User'], check=True)
                subprocess.run(['git', 'config', 'user.email', 'test@example.com'], check=True)
                
                # Create a test file
                test_file = Path(tmpdir) / 'test.txt'
                test_file.write_text('Integration test content')
                
                # Add and commit
                subprocess.run(['git', 'add', '.'], check=True)
                subprocess.run(['git', 'commit', '-m', 'Integration test commit'], check=True)
                
                # Test Git module
                workflow_result = await self.orchestrator.execute_workflow('git-test', [
                    {'module': 'git', 'task': 'status', 'context': {'path': tmpdir}}
                ])
                
                passed = workflow_result['tasks_succeeded'] == 1
                self.log_result("Git Operations", passed)
                
        except Exception as e:
            self.log_result("Git Operations", False, str(e))
    
    async def test_concurrent_workflows(self):
        """Test concurrent workflow execution."""
        try:
            # Launch multiple workflows concurrently
            workflows = []
            for i in range(5):
                workflow = self.orchestrator.execute_workflow(f'concurrent-{i}', [
                    {'module': 'homebrew', 'task': 'list', 'context': {}},
                    {'module': 'git', 'task': 'status', 'context': {}}
                ])
                workflows.append(workflow)
            
            # Wait for all to complete
            results = await asyncio.gather(*workflows)
            
            # Check all succeeded
            all_succeeded = all(r['succeeded'] for r in results)
            total_tasks = sum(r['tasks_total'] for r in results)
            
            self.log_result(
                "Concurrent Workflows", 
                all_succeeded,
                f"Executed {total_tasks} tasks across 5 workflows"
            )
            
        except Exception as e:
            self.log_result("Concurrent Workflows", False, str(e))
    
    async def test_error_recovery(self):
        """Test error handling and recovery."""
        try:
            # Execute workflow with intentional error
            workflow_result = await self.orchestrator.execute_workflow('error-test', [
                {'module': 'nonexistent', 'task': 'test', 'context': {}},
                {'module': 'homebrew', 'task': 'list', 'context': {}}  # Should still execute
            ])
            
            # Check partial success handling
            passed = (
                workflow_result['tasks_failed'] == 1 and
                workflow_result['tasks_succeeded'] >= 0
            )
            
            self.log_result("Error Recovery", passed)
            
        except Exception as e:
            self.log_result("Error Recovery", False, str(e))
    
    async def test_performance_benchmarks(self):
        """Test performance meets requirements."""
        try:
            # Warmup
            await self.orchestrator.execute_workflow('warmup', [
                {'module': 'homebrew', 'task': 'list', 'context': {}}
            ])
            
            # Benchmark 10 workflows
            start_time = time.time()
            
            for i in range(10):
                await self.orchestrator.execute_workflow(f'perf-{i}', [
                    {'module': 'homebrew', 'task': 'list', 'context': {}},
                    {'module': 'git', 'task': 'status', 'context': {}}
                ])
            
            elapsed = time.time() - start_time
            avg_ms = (elapsed / 10) * 1000
            
            # Check if under 200ms threshold
            passed = avg_ms < 200
            self.log_result(
                "Performance Benchmarks",
                passed,
                f"Average: {avg_ms:.1f}ms per workflow"
            )
            
        except Exception as e:
            self.log_result("Performance Benchmarks", False, str(e))
    
    async def test_module_interactions(self):
        """Test module interactions and data passing."""
        try:
            # Create workflow with dependent tasks
            workflow_result = await self.orchestrator.execute_workflow('interaction-test', [
                {'module': 'git', 'task': 'status', 'context': {}},
                {'module': 'homebrew', 'task': 'list', 'context': {'filter': 'git'}}
            ])
            
            passed = workflow_result['tasks_succeeded'] == 2
            self.log_result("Module Interactions", passed)
            
        except Exception as e:
            self.log_result("Module Interactions", False, str(e))
    
    async def test_configuration_loading(self):
        """Test configuration file loading and validation."""
        try:
            # Create temporary config
            config_data = {
                'framework': {
                    'version': '2.0.0',
                    'max_concurrent': 3
                },
                'modules': {
                    'homebrew': {'enabled': True},
                    'git': {'enabled': True},
                    'docker': {'enabled': False}
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                import yaml
                yaml.dump(config_data, f)
                config_path = f.name
            
            try:
                # Create orchestrator with config
                test_orchestrator = TerminalAutomationOrchestrator(config_path=config_path)
                
                # Verify configuration applied
                has_homebrew = 'homebrew' in test_orchestrator.modules
                has_git = 'git' in test_orchestrator.modules
                no_docker = 'docker' not in test_orchestrator.modules
                
                passed = has_homebrew and has_git and no_docker
                self.log_result("Configuration Loading", passed)
                
            finally:
                os.unlink(config_path)
                
        except Exception as e:
            self.log_result("Configuration Loading", False, str(e))
    
    async def test_export_functionality(self):
        """Test results export functionality."""
        try:
            # Execute workflow
            await self.orchestrator.execute_workflow('export-test', [
                {'module': 'homebrew', 'task': 'list', 'context': {}}
            ])
            
            # Export results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                export_path = f.name
            
            try:
                exported = self.orchestrator.export_results(export_path)
                
                # Verify file created and valid JSON
                with open(exported, 'r') as f:
                    data = json.load(f)
                
                passed = 'workflows' in data and 'summary' in data
                self.log_result("Export Functionality", passed)
                
            finally:
                if os.path.exists(export_path):
                    os.unlink(export_path)
                    
        except Exception as e:
            self.log_result("Export Functionality", False, str(e))
    
    async def test_monitoring_metrics(self):
        """Test monitoring and metrics collection."""
        try:
            # Execute several workflows
            for i in range(3):
                await self.orchestrator.execute_workflow(f'monitor-{i}', [
                    {'module': 'homebrew', 'task': 'list', 'context': {}}
                ])
            
            # Get execution summary
            summary = self.orchestrator.get_execution_summary()
            
            # Verify metrics collected
            has_metrics = (
                'total_workflows' in summary and
                summary['total_workflows'] >= 3 and
                'success_rate' in summary and
                'total_cost' in summary
            )
            
            self.log_result("Monitoring Metrics", has_metrics)
            
        except Exception as e:
            self.log_result("Monitoring Metrics", False, str(e))
    
    async def test_cleanup_operations(self):
        """Test cleanup and resource management."""
        try:
            # Check initial state
            initial_history = len(self.orchestrator.execution_history)
            
            # Execute workflow
            await self.orchestrator.execute_workflow('cleanup-test', [
                {'module': 'homebrew', 'task': 'list', 'context': {}}
            ])
            
            # Verify history updated
            final_history = len(self.orchestrator.execution_history)
            
            passed = final_history > initial_history
            self.log_result("Cleanup Operations", passed)
            
        except Exception as e:
            self.log_result("Cleanup Operations", False, str(e))
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("\n" + "="*60)
        print("🧪 INTEGRATION TEST SUITE")
        print("="*60 + "\n")
        
        # Initialize orchestrator
        self.orchestrator = TerminalAutomationOrchestrator()
        
        # Run all tests
        test_methods = [
            self.test_real_homebrew_operations,
            self.test_real_git_operations,
            self.test_concurrent_workflows,
            self.test_error_recovery,
            self.test_performance_benchmarks,
            self.test_module_interactions,
            self.test_configuration_loading,
            self.test_export_functionality,
            self.test_monitoring_metrics,
            self.test_cleanup_operations
        ]
        
        for test_method in test_methods:
            await test_method()
        
        # Summary
        print("\n" + "="*60)
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📊 RESULTS: {passed}/{total} tests passed ({success_rate:.1f}%)")
        
        if passed == total:
            print("✅ ALL INTEGRATION TESTS PASSED!")
        else:
            print(f"❌ {total - passed} test(s) failed")
        
        print("="*60 + "\n")
        
        # Export results
        results_file = 'integration_test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': success_rate,
                'results': self.results
            }, f, indent=2)
        
        print(f"📄 Results exported to: {results_file}")
        
        return passed == total


async def main():
    """Main entry point."""
    suite = IntegrationTestSuite()
    success = await suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())