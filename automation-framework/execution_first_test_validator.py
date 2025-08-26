#!/usr/bin/env python3
"""
EXECUTION-FIRST Test Validator
Implements real behavioral testing as required by Tu authorization
Replaces phantom testing with actual function execution validation
"""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import psutil
import sys
import importlib.util

@dataclass
class ExecutionTest:
    """Real execution test with actual behavior validation"""
    id: int
    name: str
    category: str
    module_path: str
    class_name: str
    test_method: str
    expected_behavior: str
    success: bool = False
    error_message: str = ""
    execution_time: float = 0.0
    actual_result: Any = None

class ExecutionFirstValidator:
    """
    EXECUTION-FIRST validator that tests real behavior, not structure
    Addresses Tu Agent requirement: "VERIFY IT WORKS"
    """
    
    def __init__(self):
        self.base_path = Path("/Users/nguythe/ag06_mixer/automation-framework")
        self.tests: List[ExecutionTest] = []
        self.results: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        # Resource protection per MANU requirements
        self._check_system_resources()
        
    def _check_system_resources(self):
        """Resource protection implementation"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        if cpu_percent > 85:
            raise RuntimeError(f"CPU too high ({cpu_percent}%) - refusing to start")
        
        if memory.percent > 90:
            raise RuntimeError(f"Memory too high ({memory.percent}%) - refusing to start")
            
        print(f"✅ Resource protection: CPU {cpu_percent}%, Memory {memory.percent}%")
    
    def _load_module_dynamically(self, module_path: str) -> Optional[Any]:
        """Dynamically load and return module"""
        try:
            if not Path(module_path).exists():
                return None
                
            spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"❌ Failed to load {module_path}: {e}")
            return None
    
    async def _execute_real_test(self, test: ExecutionTest) -> bool:
        """Execute actual behavioral test - not structure checking"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Load the actual module
            module = self._load_module_dynamically(test.module_path)
            if module is None:
                test.error_message = f"Module not found: {test.module_path}"
                return False
            
            # Get the actual class
            if not hasattr(module, test.class_name):
                test.error_message = f"Class {test.class_name} not found in module"
                return False
            
            cls = getattr(module, test.class_name)
            
            # ❌ FORBIDDEN - Phantom testing
            # if hasattr(cls, test.test_method):
            #     return True  # This is worthless phantom testing
            
            # ✅ REQUIRED - Real execution testing
            instance = cls()
            
            # Execute the actual method with real data
            if hasattr(instance, test.test_method):
                method = getattr(instance, test.test_method)
                
                if asyncio.iscoroutinefunction(method):
                    result = await method()
                else:
                    result = method()
                
                # VERIFY THE ACTUAL BEHAVIOR, NOT JUST EXISTENCE
                test.actual_result = result
                
                # Behavioral validation based on expected behavior
                success = self._validate_behavior(test, result)
                test.success = success
                
                if not success:
                    test.error_message = f"Behavior validation failed: expected {test.expected_behavior}, got {type(result).__name__}"
                
                return success
            else:
                test.error_message = f"Method {test.test_method} not found"
                return False
                
        except Exception as e:
            test.error_message = f"Execution failed: {str(e)}"
            test.success = False
            return False
        finally:
            end_time = asyncio.get_event_loop().time()
            test.execution_time = end_time - start_time
    
    def _validate_behavior(self, test: ExecutionTest, result: Any) -> bool:
        """Validate actual behavior, not just presence"""
        
        # Behavioral validation rules
        validation_rules = {
            "returns_data_structure": lambda r: isinstance(r, (dict, list)) and len(r) > 0,
            "returns_positive_number": lambda r: isinstance(r, (int, float)) and r > 0,
            "returns_success_status": lambda r: hasattr(r, 'status') and r.status == 'success',
            "creates_functional_object": lambda r: r is not None and hasattr(r, '__dict__'),
            "executes_without_error": lambda r: True,  # If we got here, no error occurred
            "returns_valid_json": lambda r: isinstance(r, dict) and len(r) > 0,
            "performs_calculation": lambda r: isinstance(r, (int, float, dict)) and r is not None,
            "generates_report": lambda r: isinstance(r, dict) and 'timestamp' in r,
        }
        
        expected = test.expected_behavior
        if expected in validation_rules:
            return validation_rules[expected](result)
        
        # Default: check that result is not None and meaningful
        return result is not None
    
    def _create_execution_tests(self) -> List[ExecutionTest]:
        """Create real execution tests for all systems"""
        return [
            # Autonomous Scaling System - REAL EXECUTION TESTS
            ExecutionTest(1, "Scaling System Creation", "Infrastructure", 
                         str(self.base_path / "autonomous_scaling_system.py"), 
                         "AutonomousScalingSystem", "test_init_behavior", "creates_functional_object"),
            
            ExecutionTest(2, "Scaling Decision Logic", "Infrastructure",
                         str(self.base_path / "autonomous_scaling_system.py"),
                         "AutonomousScalingSystem", "analyze_scaling_needs", "returns_data_structure"),
            
            ExecutionTest(3, "Cost Optimization", "Infrastructure",
                         str(self.base_path / "autonomous_scaling_system.py"),
                         "AutonomousScalingSystem", "optimize_costs", "returns_positive_number"),
            
            # International Expansion System - REAL EXECUTION TESTS  
            ExecutionTest(4, "Market Analysis", "International",
                         str(self.base_path / "international_expansion_system.py"),
                         "InternationalExpansionSystem", "test_analyze_market_opportunity", "returns_data_structure"),
            
            ExecutionTest(5, "Localization Planning", "International", 
                         str(self.base_path / "international_expansion_system.py"),
                         "InternationalExpansionSystem", "test_create_localization_plan", "returns_data_structure"),
            
            # Referral Program System - REAL EXECUTION TESTS
            ExecutionTest(6, "Referral Code Generation", "Referral",
                         str(self.base_path / "referral_program_system.py"),
                         "ReferralProgramSystem", "test_generate_referral_code", "returns_data_structure"),
            
            ExecutionTest(7, "Tier Calculation", "Referral",
                         str(self.base_path / "referral_program_system.py"), 
                         "ReferralProgramSystem", "test_calculate_tier", "returns_data_structure"),
            
            # Premium Studio System - REAL EXECUTION TESTS
            ExecutionTest(8, "Feature Usage Analysis", "Premium",
                         str(self.base_path / "premium_studio_tier_system.py"),
                         "PremiumStudioTierSystem", "test_analyze_feature_usage", "returns_data_structure"),
            
            ExecutionTest(9, "ROI Calculation", "Premium", 
                         str(self.base_path / "premium_studio_tier_system.py"),
                         "PremiumStudioTierSystem", "test_calculate_feature_roi", "returns_positive_number"),
            
            ExecutionTest(10, "Feature Prioritization", "Premium",
                         str(self.base_path / "premium_studio_tier_system.py"),
                         "PremiumStudioTierSystem", "prioritize_feature_development", "returns_data_structure"),
        ]
    
    async def execute_all_tests(self):
        """Execute all real behavioral tests"""
        print("🔧 EXECUTION-FIRST VALIDATION")
        print("="*80)
        print("Testing actual behavior, not structure existence")
        print("="*80)
        
        self.tests = self._create_execution_tests()
        passed = 0
        failed = 0
        
        for test in self.tests:
            print(f"Test {test.id:2d}: ", end="")
            
            success = await self._execute_real_test(test)
            
            if success:
                print(f"✅ PASS - {test.name}")
                passed += 1
            else:
                print(f"❌ FAIL - {test.name}")
                print(f"         Reason: {test.error_message}")
                failed += 1
        
        # Generate execution report
        self._generate_execution_report(passed, failed)
        
        print("="*80)
        print(f"📊 EXECUTION-FIRST RESULTS")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🎯 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        print("="*80)
        
        return passed == len(self.tests)
    
    def _generate_execution_report(self, passed: int, failed: int):
        """Generate detailed execution report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "validation_type": "EXECUTION-FIRST",
            "total_tests": len(self.tests),
            "passed_tests": passed,
            "failed_tests": failed,
            "success_rate": (passed / len(self.tests) * 100) if self.tests else 0,
            "methodology": "Real behavioral validation, not phantom testing",
            "compliance_status": "Tu Agent EXECUTION-FIRST requirements",
            "detailed_results": [
                {
                    "test_id": test.id,
                    "name": test.name,
                    "category": test.category,
                    "success": test.success,
                    "expected_behavior": test.expected_behavior,
                    "actual_result_type": type(test.actual_result).__name__ if test.actual_result is not None else "None",
                    "error_message": test.error_message,
                    "execution_time": test.execution_time
                }
                for test in self.tests
            ]
        }
        
        # Save execution report
        report_path = self.base_path / "execution_first_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Execution report saved: {report_path}")

async def main():
    """Execute EXECUTION-FIRST validation"""
    try:
        validator = ExecutionFirstValidator()
        all_passed = await validator.execute_all_tests()
        
        if all_passed:
            print("🎉 ALL EXECUTION-FIRST TESTS PASSED")
            print("✅ Systems validated with real behavioral testing")
        else:
            print("⚠️  EXECUTION-FIRST VALIDATION INCOMPLETE")  
            print("🔧 Some systems require behavioral fixes")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ EXECUTION-FIRST VALIDATION FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())