#!/usr/bin/env python3
"""
Test Specialized Agents System
Simple validation script to test the deployed specialized agent system
"""

import asyncio
import json
import sys
from pathlib import Path

async def test_system():
    """Test the specialized agents system"""
    
    print("🧪 Testing Specialized Agents System...")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Import all modules
    try:
        from research_agent import AdvancedResearchAgent
        from architecture_agent import AdvancedArchitectureAgent
        from performance_agent import AdvancedPerformanceAgent
        from quality_assurance_agent import AdvancedQualityAssuranceAgent
        from deployment_agent import AdvancedDeploymentAgent
        from unified_agent_orchestrator import UnifiedAgentOrchestrator
        
        test_results.append({"test": "Module Imports", "status": "PASS"})
        print("✅ Module Imports: PASS")
        
    except Exception as e:
        test_results.append({"test": "Module Imports", "status": "FAIL", "error": str(e)})
        print(f"❌ Module Imports: FAIL - {e}")
        return test_results
    
    # Test 2: Create agent instances
    try:
        research_agent = AdvancedResearchAgent()
        architecture_agent = AdvancedArchitectureAgent()
        performance_agent = AdvancedPerformanceAgent()
        qa_agent = AdvancedQualityAssuranceAgent()
        deployment_agent = AdvancedDeploymentAgent()
        orchestrator = UnifiedAgentOrchestrator()
        
        test_results.append({"test": "Agent Creation", "status": "PASS"})
        print("✅ Agent Creation: PASS")
        
    except Exception as e:
        test_results.append({"test": "Agent Creation", "status": "FAIL", "error": str(e)})
        print(f"❌ Agent Creation: FAIL - {e}")
        return test_results
    
    # Test 3: Test orchestrator status (without starting agents)
    try:
        status = await orchestrator.get_orchestration_status()
        
        test_results.append({"test": "Orchestrator Status", "status": "PASS"})
        print("✅ Orchestrator Status: PASS")
        print(f"   - Total agents configured: {status['total_agents']}")
        
    except Exception as e:
        test_results.append({"test": "Orchestrator Status", "status": "FAIL", "error": str(e)})
        print(f"❌ Orchestrator Status: FAIL - {e}")
    
    # Test 4: Test file system operations
    try:
        test_file = Path("test_system_validation.json")
        test_data = {"test": "file_operations", "timestamp": "2025-01-01"}
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        test_file.unlink()  # Clean up
        
        assert loaded_data == test_data
        
        test_results.append({"test": "File Operations", "status": "PASS"})
        print("✅ File Operations: PASS")
        
    except Exception as e:
        test_results.append({"test": "File Operations", "status": "FAIL", "error": str(e)})
        print(f"❌ File Operations: FAIL - {e}")
    
    # Test 5: Quick agent functionality test (without full startup)
    try:
        # Test research agent functionality
        research_agent = AdvancedResearchAgent()
        research_areas = research_agent.research_areas
        assert len(research_areas) > 0
        
        test_results.append({"test": "Agent Functionality", "status": "PASS"})
        print("✅ Agent Functionality: PASS")
        print(f"   - Research areas configured: {len(research_areas)}")
        
    except Exception as e:
        test_results.append({"test": "Agent Functionality", "status": "FAIL", "error": str(e)})
        print(f"❌ Agent Functionality: FAIL - {e}")
    
    # Test Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["status"] == "PASS")
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - System is ready for deployment!")
        return test_results, True
    else:
        print("\n⚠️ SOME TESTS FAILED - Please review errors before deployment.")
        return test_results, False

async def main():
    """Main test function"""
    try:
        results, success = await test_system()
        
        # Save test results
        with open("system_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📝 Test results saved to: system_test_results.json")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)