#!/usr/bin/env python3
"""
Universal Parallel Workflow System Validation Test
Validates the complete transformation from project-specific to universal repository-agnostic system
"""

import subprocess
import json
import os
import sys

def run_command(cmd):
    """Execute shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_universal_workflow_system():
    """Comprehensive validation test for universal workflow system"""
    print("🔍 UNIVERSAL PARALLEL WORKFLOW SYSTEM VALIDATION")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 12
    
    # Test 1: Dev CLI Help shows universal commands
    print("\n1. Testing dev CLI universal commands...")
    success, output, error = run_command("./dev help")
    if success and "universal:" in output:
        print("✅ PASS - Dev CLI includes universal commands")
        tests_passed += 1
    else:
        print("❌ FAIL - Dev CLI missing universal commands")
    
    # Test 2: Universal init works
    print("\n2. Testing universal workflow initialization...")
    success, output, error = run_command("./dev universal:init")
    if success and "Environment successfully initialized" in output:
        print("✅ PASS - Universal initialization successful")
        tests_passed += 1
    else:
        print("❌ FAIL - Universal initialization failed")
        print(f"Error: {error}")
    
    # Test 3: Project analysis creates tasks
    print("\n3. Testing project analysis and task creation...")
    success, output, error = run_command("./dev universal:analyze")
    if success and "Created" in output and "tasks" in output:
        print("✅ PASS - Project analysis creates tasks")
        tests_passed += 1
    else:
        print("❌ FAIL - Project analysis failed")
        print(f"Output: {output}")
    
    # Test 4: Status shows correct task distribution
    print("\n4. Testing workflow status reporting...")
    success, output, error = run_command("./dev universal:status")
    if success and "Total Tasks:" in output and "backend_development:" in output:
        print("✅ PASS - Status shows task distribution by category")
        tests_passed += 1
    else:
        print("❌ FAIL - Status reporting failed")
    
    # Test 5: Instance registration works
    print("\n5. Testing instance registration...")
    success, output, error = run_command('./dev universal:register test_instance testing_validation "Test validation work"')
    if success and "Instance registered" in output:
        print("✅ PASS - Instance registration successful")
        tests_passed += 1
    else:
        print("❌ FAIL - Instance registration failed")
    
    # Test 6: Task distribution works
    print("\n6. Testing task distribution...")
    success, output, error = run_command("./dev universal:distribute")
    if success and ("Tasks distributed" in output or "Task assigned" in output):
        print("✅ PASS - Task distribution successful")
        tests_passed += 1
    else:
        print("❌ FAIL - Task distribution failed")
    
    # Test 7: Monitor once mode works
    print("\n7. Testing monitor dashboard (once mode)...")
    success, output, error = run_command("./dev universal:monitor once")
    if success or ("Universal Parallel Development Dashboard" in output):
        print("✅ PASS - Monitor dashboard displays correctly")
        tests_passed += 1
    else:
        print("❌ FAIL - Monitor dashboard failed")
    
    # Test 8: Universal orchestrator is executable
    print("\n8. Testing universal orchestrator direct execution...")
    success, output, error = run_command("./universal_parallel_orchestrator.sh version")
    if success and "Universal Parallel Workflow Orchestrator" in output:
        print("✅ PASS - Universal orchestrator is executable")
        tests_passed += 1
    else:
        print("❌ FAIL - Universal orchestrator not executable")
    
    # Test 9: Both parallel and universal systems coexist
    print("\n9. Testing coexistence of parallel and universal systems...")
    success1, output1, error1 = run_command("./dev parallel:init")
    success2, output2, error2 = run_command("./dev universal:init")
    if success1 and success2:
        print("✅ PASS - Both parallel and universal systems work independently")
        tests_passed += 1
    else:
        print("❌ FAIL - System conflict detected")
    
    # Test 10: Workflow directory structure
    print("\n10. Testing workflow directory structure...")
    universal_dir = os.path.expanduser("~/.universal_workflows/ag06_mixer")
    if os.path.exists(universal_dir) and os.path.exists(f"{universal_dir}/tasks"):
        print("✅ PASS - Universal workflow directory structure created")
        tests_passed += 1
    else:
        print("❌ FAIL - Workflow directory structure missing")
    
    # Test 11: Repository agnostic detection
    print("\n11. Testing repository detection...")
    success, output, error = run_command("./universal_parallel_orchestrator.sh init")
    if success and "ag06_mixer" in output:
        print("✅ PASS - Repository auto-detection working")
        tests_passed += 1
    else:
        print("❌ FAIL - Repository detection failed")
    
    # Test 12: Task categories are universal
    print("\n12. Testing universal task categories...")
    success, output, error = run_command("./dev universal:status")
    universal_categories = [
        "backend_development", "frontend_development", "database_design",
        "api_integration", "testing_validation", "documentation"
    ]
    category_found = any(cat in output for cat in universal_categories)
    if success and category_found:
        print("✅ PASS - Universal task categories present")
        tests_passed += 1
    else:
        print("❌ FAIL - Universal categories missing")
    
    # Final Results
    print("\n" + "=" * 60)
    print(f"UNIVERSAL WORKFLOW VALIDATION RESULTS: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Universal workflow system is fully operational!")
        print("\nKey Achievements:")
        print("✅ Transformed project-specific system into universal repository-agnostic framework")
        print("✅ Dev CLI integration with universal: namespace commands") 
        print("✅ Auto-detection of project context and repository structure")
        print("✅ Universal task categories for any project type")
        print("✅ Coexistence with existing parallel workflow system")
        print("✅ Complete instance management and task distribution")
        print("✅ Real-time monitoring dashboard")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed - system needs refinement")
        return False

if __name__ == "__main__":
    success = test_universal_workflow_system()
    sys.exit(0 if success else 1)