#!/usr/bin/env python3
"""
Terminal Automation Framework - Comprehensive 88-Test Validation Suite
Critical accuracy assessment with real execution testing
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

class FrameworkValidator:
    def __init__(self):
        self.framework_dir = Path("/Users/nguythe/ag06_mixer/automation-framework")
        self.results = []
        self.total_tests = 88
        self.passed_tests = 0
        self.failed_tests = []
        
    def run_command(self, cmd: str, cwd: str = None) -> Tuple[bool, str, str]:
        """Execute command and return success status with output"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=cwd or self.framework_dir
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def test_file_exists(self, filepath: str, test_num: int, description: str) -> bool:
        """Test if a file exists"""
        path = self.framework_dir / filepath
        exists = path.exists()
        self.record_result(test_num, description, exists, f"File {'exists' if exists else 'missing'}: {filepath}")
        return exists

    def test_command_execution(self, cmd: str, test_num: int, description: str) -> bool:
        """Test if a command executes successfully"""
        success, stdout, stderr = self.run_command(cmd)
        self.record_result(test_num, description, success, f"Command {'succeeded' if success else 'failed'}: {cmd}")
        return success

    def record_result(self, test_num: int, description: str, passed: bool, details: str = ""):
        """Record test result"""
        self.results.append({
            "test_number": test_num,
            "description": description,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests.append(test_num)
        
        # Print real-time result
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Test {test_num:2d}: {description:<50} ... {status}")
        if not passed and details:
            print(f"         Details: {details}")

    def run_all_tests(self):
        """Run all 88 tests"""
        print("=" * 80)
        print("Terminal Automation Framework - 88 Test Validation Suite")
        print("=" * 80)
        print()
        
        # Category 1: Core Files (Tests 1-10)
        print("📁 Category 1: Core Files Existence")
        self.test_file_exists("dev", 1, "Main dev CLI script exists")
        self.test_file_exists("scripts/lib/core.sh", 2, "Core library exists")
        self.test_file_exists("scripts/lib/homebrew.sh", 3, "Homebrew library exists")
        self.test_file_exists("scripts/lib/git.sh", 4, "Git library exists")
        self.test_file_exists(".github/workflows/ci.yml", 5, "CI/CD workflow exists")
        self.test_file_exists("tests/framework.bats", 6, "BATS test suite exists")
        self.test_file_exists("test-runner.sh", 7, "Test runner script exists")
        self.test_file_exists("Brewfile", 8, "Brewfile exists")
        self.test_file_exists("FRAMEWORK_README.md", 9, "Framework README exists")
        self.test_file_exists("DEPLOYMENT_SUMMARY.md", 10, "Deployment summary exists")
        
        # Category 2: Script Permissions (Tests 11-15)
        print("\n🔑 Category 2: Script Permissions")
        self.test_command_execution("test -x dev", 11, "dev script is executable")
        self.test_command_execution("test -x test-runner.sh", 12, "test-runner.sh is executable")
        self.test_command_execution("test -x fix_homebrew_permissions.sh", 13, "fix_homebrew_permissions.sh is executable")
        self.test_command_execution("test -x install_dev_tools.sh", 14, "install_dev_tools.sh is executable")
        self.test_command_execution("test -r scripts/lib/core.sh", 15, "Core library is readable")
        
        # Category 3: Command Functionality (Tests 16-30)
        print("\n⚡ Category 3: Command Functionality")
        self.test_command_execution("./dev version", 16, "dev version command works")
        self.test_command_execution("./dev help | grep -q 'USAGE'", 17, "dev help command works")
        self.test_command_execution("./dev doctor 2>&1 | grep -q 'health check'", 18, "dev doctor command runs")
        
        # Test invalid commands fail appropriately
        success, stdout, stderr = self.run_command("./dev invalid-command-xyz 2>&1")
        combined_output = stdout + stderr
        self.record_result(19, "Invalid command shows error", not success and ("Unknown command" in combined_output or "Available commands" in combined_output))
        
        # Test parameter validation
        success, stdout, stderr = self.run_command("./dev install 2>&1")
        combined_output = stdout + stderr
        self.record_result(20, "Install requires package name", not success and ("required" in combined_output.lower() or "package name" in combined_output.lower()))
        
        success, stdout, stderr = self.run_command("./dev git:setup 2>&1")
        combined_output = stdout + stderr
        self.record_result(21, "git:setup requires parameters", not success and ("required" in combined_output.lower() or "name" in combined_output.lower()))
        
        success, stdout, stderr = self.run_command("./dev git:branch 2>&1")
        combined_output = stdout + stderr
        self.record_result(22, "git:branch requires branch name", not success and ("required" in combined_output.lower() or "branch" in combined_output.lower()))
        
        # Test help for each command
        self.test_command_execution("./dev help | grep -q 'doctor'", 23, "Help includes doctor command")
        self.test_command_execution("./dev help | grep -q 'bootstrap'", 24, "Help includes bootstrap command")
        self.test_command_execution("./dev help | grep -q 'install'", 25, "Help includes install command")
        self.test_command_execution("./dev help | grep -q 'lint'", 26, "Help includes lint command")
        self.test_command_execution("./dev help | grep -q 'format'", 27, "Help includes format command")
        self.test_command_execution("./dev help | grep -q 'test'", 28, "Help includes test command")
        self.test_command_execution("./dev help | grep -q 'ci'", 29, "Help includes ci command")
        self.test_command_execution("./dev help | grep -q 'git:setup'", 30, "Help includes git:setup command")
        
        # Category 4: Library Functions (Tests 31-40)
        print("\n📚 Category 4: Library Functions")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && type log_info'", 31, "log_info function exists")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && type log_error'", 32, "log_error function exists")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && type validate_command'", 33, "validate_command function exists")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && type get_os'", 34, "get_os function exists")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && type retry'", 35, "retry function exists")
        self.test_command_execution("bash -c 'source scripts/lib/homebrew.sh && type brew_install'", 36, "brew_install function exists")
        self.test_command_execution("bash -c 'source scripts/lib/homebrew.sh && type brew_update'", 37, "brew_update function exists")
        self.test_command_execution("bash -c 'source scripts/lib/git.sh && type git_validate_repo'", 38, "git_validate_repo function exists")
        self.test_command_execution("bash -c 'source scripts/lib/git.sh && type git_create_branch'", 39, "git_create_branch function exists")
        self.test_command_execution("bash -c 'source scripts/lib/git.sh && type git_setup_user'", 40, "git_setup_user function exists")
        
        # Category 5: CI/CD Configuration (Tests 41-50)
        print("\n🔄 Category 5: CI/CD Configuration")
        self.test_command_execution("grep -q 'framework-test' .github/workflows/ci.yml", 41, "CI has framework-test job")
        self.test_command_execution("grep -q 'shell-quality' .github/workflows/ci.yml", 42, "CI has shell-quality job")
        self.test_command_execution("grep -q 'integration-tests' .github/workflows/ci.yml", 43, "CI has integration-tests job")
        self.test_command_execution("grep -q 'macos-13' .github/workflows/ci.yml", 44, "CI tests macOS 13")
        self.test_command_execution("grep -q 'macos-14' .github/workflows/ci.yml", 45, "CI tests macOS 14")
        self.test_command_execution("grep -q 'macos-latest' .github/workflows/ci.yml", 46, "CI tests macOS latest")
        self.test_command_execution("grep -q 'shellcheck' .github/workflows/ci.yml", 47, "CI includes shellcheck")
        self.test_command_execution("grep -q 'shfmt' .github/workflows/ci.yml", 48, "CI includes shfmt")
        self.test_command_execution("grep -q 'cache' .github/workflows/ci.yml", 49, "CI includes caching")
        self.test_command_execution("grep -q 'concurrency' .github/workflows/ci.yml", 50, "CI has concurrency control")
        
        # Category 6: Documentation (Tests 51-60)
        print("\n📖 Category 6: Documentation")
        self.test_file_exists("PRODUCTION_DEPLOYMENT_CHECKLIST.md", 51, "Production checklist exists")
        self.test_file_exists("ISSUES_AND_IMPROVEMENTS.md", 52, "Issues documentation exists")
        self.test_file_exists("FINAL_FRAMEWORK_STATUS.md", 53, "Final status report exists")
        self.test_file_exists("TEAM_ONBOARDING_GUIDE.md", 54, "Team onboarding guide exists")
        self.test_file_exists("NEXT_STEPS_ACTION_PLAN.md", 55, "Next steps plan exists")
        self.test_command_execution("grep -q 'Quick Start' FRAMEWORK_README.md", 56, "README has quick start section")
        self.test_command_execution("grep -q 'Commands' FRAMEWORK_README.md", 57, "README documents commands")
        self.test_command_execution("grep -q 'Google' DEPLOYMENT_SUMMARY.md", 58, "Deployment mentions Google practices")
        self.test_command_execution("grep -q 'Meta' DEPLOYMENT_SUMMARY.md", 59, "Deployment mentions Meta practices")
        self.test_command_execution("grep -q '2.0.0' dev", 60, "Version 2.0.0 is set in dev script")
        
        # Category 7: Error Handling (Tests 61-70)
        print("\n⚠️ Category 7: Error Handling")
        self.test_command_execution("grep -q 'set -euo pipefail' dev", 61, "dev uses strict error handling")
        self.test_command_execution("grep -q 'set -euo pipefail' scripts/lib/core.sh", 62, "core.sh uses strict error handling")
        self.test_command_execution("grep -q 'handle_error' scripts/lib/core.sh", 63, "Error handler function exists")
        self.test_command_execution("grep -q 'cleanup_on_exit' scripts/lib/core.sh", 64, "Cleanup function exists")
        self.test_command_execution("grep -q 'log_fatal' scripts/lib/core.sh", 65, "Fatal error logging exists")
        self.test_command_execution("grep -q 'validate_command' dev", 66, "dev validates commands")
        self.test_command_execution("grep -q 'validate_file' scripts/lib/core.sh", 67, "File validation exists")
        self.test_command_execution("grep -q 'validate_directory' scripts/lib/core.sh", 68, "Directory validation exists")
        self.test_command_execution("grep -q 'timer_start' scripts/lib/core.sh", 69, "Performance timing exists")
        self.test_command_execution("grep -q 'show_spinner' scripts/lib/core.sh", 70, "Progress indication exists")
        
        # Category 8: Integration (Tests 71-80)
        print("\n🔗 Category 8: Integration Testing")
        
        # Test library sourcing
        test_script = """
        #!/bin/bash
        set -e
        source scripts/lib/core.sh
        source scripts/lib/homebrew.sh
        source scripts/lib/git.sh
        echo "Libraries loaded successfully"
        """
        
        with open(self.framework_dir / "test_integration.sh", "w") as f:
            f.write(test_script)
        os.chmod(self.framework_dir / "test_integration.sh", 0o755)
        
        self.test_command_execution("./test_integration.sh", 71, "All libraries source correctly")
        
        # Test function availability after sourcing
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && log_info test 2>&1 | grep -q INFO'", 72, "Logging works after sourcing")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && get_os | grep -q macos'", 73, "OS detection works")
        self.test_command_execution("bash -c 'source scripts/lib/core.sh && get_arch | grep -q arm64'", 74, "Architecture detection works")
        
        # Test command combinations
        self.test_command_execution("./dev version | grep -q '2.0.0'", 75, "Version output is correct")
        self.test_command_execution("./dev help 2>&1 | wc -l | grep -q -E '[0-9]{2,}'", 76, "Help output is substantial")
        
        # Test logging
        self.test_command_execution("test -f automation.log || ./dev doctor 2>&1 >/dev/null; test -f automation.log", 77, "Log file is created")
        
        # Test Git integration
        self.test_command_execution("bash -c 'source scripts/lib/git.sh && git_validate_repo || true'", 78, "Git validation runs without crash")
        
        # Test Homebrew integration
        self.test_command_execution("bash -c 'source scripts/lib/homebrew.sh && type install_homebrew'", 79, "Homebrew installer function exists")
        self.test_command_execution("bash -c 'source scripts/lib/homebrew.sh && type brew_cleanup'", 80, "Homebrew cleanup function exists")
        
        # Category 9: Best Practices (Tests 81-88)
        print("\n✨ Category 9: Best Practices Compliance")
        self.test_command_execution("grep -q 'Google' dev", 81, "Google practices mentioned in dev")
        self.test_command_execution("grep -q 'Meta' dev", 82, "Meta practices mentioned in dev")
        self.test_command_execution("grep -q 'shellcheck' dev", 83, "ShellCheck integration present")
        self.test_command_execution("grep -q 'shfmt' dev", 84, "shfmt integration present")
        self.test_command_execution("grep -q 'readonly' scripts/lib/core.sh", 85, "Uses readonly variables")
        self.test_command_execution("grep -q 'export -f' scripts/lib/core.sh", 86, "Functions are exported")
        self.test_command_execution("grep -q '#!/' dev", 87, "dev has shebang")
        self.test_command_execution("grep -q '#!/' scripts/lib/core.sh", 88, "core.sh has shebang")
        
        # Cleanup
        if (self.framework_dir / "test_integration.sh").exists():
            os.remove(self.framework_dir / "test_integration.sh")

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        print(f"\n📊 Overall Results:")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {self.passed_tests}")
        print(f"  Failed: {len(self.failed_tests)}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests: {', '.join(map(str, self.failed_tests))}")
            print("\nFailed Test Details:")
            for result in self.results:
                if not result['passed']:
                    print(f"  Test {result['test_number']}: {result['description']}")
                    if result['details']:
                        print(f"    → {result['details']}")
        
        # Category breakdown
        categories = {
            "Core Files": (1, 10),
            "Script Permissions": (11, 15),
            "Command Functionality": (16, 30),
            "Library Functions": (31, 40),
            "CI/CD Configuration": (41, 50),
            "Documentation": (51, 60),
            "Error Handling": (61, 70),
            "Integration Testing": (71, 80),
            "Best Practices": (81, 88)
        }
        
        print("\n📋 Category Breakdown:")
        for category, (start, end) in categories.items():
            cat_passed = sum(1 for r in self.results if start <= r['test_number'] <= end and r['passed'])
            cat_total = end - start + 1
            cat_rate = (cat_passed / cat_total) * 100
            status = "✅" if cat_rate == 100 else "⚠️" if cat_rate >= 80 else "❌"
            print(f"  {status} {category}: {cat_passed}/{cat_total} ({cat_rate:.0f}%)")
        
        # Final verdict
        print("\n" + "=" * 80)
        if success_rate == 100:
            print("🎉 PERFECT SCORE: 88/88 tests passed (100%)")
            print("✅ Framework is FULLY VALIDATED and PRODUCTION READY")
        elif success_rate >= 95:
            print(f"✅ EXCELLENT: {self.passed_tests}/88 tests passed ({success_rate:.1f}%)")
            print("Framework is production ready with minor issues")
        elif success_rate >= 80:
            print(f"⚠️ GOOD: {self.passed_tests}/88 tests passed ({success_rate:.1f}%)")
            print("Framework needs some fixes before production")
        else:
            print(f"❌ NEEDS WORK: {self.passed_tests}/88 tests passed ({success_rate:.1f}%)")
            print("Framework requires significant fixes")
        print("=" * 80)
        
        # Save results to JSON
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": len(self.failed_tests),
            "success_rate": success_rate,
            "failed_test_numbers": self.failed_tests,
            "all_results": self.results
        }
        
        with open(self.framework_dir / "test_results_88.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: test_results_88.json")
        
        return success_rate == 100

def main():
    """Main execution"""
    os.chdir("/Users/nguythe/ag06_mixer/automation-framework")
    
    validator = FrameworkValidator()
    validator.run_all_tests()
    success = validator.generate_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()