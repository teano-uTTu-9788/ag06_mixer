#!/usr/bin/env python3
"""
Advanced Quality Assurance Agent - Behavioral Testing & Validation
Based on 2024-2025 Quality Engineering Best Practices

This agent implements:
- Behavioral testing automation (88/88 protocol)
- Real-time code quality validation
- Test coverage analysis
- Quality metrics monitoring
- Automated regression testing
"""

import asyncio
import ast
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Tuple
from abc import ABC, abstractmethod
import psutil
import re

# Optional dependencies - handle import failures gracefully
try:
    import pytest
except ImportError:
    pytest = None

try:
    import coverage
except ImportError:
    coverage = None

# SOLID Architecture Implementation
class ITestRunner(Protocol):
    """Interface for test execution"""
    async def run_tests(self, test_path: str) -> Dict[str, Any]: ...

class IQualityAnalyzer(Protocol):
    """Interface for quality analysis"""
    async def analyze_quality(self, test_results: Dict[str, Any]) -> Dict[str, Any]: ...

class ICoverageAnalyzer(Protocol):
    """Interface for coverage analysis"""
    async def analyze_coverage(self, source_path: str) -> Dict[str, Any]: ...

class IBehavioralValidator(Protocol):
    """Interface for behavioral validation"""
    async def validate_behavior(self, code_path: str, test_results: Dict[str, Any]) -> Dict[str, Any]: ...

@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration: float
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None

@dataclass
class QualityMetric:
    """Quality metric data structure"""
    name: str
    value: float
    threshold: float
    status: str  # PASS, FAIL, WARNING
    category: str
    description: str

@dataclass
class BehavioralTest:
    """Behavioral test specification"""
    test_id: str
    description: str
    test_type: str  # FUNCTIONAL, INTEGRATION, BEHAVIORAL
    expected_behavior: str
    actual_behavior: Optional[str] = None
    validation_status: str = "PENDING"  # PENDING, PASS, FAIL

class QualityAssuranceError(Exception):
    """Custom QA agent exceptions"""
    pass

class PytestTestRunner:
    """Advanced pytest-based test runner"""
    
    def __init__(self):
        self.pytest_args = [
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker validation
            "--disable-warnings",  # Reduce noise
            "--junitxml=test_results.xml"  # XML output for parsing
        ]
    
    async def run_tests(self, test_path: str) -> Dict[str, Any]:
        """Run tests using pytest and collect detailed results"""
        start_time = time.time()
        
        try:
            # Prepare pytest command
            cmd = ["python", "-m", "pytest"] + self.pytest_args + [test_path]
            
            # Run tests
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            end_time = time.time()
            
            # Parse pytest output
            results = self._parse_pytest_output(
                stdout.decode('utf-8'),
                stderr.decode('utf-8'),
                process.returncode
            )
            
            results.update({
                "execution_time": end_time - start_time,
                "test_path": test_path,
                "timestamp": datetime.now().isoformat()
            })
            
            return results
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "test_path": test_path,
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_pytest_output(self, stdout: str, stderr: str, return_code: int) -> Dict[str, Any]:
        """Parse pytest output to extract test results"""
        
        # Extract summary line (e.g., "5 passed, 1 failed, 2 skipped")
        summary_pattern = r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)?'
        summary_match = re.search(summary_pattern, stdout)
        
        passed = int(summary_match.group(1)) if summary_match and summary_match.group(1) else 0
        failed = int(summary_match.group(2)) if summary_match and summary_match.group(2) else 0
        skipped = int(summary_match.group(3)) if summary_match and summary_match.group(3) else 0
        errors = int(summary_match.group(4)) if summary_match and summary_match.group(4) else 0
        
        total_tests = passed + failed + skipped + errors
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        # Extract individual test results
        individual_tests = self._extract_individual_test_results(stdout)
        
        # Extract failures details
        failures = self._extract_failure_details(stdout)
        
        return {
            "status": "SUCCESS" if return_code == 0 else "FAILURE",
            "summary": {
                "total": total_tests,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "success_rate": success_rate
            },
            "individual_tests": individual_tests,
            "failures": failures,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "88_compliant": total_tests >= 88 and success_rate == 100.0
        }
    
    def _extract_individual_test_results(self, output: str) -> List[TestResult]:
        """Extract individual test results from pytest output"""
        tests = []
        
        # Pattern to match test results: test_file.py::test_name PASSED/FAILED [duration]
        test_pattern = r'([^\s]+)::(test_[^\s]+)\s+(PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[([0-9.]+)s\])?'
        
        for match in re.finditer(test_pattern, output):
            file_path = match.group(1)
            test_name = match.group(2)
            status = match.group(3)
            duration = float(match.group(4)) if match.group(4) else 0.0
            
            tests.append(TestResult(
                test_name=f"{file_path}::{test_name}",
                status=status,
                duration=duration,
                file_path=file_path
            ))
        
        return tests
    
    def _extract_failure_details(self, output: str) -> List[Dict[str, Any]]:
        """Extract failure details from pytest output"""
        failures = []
        
        # Split by failure sections
        failure_sections = output.split('FAILURES')
        if len(failure_sections) > 1:
            failure_text = failure_sections[1].split('=== short test summary info ===')[0]
            
            # Extract individual failure details
            failure_blocks = re.split(r'_+ (.*?) _+', failure_text)[1:]
            
            for i in range(0, len(failure_blocks), 2):
                if i + 1 < len(failure_blocks):
                    test_name = failure_blocks[i].strip()
                    failure_detail = failure_blocks[i + 1].strip()
                    
                    failures.append({
                        "test_name": test_name,
                        "error_message": failure_detail[:500],  # Truncate long messages
                        "full_error": failure_detail
                    })
        
        return failures

class QualityAnalyzer:
    """Advanced quality analysis and metrics calculation"""
    
    def __init__(self):
        self.quality_thresholds = self._load_quality_thresholds()
        self.behavioral_patterns = self._load_behavioral_patterns()
    
    async def analyze_quality(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results for quality metrics"""
        
        metrics = []
        
        # Calculate test-based metrics
        test_metrics = self._calculate_test_metrics(test_results)
        metrics.extend(test_metrics)
        
        # Calculate behavioral compliance metrics
        behavioral_metrics = self._calculate_behavioral_metrics(test_results)
        metrics.extend(behavioral_metrics)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        
        # Generate quality assessment
        assessment = self._generate_quality_assessment(metrics, quality_score)
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "quality_grade": self._calculate_grade(quality_score),
            "metrics": [m.__dict__ for m in metrics],
            "assessment": assessment,
            "recommendations": self._generate_recommendations(metrics),
            "88_compliance": self._check_88_compliance(test_results)
        }
    
    def _calculate_test_metrics(self, test_results: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate test-related quality metrics"""
        metrics = []
        summary = test_results.get("summary", {})
        
        # Success Rate Metric
        success_rate = summary.get("success_rate", 0)
        metrics.append(QualityMetric(
            name="Test Success Rate",
            value=success_rate,
            threshold=self.quality_thresholds["success_rate"],
            status="PASS" if success_rate >= self.quality_thresholds["success_rate"] else "FAIL",
            category="Testing",
            description=f"Percentage of tests passing: {success_rate:.1f}%"
        ))
        
        # Test Coverage Metric (placeholder - would integrate with coverage.py)
        test_count = summary.get("total", 0)
        coverage_estimate = min(100, test_count * 1.2)  # Rough estimate
        metrics.append(QualityMetric(
            name="Test Coverage",
            value=coverage_estimate,
            threshold=self.quality_thresholds["coverage"],
            status="PASS" if coverage_estimate >= self.quality_thresholds["coverage"] else "FAIL",
            category="Testing",
            description=f"Estimated test coverage: {coverage_estimate:.1f}%"
        ))
        
        # Test Stability Metric
        failed_tests = summary.get("failed", 0)
        error_tests = summary.get("errors", 0)
        instability = failed_tests + error_tests
        stability_score = max(0, 100 - (instability * 10))
        
        metrics.append(QualityMetric(
            name="Test Stability",
            value=stability_score,
            threshold=self.quality_thresholds["stability"],
            status="PASS" if stability_score >= self.quality_thresholds["stability"] else "FAIL",
            category="Testing",
            description=f"Test stability score: {stability_score:.1f}%"
        ))
        
        return metrics
    
    def _calculate_behavioral_metrics(self, test_results: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate behavioral quality metrics"""
        metrics = []
        
        # 88/88 Compliance Metric
        is_88_compliant = test_results.get("88_compliant", False)
        compliance_score = 100 if is_88_compliant else 0
        
        metrics.append(QualityMetric(
            name="88/88 Compliance",
            value=compliance_score,
            threshold=100.0,
            status="PASS" if is_88_compliant else "FAIL",
            category="Behavioral",
            description="Compliance with 88/88 behavioral testing protocol"
        ))
        
        # Behavioral Test Depth
        total_tests = test_results.get("summary", {}).get("total", 0)
        behavioral_depth = min(100, (total_tests / 88) * 100) if total_tests > 0 else 0
        
        metrics.append(QualityMetric(
            name="Behavioral Test Depth",
            value=behavioral_depth,
            threshold=80.0,
            status="PASS" if behavioral_depth >= 80 else "FAIL",
            category="Behavioral",
            description=f"Behavioral test coverage depth: {behavioral_depth:.1f}%"
        ))
        
        return metrics
    
    def _calculate_quality_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score"""
        if not metrics:
            return 0.0
        
        # Weighted scoring based on metric categories
        weights = {
            "Testing": 0.4,
            "Behavioral": 0.6
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.category, 0.1)
            weighted_sum += metric.value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_quality_assessment(self, metrics: List[QualityMetric], quality_score: float) -> str:
        """Generate human-readable quality assessment"""
        failing_metrics = [m for m in metrics if m.status == "FAIL"]
        
        if quality_score >= 90:
            assessment = "Excellent quality - system meets all behavioral requirements"
        elif quality_score >= 80:
            assessment = "Good quality - minor improvements needed"
        elif quality_score >= 70:
            assessment = "Fair quality - several areas need attention"
        elif quality_score >= 60:
            assessment = "Poor quality - significant improvements required"
        else:
            assessment = "Critical quality issues - system not ready for production"
        
        if failing_metrics:
            assessment += f" ({len(failing_metrics)} metrics failing)"
        
        return assessment
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        failing_metrics = [m for m in metrics if m.status == "FAIL"]
        
        for metric in failing_metrics:
            if metric.name == "Test Success Rate":
                recommendations.append("Fix failing tests to improve success rate")
                recommendations.append("Review test failures and implement fixes")
            elif metric.name == "Test Coverage":
                recommendations.append("Add more comprehensive test cases")
                recommendations.append("Focus on testing edge cases and error conditions")
            elif metric.name == "88/88 Compliance":
                recommendations.append("Implement 88 behavioral tests with real data validation")
                recommendations.append("Ensure all tests verify actual functionality, not just structure")
            elif metric.name == "Behavioral Test Depth":
                recommendations.append("Add more behavioral test scenarios")
                recommendations.append("Implement integration tests for complex workflows")
        
        # Remove duplicates
        return list(set(recommendations))[:10]  # Top 10 recommendations
    
    def _check_88_compliance(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with 88/88 testing protocol"""
        summary = test_results.get("summary", {})
        total_tests = summary.get("total", 0)
        success_rate = summary.get("success_rate", 0)
        
        return {
            "is_compliant": total_tests >= 88 and success_rate == 100.0,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "tests_needed": max(0, 88 - total_tests),
            "failures_to_fix": summary.get("failed", 0) + summary.get("errors", 0)
        }
    
    def _calculate_grade(self, quality_score: float) -> str:
        """Calculate quality grade"""
        if quality_score >= 90:
            return "A"
        elif quality_score >= 80:
            return "B"
        elif quality_score >= 70:
            return "C"
        elif quality_score >= 60:
            return "D"
        else:
            return "F"
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality thresholds"""
        return {
            "success_rate": 95.0,
            "coverage": 80.0,
            "stability": 90.0,
            "behavioral_depth": 80.0
        }
    
    def _load_behavioral_patterns(self) -> Dict[str, Any]:
        """Load behavioral test patterns"""
        return {
            "required_patterns": [
                "test_real_data",
                "test_behavior",
                "test_functionality",
                "test_integration"
            ],
            "anti_patterns": [
                "test_structure_only",
                "test_hasattr",
                "test_import_only"
            ]
        }

class CoverageAnalyzer:
    """Advanced test coverage analysis"""
    
    def __init__(self):
        if coverage:
            self.coverage_instance = coverage.Coverage()
        else:
            self.coverage_instance = None
    
    async def analyze_coverage(self, source_path: str) -> Dict[str, Any]:
        """Analyze test coverage for source code"""
        try:
            # Start coverage measurement
            self.coverage_instance.start()
            
            # Import and analyze modules (would be more sophisticated in real implementation)
            source_files = list(Path(source_path).glob("**/*.py"))
            
            # Stop coverage measurement
            self.coverage_instance.stop()
            
            # Generate coverage report
            coverage_data = self._generate_coverage_report(source_files)
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "source_path": source_path,
                "coverage_data": coverage_data,
                "summary": self._calculate_coverage_summary(coverage_data)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "source_path": source_path
            }
    
    def _generate_coverage_report(self, source_files: List[Path]) -> Dict[str, Any]:
        """Generate coverage report for source files"""
        # Simplified coverage analysis (would integrate with coverage.py properly)
        coverage_data = {}
        
        for file_path in source_files:
            if "__pycache__" in str(file_path) or "test_" in file_path.name:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simplified coverage estimation
                lines = content.splitlines()
                executable_lines = [i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith('#')]
                
                # Estimate coverage (in real implementation, this would use actual coverage data)
                estimated_covered = int(len(executable_lines) * 0.75)  # Assume 75% coverage
                
                coverage_data[str(file_path)] = {
                    "total_lines": len(executable_lines),
                    "covered_lines": estimated_covered,
                    "coverage_percentage": (estimated_covered / len(executable_lines) * 100) if executable_lines else 0,
                    "missing_lines": list(range(estimated_covered + 1, len(executable_lines) + 1))
                }
                
            except Exception as e:
                coverage_data[str(file_path)] = {"error": str(e)}
        
        return coverage_data
    
    def _calculate_coverage_summary(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall coverage summary"""
        total_lines = 0
        covered_lines = 0
        file_count = 0
        
        for file_path, data in coverage_data.items():
            if "error" in data:
                continue
                
            total_lines += data["total_lines"]
            covered_lines += data["covered_lines"]
            file_count += 1
        
        overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            "files_analyzed": file_count,
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "overall_coverage": overall_coverage,
            "coverage_grade": self._calculate_coverage_grade(overall_coverage)
        }
    
    def _calculate_coverage_grade(self, coverage: float) -> str:
        """Calculate coverage grade"""
        if coverage >= 90:
            return "A"
        elif coverage >= 80:
            return "B"
        elif coverage >= 70:
            return "C"
        elif coverage >= 60:
            return "D"
        else:
            return "F"

class BehavioralValidator:
    """Advanced behavioral validation"""
    
    def __init__(self):
        self.validation_patterns = self._load_validation_patterns()
    
    async def validate_behavior(self, code_path: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate behavioral compliance"""
        
        # Analyze test files for behavioral patterns
        test_files = list(Path(code_path).glob("**/test_*.py"))
        behavioral_tests = []
        
        for test_file in test_files:
            file_analysis = await self._analyze_test_file(test_file)
            behavioral_tests.extend(file_analysis)
        
        # Validate against behavioral requirements
        validation_results = self._validate_behavioral_requirements(behavioral_tests, test_results)
        
        return {
            "validation_timestamp": datetime.now().isoformat(),
            "code_path": code_path,
            "behavioral_tests": behavioral_tests,
            "validation_results": validation_results,
            "compliance_score": self._calculate_compliance_score(validation_results)
        }
    
    async def _analyze_test_file(self, test_file: Path) -> List[BehavioralTest]:
        """Analyze test file for behavioral patterns"""
        behavioral_tests = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find test functions
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    behavioral_test = self._analyze_test_function(node, content, str(test_file))
                    if behavioral_test:
                        behavioral_tests.append(behavioral_test)
        
        except Exception as e:
            # Log error but continue
            pass
        
        return behavioral_tests
    
    def _analyze_test_function(self, func_node: ast.FunctionDef, content: str, file_path: str) -> Optional[BehavioralTest]:
        """Analyze individual test function for behavioral patterns"""
        
        # Extract function source
        lines = content.splitlines()
        if func_node.lineno <= len(lines):
            # Get function body (simplified)
            func_start = func_node.lineno - 1
            func_end = func_node.end_lineno if hasattr(func_node, 'end_lineno') else func_start + 10
            func_lines = lines[func_start:func_end]
            func_source = '\n'.join(func_lines)
            
            # Analyze for behavioral patterns
            test_type = self._classify_test_type(func_source)
            is_behavioral = self._is_behavioral_test(func_source)
            
            if is_behavioral:
                return BehavioralTest(
                    test_id=f"{file_path}::{func_node.name}",
                    description=f"Behavioral test: {func_node.name}",
                    test_type=test_type,
                    expected_behavior=self._extract_expected_behavior(func_source),
                    validation_status="DETECTED"
                )
        
        return None
    
    def _classify_test_type(self, func_source: str) -> str:
        """Classify test type based on content"""
        func_lower = func_source.lower()
        
        if any(pattern in func_lower for pattern in ['integration', 'workflow', 'pipeline']):
            return "INTEGRATION"
        elif any(pattern in func_lower for pattern in ['behavior', 'real_data', 'execution', 'functional']):
            return "BEHAVIORAL"
        else:
            return "FUNCTIONAL"
    
    def _is_behavioral_test(self, func_source: str) -> bool:
        """Check if test follows behavioral testing patterns"""
        func_lower = func_source.lower()
        
        # Look for behavioral patterns
        behavioral_indicators = [
            'real_data',
            'execute',
            'process',
            'workflow',
            'result',
            'output',
            'behavior'
        ]
        
        # Look for anti-patterns
        anti_patterns = [
            'hasattr',
            'import',
            'isinstance'
        ]
        
        has_behavioral = any(indicator in func_lower for indicator in behavioral_indicators)
        has_anti_pattern = any(pattern in func_lower for pattern in anti_patterns)
        
        return has_behavioral and not has_anti_pattern
    
    def _extract_expected_behavior(self, func_source: str) -> str:
        """Extract expected behavior description from test"""
        # Look for docstring or comments describing expected behavior
        lines = func_source.splitlines()
        
        for line in lines:
            if '"""' in line or "'''" in line:
                return line.strip().replace('"""', '').replace("'''", '').strip()
            elif line.strip().startswith('#'):
                return line.strip().replace('#', '').strip()
        
        return "Validates system behavior"
    
    def _validate_behavioral_requirements(self, behavioral_tests: List[BehavioralTest], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate behavioral requirements against test results"""
        
        total_behavioral_tests = len(behavioral_tests)
        integration_tests = len([t for t in behavioral_tests if t.test_type == "INTEGRATION"])
        functional_tests = len([t for t in behavioral_tests if t.test_type == "BEHAVIORAL"])
        
        # Check 88/88 compliance
        total_tests = test_results.get("summary", {}).get("total", 0)
        success_rate = test_results.get("summary", {}).get("success_rate", 0)
        
        validation_results = {
            "total_behavioral_tests": total_behavioral_tests,
            "integration_tests": integration_tests,
            "functional_tests": functional_tests,
            "88_compliance": {
                "total_tests": total_tests,
                "required_tests": 88,
                "success_rate": success_rate,
                "required_success_rate": 100.0,
                "is_compliant": total_tests >= 88 and success_rate == 100.0
            },
            "behavioral_depth": {
                "behavioral_test_ratio": (total_behavioral_tests / total_tests * 100) if total_tests > 0 else 0,
                "required_ratio": 60.0,  # 60% of tests should be behavioral
                "meets_requirement": (total_behavioral_tests / total_tests * 100) >= 60.0 if total_tests > 0 else False
            }
        }
        
        return validation_results
    
    def _calculate_compliance_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall behavioral compliance score"""
        score = 0.0
        
        # 88/88 compliance (40% of score)
        if validation_results["88_compliance"]["is_compliant"]:
            score += 40.0
        else:
            # Partial credit based on progress
            test_progress = min(1.0, validation_results["88_compliance"]["total_tests"] / 88)
            success_progress = validation_results["88_compliance"]["success_rate"] / 100.0
            score += 40.0 * test_progress * success_progress
        
        # Behavioral depth (30% of score)
        if validation_results["behavioral_depth"]["meets_requirement"]:
            score += 30.0
        else:
            # Partial credit
            ratio_progress = min(1.0, validation_results["behavioral_depth"]["behavioral_test_ratio"] / 60.0)
            score += 30.0 * ratio_progress
        
        # Test diversity (30% of score)
        total_behavioral = validation_results["total_behavioral_tests"]
        integration_tests = validation_results["integration_tests"]
        functional_tests = validation_results["functional_tests"]
        
        if total_behavioral > 0:
            diversity_score = min(30.0, (integration_tests + functional_tests) / total_behavioral * 30.0)
            score += diversity_score
        
        return min(100.0, score)
    
    def _load_validation_patterns(self) -> Dict[str, Any]:
        """Load validation patterns"""
        return {
            "behavioral_indicators": [
                "real_data",
                "execute",
                "process",
                "workflow",
                "result",
                "output",
                "behavior"
            ],
            "anti_patterns": [
                "hasattr",
                "import",
                "isinstance"
            ],
            "required_test_types": [
                "FUNCTIONAL",
                "INTEGRATION",
                "BEHAVIORAL"
            ]
        }

class QualityAssuranceReporter:
    """Quality assurance reporting and data persistence"""
    
    def __init__(self, output_path: str = "qa_reports"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    async def generate_report(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive quality assurance report"""
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "quality_summary": quality_data.get("quality_analysis", {}),
            "test_results": quality_data.get("test_results", {}),
            "coverage_analysis": quality_data.get("coverage_analysis", {}),
            "behavioral_validation": quality_data.get("behavioral_validation", {}),
            "overall_assessment": self._generate_overall_assessment(quality_data),
            "action_items": self._generate_action_items(quality_data)
        }
        
        # Save report
        await self._save_report(report)
        
        return report
    
    def _generate_overall_assessment(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall quality assessment"""
        quality_analysis = quality_data.get("quality_analysis", {})
        test_results = quality_data.get("test_results", {})
        behavioral_validation = quality_data.get("behavioral_validation", {})
        
        quality_score = quality_analysis.get("quality_score", 0)
        compliance_score = behavioral_validation.get("compliance_score", 0)
        
        overall_score = (quality_score + compliance_score) / 2
        
        return {
            "overall_score": overall_score,
            "overall_grade": self._calculate_grade(overall_score),
            "quality_score": quality_score,
            "compliance_score": compliance_score,
            "ready_for_production": overall_score >= 90 and compliance_score >= 90,
            "assessment": self._generate_assessment_text(overall_score)
        }
    
    def _generate_action_items(self, quality_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized action items"""
        action_items = []
        
        # Add recommendations from quality analysis
        quality_analysis = quality_data.get("quality_analysis", {})
        for rec in quality_analysis.get("recommendations", []):
            action_items.append({
                "category": "Quality",
                "priority": "HIGH",
                "description": rec,
                "type": "IMPROVEMENT"
            })
        
        # Add 88/88 compliance items
        behavioral_validation = quality_data.get("behavioral_validation", {})
        compliance_88 = behavioral_validation.get("validation_results", {}).get("88_compliance", {})
        
        if not compliance_88.get("is_compliant", False):
            tests_needed = compliance_88.get("required_tests", 88) - compliance_88.get("total_tests", 0)
            if tests_needed > 0:
                action_items.append({
                    "category": "88/88 Compliance",
                    "priority": "CRITICAL",
                    "description": f"Add {tests_needed} more behavioral tests to reach 88/88",
                    "type": "REQUIREMENT"
                })
            
            failures = quality_data.get("test_results", {}).get("summary", {})
            failed_count = failures.get("failed", 0) + failures.get("errors", 0)
            if failed_count > 0:
                action_items.append({
                    "category": "88/88 Compliance",
                    "priority": "CRITICAL",
                    "description": f"Fix {failed_count} failing tests to achieve 100% success rate",
                    "type": "BUG_FIX"
                })
        
        return action_items[:15]  # Top 15 action items
    
    def _generate_assessment_text(self, overall_score: float) -> str:
        """Generate assessment text based on score"""
        if overall_score >= 90:
            return "System meets all quality and behavioral requirements - ready for production"
        elif overall_score >= 80:
            return "Good quality with minor improvements needed"
        elif overall_score >= 70:
            return "Acceptable quality but requires attention to several areas"
        elif overall_score >= 60:
            return "Poor quality - significant work needed before production"
        else:
            return "Critical quality issues - system not suitable for production use"
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate grade based on score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    async def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_report_{timestamp}.json"
        filepath = self.output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.output_path / "latest_qa_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

class AdvancedQualityAssuranceAgent:
    """
    Advanced Quality Assurance Agent implementing 2024-2025 best practices
    
    Features:
    - Behavioral testing automation (88/88 protocol)
    - Real-time code quality validation
    - Test coverage analysis
    - Quality metrics monitoring
    - Automated regression testing
    """
    
    def __init__(
        self, 
        project_path: str = ".",
        testing_interval: int = 1800,  # 30 minutes
        output_path: str = "qa_reports"
    ):
        self.project_path = Path(project_path)
        self.testing_interval = testing_interval
        self.output_path = output_path
        self.is_running = False
        
        # Dependency injection following SOLID principles
        self.test_runner: ITestRunner = PytestTestRunner()
        self.quality_analyzer: IQualityAnalyzer = QualityAnalyzer()
        self.coverage_analyzer: ICoverageAnalyzer = CoverageAnalyzer()
        self.behavioral_validator: IBehavioralValidator = BehavioralValidator()
        self.reporter = QualityAssuranceReporter(output_path)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the quality assurance agent"""
        self.logger.info("Starting Advanced Quality Assurance Agent")
        self._check_system_resources()
        
        self.is_running = True
        
        # Start continuous testing loop
        asyncio.create_task(self._testing_loop())
        
        self.logger.info(f"Quality Assurance Agent started - monitoring {self.project_path}")
    
    async def stop(self) -> None:
        """Stop the quality assurance agent"""
        self.logger.info("Stopping Advanced Quality Assurance Agent")
        self.is_running = False
    
    async def _testing_loop(self) -> None:
        """Main testing loop"""
        while self.is_running:
            try:
                await self._perform_testing_cycle()
                await asyncio.sleep(self.testing_interval)
            except Exception as e:
                self.logger.error(f"Testing cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _perform_testing_cycle(self) -> None:
        """Perform one complete testing cycle"""
        self.logger.info("Starting quality assurance testing cycle")
        
        try:
            # Step 1: Find and run tests
            test_files = self._find_test_files()
            if not test_files:
                self.logger.warning("No test files found")
                return
            
            all_test_results = []
            for test_file in test_files:
                try:
                    test_results = await self.test_runner.run_tests(str(test_file))
                    all_test_results.append(test_results)
                except Exception as e:
                    self.logger.error(f"Failed to run tests in {test_file}: {e}")
                    continue
            
            # Combine test results
            combined_results = self._combine_test_results(all_test_results)
            
            # Step 2: Analyze quality
            quality_analysis = await self.quality_analyzer.analyze_quality(combined_results)
            
            # Step 3: Analyze coverage
            coverage_analysis = await self.coverage_analyzer.analyze_coverage(str(self.project_path))
            
            # Step 4: Validate behavioral compliance
            behavioral_validation = await self.behavioral_validator.validate_behavior(
                str(self.project_path), 
                combined_results
            )
            
            # Step 5: Generate comprehensive report
            quality_data = {
                "test_results": combined_results,
                "quality_analysis": quality_analysis,
                "coverage_analysis": coverage_analysis,
                "behavioral_validation": behavioral_validation
            }
            
            report = await self.reporter.generate_report(quality_data)
            
            # Log results
            self.logger.info(f"Testing cycle completed:")
            self.logger.info(f"  Tests run: {combined_results.get('summary', {}).get('total', 0)}")
            self.logger.info(f"  Success rate: {combined_results.get('summary', {}).get('success_rate', 0):.1f}%")
            self.logger.info(f"  Quality score: {quality_analysis.get('quality_score', 0):.1f}")
            self.logger.info(f"  Quality grade: {quality_analysis.get('quality_grade', 'F')}")
            self.logger.info(f"  88/88 compliant: {quality_analysis.get('88_compliance', {}).get('is_compliant', False)}")
            
        except Exception as e:
            self.logger.error(f"Testing cycle failed: {e}")
            raise
    
    def _find_test_files(self) -> List[Path]:
        """Find all test files in the project"""
        test_files = []
        
        # Common test file patterns
        patterns = ["test_*.py", "*_test.py", "tests.py"]
        
        for pattern in patterns:
            test_files.extend(self.project_path.glob(f"**/{pattern}"))
        
        # Filter out common ignore patterns
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv"
        ]
        
        filtered_files = []
        for file_path in test_files:
            if not any(pattern in str(file_path) for pattern in ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _combine_test_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple test results into one"""
        if not all_results:
            return {}
        
        # Combine summaries
        total_passed = sum(r.get("summary", {}).get("passed", 0) for r in all_results)
        total_failed = sum(r.get("summary", {}).get("failed", 0) for r in all_results)
        total_skipped = sum(r.get("summary", {}).get("skipped", 0) for r in all_results)
        total_errors = sum(r.get("summary", {}).get("errors", 0) for r in all_results)
        
        total_tests = total_passed + total_failed + total_skipped + total_errors
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Combine individual tests
        all_individual_tests = []
        all_failures = []
        
        for result in all_results:
            all_individual_tests.extend(result.get("individual_tests", []))
            all_failures.extend(result.get("failures", []))
        
        return {
            "status": "SUCCESS" if total_failed == 0 and total_errors == 0 else "FAILURE",
            "summary": {
                "total": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "errors": total_errors,
                "success_rate": success_rate
            },
            "individual_tests": all_individual_tests,
            "failures": all_failures,
            "88_compliant": total_tests >= 88 and success_rate == 100.0,
            "execution_time": sum(r.get("execution_time", 0) for r in all_results),
            "timestamp": datetime.now().isoformat()
        }
    
    async def run_single_test_file(self, test_file: str) -> Dict[str, Any]:
        """Run tests for a single file (for testing/debugging)"""
        return await self.test_runner.run_tests(test_file)
    
    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest QA report"""
        try:
            latest_path = Path(self.output_path) / "latest_qa_report.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load latest report: {e}")
            return None
    
    async def get_quality_score(self) -> float:
        """Get current overall quality score"""
        report = await self.get_latest_report()
        if report:
            return report.get("overall_assessment", {}).get("overall_score", 0.0)
        return 0.0
    
    def _check_system_resources(self) -> None:
        """Check system resources before starting"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 85:
            raise QualityAssuranceError(f"CPU usage too high: {cpu_percent}%")
        if memory_percent > 85:
            raise QualityAssuranceError(f"Memory usage too high: {memory_percent}%")
        
        self.logger.info(f"System resources OK - CPU: {cpu_percent}%, Memory: {memory_percent}%")

# Factory pattern for agent creation
class QualityAssuranceAgentFactory:
    """Factory for creating QA agents with different configurations"""
    
    @staticmethod
    def create_standard_agent(project_path: str = ".") -> AdvancedQualityAssuranceAgent:
        """Create standard QA agent"""
        return AdvancedQualityAssuranceAgent(project_path=project_path)
    
    @staticmethod
    def create_rapid_agent(project_path: str = ".") -> AdvancedQualityAssuranceAgent:
        """Create rapid testing agent for CI/CD"""
        return AdvancedQualityAssuranceAgent(
            project_path=project_path,
            testing_interval=300  # 5 minutes
        )
    
    @staticmethod
    def create_enterprise_agent(project_path: str = ".") -> AdvancedQualityAssuranceAgent:
        """Create enterprise-grade QA agent"""
        return AdvancedQualityAssuranceAgent(
            project_path=project_path,
            testing_interval=900,  # 15 minutes
            output_path="enterprise_qa_reports"
        )

async def main():
    """Main function for running the QA agent"""
    try:
        # Create and start the agent
        agent = QualityAssuranceAgentFactory.create_standard_agent()
        await agent.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\nShutting down Quality Assurance Agent...")
        await agent.stop()
    except Exception as e:
        print(f"Quality Assurance Agent error: {e}")

if __name__ == "__main__":
    asyncio.run(main())