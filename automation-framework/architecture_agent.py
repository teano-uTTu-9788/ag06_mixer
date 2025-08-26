#!/usr/bin/env python3
"""
Advanced Architecture Agent - SOLID Compliance & System Design Optimization
Based on 2024-2025 Enterprise Architecture Best Practices

This agent implements:
- Automated SOLID principles validation
- Real-time architecture assessment
- Code quality enforcement
- Design pattern recommendations
- Technical debt monitoring
"""

import ast
import asyncio
import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Tuple
from abc import ABC, abstractmethod
import psutil

# SOLID Architecture Implementation
class ICodeAnalyzer(Protocol):
    """Interface for code analysis"""
    async def analyze_file(self, file_path: str) -> Dict[str, Any]: ...

class ISOLIDValidator(Protocol):
    """Interface for SOLID principles validation"""
    async def validate_solid_compliance(self, analysis: Dict[str, Any]) -> Dict[str, Any]: ...

class IArchitectureReporter(Protocol):
    """Interface for architecture reporting"""
    async def generate_report(self, validations: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class IDesignPatternDetector(Protocol):
    """Interface for design pattern detection"""
    async def detect_patterns(self, code_analysis: Dict[str, Any]) -> List[str]: ...

@dataclass
class SOLIDViolation:
    """SOLID violation data structure"""
    principle: str
    severity: str
    description: str
    file_path: str
    line_number: int
    recommendation: str
    example_fix: str

@dataclass
class ArchitectureMetric:
    """Architecture quality metric"""
    name: str
    value: float
    threshold: float
    status: str  # PASS, WARN, FAIL
    description: str

@dataclass
class DesignPattern:
    """Design pattern detection result"""
    pattern_name: str
    confidence: float
    location: str
    quality_score: float
    recommendations: List[str]

class ArchitectureError(Exception):
    """Custom architecture agent exceptions"""
    pass

class PythonCodeAnalyzer:
    """Advanced Python code analyzer with AST parsing"""
    
    def __init__(self):
        self.class_patterns = {}
        self.method_patterns = {}
        self.import_patterns = {}
    
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze Python file for architecture patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            analysis = {
                "file_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "lines_of_code": len(source_code.splitlines()),
                "classes": self._analyze_classes(tree),
                "functions": self._analyze_functions(tree),
                "imports": self._analyze_imports(tree),
                "complexity_metrics": self._calculate_complexity(tree),
                "design_indicators": self._detect_design_indicators(tree, source_code)
            }
            
            return analysis
            
        except Exception as e:
            raise ArchitectureError(f"Failed to analyze {file_path}: {e}")
    
    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze class definitions"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "inheritance": [base.id for base in node.bases if isinstance(base, ast.Name)],
                    "decorators": [dec.id for dec in node.decorator_list if isinstance(dec, ast.Name)],
                    "docstring": ast.get_docstring(node),
                    "method_names": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                }
                classes.append(class_info)
        
        return classes
    
    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze function definitions"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "args_count": len(node.args.args),
                    "returns_annotation": node.returns is not None,
                    "docstring": ast.get_docstring(node),
                    "decorators": [dec.id for dec in node.decorator_list if isinstance(dec, ast.Name)],
                    "complexity": self._calculate_function_complexity(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _analyze_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze import statements"""
        imports = {"standard": [], "third_party": [], "local": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports["standard"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith('.'):
                        imports["local"].append(node.module)
                    else:
                        imports["third_party"].append(node.module)
        
        return imports
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculate various complexity metrics"""
        complexity = {
            "cyclomatic": 1,  # Base complexity
            "nesting_depth": 0,
            "cognitive": 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity["cyclomatic"] += 1
                complexity["cognitive"] += 1
            elif isinstance(node, ast.BoolOp):
                complexity["cyclomatic"] += len(node.values) - 1
        
        return complexity
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _detect_design_indicators(self, tree: ast.AST, source_code: str) -> Dict[str, Any]:
        """Detect design pattern indicators"""
        indicators = {
            "interfaces_detected": self._detect_interfaces(tree),
            "dependency_injection": self._detect_dependency_injection(tree),
            "factory_patterns": self._detect_factory_patterns(tree),
            "singleton_patterns": self._detect_singleton_patterns(source_code),
            "observer_patterns": self._detect_observer_patterns(tree)
        }
        
        return indicators
    
    def _detect_interfaces(self, tree: ast.AST) -> bool:
        """Detect interface-like patterns (Protocol, ABC)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for Protocol or ABC inheritance
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in ['Protocol', 'ABC']:
                        return True
                # Check for abstract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name) and dec.id == 'abstractmethod':
                                return True
        return False
    
    def _detect_dependency_injection(self, tree: ast.AST) -> bool:
        """Detect dependency injection patterns"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Look for constructor parameters that are likely dependencies
                if len(node.args.args) > 2:  # self + at least 2 dependencies
                    return True
        return False
    
    def _detect_factory_patterns(self, tree: ast.AST) -> bool:
        """Detect factory pattern indicators"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'create' in node.name.lower() or 'factory' in node.name.lower():
                    return True
        return False
    
    def _detect_singleton_patterns(self, source_code: str) -> bool:
        """Detect singleton pattern indicators"""
        singleton_indicators = [
            '__new__',
            '_instance',
            'getInstance',
            '@singleton'
        ]
        return any(indicator in source_code for indicator in singleton_indicators)
    
    def _detect_observer_patterns(self, tree: ast.AST) -> bool:
        """Detect observer pattern indicators"""
        observer_methods = ['subscribe', 'notify', 'observe', 'emit', 'listen']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(method in node.name.lower() for method in observer_methods):
                    return True
        return False

class SOLIDValidator:
    """Advanced SOLID principles validator"""
    
    def __init__(self):
        self.violation_patterns = self._load_violation_patterns()
    
    async def validate_solid_compliance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SOLID principles compliance"""
        violations = []
        
        # Single Responsibility Principle (SRP)
        srp_violations = await self._validate_srp(analysis)
        violations.extend(srp_violations)
        
        # Open/Closed Principle (OCP)
        ocp_violations = await self._validate_ocp(analysis)
        violations.extend(ocp_violations)
        
        # Liskov Substitution Principle (LSP)
        lsp_violations = await self._validate_lsp(analysis)
        violations.extend(lsp_violations)
        
        # Interface Segregation Principle (ISP)
        isp_violations = await self._validate_isp(analysis)
        violations.extend(isp_violations)
        
        # Dependency Inversion Principle (DIP)
        dip_violations = await self._validate_dip(analysis)
        violations.extend(dip_violations)
        
        # Calculate compliance score
        total_checks = len(analysis.get("classes", [])) * 5  # 5 principles per class
        compliance_score = max(0, (total_checks - len(violations)) / max(total_checks, 1))
        
        return {
            "file_path": analysis["file_path"],
            "validation_timestamp": datetime.now().isoformat(),
            "violations": [v.__dict__ for v in violations],
            "compliance_score": compliance_score,
            "severity_breakdown": self._calculate_severity_breakdown(violations),
            "recommendations": self._generate_recommendations(violations)
        }
    
    async def _validate_srp(self, analysis: Dict[str, Any]) -> List[SOLIDViolation]:
        """Validate Single Responsibility Principle"""
        violations = []
        
        for class_info in analysis.get("classes", []):
            # Check if class has too many responsibilities (methods)
            if class_info["methods"] > 10:
                violation = SOLIDViolation(
                    principle="Single Responsibility",
                    severity="HIGH",
                    description=f"Class '{class_info['name']}' has {class_info['methods']} methods, suggesting multiple responsibilities",
                    file_path=analysis["file_path"],
                    line_number=class_info["line_number"],
                    recommendation="Split class into smaller, focused classes",
                    example_fix=f"Consider extracting {class_info['methods'] - 5} methods into separate classes"
                )
                violations.append(violation)
            
            # Check for mixed concerns (e.g., data access + business logic)
            method_names = class_info.get("method_names", [])
            data_methods = [m for m in method_names if any(word in m.lower() for word in ['save', 'load', 'fetch', 'store', 'database', 'sql'])]
            business_methods = [m for m in method_names if any(word in m.lower() for word in ['calculate', 'process', 'validate', 'format', 'transform'])]
            
            if len(data_methods) > 0 and len(business_methods) > 0:
                violation = SOLIDViolation(
                    principle="Single Responsibility",
                    severity="MEDIUM",
                    description=f"Class '{class_info['name']}' mixes data access and business logic concerns",
                    file_path=analysis["file_path"],
                    line_number=class_info["line_number"],
                    recommendation="Separate data access and business logic into different classes",
                    example_fix="Create separate Repository and Service classes"
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_ocp(self, analysis: Dict[str, Any]) -> List[SOLIDViolation]:
        """Validate Open/Closed Principle"""
        violations = []
        
        # Check for extensibility patterns
        has_interfaces = analysis.get("design_indicators", {}).get("interfaces_detected", False)
        has_factory = analysis.get("design_indicators", {}).get("factory_patterns", False)
        
        if not has_interfaces and len(analysis.get("classes", [])) > 1:
            violation = SOLIDViolation(
                principle="Open/Closed",
                severity="MEDIUM",
                description="No interfaces detected - classes may not be open for extension",
                file_path=analysis["file_path"],
                line_number=1,
                recommendation="Define interfaces to enable extensibility",
                example_fix="Create Protocol interfaces for your main classes"
            )
            violations.append(violation)
        
        return violations
    
    async def _validate_lsp(self, analysis: Dict[str, Any]) -> List[SOLIDViolation]:
        """Validate Liskov Substitution Principle"""
        violations = []
        
        # This is harder to detect statically, but we can check for common anti-patterns
        for class_info in analysis.get("classes", []):
            if class_info.get("inheritance"):
                # Check for method name mismatches that might indicate LSP violations
                method_names = set(class_info.get("method_names", []))
                
                # Look for methods that might not be substitutable
                problematic_methods = [m for m in method_names if m.startswith('_') and not m.startswith('__')]
                
                if len(problematic_methods) > 3:
                    violation = SOLIDViolation(
                        principle="Liskov Substitution",
                        severity="LOW",
                        description=f"Class '{class_info['name']}' has many private methods, potentially breaking substitutability",
                        file_path=analysis["file_path"],
                        line_number=class_info["line_number"],
                        recommendation="Review inheritance hierarchy and ensure substitutability",
                        example_fix="Make methods public where appropriate or reconsider inheritance"
                    )
                    violations.append(violation)
        
        return violations
    
    async def _validate_isp(self, analysis: Dict[str, Any]) -> List[SOLIDViolation]:
        """Validate Interface Segregation Principle"""
        violations = []
        
        # Check for fat interfaces (classes with too many public methods)
        for class_info in analysis.get("classes", []):
            if class_info.get("inheritance") and class_info["methods"] > 15:
                violation = SOLIDViolation(
                    principle="Interface Segregation",
                    severity="HIGH",
                    description=f"Class '{class_info['name']}' implements/inherits too many methods ({class_info['methods']})",
                    file_path=analysis["file_path"],
                    line_number=class_info["line_number"],
                    recommendation="Split into smaller, more focused interfaces",
                    example_fix="Create 2-3 separate Protocol interfaces for different concerns"
                )
                violations.append(violation)
        
        return violations
    
    async def _validate_dip(self, analysis: Dict[str, Any]) -> List[SOLIDViolation]:
        """Validate Dependency Inversion Principle"""
        violations = []
        
        has_dependency_injection = analysis.get("design_indicators", {}).get("dependency_injection", False)
        
        if not has_dependency_injection and len(analysis.get("classes", [])) > 1:
            violation = SOLIDViolation(
                principle="Dependency Inversion",
                severity="HIGH",
                description="No dependency injection detected - classes may depend on concretions",
                file_path=analysis["file_path"],
                line_number=1,
                recommendation="Implement dependency injection with constructor parameters",
                example_fix="Pass dependencies through __init__ instead of creating them internally"
            )
            violations.append(violation)
        
        return violations
    
    def _calculate_severity_breakdown(self, violations: List[SOLIDViolation]) -> Dict[str, int]:
        """Calculate breakdown of violations by severity"""
        breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for violation in violations:
            breakdown[violation.severity] += 1
        
        return breakdown
    
    def _generate_recommendations(self, violations: List[SOLIDViolation]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Group by principle
        principle_counts = {}
        for violation in violations:
            principle_counts[violation.principle] = principle_counts.get(violation.principle, 0) + 1
        
        # Generate recommendations based on most common violations
        if principle_counts.get("Single Responsibility", 0) > 0:
            recommendations.append("Consider breaking down large classes into smaller, focused components")
        
        if principle_counts.get("Dependency Inversion", 0) > 0:
            recommendations.append("Implement dependency injection to decouple your classes")
        
        if principle_counts.get("Interface Segregation", 0) > 0:
            recommendations.append("Define smaller, more specific interfaces")
        
        if principle_counts.get("Open/Closed", 0) > 0:
            recommendations.append("Use interfaces and composition to enable extensibility")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _load_violation_patterns(self) -> Dict[str, Any]:
        """Load predefined violation patterns"""
        return {
            "srp_indicators": ["save", "load", "calculate", "validate", "format", "send", "receive"],
            "god_class_threshold": 10,
            "method_complexity_threshold": 10,
            "parameter_count_threshold": 5
        }

class ArchitectureReporter:
    """Architecture quality reporter"""
    
    def __init__(self, output_path: str = "architecture_reports"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    async def generate_report(self, validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive architecture report"""
        
        # Aggregate metrics
        total_files = len(validations)
        total_violations = sum(len(v["violations"]) for v in validations)
        avg_compliance_score = sum(v["compliance_score"] for v in validations) / max(total_files, 1)
        
        # Calculate architecture metrics
        metrics = self._calculate_architecture_metrics(validations)
        
        # Generate recommendations
        recommendations = self._generate_architecture_recommendations(validations)
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "files_analyzed": total_files,
                "total_violations": total_violations,
                "average_compliance_score": avg_compliance_score,
                "overall_grade": self._calculate_grade(avg_compliance_score)
            },
            "metrics": [m.__dict__ for m in metrics],
            "violations_by_principle": self._group_violations_by_principle(validations),
            "recommendations": recommendations,
            "detailed_validations": validations
        }
        
        # Save report to file
        await self._save_report(report)
        
        return report
    
    def _calculate_architecture_metrics(self, validations: List[Dict[str, Any]]) -> List[ArchitectureMetric]:
        """Calculate architecture quality metrics"""
        metrics = []
        
        # SOLID Compliance Metric
        avg_compliance = sum(v["compliance_score"] for v in validations) / max(len(validations), 1)
        metrics.append(ArchitectureMetric(
            name="SOLID Compliance",
            value=avg_compliance,
            threshold=0.8,
            status="PASS" if avg_compliance >= 0.8 else "FAIL",
            description="Overall SOLID principles compliance score"
        ))
        
        # Violation Density
        total_violations = sum(len(v["violations"]) for v in validations)
        violation_density = total_violations / max(len(validations), 1)
        metrics.append(ArchitectureMetric(
            name="Violation Density",
            value=violation_density,
            threshold=5.0,
            status="PASS" if violation_density <= 5.0 else "FAIL",
            description="Average violations per file"
        ))
        
        # High Severity Violations
        high_severity = sum(v["severity_breakdown"].get("HIGH", 0) for v in validations)
        metrics.append(ArchitectureMetric(
            name="High Severity Violations",
            value=high_severity,
            threshold=0,
            status="PASS" if high_severity == 0 else "FAIL",
            description="Number of high-severity violations"
        ))
        
        return metrics
    
    def _group_violations_by_principle(self, validations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group violations by SOLID principle"""
        principle_counts = {}
        
        for validation in validations:
            for violation in validation["violations"]:
                principle = violation["principle"]
                principle_counts[principle] = principle_counts.get(principle, 0) + 1
        
        return principle_counts
    
    def _generate_architecture_recommendations(self, validations: List[Dict[str, Any]]) -> List[str]:
        """Generate architecture improvement recommendations"""
        all_recommendations = []
        
        for validation in validations:
            all_recommendations.extend(validation.get("recommendations", []))
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _calculate_grade(self, compliance_score: float) -> str:
        """Calculate letter grade based on compliance score"""
        if compliance_score >= 0.9:
            return "A"
        elif compliance_score >= 0.8:
            return "B"
        elif compliance_score >= 0.7:
            return "C"
        elif compliance_score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _save_report(self, report: Dict[str, Any]) -> None:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"architecture_report_{timestamp}.json"
        filepath = self.output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.output_path / "latest_architecture_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

class DesignPatternDetector:
    """Advanced design pattern detection"""
    
    async def detect_patterns(self, code_analysis: Dict[str, Any]) -> List[DesignPattern]:
        """Detect design patterns in code"""
        patterns = []
        
        # Detect common patterns
        patterns.extend(self._detect_creational_patterns(code_analysis))
        patterns.extend(self._detect_structural_patterns(code_analysis))
        patterns.extend(self._detect_behavioral_patterns(code_analysis))
        
        return patterns
    
    def _detect_creational_patterns(self, analysis: Dict[str, Any]) -> List[DesignPattern]:
        """Detect creational design patterns"""
        patterns = []
        
        # Factory Pattern
        if analysis.get("design_indicators", {}).get("factory_patterns", False):
            patterns.append(DesignPattern(
                pattern_name="Factory",
                confidence=0.8,
                location=analysis["file_path"],
                quality_score=0.85,
                recommendations=["Consider using abstract factory for multiple product families"]
            ))
        
        # Singleton Pattern
        if analysis.get("design_indicators", {}).get("singleton_patterns", False):
            patterns.append(DesignPattern(
                pattern_name="Singleton",
                confidence=0.7,
                location=analysis["file_path"],
                quality_score=0.6,  # Lower score as singleton can be problematic
                recommendations=["Consider dependency injection instead of singleton for better testability"]
            ))
        
        return patterns
    
    def _detect_structural_patterns(self, analysis: Dict[str, Any]) -> List[DesignPattern]:
        """Detect structural design patterns"""
        patterns = []
        
        # Check for adapter-like patterns
        classes = analysis.get("classes", [])
        for class_info in classes:
            if "adapter" in class_info["name"].lower():
                patterns.append(DesignPattern(
                    pattern_name="Adapter",
                    confidence=0.9,
                    location=f"{analysis['file_path']}:{class_info['line_number']}",
                    quality_score=0.9,
                    recommendations=["Ensure adapter implements clean interface conversion"]
                ))
        
        return patterns
    
    def _detect_behavioral_patterns(self, analysis: Dict[str, Any]) -> List[DesignPattern]:
        """Detect behavioral design patterns"""
        patterns = []
        
        # Observer Pattern
        if analysis.get("design_indicators", {}).get("observer_patterns", False):
            patterns.append(DesignPattern(
                pattern_name="Observer",
                confidence=0.85,
                location=analysis["file_path"],
                quality_score=0.9,
                recommendations=["Consider using event-driven architecture for loose coupling"]
            ))
        
        return patterns

class AdvancedArchitectureAgent:
    """
    Advanced Architecture Agent implementing 2024-2025 best practices
    
    Features:
    - Automated SOLID principles validation
    - Real-time architecture assessment
    - Design pattern detection and recommendations
    - Technical debt monitoring
    - Enterprise-grade reporting
    """
    
    def __init__(
        self, 
        project_path: str = ".",
        analysis_interval: int = 1800,  # 30 minutes
        output_path: str = "architecture_reports"
    ):
        self.project_path = Path(project_path)
        self.analysis_interval = analysis_interval
        self.output_path = output_path
        self.is_running = False
        
        # Dependency injection following SOLID principles
        self.analyzer: ICodeAnalyzer = PythonCodeAnalyzer()
        self.validator: ISOLIDValidator = SOLIDValidator()
        self.reporter: IArchitectureReporter = ArchitectureReporter(output_path)
        self.pattern_detector: IDesignPatternDetector = DesignPatternDetector()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # File extensions to analyze
        self.file_extensions = ['.py']
    
    async def start(self) -> None:
        """Start the architecture agent"""
        self.logger.info("Starting Advanced Architecture Agent")
        self._check_system_resources()
        
        self.is_running = True
        
        # Start continuous analysis loop
        asyncio.create_task(self._analysis_loop())
        
        self.logger.info(f"Architecture Agent started - monitoring {self.project_path}")
    
    async def stop(self) -> None:
        """Stop the architecture agent"""
        self.logger.info("Stopping Advanced Architecture Agent")
        self.is_running = False
    
    async def _analysis_loop(self) -> None:
        """Main analysis loop"""
        while self.is_running:
            try:
                await self._perform_analysis_cycle()
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                self.logger.error(f"Analysis cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _perform_analysis_cycle(self) -> None:
        """Perform one complete analysis cycle"""
        self.logger.info("Starting architecture analysis cycle")
        
        try:
            # Step 1: Find all Python files
            python_files = self._find_python_files()
            self.logger.info(f"Found {len(python_files)} Python files to analyze")
            
            # Step 2: Analyze each file
            validations = []
            for file_path in python_files:
                try:
                    # Analyze code
                    analysis = await self.analyzer.analyze_file(str(file_path))
                    
                    # Validate SOLID compliance
                    validation = await self.validator.validate_solid_compliance(analysis)
                    
                    # Detect design patterns
                    patterns = await self.pattern_detector.detect_patterns(analysis)
                    validation["design_patterns"] = [p.__dict__ for p in patterns]
                    
                    validations.append(validation)
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {file_path}: {e}")
                    continue
            
            # Step 3: Generate comprehensive report
            if validations:
                report = await self.reporter.generate_report(validations)
                
                self.logger.info(f"Analysis cycle completed:")
                self.logger.info(f"  Files analyzed: {report['summary']['files_analyzed']}")
                self.logger.info(f"  Total violations: {report['summary']['total_violations']}")
                self.logger.info(f"  Average compliance: {report['summary']['average_compliance_score']:.2f}")
                self.logger.info(f"  Overall grade: {report['summary']['overall_grade']}")
            else:
                self.logger.warning("No files successfully analyzed")
                
        except Exception as e:
            self.logger.error(f"Analysis cycle failed: {e}")
            raise
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        
        for ext in self.file_extensions:
            python_files.extend(self.project_path.glob(f"**/*{ext}"))
        
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
        for file_path in python_files:
            if not any(pattern in str(file_path) for pattern in ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    async def analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file for testing/debugging"""
        try:
            # Analyze code
            analysis = await self.analyzer.analyze_file(file_path)
            
            # Validate SOLID compliance
            validation = await self.validator.validate_solid_compliance(analysis)
            
            # Detect design patterns
            patterns = await self.pattern_detector.detect_patterns(analysis)
            validation["design_patterns"] = [p.__dict__ for p in patterns]
            
            return validation
            
        except Exception as e:
            raise ArchitectureError(f"Failed to analyze {file_path}: {e}")
    
    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest architecture report"""
        try:
            latest_path = Path(self.output_path) / "latest_architecture_report.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load latest report: {e}")
            return None
    
    async def get_compliance_score(self) -> float:
        """Get current overall compliance score"""
        report = await self.get_latest_report()
        if report:
            return report["summary"]["average_compliance_score"]
        return 0.0
    
    def _check_system_resources(self) -> None:
        """Check system resources before starting"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 85:
            raise ArchitectureError(f"CPU usage too high: {cpu_percent}%")
        if memory_percent > 85:
            raise ArchitectureError(f"Memory usage too high: {memory_percent}%")
        
        self.logger.info(f"System resources OK - CPU: {cpu_percent}%, Memory: {memory_percent}%")

# Factory pattern for agent creation
class ArchitectureAgentFactory:
    """Factory for creating architecture agents with different configurations"""
    
    @staticmethod
    def create_standard_agent(project_path: str = ".") -> AdvancedArchitectureAgent:
        """Create standard architecture agent"""
        return AdvancedArchitectureAgent(project_path=project_path)
    
    @staticmethod
    def create_rapid_agent(project_path: str = ".") -> AdvancedArchitectureAgent:
        """Create rapid analysis agent for testing"""
        return AdvancedArchitectureAgent(
            project_path=project_path,
            analysis_interval=300  # 5 minutes
        )
    
    @staticmethod
    def create_enterprise_agent(project_path: str = ".") -> AdvancedArchitectureAgent:
        """Create enterprise-grade architecture agent"""
        return AdvancedArchitectureAgent(
            project_path=project_path,
            analysis_interval=900,  # 15 minutes
            output_path="enterprise_architecture_reports"
        )

async def main():
    """Main function for running the architecture agent"""
    try:
        # Create and start the agent
        agent = ArchitectureAgentFactory.create_standard_agent()
        await agent.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\nShutting down Architecture Agent...")
        await agent.stop()
    except Exception as e:
        print(f"Architecture Agent error: {e}")

if __name__ == "__main__":
    asyncio.run(main())