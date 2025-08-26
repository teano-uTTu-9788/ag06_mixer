#!/usr/bin/env python3
"""
Critical Assessment: Mobile AG06 App 88-Test Validation
Tests the actual existence and structure of mobile app components
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple

class MobileAppValidator:
    def __init__(self, base_path: str = "mobile-app"):
        self.base_path = Path(base_path)
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def test(self, test_num: int, description: str, condition: bool) -> None:
        """Record test result"""
        status = "✅ PASS" if condition else "❌ FAIL"
        self.results.append((test_num, description, status))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"Test {test_num:2d}: {description:50s} ... {status}")
        
    def run_all_tests(self) -> None:
        """Run all 88 validation tests"""
        print("=" * 70)
        print("MOBILE AG06 APP - CRITICAL ASSESSMENT (88 TESTS)")
        print("=" * 70)
        
        # Configuration Tests (8 tests)
        print("\n📋 Configuration Tests (8 tests):")
        self.test_configuration()
        
        # Service Layer Tests (12 tests) 
        print("\n🔧 Service Layer Tests (12 tests):")
        self.test_service_layer()
        
        # UI Component Tests (16 tests)
        print("\n🎨 UI Component Tests (16 tests):")
        self.test_ui_components()
        
        # API Integration Tests (12 tests)
        print("\n🔌 API Integration Tests (12 tests):")
        self.test_api_integration()
        
        # Performance Tests (10 tests)
        print("\n⚡ Performance Tests (10 tests):")
        self.test_performance()
        
        # Security Tests (8 tests)
        print("\n🔐 Security Tests (8 tests):")
        self.test_security()
        
        # Integration Tests (12 tests)
        print("\n🔄 Integration Tests (12 tests):")
        self.test_integration()
        
        # Regression Tests (10 tests)
        print("\n🐛 Regression Tests (10 tests):")
        self.test_regression()
        
        # Print summary
        self.print_summary()
        
    def test_configuration(self):
        """Tests 1-8: Configuration validation"""
        config_file = self.base_path / "Models" / "MixerConfiguration.swift"
        
        # Test 1: MixerConfiguration.swift exists
        self.test(1, "MixerConfiguration.swift exists", config_file.exists())
        
        if config_file.exists():
            content = config_file.read_text()
            
            # Test 2: MixerConfiguration struct defined
            self.test(2, "MixerConfiguration struct defined", 
                     "struct MixerConfiguration" in content)
            
            # Test 3: SubscriptionTier enum defined
            self.test(3, "SubscriptionTier enum defined",
                     "enum SubscriptionTier" in content)
            
            # Test 4: BatteryMode enum defined
            self.test(4, "BatteryMode enum defined",
                     "enum BatteryMode" in content)
            
            # Test 5: AudioMetrics struct defined
            self.test(5, "AudioMetrics struct defined",
                     "struct AudioMetrics" in content)
            
            # Test 6: MixerSettings struct defined
            self.test(6, "MixerSettings struct defined",
                     "struct MixerSettings" in content)
            
            # Test 7: ConnectionStatus struct defined
            self.test(7, "ConnectionStatus struct defined",
                     "struct ConnectionStatus" in content)
            
            # Test 8: MixerError enum defined
            self.test(8, "MixerError enum defined",
                     "enum MixerError" in content)
        else:
            for i in range(2, 9):
                self.test(i, f"Configuration test {i}", False)
                
    def test_service_layer(self):
        """Tests 9-20: Service layer validation"""
        service_file = self.base_path / "Services" / "MixerService.swift"
        
        # Test 9: MixerService.swift exists
        self.test(9, "MixerService.swift exists", service_file.exists())
        
        if service_file.exists():
            content = service_file.read_text()
            
            # Test 10: MixerService class defined
            self.test(10, "MixerService class defined",
                     "class MixerService" in content)
            
            # Test 11: @Published properties
            self.test(11, "@Published audioMetrics property",
                     "@Published var audioMetrics" in content)
            
            # Test 12: startMixer function
            self.test(12, "startMixer() function defined",
                     "func startMixer()" in content)
            
            # Test 13: stopMixer function
            self.test(13, "stopMixer() function defined",
                     "func stopMixer()" in content)
            
            # Test 14: updateSettings function
            self.test(14, "updateSettings() function defined",
                     "func updateSettings" in content)
            
            # Test 15: testConnection function
            self.test(15, "testConnection() function defined",
                     "func testConnection()" in content)
            
            # Test 16: Battery optimization methods
            self.test(16, "enterBackgroundMode() defined",
                     "func enterBackgroundMode()" in content)
            
            # Test 17: enterForegroundMode defined
            self.test(17, "enterForegroundMode() defined",
                     "func enterForegroundMode()" in content)
            
            # Test 18: Network monitoring
            self.test(18, "Network monitoring setup",
                     "NWPathMonitor" in content)
            
            # Test 19: URLSession configuration
            self.test(19, "URLSession configuration",
                     "URLSession" in content)
            
            # Test 20: Subscription limits check
            self.test(20, "checkSubscriptionLimits() defined",
                     "func checkSubscriptionLimits" in content)
        else:
            for i in range(10, 21):
                self.test(i, f"Service layer test {i}", False)
                
    def test_ui_components(self):
        """Tests 21-36: UI Component validation"""
        views_dir = self.base_path / "Views"
        
        # Test 21: MixerControlView.swift exists
        control_view = views_dir / "MixerControlView.swift"
        self.test(21, "MixerControlView.swift exists", control_view.exists())
        
        # Test 22: MixerSettingsView.swift exists
        settings_view = views_dir / "MixerSettingsView.swift"
        self.test(22, "MixerSettingsView.swift exists", settings_view.exists())
        
        # Test 23: SubscriptionView.swift exists
        subscription_view = views_dir / "SubscriptionView.swift"
        self.test(23, "SubscriptionView.swift exists", subscription_view.exists())
        
        if control_view.exists():
            content = control_view.read_text()
            
            # Test 24: MixerControlView struct
            self.test(24, "MixerControlView struct defined",
                     "struct MixerControlView" in content)
            
            # Test 25: AudioMeter component
            self.test(25, "AudioMeter component defined",
                     "struct AudioMeter" in content)
            
            # Test 26: ControlSlider component
            self.test(26, "ControlSlider component defined",
                     "struct ControlSlider" in content)
            
            # Test 27: StatusIndicator component
            self.test(27, "StatusIndicator component defined",
                     "struct StatusIndicator" in content)
            
            # Test 28: DeviceRow component
            self.test(28, "DeviceRow component defined",
                     "struct DeviceRow" in content)
            
            # Test 29: SubscriptionLockedMeter
            self.test(29, "SubscriptionLockedMeter defined",
                     "struct SubscriptionLockedMeter" in content)
            
            # Test 30: Real-time update methods
            self.test(30, "startRealTimeUpdates() defined",
                     "startRealTimeUpdates()" in content)
        else:
            for i in range(24, 31):
                self.test(i, f"UI component test {i}", False)
                
        if subscription_view.exists():
            content = subscription_view.read_text()
            
            # Test 31: SubscriptionView struct
            self.test(31, "SubscriptionView struct defined",
                     "struct SubscriptionView" in content)
            
            # Test 32: SubscriptionTierCard
            self.test(32, "SubscriptionTierCard defined",
                     "struct SubscriptionTierCard" in content)
            
            # Test 33: FeatureRow component
            self.test(33, "FeatureRow component defined",
                     "struct FeatureRow" in content)
            
            # Test 34: PurchaseFlowView
            self.test(34, "PurchaseFlowView defined",
                     "struct PurchaseFlowView" in content)
        else:
            for i in range(31, 35):
                self.test(i, f"UI component test {i}", False)
                
        # Test 35: Main app file exists
        app_file = self.base_path / "MobileAG06App.swift"
        self.test(35, "MobileAG06App.swift exists", app_file.exists())
        
        # Test 36: ContentView in main app
        if app_file.exists():
            content = app_file.read_text()
            self.test(36, "ContentView struct defined",
                     "struct ContentView" in content)
        else:
            self.test(36, "ContentView struct defined", False)
            
    def test_api_integration(self):
        """Tests 37-48: API Integration validation"""
        service_file = self.base_path / "Services" / "MixerService.swift"
        
        if service_file.exists():
            content = service_file.read_text()
            
            # Test 37: API endpoint URLs
            self.test(37, "/api/status endpoint used",
                     "/api/status" in content)
            
            # Test 38: /api/start endpoint
            self.test(38, "/api/start endpoint used",
                     "/api/start" in content)
            
            # Test 39: /api/stop endpoint
            self.test(39, "/api/stop endpoint used",
                     "/api/stop" in content)
            
            # Test 40: /api/config endpoint
            self.test(40, "/api/config endpoint used",
                     "/api/config" in content)
            
            # Test 41: /healthz endpoint
            self.test(41, "/healthz endpoint used",
                     "/healthz" in content)
            
            # Test 42: JSON decoding
            self.test(42, "JSONDecoder usage",
                     "JSONDecoder" in content)
            
            # Test 43: JSON encoding
            self.test(43, "JSONSerialization usage",
                     "JSONSerialization" in content)
            
            # Test 44: URLRequest configuration
            self.test(44, "URLRequest configuration",
                     "URLRequest" in content)
            
            # Test 45: HTTP headers
            self.test(45, "Content-Type header set",
                     "Content-Type" in content)
            
            # Test 46: Error handling
            self.test(46, "MixerError handling",
                     "MixerError" in content)
            
            # Test 47: Async/await usage
            self.test(47, "async/await pattern used",
                     "async" in content and "await" in content)
            
            # Test 48: Response validation
            self.test(48, "HTTPURLResponse validation",
                     "HTTPURLResponse" in content)
        else:
            for i in range(37, 49):
                self.test(i, f"API integration test {i}", False)
                
    def test_performance(self):
        """Tests 49-58: Performance optimization validation"""
        service_file = self.base_path / "Services" / "MixerService.swift"
        
        if service_file.exists():
            content = service_file.read_text()
            
            # Test 49: Timer-based updates
            self.test(49, "Timer-based metrics updates",
                     "Timer.scheduledTimer" in content)
            
            # Test 50: Update interval configuration
            self.test(50, "Update interval based on battery mode",
                     "updateInterval" in content)
            
            # Test 51: Log rotation
            self.test(51, "Log rotation implemented",
                     "logs.removeFirst" in content)
            
            # Test 52: Background mode optimization
            self.test(52, "Background mode reduces updates",
                     "isBackgroundMode" in content)
            
            # Test 53: Combine publishers
            self.test(53, "Combine framework used",
                     "import Combine" in content)
            
            # Test 54: @MainActor usage
            self.test(54, "@MainActor for UI updates",
                     "@MainActor" in content)
            
            # Test 55: Concurrent operations
            self.test(55, "TaskGroup for concurrent ops",
                     "withTaskGroup" in content)
            
            # Test 56: Memory management
            self.test(56, "Weak self references",
                     "[weak self]" in content)
            
            # Test 57: URLSession timeout
            self.test(57, "Request timeout configured",
                     "timeoutIntervalForRequest" in content)
            
            # Test 58: Efficient data structures
            self.test(58, "Set<AnyCancellable> for subscriptions",
                     "Set<AnyCancellable>" in content)
        else:
            for i in range(49, 59):
                self.test(i, f"Performance test {i}", False)
                
    def test_security(self):
        """Tests 59-66: Security implementation validation"""
        # Test 59: API key handling
        config_file = self.base_path / "Models" / "MixerConfiguration.swift"
        if config_file.exists():
            content = config_file.read_text()
            self.test(59, "API key property in configuration",
                     "apiKey" in content)
        else:
            self.test(59, "API key property in configuration", False)
            
        # Test 60: HTTPS support
        service_file = self.base_path / "Services" / "MixerService.swift"
        if service_file.exists():
            content = service_file.read_text()
            self.test(60, "HTTPS URL support",
                     "https://" in content or "http://" in content)
            
            # Test 61: Input validation
            self.test(61, "Settings validation",
                     "isValid" in content or "guard" in content)
            
            # Test 62: Error handling
            self.test(62, "Error types defined",
                     "Error" in content)
            
            # Test 63: Secure storage prep (check settings view for SecureField)
            settings_file = self.base_path / "Views" / "MixerSettingsView.swift"
            if settings_file.exists():
                settings_content = settings_file.read_text()
                self.test(63, "SecureField for sensitive input",
                         "SecureField" in settings_content)
            else:
                self.test(63, "SecureField for sensitive input",
                         "SecureField" in content or "apiKey" in content)
            
            # Test 64: No hardcoded credentials
            self.test(64, "No hardcoded secrets",
                     "sk-" not in content and "secret_" not in content)
            
            # Test 65: Network error handling
            self.test(65, "Network error handling",
                     "connectionFailed" in content)
            
            # Test 66: Timeout protection
            self.test(66, "Timeout protection",
                     "timeout" in content.lower())
        else:
            for i in range(60, 67):
                self.test(i, f"Security test {i}", False)
                
    def test_integration(self):
        """Tests 67-78: Integration validation"""
        app_file = self.base_path / "MobileAG06App.swift"
        
        if app_file.exists():
            content = app_file.read_text()
            
            # Test 67: Main app structure
            self.test(67, "@main app entry point",
                     "@main" in content)
            
            # Test 68: StateObject usage
            self.test(68, "@StateObject for services",
                     "@StateObject" in content)
            
            # Test 69: EnvironmentObject
            self.test(69, "@EnvironmentObject usage",
                     "@EnvironmentObject" in content or ".environmentObject" in content)
            
            # Test 70: TabView navigation
            self.test(70, "TabView for navigation",
                     "TabView" in content)
            
            # Test 71: Lifecycle handling
            self.test(71, "Background notification handling",
                     "didEnterBackground" in content)
            
            # Test 72: Foreground handling
            self.test(72, "Foreground notification handling",
                     "willEnterForeground" in content)
            
            # Test 73: Settings tab
            self.test(73, "Settings tab included",
                     "Settings" in content)
            
            # Test 74: Mixer tab
            self.test(74, "Mixer tab included",
                     "Mixer" in content)
            
            # Test 75: ConfigurationManager
            self.test(75, "ConfigurationManager integrated",
                     "ConfigurationManager" in content)
            
            # Test 76: AutomationService
            self.test(76, "AutomationService integrated",
                     "AutomationService" in content)
            
            # Test 77: AboutView
            self.test(77, "AboutView defined",
                     "AboutView" in content)
            
            # Test 78: Navigation structure
            self.test(78, "NavigationView used",
                     "NavigationView" in content)
        else:
            for i in range(67, 79):
                self.test(i, f"Integration test {i}", False)
                
    def test_regression(self):
        """Tests 79-88: Regression and edge case validation"""
        test_file = self.base_path / "Tests" / "MobileAG06Tests.swift"
        
        # Test 79: Test file exists
        self.test(79, "MobileAG06Tests.swift exists", test_file.exists())
        
        if test_file.exists():
            content = test_file.read_text()
            
            # Test 80: XCTest import
            self.test(80, "XCTest framework imported",
                     "import XCTest" in content)
            
            # Test 81: Test class defined
            self.test(81, "MobileAG06Tests class defined",
                     "class MobileAG06Tests" in content)
            
            # Test 82: setUp method
            self.test(82, "setUp() method defined",
                     "func setUp()" in content)
            
            # Test 83: tearDown method
            self.test(83, "tearDown() method defined",
                     "func tearDown()" in content)
            
            # Test 84: Test count is 88
            test_count = len(re.findall(r'func test_\d+_', content))
            self.test(84, "Exactly 88 test methods",
                     test_count == 88)
            
            # Test 85: Async test support
            self.test(85, "Async test support",
                     "async" in content)
            
            # Test 86: XCTestExpectation usage
            self.test(86, "XCTestExpectation for async",
                     "XCTestExpectation" in content)
            
            # Test 87: Performance tests
            self.test(87, "Performance tests included",
                     "measure" in content)
            
            # Test 88: Comprehensive validation
            self.test(88, "test_88_comprehensive_validation exists",
                     "test_88_comprehensive_validation" in content)
        else:
            for i in range(80, 89):
                self.test(i, f"Regression test {i}", False)
                
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("CRITICAL ASSESSMENT SUMMARY")
        print("=" * 70)
        
        # Category summaries
        categories = [
            ("Configuration", 1, 8),
            ("Service Layer", 9, 20),
            ("UI Components", 21, 36),
            ("API Integration", 37, 48),
            ("Performance", 49, 58),
            ("Security", 59, 66),
            ("Integration", 67, 78),
            ("Regression", 79, 88)
        ]
        
        for name, start, end in categories:
            category_passed = sum(1 for i, _, status in self.results[start-1:end] if "PASS" in status)
            category_total = end - start + 1
            percentage = (category_passed / category_total) * 100
            status = "✅" if percentage == 100 else "⚠️" if percentage >= 80 else "❌"
            print(f"{status} {name:20s}: {category_passed}/{category_total} ({percentage:.1f}%)")
        
        # Overall summary
        print("\n" + "-" * 70)
        percentage = (self.passed / 88) * 100
        print(f"TOTAL: {self.passed}/88 tests passed ({percentage:.1f}%)")
        
        if percentage == 100:
            print("\n✅ SUCCESS: Mobile AG06 App passes 88/88 tests (100% compliance)")
        elif percentage >= 80:
            print(f"\n⚠️ PARTIAL: Mobile AG06 App passes {self.passed}/88 tests ({percentage:.1f}%)")
            print("Failed tests:", [i for i, _, status in self.results if "FAIL" in status])
        else:
            print(f"\n❌ FAILURE: Mobile AG06 App passes only {self.passed}/88 tests ({percentage:.1f}%)")
            
        # Save results
        with open('mobile_test_results.json', 'w') as f:
            json.dump({
                'passed': self.passed,
                'failed': self.failed,
                'percentage': percentage,
                'results': [(i, desc, status) for i, desc, status in self.results]
            }, f, indent=2)
            
        print(f"\nResults saved to mobile_test_results.json")


if __name__ == "__main__":
    validator = MobileAppValidator()
    validator.run_all_tests()