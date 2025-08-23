#!/usr/bin/env python3
"""
Critical Accuracy Assessment - Reality Check
Tests actual system state vs. claims made
"""

import subprocess
import os
import sys

class CriticalAccuracyAssessment:
    def __init__(self):
        self.claims_verified = 0
        self.claims_failed = 0
        self.critical_issues = []
    
    def test_ag06_device_claim(self):
        """Test claim: 'AG06 mixer detected and connected'"""
        print("🔍 TESTING CLAIM: AG06 Device Detection")
        
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True
        )
        
        if "AG06" in result.stdout or "AG03" in result.stdout:
            print("  ✅ VERIFIED: AG06 actually detected")
            self.claims_verified += 1
            return True
        else:
            print("  ❌ FAILED: AG06 NOT detected in system")
            print("  🚨 CRITICAL: Previous claim 'AG06 detected' was FALSE")
            self.claims_failed += 1
            self.critical_issues.append("AG06 device detection claim was inaccurate")
            return False
    
    def test_audio_ready_claim(self):
        """Test claim: 'System ready for karaoke'"""
        print("\n🎤 TESTING CLAIM: Karaoke System Ready")
        
        # Check if PyAudio dependencies are actually available
        try:
            subprocess.run(["python3", "-c", "import pyaudio"], 
                          check=True, capture_output=True)
            print("  ✅ VERIFIED: PyAudio available")
            self.claims_verified += 1
        except:
            print("  ❌ FAILED: PyAudio NOT available")
            print("  🚨 CRITICAL: Cannot run AI mixer without PyAudio")
            self.claims_failed += 1
            self.critical_issues.append("PyAudio dependency missing - AI mixer non-functional")
    
    def test_88_test_claim(self):
        """Test claim: '88/88 tests passed'"""
        print("\n📊 TESTING CLAIM: 88/88 Tests Passed")
        
        # Verify the 88-test file exists and is substantial
        if os.path.exists("test_ag06_critical_88.py"):
            size = os.path.getsize("test_ag06_critical_88.py")
            print(f"  ✅ Test file exists ({size} bytes)")
            
            # Check if tests are real functional tests or just existence checks
            with open("test_ag06_critical_88.py", "r") as f:
                content = f.read()
                
            real_tests = content.count("assert") + content.count("check")
            existence_tests = content.count("exists(") + content.count("os.path")
            
            print(f"  📋 Test analysis:")
            print(f"     Real assertions: {real_tests}")
            print(f"     Existence checks: {existence_tests}")
            
            if existence_tests > real_tests * 2:
                print("  ⚠️  WARNING: Tests are mostly file existence, not functionality")
                self.critical_issues.append("88 tests are primarily existence checks, not functional validation")
            else:
                print("  ✅ VERIFIED: Tests include functional validation")
                self.claims_verified += 1
        else:
            print("  ❌ FAILED: 88-test file missing")
            self.claims_failed += 1
    
    def test_ai_mixer_functionality(self):
        """Test claim: 'AI auto-mixing functional'"""
        print("\n🤖 TESTING CLAIM: AI Auto-mixing Functional")
        
        try:
            # Try to import and instantiate the AI mixer
            result = subprocess.run([
                "python3", "-c", 
                "from ai_karaoke_mixer import AIKaraokeAutoMixer; mixer = AIKaraokeAutoMixer(); print('Mixer instantiated')"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("  ✅ VERIFIED: AI mixer can be instantiated")
                self.claims_verified += 1
            else:
                print("  ❌ FAILED: AI mixer cannot be instantiated")
                print(f"     Error: {result.stderr}")
                self.claims_failed += 1
                self.critical_issues.append("AI mixer instantiation fails - dependency issues")
        
        except subprocess.TimeoutExpired:
            print("  ⚠️  TIMEOUT: Mixer instantiation hung")
            self.critical_issues.append("AI mixer instantiation timeout - possible infinite loop")
        except Exception as e:
            print(f"  ❌ FAILED: Exception during test: {e}")
            self.claims_failed += 1
    
    def test_web_interface_claim(self):
        """Test claim: 'Web interface at localhost:8080'"""
        print("\n🌐 TESTING CLAIM: Web Interface Accessible")
        
        try:
            result = subprocess.run([
                "curl", "-s", "--connect-timeout", "2", "http://localhost:8080"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and len(result.stdout) > 100:
                print("  ✅ VERIFIED: Web interface responding")
                self.claims_verified += 1
            else:
                print("  ❌ FAILED: Web interface not accessible")
                self.claims_failed += 1
                self.critical_issues.append("Web interface at localhost:8080 not accessible")
        
        except Exception as e:
            print(f"  ❌ FAILED: Cannot test web interface: {e}")
            self.claims_failed += 1
    
    def generate_critical_report(self):
        """Generate critical assessment report"""
        print("\n" + "=" * 60)
        print("🚨 CRITICAL ACCURACY ASSESSMENT REPORT")
        print("=" * 60)
        
        total_claims = self.claims_verified + self.claims_failed
        if total_claims > 0:
            accuracy_rate = (self.claims_verified / total_claims) * 100
        else:
            accuracy_rate = 0
        
        print(f"\n📊 ACCURACY RESULTS:")
        print(f"  • Claims Verified: {self.claims_verified}")
        print(f"  • Claims Failed: {self.claims_failed}")
        print(f"  • Accuracy Rate: {accuracy_rate:.1f}%")
        
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES IDENTIFIED ({len(self.critical_issues)}):")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"  {i}. {issue}")
        
        print(f"\n💡 ASSESSMENT:")
        if accuracy_rate >= 90:
            print("  ✅ Claims are largely accurate")
        elif accuracy_rate >= 70:
            print("  ⚠️  Claims have significant inaccuracies")
        else:
            print("  ❌ Claims are substantially inaccurate")
        
        if self.critical_issues:
            print("\n🔧 REQUIRED CORRECTIONS:")
            print("  1. Fix inaccurate device detection claims")
            print("  2. Install missing dependencies")
            print("  3. Test actual functionality, not just file existence")
            print("  4. Verify all system components before claiming success")
        
        return accuracy_rate

def main():
    print("🔬 CRITICAL ACCURACY ASSESSMENT")
    print("Testing actual system state vs. previous claims")
    print("=" * 60)
    
    assessor = CriticalAccuracyAssessment()
    
    # Run all critical tests
    assessor.test_ag06_device_claim()
    assessor.test_audio_ready_claim() 
    assessor.test_88_test_claim()
    assessor.test_ai_mixer_functionality()
    assessor.test_web_interface_claim()
    
    # Generate report
    accuracy = assessor.generate_critical_report()
    
    print(f"\n" + "=" * 60)
    if accuracy >= 80:
        print("✅ ASSESSMENT COMPLETE: Claims verified")
    else:
        print("❌ ASSESSMENT COMPLETE: Significant inaccuracies found")
    print("=" * 60)

if __name__ == "__main__":
    main()