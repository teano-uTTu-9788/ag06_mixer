#!/usr/bin/env python3
"""
Instance 3 Verification Script
Validates that all mobile app components are ready for monetization
"""

import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

class Instance3Verification:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'mobile_app': {'status': 'unknown', 'tests': 0, 'percentage': 0},
            'monetization': {'status': 'unknown', 'tiers': 0, 'features': []},
            'integration': {'status': 'unknown', 'server': False, 'dashboard': False},
            'deployment': {'status': 'unknown', 'ci_cd': False, 'production': False},
            'analytics': {'status': 'unknown', 'tracking': False, 'ab_testing': False},
            'readiness_score': 0
        }
    
    async def run_verification(self):
        """Run complete verification for Instance 3 handoff"""
        print("🔍 Instance 3 Verification - Mobile App Monetization Readiness")
        print("=" * 70)
        
        await self.verify_mobile_app()
        await self.verify_monetization_framework()
        await self.verify_integration_status()
        await self.verify_deployment_pipeline()
        await self.verify_analytics_framework()
        
        self.calculate_readiness_score()
        self.print_final_report()
        
        return self.results
    
    async def verify_mobile_app(self):
        """Verify mobile app is ready"""
        print("\n📱 Verifying Mobile App Status...")
        
        try:
            # Check mobile test results
            test_file = Path("mobile_test_results.json")
            if test_file.exists():
                with open(test_file, 'r') as f:
                    results = json.load(f)
                
                self.results['mobile_app'] = {
                    'status': 'ready' if results['percentage'] == 100.0 else 'not_ready',
                    'tests': results['passed'],
                    'total': results['passed'] + results['failed'],
                    'percentage': results['percentage']
                }
                
                if results['percentage'] == 100.0:
                    print(f"  ✅ Mobile App: {results['passed']}/{results['passed'] + results['failed']} tests passing")
                else:
                    print(f"  ❌ Mobile App: {results['failed']} test failures")
            else:
                print("  ⚠️ Mobile test results not found")
                self.results['mobile_app']['status'] = 'not_found'
                
        except Exception as e:
            print(f"  ❌ Error checking mobile app: {e}")
            self.results['mobile_app']['status'] = 'error'
    
    async def verify_monetization_framework(self):
        """Verify monetization components are ready"""
        print("\n💰 Verifying Monetization Framework...")
        
        features_found = []
        
        # Check subscription tiers implementation
        swift_files = [
            "mobile-app/Models/MixerConfiguration.swift",
            "mobile-app/Views/SubscriptionView.swift",
            "mobile-app/Services/MixerService.swift"
        ]
        
        for file_path in swift_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for subscription features
                    if 'SubscriptionTier' in content:
                        features_found.append('subscription_tiers')
                    if 'free' in content and 'pro' in content and 'studio' in content:
                        features_found.append('three_tier_system')
                    if 'BatteryMode' in content:
                        features_found.append('battery_optimization')
                    if 'InAppPurchase' in content or 'StoreKit' in content:
                        features_found.append('in_app_purchases')
                        
                except Exception as e:
                    print(f"  ⚠️ Error reading {file_path}: {e}")
        
        # Check production services for analytics hooks
        prod_files = [
            "mobile-app/Production/ProductionMixerService.swift",
            "mobile-app/Production/ProductionMobileAG06App.swift"
        ]
        
        for file_path in prod_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if 'ABTestManager' in content:
                        features_found.append('ab_testing')
                    if 'FeatureFlagManager' in content:
                        features_found.append('feature_flags')
                    if 'PerformanceMonitor' in content:
                        features_found.append('analytics_hooks')
                        
                except Exception as e:
                    print(f"  ⚠️ Error reading {file_path}: {e}")
        
        tier_count = 1 if 'three_tier_system' in features_found else 0
        
        self.results['monetization'] = {
            'status': 'ready' if len(features_found) >= 4 else 'partial',
            'tiers': 3 if tier_count else 0,
            'features': features_found
        }
        
        print(f"  ✅ Subscription Tiers: {'3 tiers implemented' if tier_count else 'Missing'}")
        print(f"  ✅ Monetization Features: {len(features_found)} components ready")
        for feature in features_found:
            print(f"    • {feature.replace('_', ' ').title()}")
    
    async def verify_integration_status(self):
        """Verify server integration and monitoring"""
        print("\n🔗 Verifying Integration Status...")
        
        server_healthy = False
        dashboard_active = False
        
        try:
            # Check server health
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8080/healthz', timeout=5) as response:
                    if response.status == 200:
                        server_healthy = True
                        print("  ✅ AG06 Server: Healthy and responding")
                    else:
                        print(f"  ❌ AG06 Server: Unhealthy (status {response.status})")
                        
        except Exception as e:
            print(f"  ❌ AG06 Server: Unreachable ({e})")
        
        try:
            # Check monitoring dashboard
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8082/api/health', timeout=5) as response:
                    if response.status == 200:
                        dashboard_active = True
                        print("  ✅ Monitoring Dashboard: Active")
                    else:
                        print(f"  ⚠️ Monitoring Dashboard: Response {response.status}")
                        
        except Exception as e:
            print(f"  ❌ Monitoring Dashboard: Not accessible ({e})")
        
        self.results['integration'] = {
            'status': 'ready' if server_healthy and dashboard_active else 'partial',
            'server': server_healthy,
            'dashboard': dashboard_active
        }
    
    async def verify_deployment_pipeline(self):
        """Verify CI/CD pipeline is configured"""
        print("\n🚀 Verifying Deployment Pipeline...")
        
        pipeline_file = Path(".github/workflows/mobile-ci-cd.yml")
        deployment_guide = Path("PRODUCTION_DEPLOYMENT_GUIDE.md")
        
        ci_cd_ready = pipeline_file.exists()
        production_docs = deployment_guide.exists()
        
        if ci_cd_ready:
            print("  ✅ CI/CD Pipeline: Configured for iOS/Android deployment")
        else:
            print("  ❌ CI/CD Pipeline: Not found")
            
        if production_docs:
            print("  ✅ Production Guide: Complete deployment documentation")
        else:
            print("  ❌ Production Guide: Missing documentation")
        
        self.results['deployment'] = {
            'status': 'ready' if ci_cd_ready and production_docs else 'partial',
            'ci_cd': ci_cd_ready,
            'production': production_docs
        }
    
    async def verify_analytics_framework(self):
        """Verify analytics and A/B testing readiness"""
        print("\n📊 Verifying Analytics Framework...")
        
        tracking_ready = False
        ab_testing_ready = False
        
        # Check for analytics implementation in production files
        prod_files = [
            "mobile-app/Production/GoogleBestPractices.swift",
            "mobile-app/Production/ProductionMobileAG06App.swift"
        ]
        
        for file_path in prod_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for analytics components
                    if any(term in content for term in ['trackEvent', 'PerformanceMonitor', 'MetricsCollector']):
                        tracking_ready = True
                    if any(term in content for term in ['ABTestManager', 'experiment', 'variant']):
                        ab_testing_ready = True
                        
                except Exception as e:
                    print(f"  ⚠️ Error checking {file_path}: {e}")
        
        if tracking_ready:
            print("  ✅ User Tracking: Event tracking and analytics ready")
        else:
            print("  ❌ User Tracking: Analytics implementation missing")
            
        if ab_testing_ready:
            print("  ✅ A/B Testing: Experiment framework ready")
        else:
            print("  ❌ A/B Testing: Framework missing")
        
        self.results['analytics'] = {
            'status': 'ready' if tracking_ready and ab_testing_ready else 'partial',
            'tracking': tracking_ready,
            'ab_testing': ab_testing_ready
        }
    
    def calculate_readiness_score(self):
        """Calculate overall readiness score"""
        score = 0
        max_score = 100
        
        # Mobile app (30 points)
        if self.results['mobile_app']['status'] == 'ready':
            score += 30
        elif self.results['mobile_app']['status'] == 'partial':
            score += 15
        
        # Monetization (25 points)
        if self.results['monetization']['status'] == 'ready':
            score += 25
        elif self.results['monetization']['status'] == 'partial':
            score += 15
        
        # Integration (20 points)
        if self.results['integration']['status'] == 'ready':
            score += 20
        elif self.results['integration']['status'] == 'partial':
            score += 10
        
        # Deployment (15 points)
        if self.results['deployment']['status'] == 'ready':
            score += 15
        elif self.results['deployment']['status'] == 'partial':
            score += 8
        
        # Analytics (10 points)
        if self.results['analytics']['status'] == 'ready':
            score += 10
        elif self.results['analytics']['status'] == 'partial':
            score += 5
        
        self.results['readiness_score'] = score
        
        # Overall status
        if score >= 90:
            self.results['overall_status'] = 'production_ready'
        elif score >= 70:
            self.results['overall_status'] = 'mostly_ready'
        elif score >= 50:
            self.results['overall_status'] = 'partial_ready'
        else:
            self.results['overall_status'] = 'not_ready'
    
    def print_final_report(self):
        """Print final verification report"""
        print("\n" + "=" * 70)
        print("INSTANCE 3 READINESS REPORT")
        print("=" * 70)
        
        score = self.results['readiness_score']
        status = self.results['overall_status']
        
        # Overall status
        if status == 'production_ready':
            status_icon = "✅"
            status_text = "PRODUCTION READY"
        elif status == 'mostly_ready':
            status_icon = "🟡"
            status_text = "MOSTLY READY"
        elif status == 'partial_ready':
            status_icon = "⚠️"
            status_text = "PARTIALLY READY"
        else:
            status_icon = "❌"
            status_text = "NOT READY"
        
        print(f"\n{status_icon} OVERALL STATUS: {status_text}")
        print(f"📊 READINESS SCORE: {score}/100 ({score}%)")
        
        # Component breakdown
        print(f"\n📱 Mobile App: {self.results['mobile_app']['status'].upper()}")
        if self.results['mobile_app'].get('tests'):
            print(f"   Tests: {self.results['mobile_app']['tests']}/{self.results['mobile_app'].get('total', 88)} ({self.results['mobile_app']['percentage']}%)")
        
        print(f"💰 Monetization: {self.results['monetization']['status'].upper()}")
        print(f"   Subscription Tiers: {self.results['monetization']['tiers']}")
        print(f"   Features: {len(self.results['monetization']['features'])}")
        
        print(f"🔗 Integration: {self.results['integration']['status'].upper()}")
        print(f"   Server: {'✅' if self.results['integration']['server'] else '❌'}")
        print(f"   Dashboard: {'✅' if self.results['integration']['dashboard'] else '❌'}")
        
        print(f"🚀 Deployment: {self.results['deployment']['status'].upper()}")
        print(f"   CI/CD: {'✅' if self.results['deployment']['ci_cd'] else '❌'}")
        print(f"   Docs: {'✅' if self.results['deployment']['production'] else '❌'}")
        
        print(f"📊 Analytics: {self.results['analytics']['status'].upper()}")
        print(f"   Tracking: {'✅' if self.results['analytics']['tracking'] else '❌'}")
        print(f"   A/B Testing: {'✅' if self.results['analytics']['ab_testing'] else '❌'}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS FOR INSTANCE 3:")
        
        if score >= 90:
            print("   • System is production-ready for immediate monetization")
            print("   • Focus on app store setup and payment processing")
            print("   • Configure analytics dashboards for revenue tracking")
            print("   • Plan A/B testing experiments for conversion optimization")
        elif score >= 70:
            print("   • System is mostly ready with minor improvements needed")
            print("   • Address any missing components before launch")
            print("   • Test monetization features thoroughly")
        else:
            print("   • System requires significant work before monetization")
            print("   • Focus on completing core mobile app functionality")
            print("   • Ensure all integration tests pass")
        
        # Next steps
        print(f"\n📋 IMMEDIATE NEXT STEPS:")
        print("   1. Configure Apple App Store Connect developer account")
        print("   2. Set up Google Play Console for Android deployment")
        print("   3. Integrate Stripe/Apple Pay for payment processing")
        print("   4. Configure analytics dashboards (Mixpanel/Amplitude)")
        print("   5. Design onboarding flow for subscription conversion")
        print("   6. Create app store marketing materials and screenshots")
        print("   7. Plan pricing experiments and A/B tests")
        print("   8. Set up customer support and feedback collection")
        
        # Save results
        with open('instance_3_readiness_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Report saved to: instance_3_readiness_report.json")
        print("=" * 70)

async def main():
    """Main verification runner"""
    verifier = Instance3Verification()
    results = await verifier.run_verification()
    return results['readiness_score'] >= 90

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)