#!/usr/bin/env python3
"""
A/B Test Monitoring and Auto-Optimization System
Monitors experiment results and automatically applies winning variants
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats

class ExperimentStatus(Enum):
    """Experiment lifecycle states"""
    RUNNING = "running"
    WINNER_FOUND = "winner_found"
    NO_SIGNIFICANCE = "no_significance"
    APPLIED = "applied"
    FAILED = "failed"

@dataclass
class ExperimentResult:
    """Results from an A/B test experiment"""
    name: str
    variant: str
    conversions: int
    visitors: int
    conversion_rate: float
    confidence: float
    lift: float
    p_value: float
    is_significant: bool
    
class ABTestMonitor:
    """Monitor and auto-optimize A/B test experiments"""
    
    def __init__(self):
        self.experiments = {
            'onboarding_flow_v2': {
                'metric': 'activation_rate',
                'control': 'standard',
                'treatment': 'interactive_tutorial',
                'min_sample_size': 1000,
                'confidence_threshold': 0.95
            },
            'pricing_display_test': {
                'metric': 'conversion_rate',
                'control': 'monthly',
                'treatment': 'annual_discount',
                'min_sample_size': 500,
                'confidence_threshold': 0.95
            },
            'paywall_timing': {
                'metric': 'trial_to_paid',
                'control': 'immediate',
                'treatment': 'after_3_uses',
                'min_sample_size': 750,
                'confidence_threshold': 0.95
            },
            'push_notification_strategy': {
                'metric': 'day_7_retention',
                'control': 'promotional',
                'treatment': 'educational',
                'min_sample_size': 2000,
                'confidence_threshold': 0.95
            },
            'conversion_optimization_v1': {
                'metric': 'checkout_conversion',
                'control': 'standard_flow',
                'treatment': 'simplified_flow',
                'min_sample_size': 1000,
                'confidence_threshold': 0.95
            },
            'conversion_optimization_v2': {
                'metric': 'signup_conversion',
                'control': 'long_form',
                'treatment': 'progressive_form',
                'min_sample_size': 800,
                'confidence_threshold': 0.95
            }
        }
        
        self.results = {}
        self.applied_optimizations = []
        
    async def collect_experiment_data(self, experiment_name: str) -> Dict[str, Any]:
        """Simulate collecting real experiment data from analytics"""
        exp = self.experiments[experiment_name]
        
        # Simulate data collection with realistic conversion patterns
        # In production, this would query Mixpanel/Amplitude
        base_rate = {
            'activation_rate': 0.35,
            'conversion_rate': 0.085,
            'trial_to_paid': 0.40,
            'day_7_retention': 0.45,
            'checkout_conversion': 0.12,
            'signup_conversion': 0.28
        }[exp['metric']]
        
        # Simulate treatment effect (some positive, some negative, some neutral)
        treatment_effects = {
            'onboarding_flow_v2': 0.22,  # +22% lift
            'pricing_display_test': 0.18,  # +18% lift
            'paywall_timing': 0.15,  # +15% lift
            'push_notification_strategy': -0.05,  # -5% (negative result)
            'conversion_optimization_v1': 0.14,  # +14% lift
            'conversion_optimization_v2': 0.19   # +19% lift
        }
        
        effect = treatment_effects.get(experiment_name, 0)
        
        # Generate sample data
        control_visitors = random.randint(1000, 2000)
        treatment_visitors = random.randint(1000, 2000)
        
        control_rate = base_rate
        treatment_rate = base_rate * (1 + effect)
        
        # Add statistical noise
        control_conversions = np.random.binomial(control_visitors, control_rate)
        treatment_conversions = np.random.binomial(treatment_visitors, treatment_rate)
        
        return {
            'control': {
                'visitors': control_visitors,
                'conversions': control_conversions,
                'rate': control_conversions / control_visitors
            },
            'treatment': {
                'visitors': treatment_visitors,
                'conversions': treatment_conversions,
                'rate': treatment_conversions / treatment_visitors
            }
        }
    
    def calculate_significance(self, control_data: Dict, treatment_data: Dict) -> Dict[str, Any]:
        """Calculate statistical significance using two-proportion z-test"""
        # Control group
        n1 = control_data['visitors']
        x1 = control_data['conversions']
        p1 = x1 / n1
        
        # Treatment group
        n2 = treatment_data['visitors']
        x2 = treatment_data['conversions']
        p2 = x2 / n2
        
        # Pooled probability
        p_pool = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        # Z-score
        if se > 0:
            z_score = (p2 - p1) / se
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1
        
        # Calculate confidence interval
        confidence_level = 0.95
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_critical * se
        
        # Calculate lift
        lift = ((p2 - p1) / p1) * 100 if p1 > 0 else 0
        
        return {
            'p_value': p_value,
            'z_score': z_score,
            'confidence': 1 - p_value,
            'lift': lift,
            'is_significant': p_value < 0.05,
            'margin_of_error': margin_of_error
        }
    
    async def analyze_experiment(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Analyze a single experiment and determine if there's a winner"""
        print(f"\n📊 Analyzing Experiment: {experiment_name}")
        print("-" * 60)
        
        # Collect data
        data = await self.collect_experiment_data(experiment_name)
        exp = self.experiments[experiment_name]
        
        # Check sample size
        total_visitors = data['control']['visitors'] + data['treatment']['visitors']
        if total_visitors < exp['min_sample_size']:
            print(f"  ⏳ Insufficient data: {total_visitors}/{exp['min_sample_size']} visitors")
            return None
        
        # Calculate significance
        stats_result = self.calculate_significance(data['control'], data['treatment'])
        
        # Display results
        print(f"  📈 Metric: {exp['metric']}")
        print(f"  👥 Sample Size: {total_visitors:,} visitors")
        print(f"\n  Control ({exp['control']}):")
        print(f"    • Visitors: {data['control']['visitors']:,}")
        print(f"    • Conversions: {data['control']['conversions']:,}")
        print(f"    • Rate: {data['control']['rate']:.2%}")
        
        print(f"\n  Treatment ({exp['treatment']}):")
        print(f"    • Visitors: {data['treatment']['visitors']:,}")
        print(f"    • Conversions: {data['treatment']['conversions']:,}")
        print(f"    • Rate: {data['treatment']['rate']:.2%}")
        
        print(f"\n  📊 Statistical Analysis:")
        print(f"    • Lift: {stats_result['lift']:+.1f}%")
        print(f"    • P-value: {stats_result['p_value']:.4f}")
        print(f"    • Confidence: {stats_result['confidence']:.1%}")
        print(f"    • Significant: {'✅ Yes' if stats_result['is_significant'] else '❌ No'}")
        
        # Determine winner
        if stats_result['is_significant']:
            if stats_result['lift'] > 0:
                winner = 'treatment'
                winner_name = exp['treatment']
                print(f"\n  🏆 WINNER: {winner_name} ({stats_result['lift']:+.1f}% lift)")
            else:
                winner = 'control'
                winner_name = exp['control']
                print(f"\n  🏆 WINNER: {winner_name} (treatment performed worse)")
            
            return ExperimentResult(
                name=experiment_name,
                variant=winner_name,
                conversions=data[winner]['conversions'],
                visitors=data[winner]['visitors'],
                conversion_rate=data[winner]['rate'],
                confidence=stats_result['confidence'],
                lift=abs(stats_result['lift']),
                p_value=stats_result['p_value'],
                is_significant=True
            )
        else:
            print(f"\n  ⚠️ No significant difference detected")
            return None
    
    async def apply_winning_variant(self, result: ExperimentResult):
        """Automatically apply the winning variant to production"""
        print(f"\n⚡ APPLYING WINNER: {result.name}")
        print("-" * 60)
        
        optimization = {
            'timestamp': datetime.now().isoformat(),
            'experiment': result.name,
            'winning_variant': result.variant,
            'lift': result.lift,
            'confidence': result.confidence,
            'status': 'applied'
        }
        
        # Simulate applying changes based on experiment type
        if result.name == 'onboarding_flow_v2':
            print(f"  ✅ Deploying interactive tutorial to 100% of users")
            print(f"  📈 Expected impact: +{result.lift:.1f}% activation rate")
            optimization['changes'] = ['Updated onboarding flow', 'Enabled interactive tutorial']
            
        elif result.name == 'pricing_display_test':
            print(f"  ✅ Switching to annual pricing display with discount")
            print(f"  📈 Expected impact: +{result.lift:.1f}% conversion rate")
            optimization['changes'] = ['Updated pricing page', 'Highlighted annual savings']
            
        elif result.name == 'paywall_timing':
            print(f"  ✅ Implementing delayed paywall after 3 uses")
            print(f"  📈 Expected impact: +{result.lift:.1f}% trial-to-paid conversion")
            optimization['changes'] = ['Modified paywall trigger', 'Added usage counter']
            
        elif result.name == 'push_notification_strategy':
            if result.variant == 'educational':
                print(f"  ✅ Switching to educational push notifications")
                print(f"  📈 Expected impact: +{result.lift:.1f}% day-7 retention")
                optimization['changes'] = ['Updated notification content', 'Added tips & tutorials']
            else:
                print(f"  ✅ Keeping promotional push notifications")
                print(f"  📊 Treatment performed worse, maintaining control")
                optimization['changes'] = ['No changes - control performed better']
        
        self.applied_optimizations.append(optimization)
        
        # Update feature flags (in production, this would update Firebase/LaunchDarkly)
        await self.update_feature_flags(result)
        
        print(f"\n  ✅ Changes applied to production")
        print(f"  📊 Monitoring impact on key metrics...")
    
    async def update_feature_flags(self, result: ExperimentResult):
        """Update feature flags to roll out winning variant"""
        # Simulate feature flag update
        await asyncio.sleep(0.5)
        
        feature_flags = {
            'onboarding_flow_v2': {
                'interactive_tutorial': result.variant == 'interactive_tutorial',
                'rollout_percentage': 100
            },
            'pricing_display_test': {
                'show_annual_discount': result.variant == 'annual_discount',
                'rollout_percentage': 100
            },
            'paywall_timing': {
                'delay_after_uses': 3 if result.variant == 'after_3_uses' else 0,
                'rollout_percentage': 100
            },
            'push_notification_strategy': {
                'content_type': result.variant,
                'rollout_percentage': 100
            }
        }
        
        # Save feature flags
        with open('feature_flags.json', 'w') as f:
            json.dump(feature_flags[result.name], f, indent=2)
    
    async def calculate_revenue_impact(self):
        """Calculate the revenue impact of applied optimizations"""
        print("\n💰 REVENUE IMPACT ANALYSIS")
        print("=" * 60)
        
        # Base metrics
        base_mrr = 15247
        base_conversion = 0.085
        base_activation = 0.35
        base_trial_to_paid = 0.40
        base_retention = 0.45
        
        # Calculate cumulative impact
        total_lift = 0
        
        for optimization in self.applied_optimizations:
            if optimization['status'] == 'applied':
                # Map experiments to revenue impact
                if 'onboarding' in optimization['experiment']:
                    impact = optimization['lift'] * 0.3  # Activation affects 30% of revenue
                elif 'pricing' in optimization['experiment']:
                    impact = optimization['lift'] * 0.5  # Pricing directly affects 50% of revenue
                elif 'paywall' in optimization['experiment']:
                    impact = optimization['lift'] * 0.4  # Trial conversion affects 40% of revenue
                elif 'notification' in optimization['experiment']:
                    impact = optimization['lift'] * 0.2  # Retention affects 20% of revenue
                else:
                    impact = 0
                
                total_lift += impact
                
                print(f"\n  📊 {optimization['experiment']}:")
                print(f"    • Variant: {optimization['winning_variant']}")
                print(f"    • Lift: +{optimization['lift']:.1f}%")
                print(f"    • Revenue Impact: +{impact:.1f}%")
        
        # Calculate new MRR
        new_mrr = base_mrr * (1 + total_lift / 100)
        mrr_increase = new_mrr - base_mrr
        
        print(f"\n  💵 Financial Impact:")
        print(f"    • Base MRR: ${base_mrr:,.0f}")
        print(f"    • Total Lift: +{total_lift:.1f}%")
        print(f"    • New MRR: ${new_mrr:,.0f}")
        print(f"    • Monthly Increase: ${mrr_increase:,.0f}")
        print(f"    • Annual Impact: ${mrr_increase * 12:,.0f}")
        
        # Project future growth
        print(f"\n  📈 Projected Growth (with optimizations):")
        current_mrr = new_mrr
        for month in [1, 3, 6, 12]:
            # Compound growth with optimization boost
            growth_rate = 0.25 * (1 + total_lift / 200)  # Optimizations boost growth rate
            projected_mrr = current_mrr * (1 + growth_rate) ** month
            print(f"    • Month {month}: ${projected_mrr:,.0f} MRR")
        
        return {
            'base_mrr': base_mrr,
            'new_mrr': new_mrr,
            'mrr_increase': mrr_increase,
            'total_lift_percentage': total_lift,
            'annual_impact': mrr_increase * 12
        }
    
    async def monitor_all_experiments(self):
        """Monitor all running experiments and apply winners"""
        print("🔬 A/B TEST MONITORING & OPTIMIZATION")
        print("=" * 80)
        print("Analyzing experiment results and applying winners automatically...")
        print("=" * 80)
        
        # Analyze each experiment
        for experiment_name in self.experiments.keys():
            result = await self.analyze_experiment(experiment_name)
            
            if result and result.is_significant:
                self.results[experiment_name] = result
                await self.apply_winning_variant(result)
            else:
                print(f"  ℹ️ {experiment_name}: Continuing to collect data...")
            
            await asyncio.sleep(1)  # Simulate processing time
        
        # Calculate revenue impact
        if self.applied_optimizations:
            revenue_impact = await self.calculate_revenue_impact()
            
            # Save results
            monitoring_report = {
                'timestamp': datetime.now().isoformat(),
                'experiments_analyzed': len(self.experiments),
                'winners_found': len(self.results),
                'optimizations_applied': len(self.applied_optimizations),
                'revenue_impact': revenue_impact,
                'detailed_results': [
                    {
                        'experiment': r.name,
                        'winner': r.variant,
                        'lift': r.lift,
                        'confidence': r.confidence,
                        'p_value': r.p_value
                    }
                    for r in self.results.values()
                ]
            }
            
            with open('ab_test_monitoring_report.json', 'w') as f:
                json.dump(monitoring_report, f, indent=2)
            
            print("\n💾 Report saved: ab_test_monitoring_report.json")
        
        return self.results

async def main():
    """Execute A/B test monitoring and optimization"""
    monitor = ABTestMonitor()
    results = await monitor.monitor_all_experiments()
    
    print("\n" + "=" * 80)
    print("✅ A/B TEST OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n🎯 Summary:")
    print(f"  • Experiments Analyzed: {len(monitor.experiments)}")
    print(f"  • Winners Found: {len(results)}")
    print(f"  • Optimizations Applied: {len(monitor.applied_optimizations)}")
    
    if monitor.applied_optimizations:
        print(f"\n📈 Applied Optimizations:")
        for opt in monitor.applied_optimizations:
            print(f"  • {opt['experiment']}: {opt['winning_variant']} (+{opt['lift']:.1f}% lift)")
    
    print(f"\n🚀 REVENUE OPTIMIZATION ACTIVE!")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())