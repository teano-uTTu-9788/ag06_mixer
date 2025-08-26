#!/usr/bin/env python3
"""
User Acquisition Channel Optimizer
Analyzes campaign performance and automatically scales successful channels
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

class ChannelStatus(Enum):
    """User acquisition channel states"""
    ACTIVE = "active"
    PAUSED = "paused"
    SCALING = "scaling"
    OPTIMIZING = "optimizing"
    EXHAUSTED = "exhausted"

@dataclass
class ChannelPerformance:
    """Performance metrics for a user acquisition channel"""
    channel: str
    spend: float
    users_acquired: int
    cac: float
    ltv_cac_ratio: float
    roas: float
    conversion_rate: float
    quality_score: float
    scalability: str

class UserAcquisitionOptimizer:
    """Optimize and scale user acquisition channels based on performance"""
    
    def __init__(self):
        self.channels = {
            'Apple Search Ads': {
                'initial_budget': 5000,
                'current_budget': 5000,
                'min_cac': 2.50,
                'target_cac': 3.50,
                'max_cac': 5.00,
                'platform': 'ios',
                'status': ChannelStatus.ACTIVE
            },
            'Google Ads (UAC)': {
                'initial_budget': 4000,
                'current_budget': 4000,
                'min_cac': 3.00,
                'target_cac': 4.20,
                'max_cac': 6.00,
                'platform': 'android',
                'status': ChannelStatus.ACTIVE
            },
            'Facebook/Instagram': {
                'initial_budget': 3000,
                'current_budget': 3000,
                'min_cac': 4.00,
                'target_cac': 5.80,
                'max_cac': 8.00,
                'platform': 'both',
                'status': ChannelStatus.ACTIVE
            },
            'TikTok Ads': {
                'initial_budget': 6000,
                'current_budget': 6000,
                'min_cac': 2.00,
                'target_cac': 2.90,
                'max_cac': 4.00,
                'platform': 'both',
                'status': ChannelStatus.ACTIVE
            },
            'YouTube Pre-roll': {
                'initial_budget': 2000,
                'current_budget': 2000,
                'min_cac': 5.00,
                'target_cac': 6.50,
                'max_cac': 9.00,
                'platform': 'both',
                'status': ChannelStatus.ACTIVE
            }
        }
        
        self.ltv = 156  # Customer lifetime value
        self.total_budget = 20000
        self.performance_history = []
        self.optimizations_applied = []
        
    async def collect_channel_performance(self, channel_name: str) -> ChannelPerformance:
        """Simulate collecting real performance data from each channel"""
        channel = self.channels[channel_name]
        
        # Simulate realistic performance with variance
        # TikTok and Apple Search Ads perform best in simulation
        performance_multipliers = {
            'TikTok Ads': 1.35,  # Overperforming
            'Apple Search Ads': 1.15,  # Performing well
            'Google Ads (UAC)': 0.95,  # Slightly underperforming
            'Facebook/Instagram': 0.75,  # Underperforming
            'YouTube Pre-roll': 0.60  # Significantly underperforming
        }
        
        multiplier = performance_multipliers.get(channel_name, 1.0)
        
        # Calculate metrics with realistic variance
        spend = channel['current_budget'] * random.uniform(0.85, 1.0)  # Not all budget spent
        
        # CAC with performance multiplier and variance
        base_cac = channel['target_cac'] / multiplier
        actual_cac = base_cac * random.uniform(0.85, 1.15)
        
        # Users acquired based on spend and CAC
        users_acquired = int(spend / actual_cac)
        
        # Quality metrics
        conversion_rate = (0.085 * multiplier) * random.uniform(0.9, 1.1)
        quality_score = min(100, 70 * multiplier * random.uniform(0.9, 1.1))
        
        # Calculate ROAS and LTV/CAC
        ltv_cac_ratio = self.ltv / actual_cac
        revenue_per_user = self.ltv * conversion_rate
        total_revenue = revenue_per_user * users_acquired
        roas = total_revenue / spend if spend > 0 else 0
        
        # Determine scalability
        if ltv_cac_ratio > 4:
            scalability = "high"
        elif ltv_cac_ratio > 3:
            scalability = "medium"
        elif ltv_cac_ratio > 2:
            scalability = "low"
        else:
            scalability = "none"
        
        return ChannelPerformance(
            channel=channel_name,
            spend=spend,
            users_acquired=users_acquired,
            cac=actual_cac,
            ltv_cac_ratio=ltv_cac_ratio,
            roas=roas,
            conversion_rate=conversion_rate,
            quality_score=quality_score,
            scalability=scalability
        )
    
    async def analyze_channel_performance(self):
        """Analyze performance across all channels"""
        print("\n📊 CHANNEL PERFORMANCE ANALYSIS")
        print("-" * 60)
        
        performances = []
        
        for channel_name in self.channels.keys():
            perf = await self.collect_channel_performance(channel_name)
            performances.append(perf)
            
            # Display performance
            print(f"\n💰 {channel_name}:")
            print(f"  • Spend: ${perf.spend:,.0f}")
            print(f"  • Users Acquired: {perf.users_acquired:,}")
            print(f"  • CAC: ${perf.cac:.2f}")
            print(f"  • LTV/CAC: {perf.ltv_cac_ratio:.1f}x")
            print(f"  • ROAS: {perf.roas:.2f}x")
            print(f"  • Quality Score: {perf.quality_score:.0f}/100")
            print(f"  • Scalability: {perf.scalability.upper()}")
            
            # Performance indicators
            if perf.ltv_cac_ratio > 3:
                print(f"  ✅ Status: PROFITABLE - Scale up")
            elif perf.ltv_cac_ratio > 2:
                print(f"  ⚠️ Status: MARGINAL - Optimize")
            else:
                print(f"  ❌ Status: UNPROFITABLE - Reduce/Pause")
        
        return performances
    
    def calculate_optimal_budget_allocation(self, performances: List[ChannelPerformance]) -> Dict[str, float]:
        """Calculate optimal budget allocation based on performance"""
        print("\n💡 OPTIMAL BUDGET ALLOCATION")
        print("-" * 60)
        
        # Score each channel based on multiple factors
        channel_scores = {}
        
        for perf in performances:
            # Weighted scoring system
            ltv_cac_score = min(100, perf.ltv_cac_ratio * 20)  # Max 100 points
            roas_score = min(100, perf.roas * 25)  # Max 100 points
            quality_score = perf.quality_score  # Already 0-100
            
            # Scalability multiplier
            scalability_multiplier = {
                'high': 1.5,
                'medium': 1.2,
                'low': 1.0,
                'none': 0.5
            }[perf.scalability]
            
            # Calculate weighted score
            total_score = (
                ltv_cac_score * 0.4 +  # 40% weight on LTV/CAC
                roas_score * 0.3 +      # 30% weight on ROAS
                quality_score * 0.3      # 30% weight on quality
            ) * scalability_multiplier
            
            channel_scores[perf.channel] = total_score
        
        # Calculate budget allocation proportional to scores
        total_score = sum(channel_scores.values())
        
        new_allocations = {}
        for channel, score in channel_scores.items():
            if score > 50:  # Only allocate to channels scoring above threshold
                allocation = (score / total_score) * self.total_budget
                new_allocations[channel] = allocation
            else:
                new_allocations[channel] = 0  # Pause underperforming channels
        
        # Ensure we use full budget
        allocated_total = sum(new_allocations.values())
        if allocated_total < self.total_budget and allocated_total > 0:
            # Distribute remaining budget to top performers
            top_channel = max(channel_scores, key=channel_scores.get)
            new_allocations[top_channel] += (self.total_budget - allocated_total)
        
        return new_allocations
    
    async def apply_budget_optimizations(self, new_allocations: Dict[str, float]):
        """Apply the optimized budget allocations"""
        print("\n⚡ APPLYING BUDGET OPTIMIZATIONS")
        print("-" * 60)
        
        optimizations = []
        
        for channel_name, new_budget in new_allocations.items():
            old_budget = self.channels[channel_name]['current_budget']
            change = new_budget - old_budget
            change_pct = (change / old_budget * 100) if old_budget > 0 else 0
            
            # Determine action
            if new_budget == 0:
                action = "PAUSE"
                self.channels[channel_name]['status'] = ChannelStatus.PAUSED
                print(f"\n  ⏸️ {channel_name}:")
                print(f"    • Action: PAUSED (underperforming)")
                print(f"    • Budget: ${old_budget:,.0f} → $0")
            elif change > 1000:
                action = "SCALE UP"
                self.channels[channel_name]['status'] = ChannelStatus.SCALING
                print(f"\n  📈 {channel_name}:")
                print(f"    • Action: SCALING UP")
                print(f"    • Budget: ${old_budget:,.0f} → ${new_budget:,.0f}")
                print(f"    • Change: {change_pct:+.0f}%")
            elif change < -1000:
                action = "SCALE DOWN"
                self.channels[channel_name]['status'] = ChannelStatus.OPTIMIZING
                print(f"\n  📉 {channel_name}:")
                print(f"    • Action: SCALING DOWN")
                print(f"    • Budget: ${old_budget:,.0f} → ${new_budget:,.0f}")
                print(f"    • Change: {change_pct:+.0f}%")
            else:
                action = "MAINTAIN"
                print(f"\n  ➡️ {channel_name}:")
                print(f"    • Action: MAINTAIN")
                print(f"    • Budget: ${new_budget:,.0f}")
            
            # Update budget
            self.channels[channel_name]['current_budget'] = new_budget
            
            # Record optimization
            optimizations.append({
                'channel': channel_name,
                'action': action,
                'old_budget': old_budget,
                'new_budget': new_budget,
                'change': change,
                'change_percentage': change_pct
            })
        
        self.optimizations_applied.extend(optimizations)
        
        # Simulate applying changes to ad platforms
        await self.update_campaign_settings(optimizations)
        
        return optimizations
    
    async def update_campaign_settings(self, optimizations: List[Dict]):
        """Simulate updating campaign settings on ad platforms"""
        print("\n🔄 UPDATING CAMPAIGN SETTINGS")
        print("-" * 60)
        
        for opt in optimizations:
            if opt['action'] != 'MAINTAIN':
                await asyncio.sleep(0.3)  # Simulate API call
                print(f"  ✅ {opt['channel']}: Campaign updated")
        
        print("\n  ✅ All campaign settings updated across platforms")
    
    async def generate_acquisition_forecast(self):
        """Generate updated user acquisition forecast"""
        print("\n📈 UPDATED ACQUISITION FORECAST")
        print("-" * 60)
        
        # Calculate new blended CAC
        total_budget = sum(ch['current_budget'] for ch in self.channels.values())
        
        # Weighted average CAC based on new allocations
        weighted_cac = 0
        expected_users = 0
        
        for channel_name, channel in self.channels.items():
            if channel['current_budget'] > 0:
                weight = channel['current_budget'] / total_budget
                weighted_cac += channel['target_cac'] * weight
                expected_users += int(channel['current_budget'] / channel['target_cac'])
        
        print(f"\n  📊 Optimized Metrics:")
        print(f"    • Total Budget: ${total_budget:,.0f}")
        print(f"    • Blended CAC: ${weighted_cac:.2f}")
        print(f"    • Expected Users: {expected_users:,}")
        print(f"    • Cost per Install: ${total_budget/expected_users:.2f}")
        
        # Project monthly growth with optimized channels
        print(f"\n  📈 User Acquisition Projection:")
        cumulative_users = 5250  # Starting point
        
        for month in range(1, 7):
            monthly_users = expected_users
            cumulative_users += monthly_users
            
            # Apply growth as we optimize
            if month > 3:
                monthly_users = int(monthly_users * 1.15)  # 15% improvement from optimization
            
            print(f"    • Month {month}: +{monthly_users:,} users ({cumulative_users:,} total)")
        
        # Calculate ROI
        total_revenue = expected_users * self.ltv * 0.085  # Conversion rate
        roi = ((total_revenue - total_budget) / total_budget) * 100
        
        print(f"\n  💰 Financial Impact:")
        print(f"    • Monthly Spend: ${total_budget:,.0f}")
        print(f"    • Expected Revenue: ${total_revenue:,.0f}")
        print(f"    • ROI: {roi:.0f}%")
        print(f"    • Payback Period: {weighted_cac/self.ltv*12:.1f} months")
        
        return {
            'total_budget': total_budget,
            'blended_cac': weighted_cac,
            'expected_users': expected_users,
            'expected_revenue': total_revenue,
            'roi': roi
        }
    
    async def optimize_user_acquisition(self):
        """Main optimization workflow"""
        print("🎯 USER ACQUISITION OPTIMIZATION")
        print("=" * 80)
        print("Analyzing channel performance and optimizing budget allocation...")
        print("=" * 80)
        
        # Analyze current performance
        performances = await self.analyze_channel_performance()
        
        # Calculate optimal allocation
        new_allocations = self.calculate_optimal_budget_allocation(performances)
        
        # Display new allocation
        print("\n📊 NEW BUDGET ALLOCATION:")
        for channel, budget in new_allocations.items():
            old_budget = self.channels[channel]['initial_budget']
            if budget > 0:
                print(f"  • {channel}: ${old_budget:,} → ${budget:,.0f}")
            else:
                print(f"  • {channel}: ${old_budget:,} → PAUSED")
        
        # Apply optimizations
        await self.apply_budget_optimizations(new_allocations)
        
        # Generate forecast
        forecast = await self.generate_acquisition_forecast()
        
        # Save optimization report
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_analysis': [
                {
                    'channel': p.channel,
                    'spend': p.spend,
                    'users': p.users_acquired,
                    'cac': p.cac,
                    'ltv_cac_ratio': p.ltv_cac_ratio,
                    'roas': p.roas,
                    'scalability': p.scalability
                }
                for p in performances
            ],
            'optimizations': self.optimizations_applied,
            'forecast': forecast,
            'new_allocations': new_allocations
        }
        
        with open('user_acquisition_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n💾 Report saved: user_acquisition_optimization_report.json")
        
        return report

async def main():
    """Execute user acquisition optimization"""
    optimizer = UserAcquisitionOptimizer()
    report = await optimizer.optimize_user_acquisition()
    
    print("\n" + "=" * 80)
    print("✅ USER ACQUISITION OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n🎯 Optimization Summary:")
    print(f"  • Channels Analyzed: {len(optimizer.channels)}")
    print(f"  • Budget Reallocated: ${optimizer.total_budget:,}")
    print(f"  • Expected User Growth: {report['forecast']['expected_users']:,} users/month")
    print(f"  • ROI: {report['forecast']['roi']:.0f}%")
    
    # Show winning channels
    winning_channels = [
        opt['channel'] for opt in optimizer.optimizations_applied 
        if opt['action'] == 'SCALE UP'
    ]
    
    if winning_channels:
        print(f"\n🏆 Top Performing Channels:")
        for channel in winning_channels:
            print(f"  • {channel}: Scaled up for maximum growth")
    
    print(f"\n🚀 USER ACQUISITION OPTIMIZED FOR GROWTH!")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())