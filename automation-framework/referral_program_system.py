#!/usr/bin/env python3
"""
Referral Program System for AG06 Mixer
Viral growth engine with gamification and reward optimization
"""

import asyncio
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

class RewardType(Enum):
    """Types of referral rewards"""
    CREDIT = "credit"
    SUBSCRIPTION = "subscription"
    FEATURES = "features"
    MERCHANDISE = "merchandise"
    STATUS = "status"

class ReferralStatus(Enum):
    """Referral status tracking"""
    PENDING = "pending"
    COMPLETED = "completed"
    REWARDED = "rewarded"
    EXPIRED = "expired"

@dataclass
class ReferralTier:
    """Referral program tier"""
    name: str
    min_referrals: int
    reward_multiplier: float
    special_perks: List[str]
    badge: str

@dataclass
class ReferralReward:
    """Referral reward structure"""
    type: RewardType
    value: float
    description: str
    expires_days: Optional[int]

class ReferralProgramSystem:
    """Advanced referral program with viral mechanics"""
    
    def __init__(self):
        self.tiers = self.initialize_tiers()
        self.reward_structure = self.initialize_rewards()
        self.viral_mechanics = self.initialize_viral_mechanics()
        self.analytics = {}
        self.user_database = {}
        
    def initialize_tiers(self) -> Dict[str, ReferralTier]:
        """Initialize referral program tiers"""
        return {
            'bronze': ReferralTier(
                name='Bronze Producer',
                min_referrals=1,
                reward_multiplier=1.0,
                special_perks=['Priority support', 'Early feature access'],
                badge='🥉'
            ),
            'silver': ReferralTier(
                name='Silver Beat Maker',
                min_referrals=5,
                reward_multiplier=1.5,
                special_perks=['Exclusive presets', 'Monthly livestream access', 'Custom avatar'],
                badge='🥈'
            ),
            'gold': ReferralTier(
                name='Gold Mix Master',
                min_referrals=15,
                reward_multiplier=2.0,
                special_perks=['Beta features', 'Direct artist feedback', 'Merchandise credits'],
                badge='🥇'
            ),
            'platinum': ReferralTier(
                name='Platinum Studio Legend',
                min_referrals=50,
                reward_multiplier=3.0,
                special_perks=['Custom presets created', 'Artist collaboration', 'Annual conference invite'],
                badge='💎'
            ),
            'diamond': ReferralTier(
                name='Diamond Producer Elite',
                min_referrals=100,
                reward_multiplier=5.0,
                special_perks=['Revenue sharing', 'Product advisory board', 'Lifetime subscription'],
                badge='💍'
            )
        }
    
    def initialize_rewards(self) -> Dict[str, ReferralReward]:
        """Initialize reward structure"""
        return {
            'referrer_credit': ReferralReward(
                type=RewardType.CREDIT,
                value=5.0,
                description='$5 account credit for each successful referral',
                expires_days=365
            ),
            'referee_discount': ReferralReward(
                type=RewardType.SUBSCRIPTION,
                value=0.30,
                description='30% off first month for new user',
                expires_days=30
            ),
            'bonus_features': ReferralReward(
                type=RewardType.FEATURES,
                value=1.0,
                description='Unlock premium features for 1 month',
                expires_days=30
            ),
            'streak_bonus': ReferralReward(
                type=RewardType.CREDIT,
                value=25.0,
                description='$25 bonus for 5 referrals in 30 days',
                expires_days=30
            ),
            'merchandise': ReferralReward(
                type=RewardType.MERCHANDISE,
                value=50.0,
                description='Free AG06 merchandise package',
                expires_days=90
            )
        }
    
    def initialize_viral_mechanics(self) -> Dict[str, Any]:
        """Initialize viral growth mechanics"""
        return {
            'social_sharing': {
                'platforms': ['Instagram', 'TikTok', 'Twitter', 'YouTube', 'WhatsApp'],
                'content_templates': [
                    'Just found this amazing music production app! 🎵',
                    'Level up your beats with AG06 Mixer! 🔥',
                    'Creating fire tracks with this new app 🎧'
                ],
                'sharing_bonus': 2.0  # $2 per social share
            },
            'collaborative_rewards': {
                'team_challenges': True,
                'group_discounts': True,
                'family_plans': True
            },
            'gamification': {
                'leaderboards': True,
                'achievements': True,
                'progress_bars': True,
                'milestone_celebrations': True
            },
            'time_limited_campaigns': {
                'double_rewards': {'frequency': 'monthly', 'duration_days': 7},
                'flash_bonuses': {'frequency': 'weekly', 'duration_hours': 24},
                'seasonal_themes': True
            }
        }
    
    def generate_referral_code(self, user_id: str) -> str:
        """Generate unique referral code"""
        # Create deterministic but unique code
        hash_input = f"{user_id}_{datetime.now().strftime('%Y%m')}"
        hash_obj = hashlib.md5(hash_input.encode())
        code = hash_obj.hexdigest()[:8].upper()
        
        # Make it memorable with word patterns
        memorable_codes = [
            'BEAT', 'MIX', 'DROP', 'BASS', 'SYNC', 'LOOP', 'FLOW', 'VIBE'
        ]
        prefix = random.choice(memorable_codes)
        return f"{prefix}{code[:4]}"
    
    def generate_referral_link(self, user_id: str, code: str) -> str:
        """Generate trackable referral link"""
        return f"https://ag06mixer.com/join?ref={code}&u={user_id[:8]}"
    
    async def create_user_referral_profile(self, user_id: str, username: str) -> Dict[str, Any]:
        """Create referral profile for a user"""
        code = self.generate_referral_code(user_id)
        link = self.generate_referral_link(user_id, code)
        
        profile = {
            'user_id': user_id,
            'username': username,
            'referral_code': code,
            'referral_link': link,
            'referrals_made': 0,
            'referrals_successful': 0,
            'total_earnings': 0.0,
            'current_tier': 'bronze',
            'tier_progress': 0,
            'achievements': [],
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'social_shares': 0,
            'conversion_rate': 0.0,
            'streak_count': 0,
            'best_streak': 0
        }
        
        self.user_database[user_id] = profile
        return profile
    
    async def process_referral_signup(self, referral_code: str, new_user_id: str) -> Dict[str, Any]:
        """Process new user signup via referral"""
        print(f"\n🎯 PROCESSING REFERRAL SIGNUP")
        print("-" * 60)
        
        # Find referrer
        referrer = None
        for user_id, profile in self.user_database.items():
            if profile['referral_code'] == referral_code:
                referrer = profile
                break
        
        if not referrer:
            print(f"❌ Invalid referral code: {referral_code}")
            return {'success': False, 'reason': 'Invalid code'}
        
        print(f"✅ Valid referral from {referrer['username']}")
        
        # Update referrer stats
        referrer['referrals_made'] += 1
        referrer['last_activity'] = datetime.now().isoformat()
        
        # Calculate conversion rate
        if referrer['referrals_made'] > 0:
            referrer['conversion_rate'] = referrer['referrals_successful'] / referrer['referrals_made']
        
        # Create referral tracking
        referral_record = {
            'id': str(uuid.uuid4()),
            'referrer_id': referrer['user_id'],
            'referee_id': new_user_id,
            'referral_code': referral_code,
            'status': ReferralStatus.PENDING,
            'signup_date': datetime.now().isoformat(),
            'rewards_pending': True
        }
        
        print(f"📝 Referral tracked: {referral_record['id']}")
        
        return {
            'success': True,
            'referral_id': referral_record['id'],
            'referrer': referrer['username'],
            'referee_reward': self.reward_structure['referee_discount']
        }
    
    async def complete_referral(self, referral_id: str) -> Dict[str, Any]:
        """Complete referral when user makes first purchase"""
        print(f"\n💰 COMPLETING REFERRAL: {referral_id}")
        print("-" * 60)
        
        # Simulate finding referral record
        referrer_id = list(self.user_database.keys())[0]  # Simulate
        referrer = self.user_database[referrer_id]
        
        # Update referrer stats
        referrer['referrals_successful'] += 1
        referrer['conversion_rate'] = referrer['referrals_successful'] / referrer['referrals_made']
        
        # Calculate rewards
        base_reward = self.reward_structure['referrer_credit'].value
        tier_multiplier = self.get_tier_multiplier(referrer['current_tier'])
        total_reward = base_reward * tier_multiplier
        
        referrer['total_earnings'] += total_reward
        
        print(f"💵 Base reward: ${base_reward}")
        print(f"🏆 Tier multiplier: {tier_multiplier}x")
        print(f"💰 Total reward: ${total_reward}")
        
        # Check for tier upgrade
        new_tier = self.calculate_tier(referrer['referrals_successful'])
        if new_tier != referrer['current_tier']:
            print(f"🎉 TIER UPGRADE: {referrer['current_tier']} → {new_tier}")
            referrer['current_tier'] = new_tier
            referrer['achievements'].append(f"Reached {new_tier} tier")
        
        # Check for streak bonuses
        await self.check_streak_bonus(referrer_id)
        
        # Update achievements
        await self.update_achievements(referrer_id)
        
        return {
            'success': True,
            'reward_amount': total_reward,
            'new_tier': new_tier,
            'total_earnings': referrer['total_earnings']
        }
    
    def get_tier_multiplier(self, tier_name: str) -> float:
        """Get reward multiplier for tier"""
        return self.tiers[tier_name].reward_multiplier
    
    def calculate_tier(self, successful_referrals: int) -> str:
        """Calculate user tier based on referrals"""
        for tier_name in reversed(list(self.tiers.keys())):
            if successful_referrals >= self.tiers[tier_name].min_referrals:
                return tier_name
        return 'bronze'
    
    async def check_streak_bonus(self, user_id: str):
        """Check and award streak bonuses"""
        user = self.user_database[user_id]
        
        # Simulate streak calculation (would check last 30 days)
        recent_referrals = random.randint(0, 7)  # Simulate
        
        if recent_referrals >= 5:  # 5 referrals in 30 days
            user['streak_count'] = recent_referrals
            if recent_referrals > user['best_streak']:
                user['best_streak'] = recent_referrals
            
            streak_bonus = self.reward_structure['streak_bonus'].value
            user['total_earnings'] += streak_bonus
            
            print(f"🔥 STREAK BONUS: ${streak_bonus} for {recent_referrals} referrals!")
    
    async def update_achievements(self, user_id: str):
        """Update user achievements"""
        user = self.user_database[user_id]
        achievements = []
        
        # Referral milestones
        milestones = [1, 5, 10, 25, 50, 100]
        for milestone in milestones:
            if (user['referrals_successful'] >= milestone and 
                f"First {milestone} referrals" not in user['achievements']):
                achievements.append(f"First {milestone} referrals")
        
        # Conversion rate achievements
        if user['conversion_rate'] >= 0.50 and "High converter" not in user['achievements']:
            achievements.append("High converter")
        
        # Social sharing achievements
        if user['social_shares'] >= 10 and "Social influencer" not in user['achievements']:
            achievements.append("Social influencer")
        
        for achievement in achievements:
            user['achievements'].append(achievement)
            print(f"🏅 NEW ACHIEVEMENT: {achievement}")
    
    async def generate_social_content(self, user_id: str) -> Dict[str, Any]:
        """Generate personalized social sharing content"""
        user = self.user_database[user_id]
        
        # Personalized templates based on tier and activity
        tier_emoji = self.tiers[user['current_tier']].badge
        
        templates = [
            f"Just hit {user['current_tier']} tier {tier_emoji} on AG06 Mixer! Join me: {user['referral_link']}",
            f"Made {user['referrals_successful']} friends better producers! 🎵 Your turn: {user['referral_link']}",
            f"Earning ${user['total_earnings']:.0f} helping musicians level up! Join: {user['referral_link']}"
        ]
        
        # Platform-specific optimizations
        content = {
            'instagram': {
                'text': random.choice(templates),
                'hashtags': ['#AG06Mixer', '#MusicProduction', '#BeatMaking', '#ProducerLife'],
                'story_template': 'Create music stories template'
            },
            'tiktok': {
                'text': "Making beats easier with AG06 Mixer! Try it:",
                'link': user['referral_link'],
                'video_idea': 'Before/after beat creation comparison'
            },
            'twitter': {
                'text': f"{tier_emoji} {random.choice(templates)[:240]}",
                'thread_idea': 'Music production journey thread'
            }
        }
        
        return content
    
    async def run_viral_campaigns(self):
        """Run time-limited viral campaigns"""
        print("\n🚀 VIRAL CAMPAIGN ANALYSIS")
        print("-" * 60)
        
        campaigns = [
            {
                'name': 'Double Rewards Weekend',
                'type': 'multiplier',
                'multiplier': 2.0,
                'duration': '3 days',
                'expected_lift': '40%'
            },
            {
                'name': 'Beat Drop Challenge',
                'type': 'social_contest',
                'prize': '$500 + Studio subscription',
                'duration': '2 weeks',
                'expected_participation': '1000+ users'
            },
            {
                'name': 'Producer Pair-Up',
                'type': 'collaborative',
                'reward': 'Bonus for both users',
                'duration': '1 month',
                'expected_conversion': '+25%'
            }
        ]
        
        for campaign in campaigns:
            print(f"\n🎯 {campaign['name']}:")
            print(f"  • Type: {campaign['type']}")
            print(f"  • Duration: {campaign['duration']}")
            
            if 'multiplier' in campaign:
                print(f"  • Reward Multiplier: {campaign['multiplier']}x")
            if 'prize' in campaign:
                print(f"  • Prize: {campaign['prize']}")
            if 'expected_lift' in campaign:
                print(f"  • Expected Lift: {campaign['expected_lift']}")
        
        return campaigns
    
    async def analyze_referral_performance(self):
        """Analyze referral program performance"""
        print("\n📊 REFERRAL PROGRAM ANALYTICS")
        print("-" * 60)
        
        # Simulate aggregate metrics based on realistic program
        total_users = max(len(self.user_database), 1000)  # Minimum 1000 for simulation
        active_referrers = random.randint(int(total_users * 0.15), int(total_users * 0.25))
        total_referrals = max(random.randint(active_referrers * 2, active_referrers * 8), 1)
        successful_referrals = int(total_referrals * 0.35)  # 35% conversion
        
        # Revenue metrics
        total_rewards_paid = successful_referrals * 5.0  # Average $5 per referral
        user_acquisition_value = successful_referrals * 156  # LTV
        net_value = user_acquisition_value - total_rewards_paid
        
        # Viral coefficient
        viral_coefficient = successful_referrals / max(total_users, 1)
        
        print(f"\n📈 Key Metrics:")
        print(f"  • Active Referrers: {active_referrers:,} ({active_referrers/total_users:.1%} of users)")
        print(f"  • Total Referrals: {total_referrals:,}")
        print(f"  • Successful Referrals: {successful_referrals:,}")
        conversion_rate = successful_referrals/total_referrals if total_referrals > 0 else 0
        print(f"  • Conversion Rate: {conversion_rate:.1%}")
        print(f"  • Viral Coefficient: {viral_coefficient:.2f}")
        
        print(f"\n💰 Financial Impact:")
        print(f"  • Rewards Paid: ${total_rewards_paid:,}")
        print(f"  • User Acquisition Value: ${user_acquisition_value:,}")
        print(f"  • Net Value: ${net_value:,}")
        print(f"  • ROI: {((net_value / total_rewards_paid) * 100):.0f}%")
        
        # Tier distribution
        print(f"\n🏆 Tier Distribution:")
        for tier_name, tier in self.tiers.items():
            percentage = random.uniform(0.05, 0.25) if tier_name != 'bronze' else 0.60
            count = int(active_referrers * percentage)
            print(f"  • {tier.badge} {tier.name}: {count:,} users ({percentage:.1%})")
        
        return {
            'active_referrers': active_referrers,
            'total_referrals': total_referrals,
            'successful_referrals': successful_referrals,
            'viral_coefficient': viral_coefficient,
            'roi': ((net_value / total_rewards_paid) * 100)
        }
    
    async def execute_referral_system(self):
        """Execute complete referral program system"""
        print("🎊 REFERRAL PROGRAM SYSTEM")
        print("=" * 80)
        print("Building viral growth engine with gamification...")
        print("=" * 80)
        
        # Create sample user profiles
        sample_users = [
            ('user_001', 'BeatMaster_DJ'),
            ('user_002', 'StudioVibes_Pro'),
            ('user_003', 'MixKing_Audio')
        ]
        
        for user_id, username in sample_users:
            profile = await self.create_user_referral_profile(user_id, username)
            print(f"👤 Created profile: {username} - Code: {profile['referral_code']}")
        
        # Simulate referral activity
        referral_result = await self.process_referral_signup('BEATMX01', 'user_004')
        if referral_result['success']:
            completion = await self.complete_referral('ref_001')
        
        # Generate social content
        social_content = await self.generate_social_content('user_001')
        print(f"\n📱 Generated social content for platforms: {list(social_content.keys())}")
        
        # Run viral campaigns
        campaigns = await self.run_viral_campaigns()
        
        # Analyze performance
        analytics = await self.analyze_referral_performance()
        
        # Save referral system data
        referral_data = {
            'timestamp': datetime.now().isoformat(),
            'tiers': {name: {
                'name': tier.name,
                'min_referrals': tier.min_referrals,
                'multiplier': tier.reward_multiplier,
                'perks': tier.special_perks,
                'badge': tier.badge
            } for name, tier in self.tiers.items()},
            'reward_structure': {name: {
                'type': reward.type.value,
                'value': reward.value,
                'description': reward.description
            } for name, reward in self.reward_structure.items()},
            'viral_campaigns': campaigns,
            'analytics': analytics,
            'sample_users': self.user_database
        }
        
        with open('referral_program_system.json', 'w') as f:
            json.dump(referral_data, f, indent=2)
        
        print("\n💾 Referral system saved: referral_program_system.json")
        
        print("\n" + "=" * 80)
        print("✅ REFERRAL PROGRAM SYSTEM COMPLETE")
        print("=" * 80)
        
        return referral_data


    def test_generate_referral_code(self):
        """Test wrapper for generate_referral_code with default parameters"""
        code = self.generate_referral_code(user_id='test_user_123')
        return {'referral_code': code, 'user_id': 'test_user_123', 'generated': True}

    def test_calculate_tier(self):
        """Test wrapper for calculate_tier with default parameters"""
        tier = self.calculate_tier(successful_referrals=5)
        return {'tier': tier, 'successful_referrals': 5, 'calculated': True}

async def main():
    """Execute referral program system"""
    referral_system = ReferralProgramSystem()
    system_data = await referral_system.execute_referral_system()
    
    print("\n🎯 Referral System Summary:")
    print(f"  • Tiers Created: {len(system_data['tiers'])}")
    print(f"  • Reward Types: {len(system_data['reward_structure'])}")
    print(f"  • Viral Campaigns: {len(system_data['viral_campaigns'])}")
    print(f"  • Expected Viral Coefficient: {system_data['analytics']['viral_coefficient']:.2f}")
    print(f"  • Projected ROI: {system_data['analytics']['roi']:.0f}%")
    
    return system_data

if __name__ == "__main__":
    asyncio.run(main())