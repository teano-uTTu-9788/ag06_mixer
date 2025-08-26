"""
Production Recommendation System
Following Netflix, Spotify, YouTube, and Amazon practices
"""

import asyncio
import json
import time
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import heapq
import random
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationType(Enum):
    """Types of recommendations"""
    COLLABORATIVE = "collaborative"  # User-based CF
    CONTENT_BASED = "content_based"  # Item features
    HYBRID = "hybrid"  # Combined approach
    MATRIX_FACTORIZATION = "matrix_factorization"  # SVD/NMF
    DEEP_LEARNING = "deep_learning"  # Neural networks
    CONTEXTUAL = "contextual"  # Context-aware

class InteractionType(Enum):
    """Types of user interactions"""
    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    DOWNLOAD = "download"
    SKIP = "skip"
    RATING = "rating"
    PURCHASE = "purchase"

@dataclass
class User:
    """User profile"""
    user_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)
    demographics: Dict[str, str] = field(default_factory=dict)
    behavior_patterns: Dict[str, float] = field(default_factory=dict)

@dataclass
class Item:
    """Item to recommend"""
    item_id: str
    title: str
    category: str
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    popularity_score: float = 0.0
    quality_score: float = 0.0

@dataclass
class Interaction:
    """User-item interaction"""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    rating: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Recommendation:
    """Recommendation result"""
    item_id: str
    score: float
    rank: int
    algorithm: str
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class RecommendationEngine:
    """Main recommendation engine"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[Interaction] = []
        
        # Algorithm implementations
        self.collaborative_filter = CollaborativeFilter()
        self.content_filter = ContentBasedFilter()
        self.matrix_factorizer = MatrixFactorization()
        self.popularity_recommender = PopularityRecommender()
        
        # Hybrid system
        self.ensemble = EnsembleRecommender()
        
        # Real-time components
        self.online_learner = OnlineLearner()
        self.cold_start_handler = ColdStartHandler()
        
        # Performance tracking
        self.metrics = RecommendationMetrics()
        
        # Feature engineering
        self.feature_encoder = FeatureEncoder()
    
    def add_user(self, user: User):
        """Add user to system"""
        self.users[user.user_id] = user
    
    def add_item(self, item: Item):
        """Add item to system"""
        self.items[item.item_id] = item
    
    async def record_interaction(self, interaction: Interaction):
        """Record user interaction"""
        self.interactions.append(interaction)
        
        # Update online models
        await self.online_learner.update(interaction)
        
        # Track metrics
        await self.metrics.record_interaction(interaction)
    
    async def get_recommendations(self, user_id: str, num_items: int = 10,
                                 algorithm: Optional[RecommendationType] = None,
                                 context: Optional[Dict] = None) -> List[Recommendation]:
        """Get personalized recommendations"""
        
        if user_id not in self.users:
            # Cold start user
            return await self.cold_start_handler.recommend_for_new_user(
                user_id, num_items, self.items
            )
        
        user = self.users[user_id]
        
        # Get user interaction history
        user_interactions = [i for i in self.interactions if i.user_id == user_id]
        
        # Check for cold start items
        if len(user_interactions) < 5:
            return await self.cold_start_handler.recommend_for_sparse_user(
                user, num_items, self.items, user_interactions
            )
        
        # Generate recommendations based on algorithm
        if algorithm == RecommendationType.COLLABORATIVE:
            recommendations = await self.collaborative_filter.recommend(
                user_id, self.users, self.interactions, num_items
            )
        elif algorithm == RecommendationType.CONTENT_BASED:
            recommendations = await self.content_filter.recommend(
                user, self.items, user_interactions, num_items
            )
        elif algorithm == RecommendationType.MATRIX_FACTORIZATION:
            recommendations = await self.matrix_factorizer.recommend(
                user_id, self.interactions, self.items, num_items
            )
        else:
            # Default: Hybrid approach
            recommendations = await self.ensemble.recommend(
                user_id, user, self.users, self.items, 
                self.interactions, num_items, context
            )
        
        # Apply business rules and filtering
        recommendations = await self._apply_business_rules(
            user_id, recommendations, context
        )
        
        # Track recommendation serving
        await self.metrics.record_recommendations(user_id, recommendations)
        
        return recommendations
    
    async def _apply_business_rules(self, user_id: str, 
                                   recommendations: List[Recommendation],
                                   context: Optional[Dict]) -> List[Recommendation]:
        """Apply business rules and filtering"""
        
        # Remove already interacted items
        user_items = set(i.item_id for i in self.interactions if i.user_id == user_id)
        recommendations = [r for r in recommendations if r.item_id not in user_items]
        
        # Apply diversity constraints
        recommendations = self._ensure_diversity(recommendations)
        
        # Apply content policies
        recommendations = self._apply_content_policies(recommendations, context)
        
        # Re-rank and assign final positions
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1
        
        return recommendations
    
    def _ensure_diversity(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Ensure diversity in recommendations"""
        
        if len(recommendations) <= 1:
            return recommendations
        
        # Group by category
        category_counts = defaultdict(int)
        diverse_recs = []
        
        # First pass: ensure no category dominates
        for rec in recommendations:
            item = self.items.get(rec.item_id)
            if item:
                category = item.category
                
                # Limit items per category
                if category_counts[category] < max(1, len(recommendations) // 3):
                    diverse_recs.append(rec)
                    category_counts[category] += 1
                
                if len(diverse_recs) >= len(recommendations):
                    break
        
        # Fill remaining slots if needed
        remaining = len(recommendations) - len(diverse_recs)
        if remaining > 0:
            used_items = set(r.item_id for r in diverse_recs)
            for rec in recommendations:
                if rec.item_id not in used_items:
                    diverse_recs.append(rec)
                    remaining -= 1
                    if remaining == 0:
                        break
        
        return diverse_recs
    
    def _apply_content_policies(self, recommendations: List[Recommendation],
                               context: Optional[Dict]) -> List[Recommendation]:
        """Apply content policies and filters"""
        
        # Example policies - in production would be more sophisticated
        filtered_recs = []
        
        for rec in recommendations:
            item = self.items.get(rec.item_id)
            if item:
                # Quality threshold
                if item.quality_score >= 0.5:
                    # Context-based filtering
                    if context and "time_of_day" in context:
                        # Different content for different times
                        if context["time_of_day"] == "morning":
                            # Prefer energetic content
                            if "energy" in item.features and item.features["energy"] > 0.7:
                                filtered_recs.append(rec)
                        else:
                            filtered_recs.append(rec)
                    else:
                        filtered_recs.append(rec)
        
        return filtered_recs

class CollaborativeFilter:
    """User-based collaborative filtering"""
    
    async def recommend(self, user_id: str, users: Dict[str, User],
                       interactions: List[Interaction], num_items: int) -> List[Recommendation]:
        """Generate collaborative filtering recommendations"""
        
        # Build user-item matrix
        user_item_matrix = self._build_user_item_matrix(interactions)
        
        # Find similar users
        similar_users = self._find_similar_users(user_id, user_item_matrix)
        
        # Generate recommendations based on similar users
        recommendations = []
        target_user_items = set()
        
        # Get items user has already interacted with
        for interaction in interactions:
            if interaction.user_id == user_id:
                target_user_items.add(interaction.item_id)
        
        # Score items based on similar users
        item_scores = defaultdict(float)
        
        for similar_user, similarity in similar_users[:50]:  # Top 50 similar users
            for interaction in interactions:
                if (interaction.user_id == similar_user and 
                    interaction.item_id not in target_user_items):
                    
                    # Weight by similarity and interaction strength
                    weight = similarity
                    if interaction.interaction_type == InteractionType.LIKE:
                        weight *= 1.5
                    elif interaction.interaction_type == InteractionType.SHARE:
                        weight *= 2.0
                    elif interaction.interaction_type == InteractionType.SKIP:
                        weight *= -0.5
                    
                    if interaction.rating:
                        weight *= (interaction.rating / 5.0)  # Normalize to 0-1
                    
                    item_scores[interaction.item_id] += weight
        
        # Convert to recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (item_id, score) in enumerate(sorted_items[:num_items]):
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                rank=i + 1,
                algorithm="collaborative_filtering",
                explanation=f"Users similar to you liked this (score: {score:.2f})"
            ))
        
        return recommendations
    
    def _build_user_item_matrix(self, interactions: List[Interaction]) -> Dict[str, Dict[str, float]]:
        """Build user-item interaction matrix"""
        
        matrix = defaultdict(lambda: defaultdict(float))
        
        for interaction in interactions:
            weight = 1.0
            
            # Weight different interaction types
            if interaction.interaction_type == InteractionType.LIKE:
                weight = 2.0
            elif interaction.interaction_type == InteractionType.SHARE:
                weight = 3.0
            elif interaction.interaction_type == InteractionType.DOWNLOAD:
                weight = 2.5
            elif interaction.interaction_type == InteractionType.SKIP:
                weight = -1.0
            
            if interaction.rating:
                weight *= (interaction.rating / 5.0)
            
            matrix[interaction.user_id][interaction.item_id] = weight
        
        return matrix
    
    def _find_similar_users(self, user_id: str, 
                           user_item_matrix: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Find users similar to target user"""
        
        if user_id not in user_item_matrix:
            return []
        
        target_vector = user_item_matrix[user_id]
        similarities = []
        
        for other_user, other_vector in user_item_matrix.items():
            if other_user != user_id:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(target_vector, other_vector)
                if similarity > 0.1:  # Minimum similarity threshold
                    similarities.append((other_user, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two sparse vectors"""
        
        # Get common items
        common_items = set(vec1.keys()) & set(vec2.keys())
        
        if not common_items:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1[item] * vec2[item] for item in common_items)
        
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

class ContentBasedFilter:
    """Content-based filtering using item features"""
    
    async def recommend(self, user: User, items: Dict[str, Item],
                       interactions: List[Interaction], num_items: int) -> List[Recommendation]:
        """Generate content-based recommendations"""
        
        # Build user profile from interaction history
        user_profile = self._build_user_profile(user, interactions, items)
        
        # Score all items against user profile
        item_scores = {}
        
        for item_id, item in items.items():
            # Skip items user has already interacted with
            if not any(i.item_id == item_id and i.user_id == user.user_id 
                      for i in interactions):
                score = self._calculate_content_similarity(user_profile, item)
                item_scores[item_id] = score
        
        # Sort and create recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for i, (item_id, score) in enumerate(sorted_items[:num_items]):
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                rank=i + 1,
                algorithm="content_based",
                explanation=f"Matches your preferences (similarity: {score:.2f})"
            ))
        
        return recommendations
    
    def _build_user_profile(self, user: User, interactions: List[Interaction],
                           items: Dict[str, Item]) -> Dict[str, float]:
        """Build user profile from interactions"""
        
        profile = defaultdict(float)
        total_weight = 0
        
        for interaction in interactions:
            if interaction.user_id == user.user_id and interaction.item_id in items:
                item = items[interaction.item_id]
                
                # Weight by interaction type
                weight = 1.0
                if interaction.interaction_type == InteractionType.LIKE:
                    weight = 2.0
                elif interaction.interaction_type == InteractionType.SHARE:
                    weight = 3.0
                elif interaction.interaction_type == InteractionType.SKIP:
                    weight = -0.5
                
                if interaction.rating:
                    weight *= (interaction.rating / 5.0)
                
                # Add item features to profile
                for feature, value in item.features.items():
                    if isinstance(value, (int, float)):
                        profile[feature] += value * weight
                
                # Add category preference
                profile[f"category_{item.category}"] += weight
                
                total_weight += abs(weight)
        
        # Normalize profile
        if total_weight > 0:
            for feature in profile:
                profile[feature] /= total_weight
        
        return profile
    
    def _calculate_content_similarity(self, user_profile: Dict[str, float], 
                                    item: Item) -> float:
        """Calculate similarity between user profile and item"""
        
        score = 0.0
        
        # Feature similarity
        for feature, value in item.features.items():
            if feature in user_profile and isinstance(value, (int, float)):
                score += user_profile[feature] * value
        
        # Category preference
        category_feature = f"category_{item.category}"
        if category_feature in user_profile:
            score += user_profile[category_feature] * 2.0  # Category boost
        
        # Quality and popularity boost
        score += item.quality_score * 0.1
        score += min(item.popularity_score, 1.0) * 0.05  # Cap popularity influence
        
        return max(0, score)  # No negative scores

class MatrixFactorization:
    """Matrix factorization using SVD"""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.fitted = False
    
    async def recommend(self, user_id: str, interactions: List[Interaction],
                       items: Dict[str, Item], num_items: int) -> List[Recommendation]:
        """Generate matrix factorization recommendations"""
        
        if not self.fitted:
            await self._fit_model(interactions)
        
        if user_id not in self.user_to_idx:
            # Cold start - return popular items
            popular_items = sorted(items.values(), 
                                 key=lambda x: x.popularity_score, reverse=True)
            
            recommendations = []
            for i, item in enumerate(popular_items[:num_items]):
                recommendations.append(Recommendation(
                    item_id=item.item_id,
                    score=item.popularity_score,
                    rank=i + 1,
                    algorithm="matrix_factorization_fallback",
                    explanation="Popular item (cold start)"
                ))
            
            return recommendations
        
        # Get user embedding
        user_idx = self.user_to_idx[user_id]
        user_embedding = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(self.item_factors, user_embedding)
        
        # Get user's interacted items
        user_items = set(i.item_id for i in interactions if i.user_id == user_id)
        
        # Create recommendations
        item_indices = list(range(len(scores)))
        item_indices.sort(key=lambda i: scores[i], reverse=True)
        
        recommendations = []
        count = 0
        
        for idx in item_indices:
            if count >= num_items:
                break
            
            # Get item_id from index
            item_id = next(
                (item_id for item_id, i in self.item_to_idx.items() if i == idx),
                None
            )
            
            if item_id and item_id not in user_items:
                recommendations.append(Recommendation(
                    item_id=item_id,
                    score=float(scores[idx]),
                    rank=count + 1,
                    algorithm="matrix_factorization",
                    explanation=f"Latent factor match (score: {scores[idx]:.2f})"
                ))
                count += 1
        
        return recommendations
    
    async def _fit_model(self, interactions: List[Interaction]):
        """Fit matrix factorization model"""
        
        # Create user and item mappings
        users = list(set(i.user_id for i in interactions))
        items = list(set(i.item_id for i in interactions))
        
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        # Build interaction matrix
        matrix = np.zeros((len(users), len(items)))
        
        for interaction in interactions:
            user_idx = self.user_to_idx[interaction.user_id]
            item_idx = self.item_to_idx[interaction.item_id]
            
            # Weight by interaction type
            weight = 1.0
            if interaction.interaction_type == InteractionType.LIKE:
                weight = 2.0
            elif interaction.interaction_type == InteractionType.SHARE:
                weight = 3.0
            elif interaction.interaction_type == InteractionType.SKIP:
                weight = -0.5
            
            if interaction.rating:
                weight *= (interaction.rating / 5.0)
            
            matrix[user_idx, item_idx] = weight
        
        # Fit SVD
        self.model.fit(matrix)
        
        # Get factor matrices
        self.user_factors = self.model.transform(matrix)
        self.item_factors = self.model.components_.T
        
        self.fitted = True

class PopularityRecommender:
    """Popularity-based recommendations"""
    
    async def recommend(self, items: Dict[str, Item], interactions: List[Interaction],
                       num_items: int, exclude_items: Set[str] = None) -> List[Recommendation]:
        """Recommend popular items"""
        
        exclude_items = exclude_items or set()
        
        # Calculate popularity scores
        item_popularity = defaultdict(float)
        
        for interaction in interactions:
            weight = 1.0
            if interaction.interaction_type == InteractionType.LIKE:
                weight = 2.0
            elif interaction.interaction_type == InteractionType.SHARE:
                weight = 3.0
            elif interaction.interaction_type == InteractionType.SKIP:
                weight = -0.5
            
            item_popularity[interaction.item_id] += weight
        
        # Sort by popularity
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        count = 0
        
        for item_id, popularity in sorted_items:
            if count >= num_items:
                break
            
            if item_id not in exclude_items and item_id in items:
                recommendations.append(Recommendation(
                    item_id=item_id,
                    score=popularity,
                    rank=count + 1,
                    algorithm="popularity",
                    explanation=f"Popular item (score: {popularity:.1f})"
                ))
                count += 1
        
        return recommendations

class EnsembleRecommender:
    """Ensemble of multiple recommendation algorithms"""
    
    async def recommend(self, user_id: str, user: User, users: Dict[str, User],
                       items: Dict[str, Item], interactions: List[Interaction],
                       num_items: int, context: Optional[Dict] = None) -> List[Recommendation]:
        """Generate ensemble recommendations"""
        
        # Get recommendations from different algorithms
        cf_recs = await CollaborativeFilter().recommend(
            user_id, users, interactions, num_items * 2
        )
        
        content_recs = await ContentBasedFilter().recommend(
            user, items, interactions, num_items * 2
        )
        
        mf_recs = await MatrixFactorization().recommend(
            user_id, interactions, items, num_items * 2
        )
        
        # Combine recommendations with weights
        combined_scores = defaultdict(float)
        algorithm_votes = defaultdict(set)
        
        # Weight each algorithm
        weights = {
            "collaborative_filtering": 0.4,
            "content_based": 0.3,
            "matrix_factorization": 0.3
        }
        
        for recs, algo in [(cf_recs, "collaborative_filtering"), 
                          (content_recs, "content_based"),
                          (mf_recs, "matrix_factorization")]:
            
            weight = weights[algo]
            
            for rec in recs:
                # Normalize score by rank (higher rank = lower weight)
                rank_weight = 1.0 / (1.0 + rec.rank * 0.1)
                combined_scores[rec.item_id] += rec.score * weight * rank_weight
                algorithm_votes[rec.item_id].add(algo)
        
        # Boost items recommended by multiple algorithms
        for item_id in combined_scores:
            if len(algorithm_votes[item_id]) > 1:
                combined_scores[item_id] *= 1.2
        
        # Create final recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        
        for i, (item_id, score) in enumerate(sorted_items[:num_items]):
            algorithms_used = list(algorithm_votes[item_id])
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                rank=i + 1,
                algorithm="ensemble",
                explanation=f"Recommended by {', '.join(algorithms_used)} (score: {score:.2f})"
            ))
        
        return recommendations

class ColdStartHandler:
    """Handle cold start scenarios"""
    
    async def recommend_for_new_user(self, user_id: str, num_items: int,
                                   items: Dict[str, Item]) -> List[Recommendation]:
        """Recommendations for completely new users"""
        
        # Return popular items
        popular_items = sorted(items.values(), 
                             key=lambda x: x.popularity_score, reverse=True)
        
        recommendations = []
        for i, item in enumerate(popular_items[:num_items]):
            recommendations.append(Recommendation(
                item_id=item.item_id,
                score=item.popularity_score,
                rank=i + 1,
                algorithm="cold_start_popular",
                explanation="Popular item for new users"
            ))
        
        return recommendations
    
    async def recommend_for_sparse_user(self, user: User, num_items: int,
                                      items: Dict[str, Item], 
                                      interactions: List[Interaction]) -> List[Recommendation]:
        """Recommendations for users with few interactions"""
        
        # Use content-based on limited interactions + popularity
        user_categories = set()
        
        for interaction in interactions:
            if interaction.item_id in items:
                user_categories.add(items[interaction.item_id].category)
        
        # Filter items by user's categories
        candidate_items = []
        for item in items.values():
            if not user_categories or item.category in user_categories:
                candidate_items.append(item)
        
        # Sort by quality and popularity
        candidate_items.sort(
            key=lambda x: x.quality_score * 0.7 + x.popularity_score * 0.3,
            reverse=True
        )
        
        recommendations = []
        for i, item in enumerate(candidate_items[:num_items]):
            recommendations.append(Recommendation(
                item_id=item.item_id,
                score=item.quality_score * 0.7 + item.popularity_score * 0.3,
                rank=i + 1,
                algorithm="cold_start_category",
                explanation=f"Popular in {item.category} category"
            ))
        
        return recommendations

class OnlineLearner:
    """Online learning for real-time updates"""
    
    async def update(self, interaction: Interaction):
        """Update models with new interaction"""
        # In production, would update model parameters in real-time
        # For now, just log the interaction
        pass

class FeatureEncoder:
    """Feature encoding for recommendations"""
    
    def encode_user_features(self, user: User) -> np.ndarray:
        """Encode user features as vector"""
        # Simple encoding - in production would be more sophisticated
        features = []
        
        # Demographics
        features.extend([
            hash(user.demographics.get("age_group", "unknown")) % 100 / 100,
            hash(user.demographics.get("location", "unknown")) % 100 / 100
        ])
        
        # Behavior patterns
        for pattern in ["morning_listener", "evening_listener", "weekend_user"]:
            features.append(user.behavior_patterns.get(pattern, 0))
        
        return np.array(features)

class RecommendationMetrics:
    """Track recommendation performance"""
    
    def __init__(self):
        self.interactions_tracked = 0
        self.recommendations_served = 0
        self.click_through_rates = []
        self.algorithm_performance = defaultdict(list)
    
    async def record_interaction(self, interaction: Interaction):
        """Record user interaction"""
        self.interactions_tracked += 1
    
    async def record_recommendations(self, user_id: str, 
                                   recommendations: List[Recommendation]):
        """Record recommendations served"""
        self.recommendations_served += len(recommendations)
        
        for rec in recommendations:
            self.algorithm_performance[rec.algorithm].append({
                "user_id": user_id,
                "item_id": rec.item_id,
                "score": rec.score,
                "rank": rec.rank,
                "timestamp": time.time()
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recommendation metrics"""
        return {
            "interactions_tracked": self.interactions_tracked,
            "recommendations_served": self.recommendations_served,
            "algorithms_used": list(self.algorithm_performance.keys()),
            "avg_recommendations_per_user": self.recommendations_served / max(1, len(set(
                perf["user_id"] for perfs in self.algorithm_performance.values() 
                for perf in perfs
            )))
        }

# Example usage
async def main():
    """Demonstrate recommendation system"""
    
    print("🎯 Production Recommendation System")
    print("=" * 60)
    
    # Initialize system
    engine = RecommendationEngine()
    
    # Add users
    users = [
        User("user_001", preferences={"rock": 0.8, "jazz": 0.3}, 
             demographics={"age_group": "25-34", "location": "US"}),
        User("user_002", preferences={"electronic": 0.9, "ambient": 0.6},
             demographics={"age_group": "18-24", "location": "UK"}),
        User("user_003", preferences={"classical": 0.7, "jazz": 0.8},
             demographics={"age_group": "35-44", "location": "CA"})
    ]
    
    for user in users:
        engine.add_user(user)
    
    # Add items
    items = [
        Item("track_001", "Rock Anthem", "rock", 
             features={"energy": 0.9, "tempo": 140}, 
             popularity_score=0.8, quality_score=0.9),
        Item("track_002", "Electronic Dreams", "electronic",
             features={"energy": 0.7, "tempo": 128},
             popularity_score=0.6, quality_score=0.8),
        Item("track_003", "Jazz Classics", "jazz",
             features={"energy": 0.4, "tempo": 90},
             popularity_score=0.7, quality_score=0.9),
        Item("track_004", "Ambient Space", "ambient",
             features={"energy": 0.2, "tempo": 60},
             popularity_score=0.5, quality_score=0.7),
        Item("track_005", "Classical Symphony", "classical",
             features={"energy": 0.6, "tempo": 100},
             popularity_score=0.9, quality_score=0.95)
    ]
    
    for item in items:
        engine.add_item(item)
    
    print(f"\n📊 System Initialized:")
    print(f"  Users: {len(engine.users)}")
    print(f"  Items: {len(engine.items)}")
    
    # Simulate interactions
    print("\n🔄 Simulating User Interactions:")
    interactions = [
        # User 001 likes rock and jazz
        Interaction("user_001", "track_001", InteractionType.LIKE, rating=5),
        Interaction("user_001", "track_003", InteractionType.LIKE, rating=4),
        Interaction("user_001", "track_002", InteractionType.SKIP),
        
        # User 002 likes electronic and ambient
        Interaction("user_002", "track_002", InteractionType.LIKE, rating=5),
        Interaction("user_002", "track_004", InteractionType.LIKE, rating=4),
        Interaction("user_002", "track_001", InteractionType.VIEW),
        
        # User 003 likes classical and jazz
        Interaction("user_003", "track_005", InteractionType.SHARE),
        Interaction("user_003", "track_003", InteractionType.LIKE, rating=5),
        Interaction("user_003", "track_001", InteractionType.VIEW)
    ]
    
    for interaction in interactions:
        await engine.record_interaction(interaction)
    
    print(f"  Recorded {len(interactions)} interactions")
    
    # Generate recommendations
    print("\n🎯 Generating Recommendations:")
    print("-" * 40)
    
    for user_id in ["user_001", "user_002", "user_003"]:
        recommendations = await engine.get_recommendations(user_id, num_items=3)
        
        print(f"\nRecommendations for {user_id}:")
        for rec in recommendations:
            item = engine.items[rec.item_id]
            print(f"  {rec.rank}. {item.title} ({item.category})")
            print(f"     Score: {rec.score:.2f} | {rec.explanation}")
    
    # Test different algorithms
    print("\n🧪 Algorithm Comparison:")
    print("-" * 40)
    
    algorithms = [
        RecommendationType.COLLABORATIVE,
        RecommendationType.CONTENT_BASED,
        RecommendationType.MATRIX_FACTORIZATION
    ]
    
    user_id = "user_001"
    for algo in algorithms:
        recs = await engine.get_recommendations(user_id, num_items=2, algorithm=algo)
        print(f"\n{algo.value}:")
        for rec in recs:
            item = engine.items[rec.item_id]
            print(f"  {item.title} (score: {rec.score:.2f})")
    
    # Get metrics
    metrics = engine.metrics.get_metrics()
    print(f"\n📈 System Metrics:")
    print("-" * 40)
    print(f"Interactions: {metrics['interactions_tracked']}")
    print(f"Recommendations: {metrics['recommendations_served']}")
    print(f"Algorithms: {metrics['algorithms_used']}")
    
    print("\n✅ Recommendation system operational!")
    
    return engine

if __name__ == "__main__":
    asyncio.run(main())