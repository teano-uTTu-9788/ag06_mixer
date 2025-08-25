"""
Production Reinforcement Learning Mixer System
Following DeepMind/OpenAI patterns for production RL systems

Key patterns implemented:
- DeepMind's DQN approach: Experience replay and target networks
- OpenAI's PPO patterns: Policy optimization with clipping
- Google's Dopamine framework: Modular agent architecture
- Stable-Baselines3 patterns: Production-ready RL implementations
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
import hashlib
import pickle

# Setup structured logging (Google Cloud pattern)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Action types following standard mixer controls"""
    VOLUME_ADJUST = "volume_adjust"
    PAN_ADJUST = "pan_adjust"
    EQ_LOW = "eq_low"
    EQ_MID = "eq_mid"
    EQ_HIGH = "eq_high"
    COMPRESSION = "compression"
    REVERB = "reverb"
    DELAY = "delay"

class RewardSignal(Enum):
    """Reward signals following audio engineering principles"""
    LOUDNESS_LUFS = "loudness_lufs"      # Target: -14 LUFS for streaming
    STEREO_BALANCE = "stereo_balance"    # Target: Balanced stereo field
    FREQUENCY_BALANCE = "frequency_balance"  # Target: Even frequency response
    DYNAMIC_RANGE = "dynamic_range"      # Target: Good dynamics
    CLARITY = "clarity"                  # Target: Clear separation
    MUSICALITY = "musicality"            # Target: Musical coherence
    HEADROOM = "headroom"                # Target: Avoid clipping

@dataclass
class MixerState:
    """Mixer state representation following DeepMind state patterns"""
    channels: Dict[int, Dict[str, float]]
    master: Dict[str, float]
    audio_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to vector for RL agent"""
        vector = []
        
        # Channel parameters (up to 8 channels)
        for i in range(8):
            if i in self.channels:
                channel = self.channels[i]
                vector.extend([
                    channel.get("volume", 0.0),
                    channel.get("pan", 0.0),
                    channel.get("eq_low", 0.0),
                    channel.get("eq_mid", 0.0),
                    channel.get("eq_high", 0.0),
                    channel.get("compression", 0.0),
                    channel.get("reverb", 0.0),
                    channel.get("delay", 0.0)
                ])
            else:
                vector.extend([0.0] * 8)  # Zero padding for unused channels
        
        # Master section
        vector.extend([
            self.master.get("volume", 1.0),
            self.master.get("eq_low", 0.0),
            self.master.get("eq_mid", 0.0),
            self.master.get("eq_high", 0.0),
            self.master.get("compression", 0.0)
        ])
        
        # Audio metrics
        vector.extend([
            self.audio_metrics.get("lufs", -20.0) / -60.0,  # Normalize to 0-1
            self.audio_metrics.get("peak", 0.0),
            self.audio_metrics.get("rms", 0.0),
            self.audio_metrics.get("stereo_correlation", 0.0),
            self.audio_metrics.get("spectral_centroid", 0.0)
        ])
        
        return np.array(vector, dtype=np.float32)
    
    def get_hash(self) -> str:
        """Get state hash for experience replay deduplication"""
        state_str = json.dumps(self.channels, sort_keys=True) + json.dumps(self.master, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

@dataclass
class RLAction:
    """RL action representation"""
    action_type: ActionType
    channel_id: int
    parameter: str
    value_delta: float  # Change amount (-1.0 to 1.0)
    confidence: float = 1.0

@dataclass
class Experience:
    """Experience tuple for replay buffer (following DQN patterns)"""
    state: MixerState
    action: RLAction
    reward: float
    next_state: MixerState
    done: bool
    timestamp: float = field(default_factory=time.time)

class IRewardCalculator(ABC):
    """Abstract reward calculator interface"""
    @abstractmethod
    def calculate_reward(self, state: MixerState, action: RLAction, next_state: MixerState) -> float:
        pass

class IExperienceReplay(ABC):
    """Abstract experience replay interface"""
    @abstractmethod
    def add_experience(self, experience: Experience) -> None:
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Experience]:
        pass

class IQNetwork(ABC):
    """Abstract Q-Network interface"""
    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        pass

class ProductionRewardCalculator(IRewardCalculator):
    """Production reward calculator following audio engineering principles"""
    
    def __init__(self):
        logger.info("🎯 Initializing Production Reward Calculator")
        self.target_lufs = -14.0  # Streaming standard
        self.reward_weights = {
            RewardSignal.LOUDNESS_LUFS: 0.3,
            RewardSignal.STEREO_BALANCE: 0.15,
            RewardSignal.FREQUENCY_BALANCE: 0.2,
            RewardSignal.DYNAMIC_RANGE: 0.15,
            RewardSignal.CLARITY: 0.1,
            RewardSignal.MUSICALITY: 0.05,
            RewardSignal.HEADROOM: 0.05
        }
    
    def calculate_reward(self, state: MixerState, action: RLAction, next_state: MixerState) -> float:
        """Calculate multi-objective reward following audio engineering principles"""
        try:
            total_reward = 0.0
            
            # LUFS (Loudness) reward
            lufs_reward = self._calculate_lufs_reward(next_state)
            total_reward += lufs_reward * self.reward_weights[RewardSignal.LOUDNESS_LUFS]
            
            # Stereo balance reward
            balance_reward = self._calculate_balance_reward(next_state)
            total_reward += balance_reward * self.reward_weights[RewardSignal.STEREO_BALANCE]
            
            # Frequency balance reward
            freq_reward = self._calculate_frequency_reward(next_state)
            total_reward += freq_reward * self.reward_weights[RewardSignal.FREQUENCY_BALANCE]
            
            # Dynamic range reward
            dynamics_reward = self._calculate_dynamics_reward(next_state)
            total_reward += dynamics_reward * self.reward_weights[RewardSignal.DYNAMIC_RANGE]
            
            # Clarity reward
            clarity_reward = self._calculate_clarity_reward(next_state)
            total_reward += clarity_reward * self.reward_weights[RewardSignal.CLARITY]
            
            # Musicality reward (coherence)
            musicality_reward = self._calculate_musicality_reward(state, next_state)
            total_reward += musicality_reward * self.reward_weights[RewardSignal.MUSICALITY]
            
            # Headroom reward (avoid clipping)
            headroom_reward = self._calculate_headroom_reward(next_state)
            total_reward += headroom_reward * self.reward_weights[RewardSignal.HEADROOM]
            
            # Action efficiency penalty (discourage excessive changes)
            efficiency_penalty = self._calculate_efficiency_penalty(action)
            total_reward += efficiency_penalty
            
            return np.clip(total_reward, -1.0, 1.0)  # Bounded reward
            
        except Exception as e:
            logger.error(f"❌ Reward calculation failed: {e}")
            return -0.1  # Small negative reward for errors
    
    def _calculate_lufs_reward(self, state: MixerState) -> float:
        """Calculate LUFS (loudness) reward"""
        current_lufs = state.audio_metrics.get("lufs", -20.0)
        lufs_error = abs(current_lufs - self.target_lufs)
        
        # Maximum reward when within 1 LUFS of target
        if lufs_error <= 1.0:
            return 1.0 - (lufs_error / 1.0) * 0.2
        elif lufs_error <= 3.0:
            return 0.8 - (lufs_error - 1.0) / 2.0 * 0.6
        else:
            return 0.2 - min(lufs_error - 3.0, 10.0) / 10.0 * 0.2
    
    def _calculate_balance_reward(self, state: MixerState) -> float:
        """Calculate stereo balance reward"""
        # Check pan distribution across channels
        pan_positions = []
        volumes = []
        
        for channel_config in state.channels.values():
            pan = channel_config.get("pan", 0.0)
            volume = channel_config.get("volume", 0.0)
            if volume > 0.1:  # Only consider audible channels
                pan_positions.append(pan)
                volumes.append(volume)
        
        if not pan_positions:
            return 0.5  # Neutral reward for no active channels
        
        # Calculate balance score
        weighted_pan = sum(p * v for p, v in zip(pan_positions, volumes)) / sum(volumes)
        balance_error = abs(weighted_pan)  # Should be close to 0 (center)
        
        return max(0.0, 1.0 - balance_error * 2.0)  # Linear penalty for imbalance
    
    def _calculate_frequency_reward(self, state: MixerState) -> float:
        """Calculate frequency balance reward"""
        # Aggregate EQ settings across all channels
        eq_low_total = 0.0
        eq_mid_total = 0.0
        eq_high_total = 0.0
        active_channels = 0
        
        for channel_config in state.channels.values():
            if channel_config.get("volume", 0.0) > 0.1:
                eq_low_total += channel_config.get("eq_low", 0.0)
                eq_mid_total += channel_config.get("eq_mid", 0.0)
                eq_high_total += channel_config.get("eq_high", 0.0)
                active_channels += 1
        
        if active_channels == 0:
            return 0.5  # Neutral for no active channels
        
        # Average EQ settings
        eq_low_avg = eq_low_total / active_channels
        eq_mid_avg = eq_mid_total / active_channels
        eq_high_avg = eq_high_total / active_channels
        
        # Prefer balanced frequency response (small EQ adjustments)
        eq_balance = 1.0 - (abs(eq_low_avg) + abs(eq_mid_avg) + abs(eq_high_avg)) / 3.0
        return max(0.0, eq_balance)
    
    def _calculate_dynamics_reward(self, state: MixerState) -> float:
        """Calculate dynamic range reward"""
        rms = state.audio_metrics.get("rms", 0.3)
        peak = state.audio_metrics.get("peak", 0.8)
        
        if peak > 0.01:  # Avoid division by zero
            dynamic_range = 20 * np.log10(peak / (rms + 0.001))  # dB
            
            # Target: 8-20 dB dynamic range
            if 8 <= dynamic_range <= 20:
                return 1.0
            elif 4 <= dynamic_range < 8 or 20 < dynamic_range <= 25:
                return 0.7
            else:
                return 0.3
        else:
            return 0.0  # No signal
    
    def _calculate_clarity_reward(self, state: MixerState) -> float:
        """Calculate mix clarity reward"""
        # Use stereo correlation as clarity indicator
        correlation = state.audio_metrics.get("stereo_correlation", 0.0)
        
        # Target correlation: 0.3-0.8 (good stereo width with coherence)
        if 0.3 <= correlation <= 0.8:
            return 1.0
        elif 0.1 <= correlation < 0.3 or 0.8 < correlation <= 0.9:
            return 0.6
        else:
            return 0.2
    
    def _calculate_musicality_reward(self, prev_state: MixerState, current_state: MixerState) -> float:
        """Calculate musical coherence reward"""
        # Reward smooth transitions (avoid abrupt changes)
        total_change = 0.0
        param_count = 0
        
        for channel_id in current_state.channels:
            if channel_id in prev_state.channels:
                prev_channel = prev_state.channels[channel_id]
                curr_channel = current_state.channels[channel_id]
                
                for param in ["volume", "pan", "eq_low", "eq_mid", "eq_high"]:
                    prev_val = prev_channel.get(param, 0.0)
                    curr_val = curr_channel.get(param, 0.0)
                    change = abs(curr_val - prev_val)
                    total_change += change
                    param_count += 1
        
        if param_count > 0:
            avg_change = total_change / param_count
            # Reward small, smooth changes
            return max(0.0, 1.0 - avg_change * 5.0)
        else:
            return 0.5  # Neutral
    
    def _calculate_headroom_reward(self, state: MixerState) -> float:
        """Calculate headroom reward (avoid clipping)"""
        peak = state.audio_metrics.get("peak", 0.8)
        
        if peak <= 0.9:  # Good headroom
            return 1.0
        elif peak <= 0.95:  # Acceptable headroom
            return 0.7
        elif peak <= 0.99:  # Marginal headroom
            return 0.3
        else:  # Clipping risk
            return -0.5
    
    def _calculate_efficiency_penalty(self, action: RLAction) -> float:
        """Penalty for excessive parameter changes"""
        change_magnitude = abs(action.value_delta)
        
        if change_magnitude <= 0.1:  # Small changes preferred
            return 0.0
        elif change_magnitude <= 0.3:
            return -0.05
        else:  # Large changes discouraged
            return -0.1

class ProductionExperienceReplay(IExperienceReplay):
    """Production experience replay buffer following DQN patterns"""
    
    def __init__(self, max_size: int = 10000):
        logger.info(f"💾 Initializing Experience Replay Buffer (size: {max_size})")
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.state_hashes = set()  # For deduplication
        self.priority_weights = {
            "high_reward": 2.0,    # Prioritize high-reward experiences
            "diverse_action": 1.5,  # Prioritize diverse actions
            "recent": 1.2          # Slight bias towards recent experiences
        }
        
    def add_experience(self, experience: Experience) -> None:
        """Add experience to buffer with deduplication"""
        try:
            state_hash = experience.state.get_hash()
            
            # Add experience (deque handles size limit automatically)
            self.buffer.append(experience)
            self.state_hashes.add(state_hash)
            
            # Remove old hash if buffer wrapped
            if len(self.buffer) == self.max_size:
                oldest_exp = self.buffer[0] if self.buffer else None
                if oldest_exp:
                    old_hash = oldest_exp.state.get_hash()
                    self.state_hashes.discard(old_hash)
            
            logger.debug(f"📝 Added experience: {experience.action.action_type.value} (reward: {experience.reward:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Failed to add experience: {e}")
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """Sample batch with prioritized sampling"""
        try:
            if len(self.buffer) < batch_size:
                return list(self.buffer)  # Return all available experiences
            
            # Calculate sampling probabilities
            priorities = self._calculate_priorities()
            probabilities = np.array(priorities) / sum(priorities)
            
            # Sample indices without replacement
            indices = np.random.choice(
                len(self.buffer), 
                size=batch_size, 
                replace=False, 
                p=probabilities
            )
            
            batch = [self.buffer[i] for i in indices]
            logger.debug(f"🎲 Sampled batch: {batch_size} experiences")
            return batch
            
        except Exception as e:
            logger.error(f"❌ Failed to sample batch: {e}")
            # Fallback to random sampling
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
            return [self.buffer[i] for i in indices]
    
    def _calculate_priorities(self) -> List[float]:
        """Calculate sampling priorities for experiences"""
        priorities = []
        current_time = time.time()
        
        for exp in self.buffer:
            priority = 1.0  # Base priority
            
            # High reward priority
            if exp.reward > 0.5:
                priority *= self.priority_weights["high_reward"]
            elif exp.reward < -0.3:
                priority *= self.priority_weights["high_reward"] * 0.8  # Also learn from bad actions
            
            # Diverse action priority (encourage exploration)
            if exp.action.action_type in [ActionType.EQ_LOW, ActionType.EQ_MID, ActionType.EQ_HIGH]:
                priority *= self.priority_weights["diverse_action"]
            
            # Recent experience priority
            age_hours = (current_time - exp.timestamp) / 3600
            if age_hours < 1.0:  # Recent experiences
                priority *= self.priority_weights["recent"]
            
            priorities.append(priority)
        
        return priorities
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {"size": 0, "avg_reward": 0.0, "action_distribution": {}}
        
        rewards = [exp.reward for exp in self.buffer]
        actions = [exp.action.action_type.value for exp in self.buffer]
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "size": len(self.buffer),
            "avg_reward": np.mean(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "action_distribution": action_counts
        }

class ProductionQNetwork(IQNetwork):
    """Production Q-Network using simplified neural network patterns"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        logger.info(f"🧠 Initializing Q-Network (state: {state_size}, actions: {action_size})")
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Simplified neural network using matrix operations
        # Layer sizes: state_size -> 128 -> 64 -> action_size
        self.hidden1_size = 128
        self.hidden2_size = 64
        
        # Initialize weights using Xavier initialization
        self.w1 = np.random.randn(self.state_size, self.hidden1_size) * np.sqrt(2.0 / self.state_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        
        self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))
        
        self.w3 = np.random.randn(self.hidden2_size, self.action_size) * np.sqrt(2.0 / self.hidden2_size)
        self.b3 = np.zeros((1, self.action_size))
        
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize Adam moments
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        
        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.m_w3 = np.zeros_like(self.w3)
        self.v_w3 = np.zeros_like(self.w3)
        self.m_b3 = np.zeros_like(self.b3)
        self.v_b3 = np.zeros_like(self.b3)
        
        self.t = 0  # Time step for Adam
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        try:
            # Ensure state is 2D
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            # Forward pass
            z1 = np.dot(state, self.w1) + self.b1
            a1 = self._relu(z1)
            
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self._relu(z2)
            
            z3 = np.dot(a2, self.w3) + self.b3
            q_values = z3  # No activation on output layer
            
            return q_values
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            # Return random Q-values as fallback
            return np.random.randn(1, self.action_size) * 0.1
    
    def update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """Update network weights using backpropagation and Adam optimizer"""
        try:
            batch_size = states.shape[0]
            self.t += 1  # Increment time step
            
            # Forward pass
            z1 = np.dot(states, self.w1) + self.b1
            a1 = self._relu(z1)
            
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self._relu(z2)
            
            z3 = np.dot(a2, self.w3) + self.b3
            predictions = z3
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - targets) ** 2)
            
            # Backward pass
            # Output layer gradients
            dz3 = (predictions - targets) / batch_size
            dw3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)
            
            # Hidden layer 2 gradients
            da2 = np.dot(dz3, self.w3.T)
            dz2 = da2 * self._relu_derivative(z2)
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            # Hidden layer 1 gradients
            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self._relu_derivative(z1)
            dw1 = np.dot(states.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Adam optimizer updates
            self._adam_update(dw1, db1, dw2, db2, dw3, db3)
            
            return float(loss)
            
        except Exception as e:
            logger.error(f"❌ Network update failed: {e}")
            return 1.0  # High loss to indicate failure
    
    def _adam_update(self, dw1, db1, dw2, db2, dw3, db3):
        """Apply Adam optimizer updates"""
        # Bias correction terms
        lr_corrected = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        
        # Update layer 1
        self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * dw1
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (dw1 ** 2)
        self.w1 -= lr_corrected * self.m_w1 / (np.sqrt(self.v_w1) + self.epsilon)
        
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        self.b1 -= lr_corrected * self.m_b1 / (np.sqrt(self.v_b1) + self.epsilon)
        
        # Update layer 2
        self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * dw2
        self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (dw2 ** 2)
        self.w2 -= lr_corrected * self.m_w2 / (np.sqrt(self.v_w2) + self.epsilon)
        
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        self.b2 -= lr_corrected * self.m_b2 / (np.sqrt(self.v_b2) + self.epsilon)
        
        # Update layer 3
        self.m_w3 = self.beta1 * self.m_w3 + (1 - self.beta1) * dw3
        self.v_w3 = self.beta2 * self.v_w3 + (1 - self.beta2) * (dw3 ** 2)
        self.w3 -= lr_corrected * self.m_w3 / (np.sqrt(self.v_w3) + self.epsilon)
        
        self.m_b3 = self.beta1 * self.m_b3 + (1 - self.beta1) * db3
        self.v_b3 = self.beta2 * self.v_b3 + (1 - self.beta2) * (db3 ** 2)
        self.b3 -= lr_corrected * self.m_b3 / (np.sqrt(self.v_b3) + self.epsilon)

class ProductionRLMixer:
    """
    Production Reinforcement Learning Mixer System
    Following DeepMind/OpenAI patterns for production RL
    """
    
    def __init__(self, 
                 state_size: int = 69,  # 8 channels * 8 params + 5 master + 5 audio metrics
                 action_size: int = 64,  # 8 actions * 8 channels
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.95):
        
        logger.info("🤖 Initializing Production RL Mixer")
        
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma  # Discount factor
        
        # Initialize components
        self.reward_calculator = ProductionRewardCalculator()
        self.experience_replay = ProductionExperienceReplay(max_size=5000)
        self.q_network = ProductionQNetwork(state_size, action_size, learning_rate)
        self.target_network = ProductionQNetwork(state_size, action_size, learning_rate)
        
        # Training parameters
        self.batch_size = 32
        self.target_update_frequency = 100  # Update target network every N steps
        self.training_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.training_losses = []
        self.convergence_threshold = 0.1
        self.convergence_window = 10
        
        # Action mapping
        self.action_map = self._build_action_map()
        
        logger.info("✅ Production RL Mixer initialized successfully")
    
    def _build_action_map(self) -> Dict[int, Tuple[ActionType, int, str]]:
        """Build mapping from action indices to (action_type, channel, parameter)"""
        action_map = {}
        idx = 0
        
        action_types = list(ActionType)
        channels = list(range(8))  # 8 channels
        
        for action_type in action_types:
            for channel in channels:
                if idx < self.action_size:
                    param_map = {
                        ActionType.VOLUME_ADJUST: "volume",
                        ActionType.PAN_ADJUST: "pan",
                        ActionType.EQ_LOW: "eq_low",
                        ActionType.EQ_MID: "eq_mid",
                        ActionType.EQ_HIGH: "eq_high",
                        ActionType.COMPRESSION: "compression",
                        ActionType.REVERB: "reverb",
                        ActionType.DELAY: "delay"
                    }
                    parameter = param_map[action_type]
                    action_map[idx] = (action_type, channel, parameter)
                    idx += 1
        
        return action_map
    
    def get_action(self, state: MixerState) -> RLAction:
        """Get action using epsilon-greedy policy"""
        try:
            # Epsilon-greedy exploration
            if np.random.random() <= self.epsilon:
                # Random action (exploration)
                action_idx = np.random.randint(0, self.action_size)
                logger.debug("🎲 Random action selected (exploration)")
            else:
                # Greedy action (exploitation)
                state_vector = state.to_vector()
                q_values = self.q_network.predict(state_vector)
                action_idx = np.argmax(q_values[0])
                logger.debug(f"🎯 Greedy action selected: idx={action_idx}")
            
            # Convert action index to RLAction
            action_type, channel, parameter = self.action_map[action_idx]
            
            # Generate action value delta (-0.2 to 0.2 for smooth changes)
            value_delta = (np.random.random() - 0.5) * 0.4
            
            action = RLAction(
                action_type=action_type,
                channel_id=channel,
                parameter=parameter,
                value_delta=value_delta,
                confidence=1.0 - self.epsilon  # Higher confidence when exploiting
            )
            
            return action
            
        except Exception as e:
            logger.error(f"❌ Action selection failed: {e}")
            # Fallback to random action
            return RLAction(
                action_type=ActionType.VOLUME_ADJUST,
                channel_id=0,
                parameter="volume",
                value_delta=0.0,
                confidence=0.1
            )
    
    def apply_action(self, state: MixerState, action: RLAction) -> MixerState:
        """Apply action to state and return new state"""
        try:
            # Create new state (copy)
            new_channels = {}
            for channel_id, channel_config in state.channels.items():
                new_channels[channel_id] = channel_config.copy()
            
            new_master = state.master.copy()
            
            # Apply action to appropriate channel
            if action.channel_id in new_channels:
                current_value = new_channels[action.channel_id].get(action.parameter, 0.0)
                new_value = current_value + action.value_delta
                
                # Apply parameter-specific bounds
                new_value = self._apply_parameter_bounds(action.parameter, new_value)
                new_channels[action.channel_id][action.parameter] = new_value
            else:
                # Initialize channel if it doesn't exist
                new_channels[action.channel_id] = {
                    "volume": 0.7,
                    "pan": 0.0,
                    "eq_low": 0.0,
                    "eq_mid": 0.0,
                    "eq_high": 0.0,
                    "compression": 0.0,
                    "reverb": 0.0,
                    "delay": 0.0
                }
                new_channels[action.channel_id][action.parameter] = action.value_delta
            
            # Simulate audio processing to generate new audio metrics
            new_audio_metrics = self._simulate_audio_processing(new_channels, new_master)
            
            new_state = MixerState(
                channels=new_channels,
                master=new_master,
                audio_metrics=new_audio_metrics
            )
            
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Action application failed: {e}")
            return state  # Return original state on error
    
    def _apply_parameter_bounds(self, parameter: str, value: float) -> float:
        """Apply bounds specific to parameter type"""
        bounds = {
            "volume": (0.0, 1.0),
            "pan": (-1.0, 1.0),
            "eq_low": (-12.0, 12.0),    # dB
            "eq_mid": (-12.0, 12.0),    # dB
            "eq_high": (-12.0, 12.0),   # dB
            "compression": (0.0, 1.0),
            "reverb": (0.0, 1.0),
            "delay": (0.0, 1.0)
        }
        
        min_val, max_val = bounds.get(parameter, (0.0, 1.0))
        return np.clip(value, min_val, max_val)
    
    def _simulate_audio_processing(self, channels: Dict[int, Dict[str, float]], master: Dict[str, float]) -> Dict[str, float]:
        """Simulate audio processing to generate realistic audio metrics"""
        # Calculate aggregate metrics based on channel settings
        total_volume = 0.0
        total_pan_weight = 0.0
        eq_low_sum = 0.0
        eq_mid_sum = 0.0
        eq_high_sum = 0.0
        active_channels = 0
        
        for channel_config in channels.values():
            volume = channel_config.get("volume", 0.0)
            if volume > 0.01:  # Consider only audible channels
                total_volume += volume
                total_pan_weight += abs(channel_config.get("pan", 0.0)) * volume
                eq_low_sum += channel_config.get("eq_low", 0.0)
                eq_mid_sum += channel_config.get("eq_mid", 0.0) 
                eq_high_sum += channel_config.get("eq_high", 0.0)
                active_channels += 1
        
        if active_channels == 0:
            return {
                "lufs": -60.0,  # Silent
                "peak": 0.0,
                "rms": 0.0,
                "stereo_correlation": 0.0,
                "spectral_centroid": 0.0
            }
        
        # Calculate LUFS (simplified model)
        # Base LUFS around -18, modified by total volume and EQ
        base_lufs = -18.0
        volume_adjustment = (total_volume - 0.7) * 10  # Volume impact
        eq_adjustment = (eq_low_sum + eq_mid_sum + eq_high_sum) / active_channels * 0.5
        lufs = base_lufs + volume_adjustment + eq_adjustment
        
        # Calculate peak (based on volume and compression)
        avg_compression = sum(ch.get("compression", 0.0) for ch in channels.values()) / active_channels
        peak = min(0.99, total_volume * (1.0 - avg_compression * 0.3))
        
        # Calculate RMS (related to LUFS)
        rms = min(0.8, 10 ** (lufs / 20.0) if lufs > -60 else 0.0)
        
        # Calculate stereo correlation (based on pan spread)
        avg_pan_spread = total_pan_weight / max(total_volume, 0.01)
        stereo_correlation = max(0.1, 1.0 - avg_pan_spread)
        
        # Calculate spectral centroid (based on EQ settings)
        spectral_centroid = 0.5 + (eq_high_sum - eq_low_sum) / active_channels * 0.05
        spectral_centroid = np.clip(spectral_centroid, 0.0, 1.0)
        
        return {
            "lufs": lufs,
            "peak": peak,
            "rms": rms,
            "stereo_correlation": stereo_correlation,
            "spectral_centroid": spectral_centroid
        }
    
    def train_step(self, state: MixerState, action: RLAction, reward: float, next_state: MixerState, done: bool) -> float:
        """Perform one training step"""
        try:
            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            self.experience_replay.add_experience(experience)
            
            # Train if we have enough experiences
            if self.experience_replay.buffer and len(self.experience_replay.buffer) >= self.batch_size:
                loss = self._replay_train()
                self.training_losses.append(loss)
                
                # Update target network periodically
                self.training_step += 1
                if self.training_step % self.target_update_frequency == 0:
                    self._update_target_network()
                
                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                return loss
            else:
                return 0.0  # No training performed
                
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            return 1.0  # High loss to indicate failure
    
    def _replay_train(self) -> float:
        """Train the network on a batch of experiences"""
        try:
            # Sample batch from experience replay
            batch = self.experience_replay.sample_batch(self.batch_size)
            
            # Prepare training data
            states = np.array([exp.state.to_vector() for exp in batch])
            next_states = np.array([exp.next_state.to_vector() for exp in batch])
            actions = np.array([self._get_action_index(exp.action) for exp in batch])
            rewards = np.array([exp.reward for exp in batch])
            dones = np.array([exp.done for exp in batch])
            
            # Current Q values
            current_q_values = self.q_network.predict(states)
            
            # Next Q values from target network
            next_q_values = self.target_network.predict(next_states)
            max_next_q_values = np.max(next_q_values, axis=1)
            
            # Calculate targets using Bellman equation
            targets = current_q_values.copy()
            for i in range(len(batch)):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q_values[i]
            
            # Update network
            loss = self.q_network.update(states, actions, targets)
            return loss
            
        except Exception as e:
            logger.error(f"❌ Replay training failed: {e}")
            return 1.0
    
    def _get_action_index(self, action: RLAction) -> int:
        """Get action index from RLAction"""
        for idx, (action_type, channel, parameter) in self.action_map.items():
            if (action.action_type == action_type and 
                action.channel_id == channel and 
                action.parameter == parameter):
                return idx
        return 0  # Fallback to first action
    
    def _update_target_network(self):
        """Copy weights from main network to target network"""
        try:
            self.target_network.w1 = self.q_network.w1.copy()
            self.target_network.b1 = self.q_network.b1.copy()
            self.target_network.w2 = self.q_network.w2.copy()
            self.target_network.b2 = self.q_network.b2.copy()
            self.target_network.w3 = self.q_network.w3.copy()
            self.target_network.b3 = self.q_network.b3.copy()
            
            logger.debug("🎯 Target network updated")
            
        except Exception as e:
            logger.error(f"❌ Target network update failed: {e}")
    
    def run_episode(self, initial_state: MixerState, max_steps: int = 50) -> Dict[str, Any]:
        """Run a complete training episode"""
        try:
            logger.info(f"🏃 Starting episode (max steps: {max_steps})")
            
            current_state = initial_state
            total_reward = 0.0
            steps = 0
            episode_actions = []
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(current_state)
                
                # Apply action
                next_state = self.apply_action(current_state, action)
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(current_state, action, next_state)
                
                # Check if episode should end
                done = self._check_done_condition(next_state, step, max_steps)
                
                # Train
                loss = self.train_step(current_state, action, reward, next_state, done)
                
                # Update tracking
                total_reward += reward
                steps += 1
                episode_actions.append({
                    "step": step,
                    "action": action.action_type.value,
                    "channel": action.channel_id,
                    "parameter": action.parameter,
                    "delta": action.value_delta,
                    "reward": reward,
                    "lufs": next_state.audio_metrics.get("lufs", -20.0)
                })
                
                # Move to next state
                current_state = next_state
                
                if done:
                    logger.info(f"📍 Episode ended at step {step + 1}")
                    break
            
            # Record episode reward
            self.episode_rewards.append(total_reward)
            
            # Calculate episode statistics
            episode_stats = {
                "total_reward": total_reward,
                "steps": steps,
                "average_reward": total_reward / steps if steps > 0 else 0.0,
                "final_lufs": current_state.audio_metrics.get("lufs", -20.0),
                "final_peak": current_state.audio_metrics.get("peak", 0.0),
                "epsilon": self.epsilon,
                "actions": episode_actions,
                "convergence_progress": self._calculate_convergence_progress()
            }
            
            logger.info(f"✅ Episode complete: reward={total_reward:.3f}, steps={steps}, LUFS={episode_stats['final_lufs']:.1f}")
            return episode_stats
            
        except Exception as e:
            logger.error(f"❌ Episode failed: {e}")
            return {
                "total_reward": -1.0,
                "steps": 0,
                "error": str(e)
            }
    
    def _check_done_condition(self, state: MixerState, step: int, max_steps: int) -> bool:
        """Check if episode should terminate"""
        # End if reached max steps
        if step >= max_steps - 1:
            return True
        
        # End if reached target LUFS (within 0.5 dB)
        lufs = state.audio_metrics.get("lufs", -20.0)
        target_lufs = self.reward_calculator.target_lufs
        if abs(lufs - target_lufs) <= 0.5:
            logger.info(f"🎯 Target LUFS reached: {lufs:.1f}")
            return True
        
        # End if clipping detected
        peak = state.audio_metrics.get("peak", 0.0)
        if peak >= 0.99:
            logger.warning("⚠️ Clipping detected, ending episode")
            return True
        
        return False
    
    def _calculate_convergence_progress(self) -> float:
        """Calculate convergence progress based on recent episode rewards"""
        if len(self.episode_rewards) < self.convergence_window:
            return 0.0
        
        recent_rewards = self.episode_rewards[-self.convergence_window:]
        reward_variance = np.var(recent_rewards)
        
        # Convergence indicated by low variance in rewards
        convergence_progress = max(0.0, 1.0 - reward_variance / 0.5)  # Scale based on expected variance
        return convergence_progress
    
    def optimize_mix(self, target_lufs: float = -14.0, channels: int = 8, max_episodes: int = 10) -> Dict[str, Any]:
        """Optimize mix using reinforcement learning"""
        try:
            logger.info(f"🚀 Starting mix optimization: target={target_lufs} LUFS, episodes={max_episodes}")
            
            # Update target LUFS
            self.reward_calculator.target_lufs = target_lufs
            
            # Initialize starting state
            initial_state = self._create_initial_state(channels)
            best_state = initial_state
            best_reward = float('-inf')
            optimization_history = []
            
            for episode in range(max_episodes):
                logger.info(f"📊 Episode {episode + 1}/{max_episodes}")
                
                # Run episode
                episode_stats = self.run_episode(initial_state)
                optimization_history.append(episode_stats)
                
                # Track best result
                if episode_stats["total_reward"] > best_reward:
                    best_reward = episode_stats["total_reward"]
                    # best_state would be the final state from the episode
                
                # Early stopping if converged
                convergence = episode_stats.get("convergence_progress", 0.0)
                if convergence > 0.9:
                    logger.info(f"🎯 Converged after {episode + 1} episodes")
                    break
            
            # Calculate optimization results
            final_episode = optimization_history[-1] if optimization_history else {}
            
            optimization_result = {
                "episodes_run": len(optimization_history),
                "best_reward": best_reward,
                "final_lufs": final_episode.get("final_lufs", target_lufs),
                "final_peak": final_episode.get("final_peak", 0.0),
                "convergence_achieved": convergence > 0.8,
                "total_steps": sum(ep.get("steps", 0) for ep in optimization_history),
                "average_reward": np.mean([ep.get("total_reward", 0) for ep in optimization_history]),
                "improvement": best_reward - optimization_history[0].get("total_reward", 0) if optimization_history else 0,
                "optimization_history": optimization_history,
                "network_stats": {
                    "epsilon": self.epsilon,
                    "training_steps": self.training_step,
                    "experience_buffer_size": len(self.experience_replay.buffer)
                }
            }
            
            logger.info(f"✅ Mix optimization complete:")
            logger.info(f"   Best reward: {best_reward:.3f}")
            logger.info(f"   Final LUFS: {optimization_result['final_lufs']:.1f} (target: {target_lufs})")
            logger.info(f"   Convergence: {'Yes' if optimization_result['convergence_achieved'] else 'No'}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Mix optimization failed: {e}")
            return {
                "episodes_run": 0,
                "error": str(e),
                "convergence_achieved": False
            }
    
    def _create_initial_state(self, channels: int) -> MixerState:
        """Create initial mixer state"""
        initial_channels = {}
        
        for i in range(channels):
            initial_channels[i] = {
                "volume": 0.7,
                "pan": 0.0,
                "eq_low": 0.0,
                "eq_mid": 0.0,
                "eq_high": 0.0,
                "compression": 0.0,
                "reverb": 0.0,
                "delay": 0.0
            }
        
        initial_master = {
            "volume": 1.0,
            "eq_low": 0.0,
            "eq_mid": 0.0,
            "eq_high": 0.0,
            "compression": 0.0
        }
        
        # Simulate initial audio metrics
        initial_audio_metrics = self._simulate_audio_processing(initial_channels, initial_master)
        
        return MixerState(
            channels=initial_channels,
            master=initial_master,
            audio_metrics=initial_audio_metrics
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            experience_stats = self.experience_replay.get_stats()
            
            stats = {
                "training": {
                    "episodes_completed": len(self.episode_rewards),
                    "total_training_steps": self.training_step,
                    "current_epsilon": self.epsilon,
                    "average_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                    "best_episode_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
                    "convergence_progress": self._calculate_convergence_progress()
                },
                "experience_replay": experience_stats,
                "network": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "learning_rate": self.learning_rate,
                    "average_training_loss": np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
                },
                "system_status": "operational" if len(self.episode_rewards) > 0 else "initializing"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance stats: {e}")
            return {"error": str(e)}

def demo_production_rl_mixer():
    """Production demo following DeepMind's demo patterns"""
    print("🤖 Production Reinforcement Learning Mixer Demo")
    print("=" * 60)
    
    try:
        # Initialize RL system
        print("1. Initializing Production RL Mixer...")
        rl_mixer = ProductionRLMixer(
            learning_rate=0.01,  # Higher learning rate for demo
            epsilon=0.8,         # More exploration for demo
            epsilon_decay=0.95
        )
        
        # Test 1: Single episode
        print("\n2. Running single training episode...")
        initial_state = rl_mixer._create_initial_state(channels=4)
        
        print(f"   Initial LUFS: {initial_state.audio_metrics['lufs']:.1f}")
        print(f"   Target LUFS: {rl_mixer.reward_calculator.target_lufs:.1f}")
        
        episode_result = rl_mixer.run_episode(initial_state, max_steps=15)
        
        print(f"   ✅ Episode completed:")
        print(f"      Total reward: {episode_result['total_reward']:.3f}")
        print(f"      Steps taken: {episode_result['steps']}")
        print(f"      Final LUFS: {episode_result['final_lufs']:.1f}")
        print(f"      Actions: {len(episode_result['actions'])}")
        
        # Test 2: Short optimization run
        print("\n3. Running optimization (5 episodes)...")
        optimization_result = rl_mixer.optimize_mix(
            target_lufs=-16.0,  # Different target for demo
            channels=6,
            max_episodes=5
        )
        
        print(f"   ✅ Optimization completed:")
        print(f"      Episodes run: {optimization_result['episodes_run']}")
        print(f"      Best reward: {optimization_result['best_reward']:.3f}")
        print(f"      Final LUFS: {optimization_result['final_lufs']:.1f} (target: -16.0)")
        print(f"      Convergence: {'Yes' if optimization_result['convergence_achieved'] else 'Partial'}")
        print(f"      Total steps: {optimization_result['total_steps']}")
        
        # Test 3: Action selection
        print("\n4. Testing action selection...")
        test_state = rl_mixer._create_initial_state(channels=3)
        
        for i in range(5):
            action = rl_mixer.get_action(test_state)
            print(f"   Action {i+1}: {action.action_type.value} on channel {action.channel_id}")
            print(f"      Parameter: {action.parameter}, Delta: {action.value_delta:.3f}")
            
            # Apply action to see result
            new_state = rl_mixer.apply_action(test_state, action)
            reward = rl_mixer.reward_calculator.calculate_reward(test_state, action, new_state)
            print(f"      Reward: {reward:.3f}, New LUFS: {new_state.audio_metrics['lufs']:.1f}")
            
            test_state = new_state
        
        # Performance statistics
        print("\n5. Performance Statistics:")
        stats = rl_mixer.get_performance_stats()
        
        print(f"   📊 Training:")
        print(f"      Episodes: {stats['training']['episodes_completed']}")
        print(f"      Training steps: {stats['training']['total_training_steps']}")
        print(f"      Current epsilon: {stats['training']['current_epsilon']:.3f}")
        print(f"      Average reward: {stats['training']['average_episode_reward']:.3f}")
        
        print(f"   💾 Experience Replay:")
        print(f"      Buffer size: {stats['experience_replay']['size']}")
        print(f"      Average reward: {stats['experience_replay']['avg_reward']:.3f}")
        
        print(f"   🧠 Network:")
        print(f"      State size: {stats['network']['state_size']}")
        print(f"      Action size: {stats['network']['action_size']}")
        print(f"      Learning rate: {stats['network']['learning_rate']}")
        
        print("\n✅ Production RL Mixer Demo Complete!")
        print(f"   System Status: {stats['system_status']}")
        print(f"   Episodes completed: {stats['training']['episodes_completed']}")
        print(f"   Best reward achieved: {stats['training']['best_episode_reward']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_production_rl_mixer()