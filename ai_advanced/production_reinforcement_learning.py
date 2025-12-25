"""
Production Reinforcement Learning Mixer System
===============================================

A comprehensive reinforcement learning system for automated audio mixing,
following industry-standard patterns from DeepMind, OpenAI, and Google Research.

Overview
--------
This module implements a Deep Q-Network (DQN) agent that learns to optimize
audio mixer settings to achieve professional-quality audio output. The agent
learns through trial-and-error interaction with a simulated mixing environment,
using experience replay and target networks for stable training.

Architecture
------------
The system follows a modular architecture with clear separation of concerns:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     ProductionRLMixer                               │
    │  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────────┐  │
    │  │ QNetwork     │  │ TargetNetwork  │  │ ExperienceReplay        │  │
    │  │ (Policy)     │  │ (Stable Q)     │  │ (Prioritized Sampling)  │  │
    │  └──────────────┘  └────────────────┘  └─────────────────────────┘  │
    │           │               │                        │                │
    │           └───────────────┴────────────────────────┘                │
    │                           │                                          │
    │                   ┌───────┴───────┐                                  │
    │                   │ RewardCalc    │                                  │
    │                   │ (Multi-obj)   │                                  │
    │                   └───────────────┘                                  │
    └─────────────────────────────────────────────────────────────────────┘

Key Algorithms
--------------
1. **Deep Q-Network (DQN)**: Uses a neural network to approximate the Q-function
   Q(s, a) which estimates expected future rewards for state-action pairs.

2. **Experience Replay**: Stores transitions (s, a, r, s') in a buffer and
   samples random mini-batches for training, breaking correlation between
   consecutive samples for stable learning.

3. **Target Network**: Maintains a separate network with frozen weights
   for computing TD targets, updated periodically to stabilize training.

4. **Epsilon-Greedy Exploration**: Balances exploration (random actions)
   with exploitation (greedy policy) using a decaying epsilon parameter.

5. **Multi-Objective Reward**: Combines multiple audio quality metrics
   (LUFS, stereo balance, frequency response, etc.) into a single scalar
   reward using weighted summation.

State Representation
--------------------
The mixer state is represented as a 69-dimensional feature vector:

    Dimensions 0-63:  Channel parameters (8 channels × 8 parameters each)
                      - volume (0-1): Channel fader position
                      - pan (-1 to 1): Stereo position
                      - eq_low (-12 to 12 dB): Low frequency EQ
                      - eq_mid (-12 to 12 dB): Mid frequency EQ
                      - eq_high (-12 to 12 dB): High frequency EQ
                      - compression (0-1): Compressor amount
                      - reverb (0-1): Reverb send level
                      - delay (0-1): Delay send level

    Dimensions 64-68: Master section (5 parameters)
                      - volume, eq_low, eq_mid, eq_high, compression

    Dimensions 69-73: Audio metrics (5 normalized values)
                      - lufs: Integrated loudness (normalized by -60)
                      - peak: Peak level (0-1)
                      - rms: RMS level (0-1)
                      - stereo_correlation: Phase correlation (-1 to 1)
                      - spectral_centroid: Brightness indicator (0-1)

Action Space
------------
The action space consists of 64 discrete actions:

    Action Index = action_type × 8 + channel_id

    Where:
    - action_type: One of 8 parameter adjustment types (ActionType enum)
    - channel_id: Target channel (0-7)

    Each action applies a small delta (-0.2 to +0.2) to the target parameter.

Reward Signal
-------------
The reward function evaluates mix quality across 7 dimensions:

    Component           Weight   Target
    ─────────────────────────────────────────
    LUFS (loudness)     30%      -14 LUFS (streaming standard)
    Frequency Balance   20%      Minimal EQ deviation
    Stereo Balance      15%      Centered weighted pan
    Dynamic Range       15%      8-20 dB peak-to-RMS ratio
    Clarity             10%      0.3-0.8 stereo correlation
    Musicality          5%       Smooth parameter transitions
    Headroom            5%       Peak < 0.9 (avoid clipping)

    Final reward is clipped to [-1, 1] range.

Hyperparameters
---------------
Key hyperparameters and their recommended ranges:

    Parameter           Default     Range           Description
    ─────────────────────────────────────────────────────────────────────
    learning_rate       0.001       0.0001-0.01     Adam optimizer LR
    epsilon             1.0         0.01-1.0        Initial exploration rate
    epsilon_decay       0.995       0.99-0.999      Per-step epsilon decay
    epsilon_min         0.01        0.01-0.1        Minimum exploration rate
    gamma               0.95        0.9-0.99        Discount factor
    batch_size          32          16-128          Training batch size
    buffer_size         5000        1000-100000     Replay buffer capacity
    target_update_freq  100         50-500          Steps between target updates

Usage Examples
--------------
Basic usage for mix optimization:

    >>> from production_reinforcement_learning import ProductionRLMixer
    >>>
    >>> # Initialize the RL mixer agent
    >>> rl_mixer = ProductionRLMixer(
    ...     learning_rate=0.001,
    ...     epsilon=0.8,          # Start with 80% exploration
    ...     epsilon_decay=0.995   # Decay to exploitation over time
    ... )
    >>>
    >>> # Optimize a mix to target LUFS
    >>> result = rl_mixer.optimize_mix(
    ...     target_lufs=-14.0,    # Streaming standard
    ...     channels=8,           # Number of mixer channels
    ...     max_episodes=50       # Training episodes
    ... )
    >>>
    >>> print(f"Final LUFS: {result['final_lufs']:.1f}")
    >>> print(f"Convergence: {result['convergence_achieved']}")

Single episode training:

    >>> # Create custom initial state
    >>> initial_state = rl_mixer._create_initial_state(channels=4)
    >>>
    >>> # Run one training episode
    >>> episode_stats = rl_mixer.run_episode(
    ...     initial_state,
    ...     max_steps=50
    ... )
    >>>
    >>> print(f"Episode reward: {episode_stats['total_reward']:.3f}")

Manual action selection and application:

    >>> state = rl_mixer._create_initial_state(channels=4)
    >>>
    >>> # Get action from policy (epsilon-greedy)
    >>> action = rl_mixer.get_action(state)
    >>> print(f"Action: {action.action_type.value} on channel {action.channel_id}")
    >>>
    >>> # Apply action to get next state
    >>> next_state = rl_mixer.apply_action(state, action)
    >>>
    >>> # Calculate reward
    >>> reward = rl_mixer.reward_calculator.calculate_reward(
    ...     state, action, next_state
    ... )

Industry Patterns Implemented
-----------------------------
- **DeepMind DQN**: Experience replay buffer, target networks, Q-learning
- **OpenAI Baselines**: Epsilon scheduling, action normalization
- **Google Dopamine**: Modular agent architecture, abstract interfaces
- **Stable-Baselines3**: Production-ready patterns, comprehensive logging

References
----------
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Schaul et al. (2016) "Prioritized Experience Replay"
- Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
- ITU-R BS.1770-4: Algorithms for loudness measurement (LUFS)
- AES Recommendations for Streaming Audio (-14 LUFS integrated)

Module Components
-----------------
Classes:
    ActionType: Enum of mixer parameter adjustment actions
    RewardSignal: Enum of audio quality reward components
    MixerState: Dataclass representing mixer configuration and audio metrics
    RLAction: Dataclass representing an RL agent action
    Experience: Dataclass for experience replay tuples
    IRewardCalculator: Abstract interface for reward calculation
    IExperienceReplay: Abstract interface for experience replay buffer
    IQNetwork: Abstract interface for Q-network implementations
    ProductionRewardCalculator: Multi-objective audio reward calculator
    ProductionExperienceReplay: Prioritized experience replay buffer
    ProductionQNetwork: Neural network Q-function approximator with Adam
    ProductionRLMixer: Main RL agent orchestrating all components

Functions:
    demo_production_rl_mixer(): Demonstration of system capabilities

See Also
--------
- ai_advanced.production_generative_ai: Generative AI mixing suggestions
- ai_advanced.production_nlp_system: Natural language mix commands
- ml.active_optimizer: Gradient-based mix optimization

Author: AG06 Mixer Team
License: Proprietary
Version: 1.0.0
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
    """
    Enumeration of mixer parameter adjustment action types.

    Each action type corresponds to a specific audio parameter that the RL
    agent can modify on individual mixer channels. These parameters represent
    standard professional audio mixing controls.

    Attributes
    ----------
    VOLUME_ADJUST : str
        Adjusts channel fader level (0.0 to 1.0 normalized range).
        Primary control for relative loudness of a channel in the mix.

    PAN_ADJUST : str
        Adjusts stereo position (-1.0 = full left, 0.0 = center, 1.0 = full right).
        Controls spatial placement of the channel in the stereo field.

    EQ_LOW : str
        Adjusts low-frequency EQ band (-12 to +12 dB).
        Typically affects frequencies below 250 Hz (bass, warmth).

    EQ_MID : str
        Adjusts mid-frequency EQ band (-12 to +12 dB).
        Typically affects frequencies 250 Hz - 4 kHz (presence, body).

    EQ_HIGH : str
        Adjusts high-frequency EQ band (-12 to +12 dB).
        Typically affects frequencies above 4 kHz (brilliance, air).

    COMPRESSION : str
        Adjusts compression amount (0.0 = none, 1.0 = maximum).
        Controls dynamic range reduction and sustain.

    REVERB : str
        Adjusts reverb send level (0.0 = dry, 1.0 = full wet).
        Controls spatial depth and room ambience.

    DELAY : str
        Adjusts delay send level (0.0 = no delay, 1.0 = full delay).
        Controls rhythmic echo effects.

    Example
    -------
    >>> action = ActionType.VOLUME_ADJUST
    >>> print(action.value)  # 'volume_adjust'
    """
    VOLUME_ADJUST = "volume_adjust"
    PAN_ADJUST = "pan_adjust"
    EQ_LOW = "eq_low"
    EQ_MID = "eq_mid"
    EQ_HIGH = "eq_high"
    COMPRESSION = "compression"
    REVERB = "reverb"
    DELAY = "delay"


class RewardSignal(Enum):
    """
    Enumeration of reward signal components for multi-objective optimization.

    Each reward signal represents an audio quality metric that contributes to
    the overall reward function. These are based on professional audio engineering
    standards and best practices for commercial audio production.

    The weighted combination of these signals guides the RL agent toward
    producing professional-quality mixes that meet streaming platform standards.

    Attributes
    ----------
    LOUDNESS_LUFS : str
        Integrated loudness measurement per ITU-R BS.1770-4 standard.
        Target: -14 LUFS for major streaming platforms (Spotify, Apple Music).
        Weight: 30% of total reward.

    STEREO_BALANCE : str
        Balance of audio energy between left and right channels.
        Target: Volume-weighted pan should average to center (0.0).
        Weight: 15% of total reward.

    FREQUENCY_BALANCE : str
        Distribution of energy across the frequency spectrum.
        Target: Minimal EQ deviation from flat response (surgical corrections only).
        Weight: 20% of total reward.

    DYNAMIC_RANGE : str
        Difference between peak and RMS levels in dB.
        Target: 8-20 dB dynamic range for engaging, punchy mixes.
        Weight: 15% of total reward.

    CLARITY : str
        Stereo correlation indicating phase coherence and separation.
        Target: 0.3-0.8 correlation (good width with mono compatibility).
        Weight: 10% of total reward.

    MUSICALITY : str
        Smoothness of parameter transitions over time.
        Target: Gradual, musical changes rather than abrupt jumps.
        Weight: 5% of total reward.

    HEADROOM : str
        Distance between peak levels and digital maximum (0 dBFS).
        Target: Peak below 0.9 (-0.9 dBFS) to avoid clipping.
        Weight: 5% of total reward.

    Notes
    -----
    The weights are tuned to prioritize loudness compliance (streaming standards)
    while maintaining mix quality. Weights can be adjusted for different use
    cases (e.g., broadcast has different LUFS targets).

    See Also
    --------
    ProductionRewardCalculator : Implementation of reward calculation using these signals.
    """
    LOUDNESS_LUFS = "loudness_lufs"      # Target: -14 LUFS for streaming
    STEREO_BALANCE = "stereo_balance"    # Target: Balanced stereo field
    FREQUENCY_BALANCE = "frequency_balance"  # Target: Even frequency response
    DYNAMIC_RANGE = "dynamic_range"      # Target: Good dynamics
    CLARITY = "clarity"                  # Target: Clear separation
    MUSICALITY = "musicality"            # Target: Musical coherence
    HEADROOM = "headroom"                # Target: Avoid clipping

@dataclass
class MixerState:
    """
    Complete mixer state representation for RL observation space.

    This dataclass encapsulates the full state of an audio mixer at a given
    moment, including all channel parameters, master section settings, and
    derived audio metrics. Following DeepMind's state representation patterns,
    this class provides methods to convert the state to a fixed-size vector
    suitable for neural network input.

    The state follows the Markov property - it contains all information
    necessary for the agent to make optimal decisions without needing
    historical context.

    Attributes
    ----------
    channels : Dict[int, Dict[str, float]]
        Dictionary mapping channel IDs (0-7) to their parameter dictionaries.
        Each channel dictionary contains:
            - 'volume' (float): Fader level, range [0.0, 1.0]
            - 'pan' (float): Stereo position, range [-1.0, 1.0]
            - 'eq_low' (float): Low EQ in dB, range [-12.0, 12.0]
            - 'eq_mid' (float): Mid EQ in dB, range [-12.0, 12.0]
            - 'eq_high' (float): High EQ in dB, range [-12.0, 12.0]
            - 'compression' (float): Compression amount, range [0.0, 1.0]
            - 'reverb' (float): Reverb send, range [0.0, 1.0]
            - 'delay' (float): Delay send, range [0.0, 1.0]

    master : Dict[str, float]
        Master section parameters dictionary containing:
            - 'volume' (float): Master fader, typically 1.0 (unity)
            - 'eq_low' (float): Master low EQ in dB
            - 'eq_mid' (float): Master mid EQ in dB
            - 'eq_high' (float): Master high EQ in dB
            - 'compression' (float): Master bus compression amount

    audio_metrics : Dict[str, float]
        Real-time audio analysis metrics dictionary containing:
            - 'lufs' (float): Integrated loudness, range [-60.0, 0.0]
            - 'peak' (float): Peak level, range [0.0, 1.0]
            - 'rms' (float): RMS level, range [0.0, 1.0]
            - 'stereo_correlation' (float): Phase correlation, range [-1.0, 1.0]
            - 'spectral_centroid' (float): Brightness indicator, range [0.0, 1.0]

    timestamp : float
        Unix timestamp when this state was captured. Default is current time.
        Used for experience replay age-based prioritization.

    Example
    -------
    >>> channels = {
    ...     0: {"volume": 0.8, "pan": -0.5, "eq_low": 2.0, "eq_mid": 0.0,
    ...         "eq_high": 1.0, "compression": 0.3, "reverb": 0.2, "delay": 0.0},
    ...     1: {"volume": 0.6, "pan": 0.5, "eq_low": -1.0, "eq_mid": 0.0,
    ...         "eq_high": 0.0, "compression": 0.0, "reverb": 0.1, "delay": 0.0}
    ... }
    >>> master = {"volume": 1.0, "eq_low": 0.0, "eq_mid": 0.0,
    ...           "eq_high": 0.0, "compression": 0.2}
    >>> metrics = {"lufs": -14.0, "peak": 0.85, "rms": 0.4,
    ...            "stereo_correlation": 0.6, "spectral_centroid": 0.5}
    >>> state = MixerState(channels=channels, master=master, audio_metrics=metrics)
    >>> vector = state.to_vector()
    >>> print(f"State vector shape: {vector.shape}")  # (69,) or (74,)

    See Also
    --------
    to_vector : Convert state to neural network input format.
    get_hash : Generate unique identifier for deduplication.
    """
    channels: Dict[int, Dict[str, float]]
    master: Dict[str, float]
    audio_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> np.ndarray:
        """
        Convert mixer state to a fixed-size feature vector for neural network input.

        This method serializes the hierarchical state dictionary into a flat
        numpy array suitable for the Q-network. The vector has a fixed size
        regardless of how many channels are active, using zero-padding for
        unused channels.

        Vector Layout
        -------------
        The 69-74 dimensional output vector is organized as follows:

            Indices 0-63 (64 values):
                8 channels × 8 parameters per channel
                Channel order: 0, 1, 2, 3, 4, 5, 6, 7
                Parameter order per channel:
                    [volume, pan, eq_low, eq_mid, eq_high, compression, reverb, delay]

            Indices 64-68 (5 values):
                Master section parameters:
                    [volume, eq_low, eq_mid, eq_high, compression]

            Indices 69-73 (5 values):
                Audio metrics (normalized):
                    [lufs/(-60), peak, rms, stereo_correlation, spectral_centroid]

        Normalization
        -------------
        Most values are already in [0, 1] or [-1, 1] range. Special handling:
            - LUFS is normalized by dividing by -60 to get [0, ~0.3] range
            - EQ values (-12 to +12 dB) are NOT normalized (network learns scale)
            - Pan (-1 to +1) is kept as-is for semantic meaning

        Returns
        -------
        np.ndarray
            Float32 array of shape (69,) to (74,) depending on metrics.
            Fixed-size representation of the mixer state.

        Notes
        -----
        - Unused channels are zero-padded to maintain fixed vector size
        - Missing parameters default to sensible values (0.0 for most, 1.0 for master volume)
        - The order of parameters is critical for network weight interpretation

        Example
        -------
        >>> state = MixerState(
        ...     channels={0: {"volume": 0.8, "pan": 0.0}},
        ...     master={"volume": 1.0},
        ...     audio_metrics={"lufs": -14.0, "peak": 0.9}
        ... )
        >>> vec = state.to_vector()
        >>> print(f"Channel 0 volume: {vec[0]}")  # 0.8
        >>> print(f"Master volume: {vec[64]}")    # 1.0
        """
        vector = []

        # Channel parameters (up to 8 channels)
        # Each channel contributes 8 values in fixed order
        for i in range(8):
            if i in self.channels:
                channel = self.channels[i]
                vector.extend([
                    channel.get("volume", 0.0),       # Index: i*8 + 0
                    channel.get("pan", 0.0),          # Index: i*8 + 1
                    channel.get("eq_low", 0.0),       # Index: i*8 + 2
                    channel.get("eq_mid", 0.0),       # Index: i*8 + 3
                    channel.get("eq_high", 0.0),      # Index: i*8 + 4
                    channel.get("compression", 0.0),  # Index: i*8 + 5
                    channel.get("reverb", 0.0),       # Index: i*8 + 6
                    channel.get("delay", 0.0)         # Index: i*8 + 7
                ])
            else:
                vector.extend([0.0] * 8)  # Zero padding for unused channels

        # Master section (indices 64-68)
        vector.extend([
            self.master.get("volume", 1.0),      # Index: 64
            self.master.get("eq_low", 0.0),      # Index: 65
            self.master.get("eq_mid", 0.0),      # Index: 66
            self.master.get("eq_high", 0.0),     # Index: 67
            self.master.get("compression", 0.0)  # Index: 68
        ])

        # Audio metrics (indices 69-73, normalized)
        vector.extend([
            self.audio_metrics.get("lufs", -20.0) / -60.0,  # Normalize LUFS to ~[0, 0.33]
            self.audio_metrics.get("peak", 0.0),             # Already [0, 1]
            self.audio_metrics.get("rms", 0.0),              # Already [0, 1]
            self.audio_metrics.get("stereo_correlation", 0.0),  # [-1, 1]
            self.audio_metrics.get("spectral_centroid", 0.0)    # [0, 1]
        ])

        return np.array(vector, dtype=np.float32)

    def get_hash(self) -> str:
        """
        Generate a short hash of the state for deduplication and caching.

        Creates an 8-character MD5 hash based on channel and master parameters.
        Audio metrics are excluded since they are derived from the parameters.
        This hash is used by the experience replay buffer to identify and
        optionally deduplicate similar states.

        Returns
        -------
        str
            8-character hexadecimal hash string uniquely identifying this state
            configuration (ignoring audio metrics and timestamp).

        Notes
        -----
        - Hash is deterministic for identical channel/master configurations
        - Uses sorted JSON serialization for consistent ordering
        - Truncated to 8 characters for memory efficiency (still ~4 billion unique values)
        - Collisions are extremely rare for practical buffer sizes

        Example
        -------
        >>> state1 = MixerState(channels={0: {"volume": 0.5}}, master={}, audio_metrics={})
        >>> state2 = MixerState(channels={0: {"volume": 0.5}}, master={}, audio_metrics={})
        >>> assert state1.get_hash() == state2.get_hash()  # Same config = same hash
        """
        state_str = json.dumps(self.channels, sort_keys=True) + json.dumps(self.master, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

@dataclass
class RLAction:
    """
    Represents a discrete action taken by the RL agent.

    Each action modifies a single parameter on a single channel by a small
    delta value. The action space is designed to encourage smooth, musical
    parameter changes rather than abrupt jumps.

    Attributes
    ----------
    action_type : ActionType
        The type of parameter adjustment (e.g., VOLUME_ADJUST, EQ_LOW).
        Determines which parameter on the target channel will be modified.

    channel_id : int
        Target channel index (0-7) where the action is applied.
        Must be a valid channel in the mixer state.

    parameter : str
        The specific parameter name to modify (e.g., 'volume', 'eq_low').
        Derived from action_type for consistency.

    value_delta : float
        The change amount to apply, typically in range [-0.2, 0.2].
        Small deltas encourage gradual, musical changes.
        Actual range depends on parameter type:
            - volume, pan: [-0.2, 0.2] typical
            - EQ: [-2.0, 2.0] dB typical
            - effects: [-0.1, 0.1] typical

    confidence : float, default=1.0
        Agent's confidence in this action, range [0, 1].
        Higher values indicate exploitation (greedy policy).
        Lower values indicate exploration (random policy).
        Used for logging and analysis, not training.

    Example
    -------
    >>> action = RLAction(
    ...     action_type=ActionType.VOLUME_ADJUST,
    ...     channel_id=0,
    ...     parameter="volume",
    ...     value_delta=0.05,
    ...     confidence=0.85
    ... )
    >>> print(f"Increase channel {action.channel_id} volume by {action.value_delta}")
    """
    action_type: ActionType
    channel_id: int
    parameter: str
    value_delta: float  # Change amount (-1.0 to 1.0)
    confidence: float = 1.0


@dataclass
class Experience:
    """
    Experience tuple for the replay buffer following DQN patterns.

    Stores a single transition (s, a, r, s', done) from the environment.
    These tuples are collected during interaction and sampled for training,
    breaking the correlation between consecutive samples.

    This follows the standard RL experience tuple format from the original
    DQN paper (Mnih et al., 2015), extended with a timestamp for
    prioritization strategies.

    Attributes
    ----------
    state : MixerState
        The state before the action was taken (s).
        Complete mixer configuration at the decision point.

    action : RLAction
        The action taken in this state (a).
        Describes the parameter modification applied.

    reward : float
        The reward received after taking the action (r).
        Scalar value in range [-1, 1] representing mix quality improvement.
        Positive rewards indicate progress toward target; negative indicate regression.

    next_state : MixerState
        The resulting state after the action (s').
        Complete mixer configuration after applying the action.

    done : bool
        Whether this transition ended the episode.
        True if: target LUFS reached, clipping detected, or max steps hit.

    timestamp : float
        Unix timestamp when this experience was recorded.
        Used for age-based prioritization in sampling.
        Recent experiences may be weighted more heavily.

    Notes
    -----
    The experience replay buffer breaks temporal correlations by randomly
    sampling from a large pool of experiences. This is crucial for stable
    training as consecutive experiences are highly correlated.

    Example
    -------
    >>> exp = Experience(
    ...     state=current_state,
    ...     action=action_taken,
    ...     reward=0.15,
    ...     next_state=resulting_state,
    ...     done=False
    ... )
    >>> replay_buffer.add_experience(exp)

    See Also
    --------
    ProductionExperienceReplay : Buffer implementation with prioritized sampling.
    """
    state: MixerState
    action: RLAction
    reward: float
    next_state: MixerState
    done: bool
    timestamp: float = field(default_factory=time.time)


class IRewardCalculator(ABC):
    """
    Abstract interface for reward calculation strategies.

    Defines the contract for computing scalar rewards from state transitions.
    Implementations can use different audio quality metrics and weighting
    strategies while maintaining a consistent interface.

    This follows the Strategy pattern, allowing runtime swapping of reward
    functions for experimentation or domain adaptation.

    Methods
    -------
    calculate_reward(state, action, next_state) -> float
        Compute the reward for a state transition.

    See Also
    --------
    ProductionRewardCalculator : Multi-objective audio quality reward implementation.
    """

    @abstractmethod
    def calculate_reward(self, state: MixerState, action: RLAction, next_state: MixerState) -> float:
        """
        Calculate reward for a state-action-next_state transition.

        Parameters
        ----------
        state : MixerState
            The state before the action was applied.

        action : RLAction
            The action that was taken.

        next_state : MixerState
            The resulting state after the action.

        Returns
        -------
        float
            Scalar reward value, typically in range [-1, 1].
            Positive values indicate improvement toward the goal.
        """
        pass


class IExperienceReplay(ABC):
    """
    Abstract interface for experience replay buffer implementations.

    Defines the contract for storing and sampling experiences for training.
    Implementations can use different sampling strategies (uniform, prioritized)
    while maintaining a consistent interface.

    The experience replay mechanism is crucial for stable DQN training,
    as it decorrelates consecutive samples and enables efficient reuse
    of past experiences.

    Methods
    -------
    add_experience(experience) -> None
        Store a new experience in the buffer.

    sample_batch(batch_size) -> List[Experience]
        Sample a batch of experiences for training.

    See Also
    --------
    ProductionExperienceReplay : Implementation with prioritized sampling.
    """

    @abstractmethod
    def add_experience(self, experience: Experience) -> None:
        """
        Add a new experience to the replay buffer.

        Parameters
        ----------
        experience : Experience
            The experience tuple to store.

        Notes
        -----
        Implementations should handle buffer overflow (e.g., FIFO eviction).
        """
        pass

    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences for training.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        List[Experience]
            Batch of sampled experiences. May be smaller than batch_size
            if the buffer doesn't contain enough experiences.
        """
        pass


class IQNetwork(ABC):
    """
    Abstract interface for Q-network implementations.

    Defines the contract for neural networks that approximate the Q-function
    Q(s, a) estimating expected future rewards. Implementations can use
    different architectures (MLP, CNN, Transformer) while maintaining
    a consistent interface.

    The Q-network is the core learnable component in DQN, mapping states
    to action values through supervised learning on Bellman targets.

    Methods
    -------
    predict(state) -> np.ndarray
        Forward pass to get Q-values for all actions.

    update(states, actions, targets) -> float
        Backward pass to update network weights.

    See Also
    --------
    ProductionQNetwork : MLP implementation with Adam optimizer.
    """

    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions given a state.

        Parameters
        ----------
        state : np.ndarray
            State vector of shape (state_size,) or (batch_size, state_size).

        Returns
        -------
        np.ndarray
            Q-values of shape (1, action_size) or (batch_size, action_size).
            Higher values indicate more promising actions.
        """
        pass

    @abstractmethod
    def update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """
        Update network weights using backpropagation.

        Parameters
        ----------
        states : np.ndarray
            Batch of state vectors, shape (batch_size, state_size).

        actions : np.ndarray
            Batch of action indices, shape (batch_size,).

        targets : np.ndarray
            Target Q-values from Bellman equation, shape (batch_size, action_size).

        Returns
        -------
        float
            Training loss (MSE between predictions and targets).
        """
        pass

class ProductionRewardCalculator(IRewardCalculator):
    """
    Multi-objective reward calculator for professional audio mixing.

    This class implements a weighted-sum reward function that evaluates
    audio mix quality across 7 dimensions based on professional audio
    engineering standards and streaming platform requirements.

    The reward function is designed to guide the RL agent toward mixes that:
    1. Meet loudness standards (LUFS compliance)
    2. Have balanced stereo imaging
    3. Maintain even frequency distribution
    4. Preserve appropriate dynamic range
    5. Achieve good stereo clarity
    6. Sound musical (smooth transitions)
    7. Avoid clipping (adequate headroom)

    Algorithm Overview
    ------------------
    The total reward is computed as a weighted sum of individual components:

        R_total = Σ (w_i × R_i) + efficiency_penalty

    Where:
        - R_i is the reward for component i (normalized to [0, 1])
        - w_i is the weight for component i (weights sum to 1.0)
        - efficiency_penalty is a small negative term for large parameter changes

    The final reward is clipped to [-1, 1] to maintain training stability.

    Reward Components
    -----------------
    1. **LUFS Reward (30%)**:
       Measures deviation from target integrated loudness.
       - Perfect (reward=1.0): Within 1 dB of target
       - Good (reward=0.2-0.8): Within 3 dB of target
       - Poor (reward<0.2): More than 3 dB deviation

    2. **Stereo Balance (15%)**:
       Measures volume-weighted center of panning.
       - Perfect (reward=1.0): Weighted pan at center
       - Degraded linearly with imbalance

    3. **Frequency Balance (20%)**:
       Measures deviation from flat EQ response.
       - Perfect (reward=1.0): No EQ adjustments
       - Degraded by average absolute EQ deviation

    4. **Dynamic Range (15%)**:
       Measures peak-to-RMS ratio in dB.
       - Perfect (reward=1.0): 8-20 dB range
       - Good (reward=0.7): 4-8 or 20-25 dB range
       - Poor (reward=0.3): Outside optimal range

    5. **Clarity (10%)**:
       Uses stereo correlation as proxy for clarity.
       - Perfect (reward=1.0): Correlation 0.3-0.8
       - Degraded for too narrow or too wide stereo

    6. **Musicality (5%)**:
       Penalizes abrupt parameter changes.
       - Perfect (reward=1.0): No parameter change
       - Degraded by average parameter delta

    7. **Headroom (5%)**:
       Penalizes peak levels approaching clipping.
       - Perfect (reward=1.0): Peak ≤ 0.9
       - Warning (reward=0.3-0.7): Peak 0.9-0.99
       - Penalty (reward=-0.5): Peak ≥ 0.99 (clipping)

    Attributes
    ----------
    target_lufs : float
        Target integrated loudness in LUFS. Default is -14.0 for streaming.
        Can be adjusted for different platforms:
            - Spotify/Apple Music: -14 LUFS
            - YouTube: -13 to -15 LUFS
            - Broadcast (EBU R128): -23 LUFS
            - CD/Master: -9 to -12 LUFS

    reward_weights : Dict[RewardSignal, float]
        Weights for each reward component. Must sum to 1.0.
        Default weights prioritize loudness compliance and frequency balance.

    Example
    -------
    >>> calculator = ProductionRewardCalculator()
    >>> calculator.target_lufs = -16.0  # YouTube target
    >>>
    >>> reward = calculator.calculate_reward(prev_state, action, curr_state)
    >>> print(f"Reward: {reward:.3f}")  # e.g., 0.723

    Notes
    -----
    The reward weights were tuned empirically based on:
    - Professional mastering engineer feedback
    - Streaming platform loudness requirements
    - Perceptual importance of each quality dimension

    See Also
    --------
    RewardSignal : Enum defining all reward components.
    """

    def __init__(self):
        """
        Initialize the reward calculator with default streaming targets.

        Sets up the target LUFS for major streaming platforms (-14 LUFS)
        and configures the default weight distribution across quality metrics.
        """
        logger.info("🎯 Initializing Production Reward Calculator")
        self.target_lufs = -14.0  # Streaming standard (Spotify, Apple Music)

        # Weight distribution prioritizes loudness and frequency response
        # Weights must sum to 1.0 for normalized reward output
        self.reward_weights = {
            RewardSignal.LOUDNESS_LUFS: 0.3,       # Primary: streaming compliance
            RewardSignal.STEREO_BALANCE: 0.15,    # Secondary: imaging
            RewardSignal.FREQUENCY_BALANCE: 0.2,  # Primary: tonal quality
            RewardSignal.DYNAMIC_RANGE: 0.15,     # Secondary: punch/energy
            RewardSignal.CLARITY: 0.1,            # Tertiary: separation
            RewardSignal.MUSICALITY: 0.05,        # Tertiary: smoothness
            RewardSignal.HEADROOM: 0.05           # Safety: clipping prevention
        }
    
    def calculate_reward(self, state: MixerState, action: RLAction, next_state: MixerState) -> float:
        """
        Calculate the total reward for a state transition.

        Combines 7 audio quality metrics into a single scalar reward using
        weighted summation, plus an efficiency penalty for large parameter changes.

        Parameters
        ----------
        state : MixerState
            The mixer state before the action was applied.

        action : RLAction
            The action that was taken (used for efficiency penalty calculation).

        next_state : MixerState
            The resulting mixer state after the action.

        Returns
        -------
        float
            Total reward in range [-1, 1]. Positive values indicate improvement
            toward professional mix quality; negative indicates degradation.

        Notes
        -----
        The reward computation is wrapped in a try-except to ensure training
        stability. On error, returns -0.1 (small penalty) rather than crashing.

        Example
        -------
        >>> reward = calculator.calculate_reward(prev_state, action, curr_state)
        >>> if reward > 0.5:
        ...     print("Good action - mix improved significantly!")
        """
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
    """
    Prioritized experience replay buffer for stable DQN training.

    This class implements an experience replay buffer with priority-based
    sampling, following the principles from Schaul et al. (2016) "Prioritized
    Experience Replay" but with simplified priority computation suitable for
    production use.

    The buffer stores state transitions and allows random sampling to break
    the correlation between consecutive training samples, which is essential
    for stable Q-learning convergence.

    Prioritized Sampling Algorithm
    ------------------------------
    Instead of uniform random sampling, experiences are sampled with
    probabilities proportional to their computed priority:

        P(i) = priority(i) / Σ priority(j)

    Priority is computed based on three factors:

    1. **Reward Magnitude** (weight: 2.0x):
       Experiences with high rewards (>0.5) or significant negative rewards
       (<-0.3) are more likely to be sampled. This focuses training on
       the most informative transitions.

    2. **Action Diversity** (weight: 1.5x):
       EQ-related actions are prioritized to ensure the agent explores
       frequency adjustments, which are often underrepresented compared
       to volume changes.

    3. **Recency** (weight: 1.2x):
       Experiences less than 1 hour old receive a slight boost, helping
       the agent adapt to recent policy changes faster.

    Memory Management
    -----------------
    The buffer uses a deque with fixed maximum size, providing O(1) FIFO
    eviction when full. State hashes are tracked for optional deduplication.

    Attributes
    ----------
    max_size : int
        Maximum number of experiences to store. Oldest experiences are
        evicted when this limit is reached. Default is 10,000.

    buffer : deque
        The underlying storage for Experience tuples.

    state_hashes : set
        Set of state hashes for fast deduplication lookup.

    priority_weights : Dict[str, float]
        Multipliers for each priority factor:
            - "high_reward": 2.0 (double weight for impactful experiences)
            - "diverse_action": 1.5 (boost for EQ actions)
            - "recent": 1.2 (slight boost for recent experiences)

    Example
    -------
    >>> replay = ProductionExperienceReplay(max_size=5000)
    >>>
    >>> # Add experiences during interaction
    >>> for experience in collected_experiences:
    ...     replay.add_experience(experience)
    >>>
    >>> # Sample for training
    >>> batch = replay.sample_batch(batch_size=32)
    >>> print(f"Sampled {len(batch)} experiences")

    Performance Characteristics
    ---------------------------
    - add_experience: O(1) amortized
    - sample_batch: O(n) where n is buffer size (priority calculation)
    - Memory: O(max_size × experience_size)

    For very large buffers (>100k), consider segment-tree based prioritization
    from the original PER paper for O(log n) sampling.

    References
    ----------
    - Schaul et al. (2016) "Prioritized Experience Replay"
    - Mnih et al. (2015) "Human-level control through deep RL" (original DQN)

    See Also
    --------
    Experience : The tuple type stored in this buffer.
    IExperienceReplay : The interface this class implements.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the experience replay buffer.

        Parameters
        ----------
        max_size : int, default=10000
            Maximum capacity of the buffer. When full, oldest experiences
            are evicted in FIFO order. Recommended range: 1000-100000.
            - Smaller buffers: Faster adaptation, higher variance
            - Larger buffers: More stable training, slower adaptation
        """
        logger.info(f"💾 Initializing Experience Replay Buffer (size: {max_size})")
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.state_hashes = set()  # For deduplication

        # Priority multipliers for sampling strategy
        # These values were tuned empirically for audio mixing tasks
        self.priority_weights = {
            "high_reward": 2.0,     # Prioritize high-reward experiences
            "diverse_action": 1.5,  # Prioritize diverse actions (EQ)
            "recent": 1.2           # Slight bias towards recent experiences
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
    """
    Multi-layer perceptron Q-network for action-value function approximation.

    This class implements a fully-connected neural network that approximates
    the Q-function Q(s, a), mapping states to action values. The network is
    trained using backpropagation with the Adam optimizer.

    Network Architecture
    --------------------
    The network uses a 3-layer MLP with ReLU activations:

        Input Layer:    state_size neurons (69 for mixer state)
              |
              v
        Hidden Layer 1: 128 neurons + ReLU activation
              |
              v
        Hidden Layer 2: 64 neurons + ReLU activation
              |
              v
        Output Layer:   action_size neurons (64 for mixer actions)

    The architecture follows common DQN practice with decreasing layer sizes
    forming a "funnel" that progressively abstracts features.

    Mathematical Details
    --------------------
    Forward pass:
        z1 = W1 · x + b1
        a1 = ReLU(z1)
        z2 = W2 · a1 + b2
        a2 = ReLU(z2)
        Q(s) = W3 · a2 + b3

    Backward pass uses chain rule to compute gradients:
        dL/dW3 = a2^T · dL/dz3
        dL/dW2 = a1^T · (dL/dz3 · W3^T ⊙ ReLU'(z2))
        dL/dW1 = x^T · (... chain continues ...)

    Weight Initialization
    ---------------------
    Uses Xavier/Glorot initialization for stable gradient flow:
        W ~ N(0, sqrt(2/n_in))

    This scaling prevents vanishing/exploding gradients in deep networks.

    Adam Optimizer
    --------------
    Uses the Adam optimizer (Kingma & Ba, 2014) for adaptive learning:

        m_t = β1 · m_{t-1} + (1-β1) · g_t     (1st moment estimate)
        v_t = β2 · v_{t-1} + (1-β2) · g_t²    (2nd moment estimate)
        m̂_t = m_t / (1 - β1^t)                (bias correction)
        v̂_t = v_t / (1 - β2^t)                (bias correction)
        θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

    Default hyperparameters (β1=0.9, β2=0.999, ε=1e-8) work well for RL.

    Attributes
    ----------
    state_size : int
        Dimension of input state vectors.

    action_size : int
        Number of discrete actions (output dimension).

    learning_rate : float
        Base learning rate for Adam optimizer. Default: 0.001.
        Recommended range: 0.0001 - 0.01.

    hidden1_size : int
        Number of neurons in first hidden layer. Default: 128.

    hidden2_size : int
        Number of neurons in second hidden layer. Default: 64.

    w1, b1 : np.ndarray
        Weights and biases for first layer.

    w2, b2 : np.ndarray
        Weights and biases for second layer.

    w3, b3 : np.ndarray
        Weights and biases for output layer.

    beta1, beta2, epsilon : float
        Adam optimizer hyperparameters.

    m_*, v_* : np.ndarray
        First and second moment estimates for Adam.

    t : int
        Training step counter for Adam bias correction.

    Example
    -------
    >>> network = ProductionQNetwork(
    ...     state_size=69,     # Mixer state dimension
    ...     action_size=64,    # 8 actions × 8 channels
    ...     learning_rate=0.001
    ... )
    >>>
    >>> # Forward pass: get Q-values for all actions
    >>> state = np.random.randn(69)
    >>> q_values = network.predict(state)
    >>> best_action = np.argmax(q_values[0])
    >>>
    >>> # Backward pass: update from target Q-values
    >>> states = np.random.randn(32, 69)  # Batch of 32 states
    >>> targets = np.random.randn(32, 64)  # Target Q-values
    >>> loss = network.update(states, None, targets)
    >>> print(f"Training loss: {loss:.4f}")

    Notes
    -----
    - This is a pure NumPy implementation, not using deep learning frameworks
    - For production at scale, consider PyTorch/TensorFlow for GPU acceleration
    - The network should be copied to create target networks (see DQN algorithm)

    References
    ----------
    - Mnih et al. (2015) "Human-level control through deep RL"
    - Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    - Glorot & Bengio (2010) "Understanding the difficulty of training DNNs"

    See Also
    --------
    IQNetwork : The interface this class implements.
    ProductionRLMixer : Uses this network for Q-learning.
    """

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the Q-network with random weights.

        Parameters
        ----------
        state_size : int
            Dimension of the input state vector (e.g., 69 for mixer state).

        action_size : int
            Number of discrete actions (e.g., 64 for 8 actions × 8 channels).

        learning_rate : float, default=0.001
            Learning rate for Adam optimizer. Higher values train faster
            but may be unstable; lower values are more stable but slower.
            Recommended: 0.0001 to 0.01.
        """
        logger.info(f"🧠 Initializing Q-Network (state: {state_size}, actions: {action_size})")
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Network architecture: funnel-shaped MLP
        # Larger first layer captures more state features
        # Smaller second layer provides abstraction
        self.hidden1_size = 128
        self.hidden2_size = 64

        # Initialize weights using Xavier/Glorot initialization
        # Scale: sqrt(2/n_in) for ReLU activation (He initialization variant)
        self.w1 = np.random.randn(self.state_size, self.hidden1_size) * np.sqrt(2.0 / self.state_size)
        self.b1 = np.zeros((1, self.hidden1_size))

        self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))

        self.w3 = np.random.randn(self.hidden2_size, self.action_size) * np.sqrt(2.0 / self.hidden2_size)
        self.b3 = np.zeros((1, self.action_size))

        # Adam optimizer hyperparameters (Kingma & Ba, 2014)
        self.beta1 = 0.9     # Exponential decay rate for 1st moment
        self.beta2 = 0.999   # Exponential decay rate for 2nd moment
        self.epsilon = 1e-8  # Small constant for numerical stability

        # Initialize first moment (mean) estimates for all parameters
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

        self.t = 0  # Time step counter for Adam bias correction
    
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
    Complete Deep Q-Network agent for automated audio mixing.

    This is the main orchestrator class that combines all RL components
    (Q-network, target network, experience replay, reward calculator) into
    a cohesive agent following the DQN algorithm from DeepMind.

    The agent learns to optimize audio mixer settings by interacting with
    a simulated mixing environment, collecting experiences, and updating
    its policy through Q-learning with experience replay and target networks.

    DQN Algorithm Overview
    ----------------------
    The agent follows the Deep Q-Network algorithm:

    1. **Observe** current mixer state s
    2. **Select action** using epsilon-greedy policy:
       - With probability ε: random action (exploration)
       - With probability 1-ε: argmax_a Q(s, a) (exploitation)
    3. **Execute action** and observe reward r and next state s'
    4. **Store transition** (s, a, r, s', done) in replay buffer
    5. **Sample mini-batch** from replay buffer
    6. **Compute targets** using target network:
       y = r + γ * max_a' Q_target(s', a') if not done, else r
    7. **Update Q-network** to minimize (Q(s, a) - y)²
    8. **Periodically update** target network with Q-network weights
    9. **Decay epsilon** to reduce exploration over time

    Component Integration
    ---------------------
    The mixer integrates four main components:

        ┌─────────────────────────────────────────────────────────────┐
        │                   ProductionRLMixer                         │
        │  ┌─────────────────────────────────────────────────────┐   │
        │  │ Interaction Loop:                                    │   │
        │  │   state → get_action() → apply_action() → next_state │   │
        │  └───────────────────────────┬─────────────────────────┘   │
        │                              │                              │
        │  ┌───────────┐  ┌────────────┴────────────┐  ┌──────────┐  │
        │  │ Reward    │  │ Experience              │  │ Training │  │
        │  │ Calculator│←─│ Replay Buffer           │─→│ Loop     │  │
        │  └───────────┘  └─────────────────────────┘  └────┬─────┘  │
        │                                                    │        │
        │  ┌───────────────────┐    ┌────────────────────────┴────┐  │
        │  │ Target Network    │←───│ Q-Network                   │  │
        │  │ (frozen weights)  │    │ (learnable weights)         │  │
        │  └───────────────────┘    └─────────────────────────────┘  │
        └─────────────────────────────────────────────────────────────┘

    Epsilon-Greedy Exploration
    --------------------------
    The agent balances exploration and exploitation using epsilon-greedy:

        ε starts at 1.0 (100% random actions)
        After each step: ε = max(ε_min, ε × decay)
        Typical schedule: 1.0 → 0.01 over ~500 steps

    This allows the agent to explore broadly early in training, then
    converge to exploiting learned Q-values as training progresses.

    Attributes
    ----------
    state_size : int
        Dimension of state vectors (default: 69 for mixer state).

    action_size : int
        Number of discrete actions (default: 64 = 8 actions × 8 channels).

    learning_rate : float
        Learning rate for Q-network updates.

    epsilon : float
        Current exploration rate (probability of random action).

    epsilon_decay : float
        Multiplicative decay applied to epsilon after each step.

    epsilon_min : float
        Minimum epsilon value (ensures some exploration continues).

    gamma : float
        Discount factor for future rewards (0.95 = value 20 steps ahead).

    reward_calculator : ProductionRewardCalculator
        Computes multi-objective rewards from state transitions.

    experience_replay : ProductionExperienceReplay
        Stores and samples experiences for training.

    q_network : ProductionQNetwork
        The main learnable Q-function approximator.

    target_network : ProductionQNetwork
        Frozen copy of Q-network for stable TD targets.

    batch_size : int
        Number of experiences per training update (default: 32).

    target_update_frequency : int
        Steps between target network updates (default: 100).

    training_step : int
        Counter of total training steps completed.

    episode_rewards : List[float]
        History of total rewards per episode for monitoring.

    training_losses : List[float]
        History of training losses for convergence analysis.

    action_map : Dict[int, Tuple[ActionType, int, str]]
        Mapping from action indices to (action_type, channel, parameter).

    Example
    -------
    Basic training loop:

        >>> mixer = ProductionRLMixer(
        ...     learning_rate=0.001,
        ...     epsilon=0.8,
        ...     epsilon_decay=0.995
        ... )
        >>>
        >>> # Optimize to target LUFS
        >>> result = mixer.optimize_mix(
        ...     target_lufs=-14.0,
        ...     channels=8,
        ...     max_episodes=50
        ... )
        >>>
        >>> print(f"Final LUFS: {result['final_lufs']:.1f}")
        >>> print(f"Best reward: {result['best_reward']:.3f}")

    Manual episode control:

        >>> state = mixer._create_initial_state(channels=4)
        >>> for step in range(100):
        ...     action = mixer.get_action(state)
        ...     next_state = mixer.apply_action(state, action)
        ...     reward = mixer.reward_calculator.calculate_reward(
        ...         state, action, next_state
        ...     )
        ...     loss = mixer.train_step(state, action, reward, next_state, done=False)
        ...     state = next_state

    Hyperparameter Tuning Guide
    ---------------------------
    - **learning_rate**: Start with 0.001; reduce if training unstable
    - **epsilon_decay**: 0.995 for ~400 step decay; 0.999 for slower exploration
    - **gamma**: 0.95 typical; higher values for long-horizon optimization
    - **batch_size**: 32 typical; increase for more stable gradients
    - **target_update_frequency**: 100 typical; increase for more stability

    See Also
    --------
    ProductionQNetwork : The neural network used for Q-value estimation.
    ProductionExperienceReplay : The replay buffer used for training.
    ProductionRewardCalculator : The reward function definition.
    """

    def __init__(self,
                 state_size: int = 69,  # 8 channels * 8 params + 5 master + 5 audio metrics
                 action_size: int = 64,  # 8 actions * 8 channels
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.95):
        """
        Initialize the RL mixer agent with all components.

        Parameters
        ----------
        state_size : int, default=69
            Dimension of state vectors. Default is 69:
            (8 channels × 8 params) + 5 master params + 5 audio metrics = 69

        action_size : int, default=64
            Number of discrete actions. Default is 64:
            8 action types × 8 channels = 64 actions

        learning_rate : float, default=0.001
            Learning rate for the Q-network Adam optimizer.
            Recommended: 0.0001 to 0.01.

        epsilon : float, default=1.0
            Initial exploration rate. 1.0 means 100% random actions.
            Recommended: Start at 0.8-1.0 for new training.

        epsilon_decay : float, default=0.995
            Decay multiplier applied after each training step.
            0.995 → epsilon reaches 0.01 in ~400 steps.
            0.999 → epsilon reaches 0.01 in ~4000 steps.

        epsilon_min : float, default=0.01
            Minimum exploration rate. Prevents fully greedy behavior.
            Recommended: 0.01-0.1.

        gamma : float, default=0.95
            Discount factor for future rewards.
            0.95 → actions 20+ steps ahead still valued.
            0.99 → longer planning horizon but slower convergence.
        """
        logger.info("🤖 Initializing Production RL Mixer")

        # Core hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon              # Exploration rate
        self.epsilon_decay = epsilon_decay  # Per-step decay
        self.epsilon_min = epsilon_min      # Floor for exploration
        self.gamma = gamma                  # Discount factor

        # Initialize DQN components
        self.reward_calculator = ProductionRewardCalculator()
        self.experience_replay = ProductionExperienceReplay(max_size=5000)
        self.q_network = ProductionQNetwork(state_size, action_size, learning_rate)
        self.target_network = ProductionQNetwork(state_size, action_size, learning_rate)

        # Training configuration
        self.batch_size = 32
        self.target_update_frequency = 100  # Steps between target updates
        self.training_step = 0

        # Performance monitoring
        self.episode_rewards = []
        self.training_losses = []
        self.convergence_threshold = 0.1
        self.convergence_window = 10

        # Build action index → (type, channel, parameter) mapping
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