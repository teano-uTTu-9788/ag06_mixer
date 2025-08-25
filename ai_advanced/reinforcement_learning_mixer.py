# Import production reinforcement learning system
from ai_advanced.production_reinforcement_learning import (
    ProductionRLMixer,
    MixerState,
    RLAction,
    ActionType,
    RewardSignal,
    demo_production_rl_mixer
)

# Create aliases for backward compatibility
QLearningMixer = ProductionRLMixer
ReinforcementLearningMixer = ProductionRLMixer

# Module-level demo function
def demo_rl_mixer():
    """Run RL mixer demo"""
    return demo_production_rl_mixer()