# Import production generative AI system
from ai_advanced.production_generative_ai import (
    ProductionGenerativeMixAI,
    MixStyle,
    GenerationRequest,
    GenerationResult,
    MixTemplate,
    demo_production_generative_ai
)

# Create aliases for backward compatibility
GenerativeMixAI = ProductionGenerativeMixAI
AIStyle = MixStyle

# Module-level demo function
def demo_generative_ai():
    """Run generative AI demo"""
    return demo_production_generative_ai()