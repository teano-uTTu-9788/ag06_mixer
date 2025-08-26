# Import production NLP system
from ai_advanced.production_nlp_system import (
    ProductionNLP,
    IntentType,
    EntityType,
    ParsedCommand,
    demo_production_nlp
)

# Create aliases for backward compatibility
NLPVoiceControl = ProductionNLP
CommandIntent = IntentType
VoiceControlSystem = ProductionNLP
ProcessingResult = ParsedCommand

# Module-level demo function
def demo_nlp_voice():
    """Run NLP voice demo"""
    import asyncio
    return asyncio.run(demo_production_nlp())