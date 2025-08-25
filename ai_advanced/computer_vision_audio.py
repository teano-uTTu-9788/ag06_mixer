# Import production computer vision system
from ai_advanced.production_computer_vision import (
    ProductionComputerVision,
    GestureType,
    ExpressionType,
    DetectionResult,
    demo_production_vision
)

# Create aliases for backward compatibility
ComputerVisionAudioMixer = ProductionComputerVision
HandGestureType = GestureType
FacialExpression = ExpressionType
ProcessingResult = DetectionResult

# Module-level demo function
def demo_computer_vision():
    """Run computer vision demo"""
    import asyncio
    return asyncio.run(demo_production_vision())