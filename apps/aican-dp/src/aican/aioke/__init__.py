"""
AiOke - Vietnamese Karaoke System with AG06 Integration

Professional karaoke system with:
- Yamaha AG06/AG06MK2 audio interface integration
- Real-time audio processing and spectrum analysis
- Vietnamese lyrics database with diacritic support
- Voice/music classification
- Professional karaoke workflow

Components:
- audio/: AG06 audio processor and real-time analysis
- api/: FastAPI endpoints for karaoke processing
- agents/: Integration agents for workflow automation

References:
- AG06_RESEARCH_IMPLEMENTATION_SUMMARY.md
- ADVANCED_KARAOKE_SYSTEM.md
- KARAOKE_API_DOCUMENTATION.md
"""

__version__ = "1.0.0"
__author__ = "AiCan Team"

from .audio.ag06_processor import OptimizedAG06Processor

__all__ = ["OptimizedAG06Processor"]
