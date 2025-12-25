"""
AiOke Audio Processing Module

Real-time audio processing for AG06 mixer with:
- 64-band spectrum analysis
- Voice/music classification
- FFT-based frequency analysis
- Professional audio quality (<3ms latency)
"""

from .ag06_processor import OptimizedAG06Processor

__all__ = ["OptimizedAG06Processor"]
