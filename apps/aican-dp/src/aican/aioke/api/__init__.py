"""
AiOke API Module

FastAPI endpoints for Vietnamese karaoke system:
- /api/aioke/devices - List available AG06 devices
- /api/aioke/start - Start audio processing
- /api/aioke/stop - Stop audio processing
- /api/aioke/status - Get current status and metrics
- /api/aioke/spectrum - Get real-time spectrum data

All endpoints follow RESTful conventions and include:
- Request validation with Pydantic
- Error handling with proper HTTP status codes
- OpenAPI documentation
- Performance metrics
"""

from .routes import router

__all__ = ["router"]
