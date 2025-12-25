"""
AiOke FastAPI Routes

RESTful API for Vietnamese karaoke system with AG06 integration.

Endpoints:
- GET /devices - List available AG06/AG03 devices
- POST /start - Start audio processing on device
- POST /stop - Stop audio processing
- GET /status - Get processor status and metrics
- GET /spectrum - Get real-time spectrum data

Performance targets:
- P95 latency: <5ms
- Throughput: 72kHz+
- Error rate: <1%
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sounddevice as sd

from ..audio.ag06_processor import OptimizedAG06Processor, detect_ag06_device

router = APIRouter(prefix="/api/aioke", tags=["AiOke"])

# Global processor instance (singleton pattern)
_processor: Optional[OptimizedAG06Processor] = None


# Pydantic models
class DeviceInfo(BaseModel):
    """Audio device information."""
    index: int = Field(..., description="Device index")
    name: str = Field(..., description="Device name")
    channels: int = Field(..., description="Input channels")
    sample_rate: float = Field(..., description="Sample rate in Hz")


class StartRequest(BaseModel):
    """Request to start audio processing."""
    device_index: int = Field(..., description="Audio device index", ge=0)
    sample_rate: int = Field(48000, description="Sample rate in Hz", ge=8000, le=192000)
    block_size: int = Field(256, description="Processing block size", ge=64, le=2048)


class ProcessorStatus(BaseModel):
    """Processor status and metrics."""
    is_running: bool = Field(..., description="Processor running status")
    device_index: Optional[int] = Field(None, description="Current device index")
    avg_latency_ms: float = Field(..., description="Average latency in ms")
    p95_latency_ms: float = Field(..., description="P95 latency in ms")
    p99_latency_ms: float = Field(..., description="P99 latency in ms")
    last_classification: Optional[str] = Field(None, description="Last audio classification")


class SpectrumData(BaseModel):
    """Real-time spectrum analysis data."""
    spectrum: List[float] = Field(..., description="64-band spectrum (0-100)")
    level_db: float = Field(..., description="RMS level in dB")
    classification: str = Field(..., description="Audio classification")
    timestamp: float = Field(..., description="Unix timestamp")


# API endpoints
@router.get(
    "/devices",
    response_model=List[DeviceInfo],
    summary="List Audio Devices",
    description="List all available AG06/AG03 audio devices"
)
async def list_devices() -> List[DeviceInfo]:
    """
    List all available audio input devices.

    Searches for AG06/AG03/Yamaha devices specifically.

    Returns:
        List of available devices with metadata
    """
    try:
        devices = sd.query_devices()
        ag06_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = device['name'].lower()
                # Prioritize AG06/AG03/Yamaha devices
                if any(keyword in device_name for keyword in ['ag06', 'ag03', 'yamaha']):
                    ag06_devices.insert(0, DeviceInfo(
                        index=i,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=device['default_samplerate']
                    ))
                else:
                    ag06_devices.append(DeviceInfo(
                        index=i,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rate=device['default_samplerate']
                    ))

        return ag06_devices

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query audio devices: {str(e)}"
        )


@router.post(
    "/start",
    response_model=ProcessorStatus,
    summary="Start Audio Processing",
    description="Start real-time audio processing on specified device"
)
async def start_processing(request: StartRequest) -> ProcessorStatus:
    """
    Start audio processing on specified device.

    Creates or reuses processor instance and begins monitoring.

    Args:
        request: Start request with device index and parameters

    Returns:
        Initial processor status

    Raises:
        HTTPException: If already running or device invalid
    """
    global _processor

    # Check if already running
    if _processor and _processor.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Processor already running. Stop it first."
        )

    try:
        # Create new processor
        _processor = OptimizedAG06Processor(
            device_index=request.device_index,
            sample_rate=request.sample_rate,
            block_size=request.block_size
        )

        # Start monitoring
        _processor.start_monitoring()

        # Get initial metrics
        metrics = _processor.get_performance_metrics()

        return ProcessorStatus(
            is_running=True,
            device_index=request.device_index,
            avg_latency_ms=metrics['avg_latency_ms'],
            p95_latency_ms=metrics['p95_latency_ms'],
            p99_latency_ms=metrics['p99_latency_ms'],
            last_classification=metrics['last_classification']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processor: {str(e)}"
        )


@router.post(
    "/stop",
    response_model=Dict[str, str],
    summary="Stop Audio Processing",
    description="Stop real-time audio processing"
)
async def stop_processing() -> Dict[str, str]:
    """
    Stop audio processing.

    Stops the processor and releases audio resources.

    Returns:
        Success message

    Raises:
        HTTPException: If processor not running
    """
    global _processor

    if not _processor or not _processor.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Processor not running"
        )

    try:
        _processor.stop_monitoring()
        return {"message": "Processor stopped successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop processor: {str(e)}"
        )


@router.get(
    "/status",
    response_model=ProcessorStatus,
    summary="Get Processor Status",
    description="Get current processor status and performance metrics"
)
async def get_status() -> ProcessorStatus:
    """
    Get processor status and metrics.

    Returns:
        Current status including latency metrics

    Raises:
        HTTPException: If processor not initialized
    """
    global _processor

    if not _processor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processor not initialized"
        )

    metrics = _processor.get_performance_metrics()

    return ProcessorStatus(
        is_running=_processor.is_running,
        device_index=_processor.device_index,
        avg_latency_ms=metrics['avg_latency_ms'],
        p95_latency_ms=metrics['p95_latency_ms'],
        p99_latency_ms=metrics['p99_latency_ms'],
        last_classification=metrics['last_classification']
    )


@router.get(
    "/spectrum",
    response_model=SpectrumData,
    summary="Get Spectrum Data",
    description="Get real-time spectrum analysis data"
)
async def get_spectrum() -> SpectrumData:
    """
    Get real-time spectrum data.

    Returns current audio spectrum analysis including:
    - 64-band logarithmic spectrum (20Hz-20kHz)
    - RMS level in dB
    - Voice/music classification
    - Timestamp

    Returns:
        Current spectrum data

    Raises:
        HTTPException: If processor not running
    """
    global _processor

    if not _processor or not _processor.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Processor not running"
        )

    try:
        # Get latest processed data
        data = _processor.process_audio_block()

        return SpectrumData(
            spectrum=data['spectrum'],
            level_db=data['level_db'],
            classification=data['classification'],
            timestamp=data['timestamp']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get spectrum data: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Check AiOke service health"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    global _processor

    return {
        "status": "healthy",
        "service": "AiOke",
        "version": "1.0.0",
        "processor_initialized": _processor is not None,
        "processor_running": _processor.is_running if _processor else False
    }
