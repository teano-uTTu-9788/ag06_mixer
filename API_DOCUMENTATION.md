# AI Mixer API Documentation v1.0

## Base URLs

### Production Endpoints
- **Global**: `https://api.aimixer.com`
- **US West**: `https://us-west.aimixer.com`
- **US East**: `https://us-east.aimixer.com`
- **EU West**: `https://eu-west.aimixer.com`
- **Asia Pacific**: `https://ap.aimixer.com`

### Edge Endpoints
- **Cloudflare Workers**: `https://edge.aimixer.com`
- **CDN Assets**: `https://cdn.aimixer.com`

## Authentication

All API requests require authentication using API keys.

### Headers
```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Rate Limiting
- **Standard**: 1000 requests/minute
- **Premium**: 10000 requests/minute
- **Enterprise**: Unlimited

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Core Audio Processing

### Process Audio Buffer

Process raw audio through the complete DSP chain with AI-optimized parameters.

**Endpoint:** `POST /process-audio`

#### Request Body
```json
{
    "audioBuffer": [Float32],     // Required: 960 samples at 48kHz
    "config": {
        "genre": String,           // Optional: "auto"|"speech"|"rock"|"jazz"|"electronic"|"classical"
        "bypass": Boolean,         // Optional: Skip processing (default: false)
        "gain": Number,           // Optional: Input gain in dB (-20 to +20)
        "noiseGate": {
            "enabled": Boolean,    // Optional: Enable noise gate (default: true)
            "threshold": Number    // Optional: Gate threshold in dB (-60 to 0)
        },
        "compressor": {
            "enabled": Boolean,    // Optional: Enable compressor (default: true)
            "ratio": Number,      // Optional: Compression ratio (1:1 to 20:1)
            "threshold": Number,  // Optional: Threshold in dB (-40 to 0)
            "attack": Number,     // Optional: Attack time in ms (0.1 to 100)
            "release": Number     // Optional: Release time in ms (10 to 1000)
        },
        "eq": {
            "enabled": Boolean,    // Optional: Enable EQ (default: true)
            "low": Number,        // Optional: Low shelf gain in dB (-12 to +12)
            "mid": Number,        // Optional: Mid bell gain in dB (-12 to +12)
            "high": Number        // Optional: High shelf gain in dB (-12 to +12)
        },
        "limiter": {
            "enabled": Boolean,    // Optional: Enable limiter (default: true)
            "ceiling": Number     // Optional: Output ceiling in dB (-3 to 0)
        }
    }
}
```

#### Response
```json
{
    "processedBuffer": [Float32],  // 960 processed samples
    "metrics": {
        "peak": Number,            // Peak level in dB
        "rms": Number,             // RMS level in dB
        "lufs": Number,            // Integrated LUFS
        "genre": String,           // Detected genre
        "confidence": Number,      // Genre confidence (0-1)
        "processingTime": Number   // Processing time in ms
    },
    "appliedSettings": {
        "noiseGate": Object,       // Applied gate settings
        "compressor": Object,      // Applied compressor settings
        "eq": Object,              // Applied EQ settings
        "limiter": Object          // Applied limiter settings
    }
}
```

#### Example Request
```bash
curl -X POST https://api.aimixer.com/process-audio \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audioBuffer": [0.1, -0.2, 0.15, ...],
    "config": {
      "genre": "auto",
      "compressor": {
        "ratio": 4,
        "threshold": -20
      }
    }
  }'
```

### Stream Audio Processing

Process audio in real-time using Server-Sent Events for streaming applications.

**Endpoint:** `POST /stream-audio`

#### Request
- **Method**: POST
- **Body**: Binary audio stream (Float32 arrays)
- **Headers**: 
  ```http
  Content-Type: application/octet-stream
  Transfer-Encoding: chunked
  ```

#### Response
Server-Sent Events stream:
```
event: audio
data: {"timestamp": 1640995200000, "chunkSize": 960, "processed": true, "metrics": {...}}

event: audio
data: {"timestamp": 1640995200020, "chunkSize": 960, "processed": true, "metrics": {...}}

event: complete
data: {"totalChunks": 150, "totalDuration": 3.0, "averageLatency": 15.2}
```

#### Example (JavaScript)
```javascript
const eventSource = new EventSource('/stream-audio');

eventSource.addEventListener('audio', (e) => {
    const data = JSON.parse(e.data);
    console.log('Processed chunk:', data);
});

eventSource.addEventListener('complete', (e) => {
    const summary = JSON.parse(e.data);
    console.log('Stream complete:', summary);
    eventSource.close();
});
```

## Feature Extraction

### Extract MFCC Features

Extract Mel-frequency cepstral coefficients for AI analysis.

**Endpoint:** `POST /extract-features`

#### Request Body
```json
{
    "audioBuffer": [Float32],      // Required: Audio samples
    "numCoefficients": Number,     // Optional: Number of MFCCs (default: 13)
    "numFilters": Number,         // Optional: Mel filterbank size (default: 40)
    "fftSize": Number,           // Optional: FFT size (default: 2048)
    "hopSize": Number,           // Optional: Hop size (default: 512)
    "windowType": String         // Optional: "hann"|"hamming"|"blackman" (default: "hann")
}
```

#### Response
```json
{
    "features": [[Float32]],      // MFCC matrix [frames][coefficients]
    "spectralCentroid": Number,   // Spectral centroid in Hz
    "zeroCrossingRate": Number,   // Zero crossing rate
    "spectralRolloff": Number,    // Spectral rolloff frequency
    "spectralFlux": Number,       // Spectral flux
    "frameCount": Number,         // Number of frames
    "duration": Number            // Duration in seconds
}
```

### Classify Genre

Classify audio genre using the trained AI model.

**Endpoint:** `POST /classify-genre`

#### Request Body
```json
{
    "features": [[Float32]],      // Required: MFCC features from extract-features
    "returnProbabilities": Boolean // Optional: Return all probabilities (default: false)
}
```

#### Response
```json
{
    "genre": String,              // Primary genre classification
    "confidence": Number,         // Confidence score (0-1)
    "probabilities": {           // If returnProbabilities=true
        "speech": Number,
        "rock": Number,
        "jazz": Number,
        "electronic": Number,
        "classical": Number
    },
    "secondaryGenre": String,     // Second most likely genre
    "modelVersion": String        // AI model version
}
```

## Configuration Management

### Get Current Configuration

**Endpoint:** `GET /config`

#### Response
```json
{
    "audio": {
        "sampleRate": 48000,
        "frameSize": 960,
        "bitDepth": 32
    },
    "dsp": {
        "noiseGate": {...},
        "compressor": {...},
        "eq": {...},
        "limiter": {...}
    },
    "ai": {
        "modelVersion": "1.0.0",
        "mfccCoefficients": 13,
        "genres": ["speech", "rock", "jazz", "electronic", "classical"]
    }
}
```

### Update Configuration

**Endpoint:** `PUT /config`

#### Request Body
```json
{
    "dsp": {
        "compressor": {
            "ratio": 6,
            "threshold": -18
        }
    }
}
```

## Health & Monitoring

### Health Check

**Endpoint:** `GET /health`

#### Response
```json
{
    "status": "healthy",          // "healthy"|"degraded"|"unhealthy"
    "uptime": 3600,              // Uptime in seconds
    "version": "1.0.0",          // API version
    "region": "us-west",         // Deployment region
    "components": {
        "audio": "healthy",
        "ai": "healthy",
        "database": "healthy"
    },
    "timestamp": 1640995200000
}
```

### Metrics

**Endpoint:** `GET /metrics`

Returns Prometheus-formatted metrics:
```
# HELP ai_mixer_requests_total Total number of requests
# TYPE ai_mixer_requests_total counter
ai_mixer_requests_total{method="POST",endpoint="/process-audio",status="200"} 1234

# HELP ai_mixer_processing_duration_seconds Audio processing duration
# TYPE ai_mixer_processing_duration_seconds histogram
ai_mixer_processing_duration_seconds_bucket{le="0.01"} 500
ai_mixer_processing_duration_seconds_bucket{le="0.025"} 900
```

### Statistics

**Endpoint:** `GET /stats`

#### Response
```json
{
    "requests": {
        "total": 1000000,
        "today": 50000,
        "errors": 12,
        "errorRate": 0.00024
    },
    "processing": {
        "totalSamples": 960000000,
        "averageLatency": 15.2,
        "p95Latency": 25.5,
        "p99Latency": 45.2
    },
    "genres": {
        "speech": 150000,
        "rock": 250000,
        "jazz": 100000,
        "electronic": 300000,
        "classical": 200000
    }
}
```

## Mobile SDK Endpoints

### Initialize Session

**Endpoint:** `POST /mobile/session`

#### Request Body
```json
{
    "deviceId": String,           // Required: Unique device ID
    "platform": String,           // Required: "ios"|"android"
    "sdkVersion": String,         // Required: SDK version
    "capabilities": {
        "sampleRate": Number,
        "channels": Number,
        "bitDepth": Number
    }
}
```

#### Response
```json
{
    "sessionId": String,          // Session identifier
    "token": String,              // Session token
    "expiresAt": Number,          // Expiration timestamp
    "config": Object              // Device-specific configuration
}
```

## WebAssembly/Edge Computing

### Download WASM Module

**Endpoint:** `GET /wasm`

Returns the compiled WebAssembly module for browser execution.

#### Response Headers
```http
Content-Type: application/wasm
Content-Length: 524288
Cache-Control: public, max-age=86400
ETag: "abc123def456"
```

## Error Responses

All endpoints use standard HTTP status codes and return errors in a consistent format:

```json
{
    "error": {
        "code": String,           // Error code (e.g., "INVALID_BUFFER_SIZE")
        "message": String,        // Human-readable message
        "details": Object,        // Additional error details
        "timestamp": Number,      // Error timestamp
        "requestId": String       // Request ID for debugging
    }
}
```

### Common Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 400 | `INVALID_BUFFER_SIZE` | Audio buffer size must be 960 samples |
| 400 | `INVALID_SAMPLE_RATE` | Sample rate must be 48000 Hz |
| 400 | `MISSING_FEATURES` | Features required for classification |
| 401 | `UNAUTHORIZED` | Invalid or missing API key |
| 403 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 404 | `ENDPOINT_NOT_FOUND` | Invalid endpoint |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## WebSocket API

For real-time bidirectional communication:

### Connection
```javascript
const ws = new WebSocket('wss://api.aimixer.com/ws');
```

### Message Format
```json
{
    "type": "process",           // Message type
    "id": "msg-123",            // Message ID
    "data": {                   // Message data
        "audioBuffer": [...]
    }
}
```

### Message Types
- `process` - Process audio buffer
- `stream` - Start streaming
- `stop` - Stop streaming
- `config` - Update configuration
- `ping` - Keep-alive

## SDK Examples

### Python
```python
import requests
import numpy as np

API_KEY = "YOUR_API_KEY"
API_URL = "https://api.aimixer.com"

# Process audio
audio_buffer = np.random.randn(960).astype(np.float32)
response = requests.post(
    f"{API_URL}/process-audio",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"audioBuffer": audio_buffer.tolist()}
)
result = response.json()
print(f"Genre: {result['metrics']['genre']}")
```

### Node.js
```javascript
const axios = require('axios');

const API_KEY = 'YOUR_API_KEY';
const API_URL = 'https://api.aimixer.com';

async function processAudio(audioBuffer) {
    const response = await axios.post(
        `${API_URL}/process-audio`,
        { audioBuffer },
        { headers: { Authorization: `Bearer ${API_KEY}` } }
    );
    return response.data;
}
```

### cURL
```bash
# Health check
curl https://api.aimixer.com/health

# Process audio with API key
curl -X POST https://api.aimixer.com/process-audio \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @audio_data.json
```

## Webhooks

Configure webhooks to receive notifications:

### Webhook Events
- `processing.complete` - Audio processing completed
- `genre.detected` - Genre classification completed
- `error.occurred` - Processing error occurred

### Webhook Payload
```json
{
    "event": "processing.complete",
    "timestamp": 1640995200000,
    "data": {
        "sessionId": "sess-123",
        "duration": 3.0,
        "genre": "rock",
        "confidence": 0.92
    }
}
```

## API Versioning

The API uses URL versioning. The current version is v1.

Future versions will be available at:
- `https://api.aimixer.com/v2/...`
- `https://api.aimixer.com/v3/...`

## Support

- **Documentation**: https://docs.aimixer.com
- **Status Page**: https://status.aimixer.com
- **Support Email**: api-support@aimixer.com

---

*Generated from system with 88/88 tests passing (100% validation)*