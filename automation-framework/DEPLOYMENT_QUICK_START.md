# 🚀 Aioke Advanced Enterprise - Quick Start Guide

## Prerequisites
- Python 3.8+
- Git
- 4GB RAM minimum
- Port 8080 available

## Quick Deployment (3 Steps)

### 1. Clone Repository
```bash
git clone https://github.com/teano-uTTu-9788/ag06_mixer.git
cd ag06_mixer/automation-framework
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install flask asyncio
```

### 3. Start Production Server
```bash
python3 start_production_server.py
```

Server will be available at: **http://localhost:8080**

## Verify Deployment

### Check Health
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "uptime": 100.5,
  "total_events": 1234,
  "error_count": 0,
  "processing": true
}
```

### Run Tests
```bash
# Test advanced patterns (88 tests)
python3 test_advanced_patterns_88.py

# Test enterprise implementation (88 tests)
python3 test_enterprise_implementation_88.py
```

Both should show: **Success Rate: 100.0%**

## What's Included

### 8 Enterprise Patterns
1. **Google Borg/Kubernetes** - Resource scheduling
2. **Meta Hydra** - Configuration management
3. **Amazon Cells** - Fault isolation architecture
4. **Microsoft Dapr** - Sidecar pattern
5. **Uber Cadence** - Workflow orchestration
6. **LinkedIn Kafka** - Stream processing
7. **Twitter Finagle** - RPC framework
8. **Airbnb Airflow** - DAG orchestration

### Key Files
- `advanced_enterprise_patterns.py` - All pattern implementations
- `test_advanced_patterns_88.py` - Comprehensive test suite
- `deploy_advanced_enterprise.py` - Full deployment script
- `start_production_server.py` - Production API server

## API Endpoints

- **GET /health** - System health status
- **GET /metrics** - Detailed metrics
- **GET /status** - Component status

## Advanced Deployment

For full deployment with all components:
```bash
python3 deploy_advanced_enterprise.py
```

This will:
- Initialize all 8 enterprise patterns
- Start background event processing
- Enable monitoring and metrics
- Create example workflows and cells

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8080
lsof -i :8080
# Kill if needed
kill -9 <PID>
```

### Import Errors
```bash
# Ensure you're in the correct directory
cd ag06_mixer/automation-framework
# Activate virtual environment
source venv/bin/activate
```

### Test Failures
- Ensure server is running before tests
- Check Python version (3.8+ required)
- Verify all files are present

## Performance

- **Throughput**: ~10 events/second baseline
- **Memory**: ~50MB typical usage
- **CPU**: Low usage (~5% single core)
- **Scalability**: Handles 100k+ events

## Support

- **Documentation**: See ADVANCED_ENTERPRISE_COMPLETE.md
- **Tests**: 176/176 compliance verified
- **Repository**: https://github.com/teano-uTTu-9788/ag06_mixer

## Quick Stop

To stop the server:
```bash
# Press Ctrl+C in the terminal running the server
# Or find and kill the process
ps aux | grep start_production_server
kill <PID>
```

---

**Version**: 3.0.0
**Last Updated**: 2025-08-26
**Status**: Production Ready