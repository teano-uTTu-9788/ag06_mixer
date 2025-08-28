# 🎤 AiOke Cross-Device Testing Guide

## 🚀 Current System Status
- **Mac Server**: Running on `192.168.1.10`
- **Main Application**: http://localhost:9099 (Mac) / http://192.168.1.10:9099 (iPad)
- **Metrics Dashboard**: http://localhost:9100/metrics (Mac) / http://192.168.1.10:9100/metrics (iPad)
- **WebSocket Streaming**: http://localhost:9098 (Mac) / http://192.168.1.10:9098 (iPad)

## 🖥️ Mac Testing URLs

### Core Application
- **Main UI**: http://localhost:9099
- **Health Check**: http://localhost:9099/healthz
- **YouTube Search**: http://localhost:9099/api/youtube/search?q=karaoke
- **AI Processing**: http://localhost:9099/api/ai/process (POST)
- **Hardware Detection**: http://localhost:9099/api/hardware/detect
- **Feature Flags**: http://localhost:9099/api/features/test_user

### Enterprise Monitoring
- **Google SRE Metrics**: http://localhost:9100/metrics
- **Meta WebSocket Stats**: http://localhost:9098/api/websocket/stats
- **Circuit Breaker Status**: http://localhost:9099/api/health/circuit-breaker

## 📱 iPad Testing URLs

### Core Application (iPad Safari)
- **Main UI**: http://192.168.1.10:9099
- **Health Check**: http://192.168.1.10:9099/healthz
- **YouTube Search**: http://192.168.1.10:9099/api/youtube/search?q=karaoke
- **Hardware Detection**: http://192.168.1.10:9099/api/hardware/detect
- **Feature Flags**: http://192.168.1.10:9099/api/features/test_user

### Enterprise Monitoring (iPad)
- **Google SRE Metrics**: http://192.168.1.10:9100/metrics
- **Meta WebSocket Stats**: http://192.168.1.10:9098/api/websocket/stats

## 🧪 Test Scenarios

### Scenario 1: Basic Functionality Test
1. **Mac**: Open http://localhost:9099 - Verify main UI loads
2. **iPad**: Open http://192.168.1.10:9099 - Verify same UI loads on mobile
3. **Both**: Test YouTube search functionality
4. **Both**: Verify feature flags are working

### Scenario 2: Real-Time WebSocket Test
1. **Mac**: Connect to WebSocket at ws://localhost:9098/ws
2. **iPad**: Connect to WebSocket at ws://192.168.1.10:9098/ws
3. **Verify**: Real-time audio levels streaming on both devices
4. **Test**: Dual-device real-time synchronization

### Scenario 3: Enterprise Monitoring
1. **Mac**: Check metrics at http://localhost:9100/metrics
2. **iPad**: Access same metrics at http://192.168.1.10:9100/metrics
3. **Verify**: Prometheus metrics collection working
4. **Test**: Circuit breaker functionality

### Scenario 4: AI Audio Processing
1. **Mac**: POST to http://localhost:9099/api/ai/process with audio data
2. **iPad**: Same endpoint at http://192.168.1.10:9099/api/ai/process
3. **Verify**: Microsoft AI patterns working on both
4. **Test**: Cross-device processing coordination

## 🔄 Simultaneous Testing Commands

### Mac Terminal Commands
```bash
# Test all endpoints
curl -s http://localhost:9099/healthz
curl -s http://localhost:9099/api/hardware/detect
curl -s http://localhost:9099/api/features/test_user
curl -s http://localhost:9100/metrics | head -20
curl -s http://localhost:9098/api/websocket/stats

# WebSocket test
python3 -c "
import asyncio
import websockets
import json

async def test_websocket():
    uri = 'ws://localhost:9098/ws'
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({'type': 'ping'}))
        response = await websocket.recv()
        print('Mac WebSocket:', response)

asyncio.run(test_websocket())
"
```

### iPad Safari Testing
1. Open Safari on iPad
2. Navigate to: http://192.168.1.10:9099
3. Test all UI elements
4. Open developer console (if available)
5. Test WebSocket connections

## 📊 Expected Results

### Mac Results
- ✅ Full UI functionality with glassmorphism design
- ✅ Google SRE metrics collection
- ✅ Meta WebSocket real-time streaming
- ✅ Microsoft AI audio processing
- ✅ Netflix circuit breaker patterns
- ✅ Apple hardware detection (AG06 not found - expected)
- ✅ OpenAI function calling simulation

### iPad Results
- ✅ Responsive PWA-like experience
- ✅ Touch-optimized controls
- ✅ Real-time WebSocket connection
- ✅ Cross-device synchronization
- ✅ Mobile-optimized YouTube search
- ✅ Feature flag A/B testing

## 🚨 Troubleshooting

### Common Issues
1. **iPad can't connect**: Check Mac firewall settings
2. **WebSocket fails**: Verify ports 9098, 9099, 9100 are open
3. **Slow response**: Check network congestion
4. **Missing features**: Verify feature flags are enabled

### Debug Commands
```bash
# Check if services are running
ps aux | grep python3 | grep -E "(comprehensive|websocket)"

# Check port availability
lsof -i :9099
lsof -i :9100
lsof -i :9098

# Test network connectivity from iPad
# (Run on iPad in Safari console)
fetch('http://192.168.1.10:9099/healthz')
  .then(r => r.text())
  .then(console.log)
```

## 🎯 Success Criteria

### Mac Testing ✅
- [ ] Main UI loads with glassmorphism design
- [ ] YouTube search returns results
- [ ] AI processing endpoints respond
- [ ] Hardware detection completes
- [ ] Feature flags load correctly
- [ ] WebSocket streaming active
- [ ] Metrics collection working

### iPad Testing ✅
- [ ] Cross-device UI loads properly
- [ ] Touch interactions work smoothly
- [ ] WebSocket connection stable
- [ ] Real-time sync with Mac
- [ ] Mobile-optimized experience
- [ ] PWA capabilities functional

### Simultaneous Testing ✅
- [ ] Both devices connected simultaneously
- [ ] Real-time data synchronization
- [ ] Load balancing working
- [ ] Cross-device feature consistency
- [ ] Enterprise monitoring active on both

## 📱 Ready for Testing!

**Mac**: System ready at http://localhost:9099
**iPad**: System accessible at http://192.168.1.10:9099
**Monitoring**: Enterprise dashboards active

Start testing now! 🚀