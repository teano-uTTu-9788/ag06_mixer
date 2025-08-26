# ChatGPT Native App Integration Guide

## 🎯 Complete Setup for ChatGPT Native App Direct Coding

This guide enables ChatGPT's native application to execute code directly through your secure enterprise API.

## ✅ System Status

**API Server:** ✅ Running on port 8090 (PID 83805)
**Authentication:** ✅ Token-based with Bearer authentication
**Endpoints:** ✅ /execute, /health, /metrics, /status
**Security:** ✅ Rate limiting, sandboxing, input validation

## 🚀 Quick Start (3 Steps)

### 1. Get Your API Token
```bash
grep CHATGPT_API_TOKEN .env.enterprise | cut -d= -f2
```
**Save this token** - you'll need it for ChatGPT configuration.

### 2. Create Public Tunnel
```bash
# Run the automated setup script
./quick_tunnel_setup.sh
```
This will:
- Verify your server is running
- Install ngrok/localtunnel if needed
- Create secure HTTPS tunnel
- Display the public URL to use

### 3. Configure Custom GPT
1. **Go to ChatGPT → Create a GPT**
2. **Name:** "Code Executor Pro"
3. **Add Action:** Import `chatgpt_openapi_spec.yaml`
4. **Set Authentication:** Use your token from step 1
5. **Update Server URL:** Use tunnel URL from step 2

## 📋 Detailed Setup Instructions

### Custom GPT Configuration

**Instructions to paste in ChatGPT:**
```
You are a Code Executor Pro with access to a secure code execution API. You can:

1. Execute Python and JavaScript code in a sandboxed environment
2. Provide real-time results with execution time metrics
3. Handle errors gracefully with detailed explanations
4. Support complex code with imports and multi-line scripts

USAGE GUIDELINES:
- Always use the /execute endpoint for running code
- Show both the code and execution results
- Handle timeouts and errors professionally
- Provide debugging help when code fails

When users ask you to run code:
1. Show the code you're about to execute
2. Call the /execute API endpoint
3. Display results clearly
4. Explain any errors or offer improvements
```

### Authentication Setup
- **Type:** API Key
- **Auth Type:** Bearer
- **Header:** Authorization
- **Value:** `Bearer cgt_your_token_here`

### Server URL Configuration
Replace in OpenAPI spec:
```yaml
servers:
  - url: https://your-actual-tunnel.ngrok.io
    description: Your tunnel endpoint
```

## 🧪 Test Your Setup

### Python Test:
```
Execute this Python code:
print("Hello from ChatGPT Native App!")
import math
print(f"π = {math.pi:.4f}")
```

### JavaScript Test:
```
Run this JavaScript:
const data = [1, 2, 3, 4, 5];
const doubled = data.map(x => x * 2);
console.log(`Doubled: ${doubled}`);
```

## ✅ Success Indicators

You'll know it's working when:
- ✅ ChatGPT can execute Python and JavaScript code
- ✅ Results appear with execution time metrics
- ✅ Error messages are helpful and detailed
- ✅ Rate limiting prevents abuse
- ✅ All security features are active

## 🔧 API Endpoints Available

### POST /execute
Execute code with full sandboxing and security
```json
{
  "code": "print('Hello World')",
  "language": "python",
  "timeout": 30
}
```

### GET /health
Check system health and uptime
```json
{
  "status": "healthy",
  "uptime": 86400.5,
  "version": "3.0.0"
}
```

### GET /metrics
Prometheus metrics for monitoring
```
chatgpt_requests_total 1234
chatgpt_request_duration_seconds 0.145
```

### GET /status
Detailed system status including:
- Circuit breaker state
- Rate limiter status
- Performance metrics

## 🛡️ Security Features

- **Token Authentication:** Bearer tokens with secure generation
- **Rate Limiting:** 10 requests/minute per IP
- **Code Sandboxing:** Isolated execution environment
- **Input Validation:** OWASP-compliant security checks
- **Circuit Breaker:** Automatic service protection
- **Monitoring:** Comprehensive logging and tracing

## 📊 Enterprise Features

- **Observability:** OpenTelemetry tracing, Prometheus metrics
- **Reliability:** Circuit breakers, rate limiting, health checks
- **Security:** Token auth, input validation, sandboxed execution
- **Performance:** Sub-second response times, optimized for scale

## 🚨 Troubleshooting

**Authentication Error:**
- Verify token format: `Bearer cgt_your_token`
- Check token in Custom GPT settings

**Connection Refused:**
- Ensure tunnel is running
- Verify server on port 8090
- Check firewall settings

**Rate Limited:**
- Wait 60 seconds between requests
- Check rate limit configuration

**Code Execution Fails:**
- Review security validation
- Check for restricted imports
- See `/status` endpoint for details

## 🎉 Result: Direct ChatGPT Coding

Once configured, ChatGPT's native application can:
1. **Generate code** based on your requests
2. **Execute code directly** through your secure API
3. **Show real results** with performance metrics
4. **Debug issues** and suggest improvements
5. **Iterate quickly** on code solutions

**Your ChatGPT can now code directly in its native interface!** 🚀