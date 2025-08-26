# 🚀 Quick Setup Guide - ChatGPT Custom GPT Code Executor

## Step 1: Create Custom GPT
1. Go to **ChatGPT → Create a GPT**
2. **Name:** `Code Executor Pro`
3. **Description:** `I can execute Python and JavaScript code securely through a production API`

## Step 2: Instructions
**Copy and paste this into the Instructions field:**

```
You are Code Executor Pro, an advanced code execution assistant with access to a secure, enterprise-grade API that can execute Python and JavaScript code in real-time.

CORE CAPABILITIES:
- Python & JavaScript execution in sandboxed environments
- Real-time results with execution timing metrics
- Enterprise security with authentication and rate limiting
- Error handling with detailed debugging information

USAGE PROCESS:
1. Show the code first - Always display what you're about to execute
2. Explain the purpose - Brief description of what the code does
3. Execute via API - Use the /execute endpoint with proper authentication
4. Display results clearly - Show output, execution time, and any errors
5. Provide debugging help - Explain errors and suggest fixes when needed

SECURITY FEATURES:
- Input validation preventing dangerous code patterns
- Sandboxed execution isolating code from the system
- Rate limiting (10 requests/minute) preventing abuse
- Circuit breaker protection for service reliability

EXAMPLE INTERACTION:
User: "Execute Python code to calculate fibonacci numbers"
Response: "I'll create a fibonacci calculator:
[Shows code] → [Executes via API] → [Shows results with timing]"

IMPORTANT: Always execute code through the API action, show real results, and provide helpful explanations for any errors or outputs.
```

## Step 3: Configure Action
1. Click **"Create new action"**
2. **Import from URL:** `https://ag06-chatgpt.loca.lt`
3. **Or paste this OpenAPI schema:**

```yaml
openapi: 3.0.3
info:
  title: Enterprise ChatGPT Coding API
  version: 3.0.0
servers:
  - url: https://ag06-chatgpt.loca.lt
paths:
  /execute:
    post:
      summary: Execute code securely
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                code:
                  type: string
                  description: The code to execute
                language:
                  type: string
                  enum: [python, javascript]
                  default: python
                timeout:
                  type: integer
                  default: 30
              required: [code]
      responses:
        '200':
          description: Code executed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  output:
                    type: string
                  error:
                    type: string
                  execution_time:
                    type: number
                  language:
                    type: string
                  trace_id:
                    type: string
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service is healthy
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: Authorization
security:
  - ApiKeyAuth: []
```

## Step 4: Authentication
1. **Authentication Type:** `API Key`
2. **API Key:** `cgt_9374d891cc8d42d78987583378c71bb3`
3. **Auth Type:** `Bearer`
4. **Custom Header Name:** `Authorization`

## Step 5: Test
Try these test prompts:

**Python Test:**
```
Execute this Python code:
print("Hello from ChatGPT!")
import math
print(f"π = {math.pi:.4f}")
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")
```

**JavaScript Test:**
```
Run this JavaScript code:
const data = [1, 2, 3, 4, 5];
const doubled = data.map(x => x * 2);
const sum = doubled.reduce((a, b) => a + b, 0);
console.log(`Original: ${data}`);
console.log(`Doubled: ${doubled}`);
console.log(`Sum: ${sum}`);
```

## ✅ Success Indicators
- Code executes and shows output
- Execution time is displayed
- Errors are handled gracefully
- Authentication works without issues

## 🔧 Troubleshooting
- **Authentication Error:** Verify the API key is correct
- **Connection Error:** Check tunnel URL is accessible
- **Rate Limit:** Wait 60 seconds between requests
- **Code Error:** Check for syntax errors in your code

## 🎉 You're Ready!
Your ChatGPT Custom GPT can now execute Python and JavaScript code directly!