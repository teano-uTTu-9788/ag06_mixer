#!/bin/bash
# Test production endpoints

PORT=${API_PORT:-5001}
HOST=${API_HOST:-127.0.0.1}
BASE="http://${HOST}:${PORT}"

echo "🧪 Testing AG06 Production Server"
echo "================================="

# 1. Health check
echo -n "1. Health check: "
curl -sS ${BASE}/healthz | python3 -c "import sys,json; d=json.load(sys.stdin); print('✅ OK' if d['ok'] else '❌ FAIL')"

# 2. Status endpoint
echo -n "2. Status endpoint: "
curl -sS ${BASE}/api/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"✅ Uptime: {d['uptime']['seconds']:.1f}s\")"

# 3. SSE stream test (5 second test)
echo -n "3. SSE stream: "
timeout 2 curl -sS ${BASE}/api/stream 2>/dev/null | head -n 5 | grep -q "data:" && echo "✅ Streaming" || echo "❌ Not streaming"

# 4. Analyze endpoint
echo -n "4. Analyze endpoint: "
curl -sS -X POST ${BASE}/api/analyze -H "Content-Type: application/json" -d '{}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"✅ RMS: {d['rms_db']:.1f} dB, Class: {d['classification']}\")"

# 5. Performance check
echo -n "5. Drop rate: "
curl -sS ${BASE}/api/status | python3 -c "
import sys,json
d=json.load(sys.stdin)
if 'audio' in d and 'drop_rate' in d['audio']:
    rate = d['audio']['drop_rate']
    status = '✅' if rate < 0.01 else '⚠️'
    print(f'{status} {rate*100:.2f}%')
else:
    print('⏳ No audio stats yet')"

echo ""
echo "Dashboard: ${BASE}/"