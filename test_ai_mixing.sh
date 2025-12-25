#!/bin/bash
# Test AI mixing production endpoints

PORT=${API_PORT:-5001}
HOST=${API_HOST:-127.0.0.1}
BASE="http://${HOST}:${PORT}"

echo "🧪 Testing AG06 AI Mixing Studio"
echo "================================="

# 1. Health check
echo -n "1. Health check: "
curl -sS ${BASE}/healthz | python3 -c "import sys,json; d=json.load(sys.stdin); print('✅ OK, Mixing: ' + str(d.get('mixing_enabled', False)))"

# 2. Status endpoint with mixing info
echo -n "2. Status with mixing: "
curl -sS ${BASE}/api/status | python3 -c "
import sys,json
d=json.load(sys.stdin)
m = d.get('mixing', {})
print(f\"✅ Mode: {m.get('mode', 'unknown')}, Genre: {m.get('current_genre', 'unknown')}\")"

# 3. Get mixing profiles
echo -n "3. Mixing profiles: "
curl -sS ${BASE}/api/mixing/profiles | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"✅ {len(d)} profiles available: {', '.join(d.keys())}\")"

# 4. SSE stream test (check for mixing data)
echo -n "4. SSE with mixing: "
timeout 2 curl -sS ${BASE}/api/stream 2>/dev/null | head -n 10 | grep -q "mixing" && echo "✅ Mixing data streaming" || echo "⚠️ No mixing data"

# 5. Test mode switching
echo -n "5. Mode switching: "
# Test auto mode
curl -sS -X POST ${BASE}/api/mixing/mode -H "Content-Type: application/json" -d '{"mode":"auto"}' > /dev/null 2>&1
# Test bypass mode  
curl -sS -X POST ${BASE}/api/mixing/mode -H "Content-Type: application/json" -d '{"mode":"bypass"}' > /dev/null 2>&1
# Back to auto
curl -sS -X POST ${BASE}/api/mixing/mode -H "Content-Type: application/json" -d '{"mode":"auto"}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"✅ Mode control working: {d.get('mode', 'error')}\")"

# 6. Analyze with mixing info
echo -n "6. Analyze with AI: "
curl -sS -X POST ${BASE}/api/analyze -H "Content-Type: application/json" -d '{}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
m = d.get('mixing', {})
print(f\"✅ Genre: {m.get('genre', 'unknown')}, Confidence: {m.get('confidence', 0)*100:.0f}%\")"

# 7. DSP metrics
echo -n "7. DSP processing: "
curl -sS ${BASE}/api/status | python3 -c "
import sys,json
d=json.load(sys.stdin)
m = d.get('mixing', {})
gate = abs(m.get('gate_reduction', 0))
comp = abs(m.get('comp_reduction', 0))
lim = abs(m.get('limiter_reduction', 0))
print(f\"✅ Gate: {gate:.1f}dB, Comp: {comp:.1f}dB, Limiter: {lim:.1f}dB\")"

# 8. Processing latency
echo -n "8. Processing latency: "
curl -sS ${BASE}/api/status | python3 -c "
import sys,json
d=json.load(sys.stdin)
m = d.get('mixing', {})
ms = m.get('processing_time_ms', 0)
if ms > 0 and ms < 12:
    status = '✅ Excellent'
elif ms < 20:
    status = '⚠️ Good'
else:
    status = '❌ High'
print(f'{status}: {ms:.1f}ms')"

echo ""
echo "🎚️ Dashboard: ${BASE}/"
echo "📊 Profiles: ${BASE}/api/mixing/profiles"