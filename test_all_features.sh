#!/bin/bash
# Comprehensive AiOke Feature Test Script

BASE_URL="http://localhost:9090"
PASSED=0
FAILED=0

echo "🎤 AiOke Comprehensive Feature Testing"
echo "======================================"
echo ""

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="$5"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    fi
    
    status_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" == "$expected_status" ]; then
        echo "✅ $name"
        ((PASSED++))
    else
        echo "❌ $name (got $status_code, expected $expected_status)"
        ((FAILED++))
    fi
}

# Test Health & Status
echo "1️⃣ Health & Status Tests"
test_endpoint "Health check" "GET" "/api/health" "" "200"
test_endpoint "Stats endpoint" "GET" "/api/stats" "" "200"
echo ""

# Test YouTube Integration
echo "2️⃣ YouTube Integration Tests"
test_endpoint "Search karaoke" "POST" "/api/youtube/search" '{"query":"bohemian rhapsody"}' "200"
test_endpoint "Add to queue" "POST" "/api/youtube/queue" '{"video_id":"test123","title":"Test Song"}' "200"
test_endpoint "Get queue" "GET" "/api/youtube/queue" "" "200"
echo ""

# Test Mixer Controls
echo "3️⃣ Mixer Control Tests"
test_endpoint "Get mixer settings" "GET" "/api/mix" "" "200"
test_endpoint "Update reverb" "POST" "/api/mix" '{"reverb":0.7}' "200"
test_endpoint "Apply party effect" "POST" "/api/effects" '{"effect":"party"}' "200"
test_endpoint "Apply clean effect" "POST" "/api/effects" '{"effect":"clean"}' "200"
test_endpoint "Apply no_vocals effect" "POST" "/api/effects" '{"effect":"no_vocals"}' "200"
echo ""

# Test Voice Commands
echo "4️⃣ Voice Command Tests"
test_endpoint "Play command" "POST" "/api/voice" '{"command":"play bohemian rhapsody"}' "200"
test_endpoint "Skip command" "POST" "/api/voice" '{"command":"skip song"}' "200"
test_endpoint "Volume up" "POST" "/api/voice" '{"command":"volume up"}' "200"
test_endpoint "Add reverb" "POST" "/api/voice" '{"command":"add reverb"}' "200"
test_endpoint "Remove vocals" "POST" "/api/voice" '{"command":"remove vocals"}' "200"
echo ""

# Test Error Handling
echo "5️⃣ Error Handling Tests"
test_endpoint "Invalid endpoint" "GET" "/api/invalid" "" "404"
test_endpoint "Empty search" "POST" "/api/youtube/search" '{}' "400"
test_endpoint "Invalid JSON" "POST" "/api/mix" 'not json' "400"
echo ""

# Test Performance
echo "6️⃣ Performance Tests"
echo -n "Response time test: "
time_ms=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/api/health")
time_ms=$(echo "$time_ms * 1000" | bc)
if (( $(echo "$time_ms < 100" | bc -l) )); then
    echo "✅ Health check < 100ms (${time_ms}ms)"
    ((PASSED++))
else
    echo "❌ Health check > 100ms (${time_ms}ms)"
    ((FAILED++))
fi
echo ""

# Test Static Files
echo "7️⃣ Static File Tests"
test_endpoint "Interface HTML" "GET" "/aioke_enhanced_interface.html" "" "200"
test_endpoint "Manifest.json" "GET" "/manifest.json" "" "200"
test_endpoint "Service worker" "GET" "/sw.js" "" "200"
echo ""

# Test Advanced Features
echo "8️⃣ Advanced Feature Tests"

# Test stats increment
initial_count=$(curl -s "$BASE_URL/api/stats" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_requests', 0))")
curl -s "$BASE_URL/api/health" > /dev/null
curl -s "$BASE_URL/api/health" > /dev/null
new_count=$(curl -s "$BASE_URL/api/stats" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_requests', 0))")

if [ "$new_count" -gt "$initial_count" ]; then
    echo "✅ Stats tracking works"
    ((PASSED++))
else
    echo "❌ Stats tracking not incrementing"
    ((FAILED++))
fi

# Test mixer persistence
curl -s -X POST "$BASE_URL/api/mix" -H "Content-Type: application/json" -d '{"reverb":0.42}' > /dev/null
reverb_value=$(curl -s "$BASE_URL/api/mix" | python3 -c "import sys, json; print(json.load(sys.stdin).get('reverb', 0))")
if [ "$(echo "$reverb_value > 0.4" | bc)" -eq 1 ]; then
    echo "✅ Mixer settings persist"
    ((PASSED++))
else
    echo "❌ Mixer settings not persisting"
    ((FAILED++))
fi

echo ""
echo "======================================"
echo "📊 Final Results"
echo "======================================"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
TOTAL=$((PASSED + FAILED))
PERCENTAGE=$((PASSED * 100 / TOTAL))
echo "  Total:  $TOTAL"
echo "  Success Rate: ${PERCENTAGE}%"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL"
else
    echo "⚠️  $FAILED tests failed - Review needed"
fi

echo ""
echo "📱 System Access URLs:"
echo "   Local: http://localhost:9090/aioke_enhanced_interface.html"
echo "   iPad:  http://$(ipconfig getifaddr en0 2>/dev/null || echo YOUR_IP):9090/aioke_enhanced_interface.html"