#!/bin/bash
# Test script for ContextLens Proxy Server

set -e

PROXY_URL="${PROXY_URL:-http://localhost:8080}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "=== ContextLens Proxy Test Suite ==="
echo ""

# Check if proxy is running
echo "1. Checking proxy health..."
if curl -s -f "$PROXY_URL/health" > /dev/null 2>&1; then
    echo "   [OK] Proxy is healthy"
else
    echo "   [FAIL] Proxy is not responding at $PROXY_URL"
    echo "   Start with: docker-compose -f docker-compose.proxy.yml up"
    exit 1
fi

# List available models
echo ""
echo "2. Listing available models..."
curl -s "$PROXY_URL/api/tags" | head -20

# Test generate endpoint (non-streaming)
echo ""
echo "3. Testing /api/generate (non-streaming)..."
curl -s -X POST "$PROXY_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2:1b",
        "prompt": "Hello, world!",
        "stream": false,
        "options": {"num_predict": 20}
    }' | head -50

# Test generate endpoint (streaming)
echo ""
echo "4. Testing /api/generate (streaming)..."
curl -s -X POST "$PROXY_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2:1b",
        "prompt": "Say hello in one word:",
        "stream": true,
        "options": {"num_predict": 5}
    }' | head -10

# Test chat endpoint
echo ""
echo "5. Testing /api/chat..."
curl -s -X POST "$PROXY_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2:1b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    }' | head -50

echo ""
echo "=== All tests completed ==="
