#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
sleep 5

echo "Pulling test model: llama3.2:3b"
ollama pull llama3.2:3b

echo ""
echo "=== ContextLens Test Suite ==="
echo "Step 1: Scanning model..."
contextlens scan llama3.2:3b

echo ""
echo "Step 2: Applying compression..."
contextlens apply llama3.2:3b --skip-benchmark

echo ""
echo "Step 3: Integrating with Ollama..."
contextlens integrate ollama --model llama3.2:3b

echo ""
echo "Step 4: Checking status..."
contextlens status

echo ""
echo "Step 5: Testing inference..."
ollama run llama3.2:3b "Say hello in one word"

echo ""
echo "=== All tests completed! ==="

# Keep container running for manual testing
tail -f /dev/null
