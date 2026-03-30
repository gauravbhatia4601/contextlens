#!/bin/bash
# Quick setup script for ContextLens Docker testing

set -e

echo "🔧 Building ContextLens test container..."
docker compose build

echo ""
echo "🚀 Starting container..."
docker compose up -d

echo ""
echo "⏳ Waiting for Ollama to start and pull model (this takes 2-3 minutes)..."
sleep 10

# Wait for Ollama to be ready
until docker exec contextlens-test curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Waiting for Ollama..."
    sleep 5
done

echo ""
echo "✅ Container is ready!"
echo ""
echo "📊 View live logs:"
echo "   docker compose logs -f"
echo ""
echo "📝 Run tests manually:"
echo "   docker exec -it contextlens-test /app/start_tests.sh"
echo ""
echo "🔍 Check status:"
echo "   docker exec -it contextlens-test contextlens status"
echo ""
echo "🗑️  Cleanup when done:"
echo "   docker compose down -v"
echo ""
echo "🌐 Ollama API available at: http://localhost:11435"
