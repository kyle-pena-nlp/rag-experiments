#!/bin/bash

# Start 4 Ollama instances for Metal GPU acceleration
echo "🚀 Starting native Ollama instances with Metal acceleration..."

# Kill any existing Ollama processes
pkill -9 ollama 2>/dev/null
sleep 1

# Start 4 instances on different ports
echo "Starting instance 1 on port 11434..."
OLLAMA_HOST=127.0.0.1:11434 ollama serve > /tmp/ollama-11434.log 2>&1 &

echo "Starting instance 2 on port 11435..."
OLLAMA_HOST=127.0.0.1:11435 ollama serve > /tmp/ollama-11435.log 2>&1 &

echo "Starting instance 3 on port 11436..."
OLLAMA_HOST=127.0.0.1:11436 ollama serve > /tmp/ollama-11436.log 2>&1 &

echo "Starting instance 4 on port 11437..."
OLLAMA_HOST=127.0.0.1:11437 ollama serve > /tmp/ollama-11437.log 2>&1 &

# Wait for instances to start
sleep 3

echo "✓ All instances started"
echo ""
echo "Pulling nomic-embed-text model..."
ollama pull nomic-embed-text

echo ""
echo "✅ Setup complete!"
echo ""
echo "Active instances:"
ps aux | grep "ollama serve" | grep -v grep
echo ""
echo "Test with: curl http://localhost:11434/api/tags"
echo "Logs in: /tmp/ollama-*.log"