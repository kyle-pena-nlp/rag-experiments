#!/bin/bash
# Stop all native Ollama instances

echo "🛑 Stopping native Ollama instances..."

# Find and kill all ollama serve processes
OLLAMA_PIDS=$(pgrep -f "ollama serve")

if [ -z "$OLLAMA_PIDS" ]; then
    echo "   ℹ️  No Ollama instances running"
    exit 0
fi

echo "   Found PIDs: $OLLAMA_PIDS"

# Kill each process
for pid in $OLLAMA_PIDS; do
    echo "   Stopping PID $pid..."
    kill -9 $pid 2>/dev/null
done

# Wait a moment
sleep 1

# Verify they're stopped
REMAINING=$(pgrep -f "ollama serve")
if [ -z "$REMAINING" ]; then
    echo "✅ All Ollama instances stopped"
    
    # Clean up log files
    if ls /tmp/ollama-*.log 1> /dev/null 2>&1; then
        echo "   Cleaning up log files..."
        rm -f /tmp/ollama-*.log
    fi
else
    echo "⚠️  Some processes may still be running: $REMAINING"
fi
