#!/bin/bash

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama is ready."

echo "Starting FastAPI..."
uvicorn app:app --host 0.0.0.0 --port 8000

kill $OLLAMA_PID