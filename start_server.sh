#!/bin/bash

# Navigate to the service directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Activate the virtual environment
source venv/bin/activate

# Start the FastAPI server on port 8000
echo "Starting SadTalker API Server on http://0.0.0.0:8000"
echo "Endpoint: POST http://0.0.0.0:8000/generate"

# Use uvicorn to run the server
export PYTORCH_ENABLE_MPS_FALLBACK=1
./venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
