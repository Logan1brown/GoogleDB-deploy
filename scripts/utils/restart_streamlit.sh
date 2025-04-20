#!/bin/bash

# Usage: ./scripts/utils/restart_streamlit.sh
# This version is simplified for the deployment package and always starts the main dashboard

# Load environment variables if they exist
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Use STREAMLIT_PORT from environment, default to 8501
PORT=${STREAMLIT_PORT:-8501}
MAX_WAIT=10  # Maximum seconds to wait for port to free up

echo "Stopping Streamlit..."
# Kill any existing Streamlit processes
pkill -f streamlit

echo "Waiting for processes to end..."
# Give pkill time to complete
sleep 2

# Start the dashboard
echo "Starting dashboard..."
cd "$(dirname "$0")/../.." && \
PYTHONPATH=$PYTHONPATH:$(pwd)/src streamlit run src/dashboard/app.py
