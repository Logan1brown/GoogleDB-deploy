#!/bin/bash

# Kill any running streamlit processes
pkill -f streamlit

# Wait a moment for processes to clean up
sleep 2

# Start the dashboard
cd "$(dirname "$0")/.." && \
source venv/bin/activate && \
PYTHONPATH=$PYTHONPATH:$(pwd)/src streamlit run src/dashboard/app.py
