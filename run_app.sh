#!/bin/bash

# United Airlines Flight Difficulty Score System - Streamlit App
# Startup script for running the web application

echo "Starting United Airlines Flight Difficulty Score System..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
else
    echo "Virtual environment found..."
fi

# Install/update requirements
echo "Installing/updating dependencies..."
.venv/bin/python -m pip install -r requirements.txt

echo "Starting Streamlit application..."
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo "=================================================="

# Start Streamlit
.venv/bin/python -m streamlit run app.py --server.port 8501 --server.address localhost
