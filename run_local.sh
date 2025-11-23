#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv .venv"
    echo "Then: source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TF_USE_LEGACY_KERAS='1'

# Configure X11
if command -v xhost &> /dev/null; then
    xhost +local: 2>/dev/null
else
    echo "Warning: xhost not found. Display may not work properly."
    echo "Install with: sudo apt-get install x11-xserver-utils"
fi

# Run the application
echo "Starting application..."
python src/main.py

# Deactivate when done
deactivate