# Local Setup Guide

This guide will help you set up and run the project directly on your local machine without Docker.

## Prerequisites

### 1. System Dependencies

Install the required system libraries for OpenCV:

```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-xserver-utils
```

### 2. NVIDIA GPU Drivers

Make sure you have NVIDIA GPU drivers installed (version 525+ recommended for CUDA 12.1):

```bash
# Check driver version
nvidia-smi
```

If not installed, follow the [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

### 3. Python 3.10

Ensure Python 3.10 is installed:

```bash
python3 --version
```

## Setup Instructions

### 1. Create a Virtual Environment

```bash
cd /home/thiagofernandes101/projects/fiap/analise-video
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TF_USE_LEGACY_KERAS='1'
```

You can add these to your `~/.bashrc` or create a `.env` file for convenience.

### 4. Configure X11 for Display

```bash
xhost +local:
```

## Running the Project

### Activate Virtual Environment

```bash
cd /home/thiagofernandes101/projects/fiap/analise-video
source venv/bin/activate
```

### Run the Main Script

```bash
python src/main.py
```

## Convenience Script

You can create a `run_local.sh` script for easier execution:

```bash
#!/bin/bash
cd /home/thiagofernandes101/projects/fiap/analise-video
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TF_USE_LEGACY_KERAS='1'
xhost +local:
python src/main.py
```

Make it executable:

```bash
chmod +x run_local.sh
```

Then simply run:

```bash
./run_local.sh
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA memory issues, try using a smaller YOLO model in `src/config.py`:
- `yolov8n-pose.pt` (smallest, fastest)
- `yolov8s-pose.pt` (small)
- `yolov8m-pose.pt` (medium, current)

### Display Issues

If the video window doesn't appear:
1. Make sure X11 is properly configured
2. Run `xhost +local:` before starting the script
3. Check that `DISPLAY` environment variable is set (usually `:0`)

### Import Errors

If you get module import errors:
1. Make sure `PYTHONPATH` includes the `src` directory
2. Activate the virtual environment before running

## Deactivating Virtual Environment

When done:

```bash
deactivate
```
