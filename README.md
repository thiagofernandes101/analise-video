# Video Analysis System

A real-time video analysis system that performs person detection, pose estimation, face detection, and emotion analysis using deep learning models. Built with Python, leveraging YOLO for pose detection, DeepFace for emotion analysis, and optimized for NVIDIA GPU acceleration.

## 🎯 Features

- **Person Detection & Tracking**: Multi-person detection and tracking using YOLOv8-pose
- **Pose Estimation**: 17-point keypoint detection for body pose analysis
- **Activity Recognition**: Real-time posture and action detection (standing, sitting, hand gestures)
- **Face Detection**: Robust face detection using DeepFace with multiple backend options
- **Emotion Analysis**: Real-time facial emotion recognition
- **Face-Person Matching**: Intelligent matching of detected faces to tracked persons
- **GPU Acceleration**: Optimized for NVIDIA GPUs (tested on RTX 3060)
- **Real-time Visualization**: HUD overlay with detection results and analytics

## 📋 Prerequisites

### Required

- **Docker** and **Docker Compose** (recommended) OR Python 3.10+
- **NVIDIA GPU** with compute capability 6.0+ (e.g., RTX 3060)
- **NVIDIA Container Toolkit** (for Docker)
- **NVIDIA GPU Drivers** (version 525+ recommended for CUDA 12.1)
- **X11 Server** for display (pre-installed on most Linux desktops)

### System Requirements

- **GPU VRAM**: 6GB minimum (tested on RTX 3060 6GB VRAM)
- **RAM**: 8GB minimum
- **OS**: Linux (Ubuntu 20.04+ recommended)

## 🚀 Quick Start

### Option 1: Docker (Recommended)

The easiest way to run the project is using Docker Compose.

#### 1. Configure X11 Permissions

```bash
xhost +local:docker
```

#### 2. Run the Application

```bash
docker-compose up --build
```

The application will:
- Build the Docker image with all dependencies
- Download required models automatically
- Start processing the video from `videos/` directory
- Display the analyzed video in a window

#### 3. Stop the Application

Press `Ctrl+C` or:

```bash
docker-compose down
```

For detailed Docker usage and troubleshooting, see [DOCKER_COMPOSE_GUIDE.md](DOCKER_COMPOSE_GUIDE.md).

### Option 2: Local Setup

If you prefer to run without Docker:

#### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev x11-xserver-utils
```

#### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export TF_USE_LEGACY_KERAS='1'
xhost +local:
```

#### 5. Run the Application

```bash
python src/main.py
```

For detailed local setup instructions and troubleshooting, see [LOCAL_SETUP.md](LOCAL_SETUP.md).

## 📁 Project Structure

```
analise-video/
├── src/
│   ├── main.py                      # Application entry point
│   ├── config.py                    # Configuration management
│   ├── activity_recognizer.py       # Activity recognition coordinator
│   ├── visualizer.py                # Visualization coordinator
│   ├── detectors/                   # Detection implementations
│   │   ├── person_detector.py       # YOLO person/pose detector
│   │   └── face_detector.py         # DeepFace face detector
│   ├── analyzers/                   # Analysis implementations
│   │   └── emotion_analyzer.py      # Emotion analysis with async processing
│   ├── services/                    # Business logic services
│   │   ├── keypoint_history_manager.py
│   │   ├── posture_detector.py
│   │   ├── action_detector.py
│   │   └── face_person_matcher.py
│   ├── renderers/                   # Visualization renderers
│   │   ├── hud_renderer.py
│   │   ├── face_renderer.py
│   │   ├── pose_renderer.py
│   │   └── activity_renderer.py
│   ├── models/                      # Data models
│   │   ├── bounding_box.py
│   │   ├── person_tracking.py
│   │   ├── face_detection.py
│   │   ├── emotion_result.py
│   │   └── activity_result.py
│   └── interfaces/                  # Protocol definitions
│       ├── detector.py
│       └── analyzer.py
├── videos/                          # Video files directory
├── Dockerfile                       # Docker image definition
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── yolov8n-pose.pt                 # YOLO pose model weights
└── README.md                       # This file
```

## 🎮 Usage

### Video Selection

Place your video file in the `videos/` directory. The application will automatically use the first video it finds. Supported formats:
- `.mp4`
- `.avi`
- `.mov`

### Keyboard Controls

While the video is playing:
- **`q`** or **`ESC`**: Quit the application
- **Window close button**: Also quits the application

### Configuration

Edit `src/config.py` to customize:
- Model backends (YOLO model size, DeepFace backend)
- Analysis intervals (emotion analysis frequency)
- Visualization settings (colors, font sizes, opacity)
- Detection thresholds

Example configurations:

```python
# Use a smaller/faster YOLO model
YOLO_MODEL_NAME = "yolov8n-pose.pt"  # nano (fastest)
# or
YOLO_MODEL_NAME = "yolov8s-pose.pt"  # small
# or
YOLO_MODEL_NAME = "yolov8m-pose.pt"  # medium (default)

# Change emotion analysis frequency
ANALYSIS_INTERVAL_FRAMES = 30  # Analyze every 30 frames

# Change face detection backend
FACE_DETECTION_BACKEND = "retinaface"  # or "opencv", "ssd", "mtcnn"
```

## 🏗️ Architecture

The project follows **Clean Code** and **SOLID** principles:

- **Single Responsibility**: Each class has one clear purpose
- **Dependency Inversion**: Components depend on abstractions (Protocols), not concrete implementations
- **Open/Closed**: Easy to extend with new detectors or analyzers
- **Clean Data Models**: Type-safe data classes instead of primitives
- **Service Layer**: Business logic separated from detection/analysis
- **Async Processing**: Non-blocking emotion analysis for real-time performance

## 🔧 Troubleshooting

### Display Issues

If the video window doesn't appear:
```bash
# Run this before starting
xhost +local:docker  # For Docker
# or
xhost +local:        # For local
```

### GPU Not Detected

Check if NVIDIA GPU is available:
```bash
# For Docker
docker-compose run --rm analise-video nvidia-smi

# For local
nvidia-smi
```

### CUDA Out of Memory

If you get CUDA memory errors:
1. Use a smaller YOLO model: Change `YOLO_MODEL_NAME` in `config.py` to `"yolov8n-pose.pt"`
2. Reduce emotion analysis frequency: Increase `ANALYSIS_INTERVAL_FRAMES` in `config.py`

### Import Errors (Local Setup)

Make sure `PYTHONPATH` is set correctly:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 📦 Dependencies

Core dependencies:
- **PyTorch** (with CUDA 12.1 support)
- **Ultralytics** (YOLOv8)
- **DeepFace** (Face detection and emotion analysis)
- **OpenCV** (Video processing and visualization)
- **TensorFlow** (DeepFace backend)
- **MediaPipe, MTCNN, RetinaFace** (Face detection backends)

See [`requirements.txt`](requirements.txt) for full list.

## 📝 License

This is a FIAP academic project.

## 🤝 Contributing

This is an academic project. For suggestions or issues, please contact the project maintainer.

## 📚 Further Reading

- [Docker Compose Guide](DOCKER_COMPOSE_GUIDE.md) - Detailed Docker usage and troubleshooting
- [Local Setup Guide](LOCAL_SETUP.md) - Detailed local development setup
