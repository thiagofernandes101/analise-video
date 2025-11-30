# Docker Compose Guide

This guide explains how to use Docker Compose to manage your video analysis project.

## Prerequisites

1. **Docker and Docker Compose** installed
2. **NVIDIA Container Toolkit** configured (for GPU support)
3. **X11 server** for display (usually pre-installed on Linux desktops)
4. **NVIDIA GPU drivers** (optional, for GPU acceleration)

> **Note**: This project supports NVIDIA GPUs or CPU-only operation. AMD GPUs are not currently supported.

## Quick Start

### 1. Configure X11 Permissions (One-time)

```bash
xhost +local:docker
```

Or add to your `~/.bashrc` for automatic setup:
```bash
echo "xhost +local:docker 2>/dev/null" >> ~/.bashrc
```

### 2. Create Environment File (Optional)

```bash
cp .env.example .env
```

Edit `.env` if your DISPLAY is not `:0`.

### 3. Run the Application

**For NVIDIA GPU (recommended):**
```bash
docker compose up --build app-gpu
```

**For CPU-only (if no GPU available):**
```bash
docker compose up --build app-cpu
```

**Default (uses GPU service):**
```bash
docker compose up --build
```

## Service Options

The `docker-compose.yml` defines two services:

### `app-gpu` (Default)
Optimized for NVIDIA GPU acceleration:
- Installs PyTorch with CUDA 12.1 support
- Requires NVIDIA Container Toolkit
- Automatically uses GPU if available
- Significantly faster processing

### `app-cpu`
CPU-only fallback:
- Installs PyTorch CPU-only version
- Works on any machine without GPU
- Slower processing but fully functional
- Useful for development or testing

## Common Commands

### Build the Image

**GPU version:**
```bash
docker compose build app-gpu
```

**CPU version:**
```bash
docker compose build app-cpu
```

### Start the Application

**GPU (default):**
```bash
docker compose up app-gpu
```

**CPU:**
```bash
docker compose up app-cpu
```

### Stop the Application

Press `Ctrl+C` in the terminal, or if running in detached mode:
```bash
docker compose down
```

### View Logs

```bash
docker compose logs -f app-gpu
# or
docker compose logs -f app-cpu
```

### Run a One-off Command

```bash
docker-compose run --rm analise-video python -c "import torch; print(torch.cuda.is_available())"
```

### Rebuild After Code Changes

The `src/` directory is mounted as a volume, so **you don't need to rebuild** after editing Python files. Just restart:

```bash
docker-compose restart
```

### Rebuild After Dependency Changes

If you modify the Dockerfile or add new dependencies:

**GPU:**
```bash
docker compose up --build app-gpu
```

**CPU:**
```bash
docker compose up --build app-cpu
```

### Remove Everything (Clean Slate)

```bash
docker-compose down --rmi all --volumes
```

## Advantages of Docker Compose

✅ **No manual script management** - Docker Compose handles everything  
✅ **Development-friendly** - Source code is mounted, edit without rebuild  
✅ **Environment variables** - Managed via `.env` file  
✅ **Consistent configuration** - Single source of truth in `docker-compose.yml`  
✅ **Flexible deployment** - Easy switch between GPU and CPU modes  
✅ **Better logging** - `docker compose logs`  
✅ **Service orchestration** - Start/stop/restart services easily  

## How It Works

### Volume Mounts

The `docker-compose.yml` mounts several directories:

- **`./src:/app/src`** - Your source code (edit and restart, no rebuild needed)
- **`./videos:/app/videos`** - Video files
- **`~/.deepface:/root/.deepface`** - DeepFace model cache (persists between runs)
- **`/tmp/.X11-unix:/tmp/.X11-unix`** - X11 socket for display

### GPU Access

GPU access is configured via the `deploy` section:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Display Configuration

X11 display is configured through:
1. Environment variable: `DISPLAY=${DISPLAY}`
2. Volume mount: `/tmp/.X11-unix:/tmp/.X11-unix`
3. Network mode: `host` (allows access to X server)

## Troubleshooting

### Display not working

1. Check X11 permissions:
   ```bash
   xhost +local:docker
   ```

2. Verify DISPLAY variable:
   ```bash
   echo $DISPLAY
   ```

3. Check if X11 socket exists:
   ```bash
   ls /tmp/.X11-unix/
   ```

### GPU not detected

1. Verify you're using the GPU service:
   ```bash
   docker compose up app-gpu
   ```

2. Verify NVIDIA runtime:
   ```bash
   docker info | grep -i runtime
   ```

3. Test GPU access:
   ```bash
   docker compose run --rm app-gpu nvidia-smi
   ```

4. If GPU is unavailable, use CPU service:
   ```bash
   docker compose up app-cpu
   ```

### Permission errors

If you get permission errors with mounted volumes:
```bash
sudo chown -R $USER:$USER src/ videos/
```

## Development Workflow

### Typical workflow:

1. **Start the application:**
   ```bash
   docker-compose up
   ```

2. **Edit your code** in `src/` using your favorite editor

3. **Restart to apply changes:**
   ```bash
   docker-compose restart
   ```

4. **Stop when done:**
   ```bash
   docker-compose down
   ```

### For dependency changes:

1. **Edit `Dockerfile`** to add new packages

2. **Rebuild and run:**
   ```bash
   docker-compose up --build
   ```

## Next Steps

You can now use standard Docker Compose commands for all container management. The setup is maintainable and development-friendly!

For local development without Docker, see [LOCAL_SETUP.md](LOCAL_SETUP.md).
