# Docker Compose Guide

This guide explains how to use Docker Compose to manage your video analysis project.

## Prerequisites

1. **Docker and Docker Compose** installed
2. **NVIDIA Container Toolkit** configured
3. **X11 server** for display (usually pre-installed on Linux desktops)

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

**Build and run:**
```bash
docker-compose up --build
```

**Run without rebuilding (if already built):**
```bash
docker-compose up
```

**Run in detached mode (background):**
```bash
docker-compose up -d
```

## Common Commands

### Build the Image

```bash
docker-compose build
```

### Start the Application

```bash
docker-compose up
```

### Stop the Application

Press `Ctrl+C` in the terminal, or if running in detached mode:
```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f
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

```bash
docker-compose up --build
```

### Remove Everything (Clean Slate)

```bash
docker-compose down --rmi all --volumes
```

## Advantages Over run.sh

✅ **No manual script management** - Docker Compose handles everything  
✅ **Development-friendly** - Source code is mounted, edit without rebuild  
✅ **Environment variables** - Managed via `.env` file  
✅ **Consistent configuration** - Single source of truth in `docker-compose.yml`  
✅ **Easy to extend** - Add databases, Redis, etc. as needed  
✅ **Better logging** - `docker-compose logs`  
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

1. Verify NVIDIA runtime:
   ```bash
   docker info | grep -i runtime
   ```

2. Test GPU access:
   ```bash
   docker-compose run --rm analise-video nvidia-smi
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

## Comparing with run.sh

| Feature | run.sh | docker-compose |
|---------|--------|----------------|
| Build image | ✅ Manual | ✅ Automatic |
| Run container | ✅ Manual | ✅ Automatic |
| GPU support | ✅ | ✅ |
| X11 display | ✅ | ✅ |
| Volume mounts | ⚠️ Limited | ✅ Full dev setup |
| Code hot-reload | ❌ | ✅ |
| Configuration management | ❌ | ✅ .env file |
| Service orchestration | ❌ | ✅ |
| Easier to maintain | ❌ | ✅ |

## Next Steps

You can now use standard Docker Compose commands instead of the `run.sh` script. The setup is more maintainable and development-friendly!
