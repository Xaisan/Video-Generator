#!/usr/bin/env fish
# ─── run.fish — Launch the Wan 2.2 I2V Web UI ─────────────────────
#
# Usage:
#   ./run.fish                              # Start web UI on port 7860
#   ./run.fish --build                      # Rebuild Docker image first
#   ./run.fish --port 8080                  # Use a custom port
#   ./run.fish --cli                        # Run the CLI generator instead
#   ./run.fish --cli -i input/photo.jpg     # CLI with overrides
#
# Prerequisites:
#   • Docker with AMD GPU passthrough (/dev/kfd, /dev/dri)
#   • Models downloaded to ./models/
#   • Input image in ./input/

set -l SCRIPT_DIR (dirname (status -f))
set -l IMAGE_NAME "wan22-i2v-amd"
set -l IMAGE_TAG "latest"

# Parse flags
set -l DO_BUILD 0
set -l CLI_MODE 0
set -l PORT 7860
set -l EXTRA_ARGS
set -l i 1
while test $i -le (count $argv)
    switch $argv[$i]
        case "--build"
            set DO_BUILD 1
        case "--cli"
            set CLI_MODE 1
        case "--port"
            set i (math $i + 1)
            set PORT $argv[$i]
        case "*"
            set EXTRA_ARGS $EXTRA_ARGS $argv[$i]
    end
    set i (math $i + 1)
end

# Build if requested or image doesn't exist
if test $DO_BUILD -eq 1; or not docker image inspect $IMAGE_NAME:$IMAGE_TAG >/dev/null 2>&1
    echo "🔨 Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    docker build -t $IMAGE_NAME:$IMAGE_TAG $SCRIPT_DIR
    if test $status -ne 0
        echo "❌ Docker build failed!"
        exit 1
    end
end

# Create output / sessions dirs
mkdir -p $SCRIPT_DIR/output $SCRIPT_DIR/input $SCRIPT_DIR/sessions

# Determine entrypoint
set -l ENTRYPOINT_ARGS
set -l RUN_FLAGS "-it"
if test $CLI_MODE -eq 1
    set ENTRYPOINT_ARGS --entrypoint python
    set EXTRA_ARGS generate.py $EXTRA_ARGS
    echo "🚀 Running Wan 2.2 I2V CLI generator (AMD ROCm)"
else
    echo "🌐 Starting Wan 2.2 I2V Web UI on port $PORT (AMD ROCm)"
    set RUN_FLAGS "-d"
end

# Run with AMD GPU passthrough
docker run --rm $RUN_FLAGS \
    --name wan22-i2v \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -p $PORT:7860 \
    -v $SCRIPT_DIR/models:/app/models:ro \
    -v $SCRIPT_DIR/input:/app/input \
    -v $SCRIPT_DIR/output:/app/output \
    -v $SCRIPT_DIR/sessions:/app/sessions \
    -v $SCRIPT_DIR/config.yaml:/app/config.yaml:ro \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --shm-size 16g \
    $ENTRYPOINT_ARGS \
    $IMAGE_NAME:$IMAGE_TAG \
    $EXTRA_ARGS

if test $CLI_MODE -eq 0
    echo "✅ Container started. Open http://localhost:$PORT"
    echo "   To view logs:  docker logs -f wan22-i2v"
    echo "   To stop:       docker stop wan22-i2v"
end
