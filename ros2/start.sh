#!/usr/bin/env bash
set -e

IMAGE="physical-ai-ros2:jazzy"
CONTAINER="physical-ai-ros2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow local Docker containers to display X11 windows.
xhost +local:docker >/dev/null

docker run --rm -it \
  --name "${CONTAINER}" \
  --network host \
  --ipc=host \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${SCRIPT_DIR}/ws:/ws" \
  "${IMAGE}"
