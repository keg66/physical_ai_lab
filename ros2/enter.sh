#!/usr/bin/env bash
set -e

CONTAINER="physical-ai-ros2"

if ! docker ps \
    --filter "name=^/${CONTAINER}$" \
    --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "Container '${CONTAINER}' is not running."
  echo "Start it first with ./start.sh"
  exit 1
fi

docker exec -it "${CONTAINER}" bash
