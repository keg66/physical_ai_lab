#!/usr/bin/env bash
set -euo pipefail

IMAGE="physical-ai-ros2:jazzy"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${IMAGE}..."
docker build \
  -t "${IMAGE}" \
  "${SCRIPT_DIR}"

echo "Built ${IMAGE}"