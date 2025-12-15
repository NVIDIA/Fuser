#!/bin/bash
# Start CUDA MPS with per-context SM partitioning for nvFuser
#
# Usage:
#   ./start_mps.sh        # Start MPS
#   ./start_mps.sh stop   # Stop MPS

set -e

if [ "$1" = "stop" ]; then
    echo "Stopping MPS..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "MPS stopped"
    exit 0
fi

# MPS configuration
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1

mkdir -p /tmp/nvidia-mps

echo "Starting MPS server..."
nvidia-cuda-mps-control -d

echo "MPS started. Usage:"
echo "  export NVFUSER_ENABLE=mps_sm_affinity(8)"
