#!/bin/bash

BINARY="${BUILD_DIRECTORY}/experiment_matmul_cudaMalloc"

LOG2_M_VALUES=(2 6 10)
LOG2_N_VALUES=(2 6 10)
LOG2_K_VALUES=(2 6 10)
USE_STREAM_VALUES=(0 1) # Bool
USE_MATMUL_OUT_VALUES=(0 1) # Bool

./install-nsys.sh
NSYS=$(sudo which nsys)
NSYS_CMD="${NSYS} profile --stats=false -w true -t cublas,cuda,nvtx,osrt"

OUTPUT_DIRECTORY="nsight/profiles/hostname=$(hostname)_$(date '+%Y-%m-%d--%H-%M-%S')"
mkdir -p ${OUTPUT_DIRECTORY}

for LOG2_M in "${LOG2_M_VALUES[@]}"; do
    for LOG2_N in "${LOG2_N_VALUES[@]}"; do
        for LOG2_K in "${LOG2_K_VALUES[@]}"; do
            for USE_STREAM in "${USE_STREAM_VALUES[@]}"; do
                for USE_MATMUL_OUT in "${USE_MATMUL_OUT_VALUES[@]}"; do
                    FILE_NAME="LOG2_M=${LOG2_M}__LOG2_N=${LOG2_N}__LOG2_K=${LOG2_K}__USE_STREAM=${USE_STREAM}__USE_MATMUL_OUT=${USE_MATMUL_OUT}"
                    CMD="${NSYS_CMD}\
                          -o ${OUTPUT_DIRECTORY}/${FILE_NAME}\
                          ${BINARY} ${LOG2_M} ${LOG2_N} ${LOG2_K} ${USE_STREAM} ${USE_MATMUL_OUT}"
                    echo $CMD
                    $CMD
                done
            done
        done
    done
done
