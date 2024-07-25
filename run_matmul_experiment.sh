#!/bin/bash

BINARY="${BUILD_DIRECTORY}/experiment_matmul_cudaMalloc"

M_VALUES=(2 3 4)
N_VALUES=(3 4)
K_VALUES=(4 5)
USE_STREAM_VALUES=(0 1) # Bool
USE_MATMUL_OUT_VALUES=(0 1) # Bool

NSYS=$(sudo which nsys)
NSYS_CMD="${NSYS} profile --stats=false -w true -t cublas,cuda,nvtx,osrt"

OUTPUT_DIRECTORY="nsight/profiles/hostname=$(hostname)_$(date '+%Y-%m-%d--%H-%M-%S')"
mkdir -p ${OUTPUT_DIRECTORY}

for M in "${M_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
        for K in "${K_VALUES[@]}"; do
            for USE_STREAM in "${USE_STREAM_VALUES[@]}"; do
                for USE_MATMUL_OUT in "${USE_MATMUL_OUT_VALUES[@]}"; do
                    echo "Running with M=$M, N=$N, K=$K, use_stream=$USE_STREAM, use_matmul_out=$USE_MATMUL_OUT"
                    FILE_NAME="M=${M}__N=${N}__K=${K}__USE_STREAM=${USE_STREAM}__USE_MATMUL_OUT=${USE_MATMUL_OUT}"
                    CMD="${NSYS_CMD} \
                          -o ${OUTPUT_DIRECTORY}/${FILE_NAME} \
                          ${BINARY} ${M} ${N} ${K} ${USE_STREAM} ${USE_MATMUL_OUT}"
                    echo $CMD
                    $CMD
                done
            done
        done
    done
done
