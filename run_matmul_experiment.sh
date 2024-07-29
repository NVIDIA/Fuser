# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

BINARY="${BUILD_DIRECTORY}/experiment_matmul_cudaMalloc"

LOG2_M_VALUES=(8)
LOG2_N_VALUES=(8)
LOG2_K_VALUES=(8)
STREAM_MODE_VALUES=("no_streams" "post_on_different_streams" "allocate_and_post_on_different_streams")
COMPUTE_MODE_VALUES=("matmul" "matmul_out" "unfused")
ENABLE_MEMORY_CACHE_VALUES=(0 1) # Bool

./install-nsys.sh
NSYS=$(sudo which nsys)
NSYS_CMD="${NSYS} profile --stats=false -w true -t cublas,cuda,nvtx,osrt"

OUTPUT_DIRECTORY="nsight/profiles/hostname=$(hostname)_$(date '+%Y-%m-%d--%H-%M-%S')"
mkdir -p ${OUTPUT_DIRECTORY}

for LOG2_M in "${LOG2_M_VALUES[@]}"; do
    for LOG2_N in "${LOG2_N_VALUES[@]}"; do
        for LOG2_K in "${LOG2_K_VALUES[@]}"; do
            for STREAM_MODE in "${STREAM_MODE_VALUES[@]}"; do
                for COMPUTE_MODE in "${COMPUTE_MODE_VALUES[@]}"; do
                    for ENABLE_MEMORY_CACHE in "${ENABLE_MEMORY_CACHE_VALUES[@]}"; do
                        FILE_NAME="LOG2_M=${LOG2_M}__LOG2_N=${LOG2_N}__LOG2_K=${LOG2_K}__STREAM_MODE=${STREAM_MODE}__COMPUTE_MODE=${COMPUTE_MODE}__ENABLE_MEMORY_CACHE=${ENABLE_MEMORY_CACHE}"
                        if [[ "$ENABLE_MEMORY_CACHE" -eq 1 ]]; then
                            export PYTORCH_NO_CUDA_MEMORY_CACHING=
                        else
                            export PYTORCH_NO_CUDA_MEMORY_CACHING=1
                        fi
                        CMD="${NSYS_CMD}\
                              -o ${OUTPUT_DIRECTORY}/${FILE_NAME}\
                              ${BINARY} ${LOG2_M} ${LOG2_N} ${LOG2_K} ${STREAM_MODE} ${COMPUTE_MODE}"
                        echo $CMD
                        $CMD
                    done
                done
            done
        done
    done
done
