# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from enum import auto, Enum


class Parallelism(Enum):
    # https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#tensor-parallelism
    TENSOR_PARALLEL = auto()
    # https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#sequence-parallelism
    SEQUENCE_PARALLEL = auto()


def get_benchmark_fn(func, /, profile: bool):
    def wrapper(*args, **kwargs):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return result

    return wrapper


# Returns two functors, the first with profiler off and the second with profiler
# on. The first functor is usually used for warmup and the second for actual
# benchmarking. This way, one can collect stats of non-warmup
# benchmark iterations using `nsys profile --capture-range=cudaProfilerApi`.
#
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html#handling-application-launchers-mpirun-deepspeed-etc
# has described several ways to profile multi-process applications.
#
# For single-node profiling, I recommend putting `nsys profile` before
# `mpirun`, e.g., `nsys profile ... mpirun -np 8 ...` instead of `mpirun -np 8
# nsys profile ...` or `mpirun -np 1 nsys profile ... : -np 7 ...`. This config
# tries to collect and align traces on different GPUs so it gives the most
# complete picture.  See
# https://github.com/NVIDIA/Fuser/pull/5751/files#r2663586669 for my
# experiment.
def get_benchmark_fns(func):
    return get_benchmark_fn(func, profile=False), get_benchmark_fn(func, profile=True)
