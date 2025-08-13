# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.distributed as dist


def get_benchmark_fn(func, /, profile: bool):
    def wrapper(*args, **kwargs):
        dist.barrier()
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
# benchmarking. This way, one
# can collect stats of the first few non-warmup benchmark iterations using
# ```bash
# mpirun -np 1 nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:<iterations> pytest tests/python/multidevice/<test_file>.py -k <filter> --only-mpi : -np <processes - 1> pytest tests/python/multidevice/<test_file>.py -k <filter> --only-mpi
# ```
# and then display the stats using e.g. `nsys stats --report=cuda_gpu_kern_sum
# report1.nsys-rep`.
def get_benchmark_fns(func):
    return get_benchmark_fn(func, profile=False), get_benchmark_fn(func, profile=True)
