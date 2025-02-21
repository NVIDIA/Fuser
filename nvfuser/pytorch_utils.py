# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch

from ._C import DataType

import ctypes
import functools
import gc
from typing import Type, Union, Tuple

NumberTypeType = Union[Type[bool], Type[int], Type[float], Type[complex]]

_torch_dtype_to_nvfuser_dtype_map = {
    torch.cdouble: DataType.ComplexDouble,
    torch.cfloat: DataType.ComplexFloat,
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.bfloat16: DataType.BFloat16,
    torch.float8_e4m3fn: DataType.Float8_e4m3fn,
    torch.float8_e5m2: DataType.Float8_e5m2,
    torch.long: DataType.Int,
    torch.int: DataType.Int32,
    torch.bool: DataType.Bool,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}


def python_scalar_to_nvfuser_dtype(a: Union[int, float, complex, bool]):
    return _torch_dtype_to_nvfuser_dtype_map[type(a)]


def torch_dtype_to_nvfuser_dtype(dtype: Union[torch.dtype, NumberTypeType]):
    """
    Translates from torch.dtype to nvFuser's DataType enum
    """
    return _torch_dtype_to_nvfuser_dtype_map[dtype]


def get_device_properties() -> Tuple[int, float]:
    """
    Computes device properties using ctypes and cuda.
    Note: Consider using CUDA-Python when CUDA support >= 12.0.
    """
    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(libnames))

    # Device attribute enums (taken from cuda.h)
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3

    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39

    device_properties = {}
    device = torch.cuda.current_device()
    cuda_properties = torch.cuda.get_device_properties(device)

    device_properties["gpu_name"] = cuda_properties.name
    device_properties["gpu_compute_capability_major"] = cuda_properties.major
    device_properties["gpu_compute_capability_minor"] = cuda_properties.minor
    device_properties["gpu_gmem_bytes"] = cuda_properties.total_memory
    device_properties["gpu_sm_count"] = cuda_properties.multi_processor_count

    max_threads_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_threads_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device,
    )
    device_properties["gpu_max_threads_per_block"] = max_threads_per_block.value

    smem_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(smem_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        device,
    )
    device_properties["gpu_smem_bytes_per_block"] = smem_per_block.value

    max_reg_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_reg_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        device,
    )
    device_properties["gpu_regs_per_block"] = max_reg_per_block.value

    max_clock_khz = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_clock_khz),
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        device,
    )
    device_properties["gpu_clock_rate_khz"] = max_clock_khz.value

    l2_cache_size = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(l2_cache_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
    )
    device_properties["gpu_l2_bytes"] = l2_cache_size.value

    memory_clock_rate = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_clock_rate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
    )
    device_properties["gpu_mem_clock_khz"] = memory_clock_rate.value

    memory_bus_width = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_bus_width),
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        device,
    )
    device_properties["gpu_mem_bus_width"] = memory_bus_width.value

    max_threads_per_sm = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_threads_per_sm),
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        device,
    )
    device_properties["gpu_max_threads_per_sm"] = max_threads_per_sm.value

    # Compute peak bandwidth in GBps
    peak_bandwidth = (2 * memory_bus_width.value * memory_clock_rate.value) / (1e6 * 8)
    device_properties["gpu_peak_bandwidth_gbps"] = peak_bandwidth

    return device_properties


DEVICE_PROPERTIES = None
if torch.cuda.is_available():
    # Loading libraries will raise errors on non-CUDA machines.
    DEVICE_PROPERTIES = get_device_properties()


def retry_on_oom_or_skip_test(func):
    """Decorator: upon torch.OutOfMemoryError clear the cache and retry test"""

    @functools.wraps(func)
    def retried_func(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
        except torch.OutOfMemoryError:
            pass
        else:
            return output

        # We have hit an OOM error, so clear the cache and retry
        gc.collect()
        torch.cuda.empty_cache()

        try:
            output = func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            # If we hit an OOM this time, then skip the test
            import pytest

            pytest.skip(f"Test failed due to OutOfMemoryError: {e}")
            return

        return output

    return retried_func
