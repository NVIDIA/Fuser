# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
from nvfuser_direct import FusionDefinition, DataType, TensorView
from looseversion import LooseVersion


def microarchitecture_is(major, minor):
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major == major and prop.minor == minor


def microarchitecture_is_pre(major):
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < major


def is_pre_volta():
    return microarchitecture_is_pre(7)


def is_pre_ampere():
    return microarchitecture_is_pre(8)


def is_pre_hopper():
    return microarchitecture_is_pre(9)


def is_pre_blackwell():
    return microarchitecture_is_pre(10)


# 10.0 (B200 and GB200)
# 10.3 (B300 and GB300)
# 12.0 (RTX PRO 6000 and RTX 50XX)
# 12.1 (DGX Spark)
def is_blackwell():
    return (
        microarchitecture_is(10, 0)
        or microarchitecture_is(10, 3)
        or microarchitecture_is(12, 0)
        or microarchitecture_is(12, 1)
    )


# Get string representation for FusionDefinition
# Run captured python definition
# Check that the result of captured python definition matches original results
def check_captured_python_definition(reference_outputs, fd, inputs, device=None):
    try:
        fd_str = fd.__repr__()
        func_name = "nvfuser_fusion"
        exec(fd_str)

        # Execute the python definition that was captured
        with FusionDefinition() as fd_cap:
            eval(func_name)(fd_cap)

        torch.manual_seed(0)
        captured_outputs = fd_cap.execute(inputs, device=device)

        if len(reference_outputs) != len(captured_outputs):
            return False

        # Check that the values of all outputs match
        for ref_out, cap_out in zip(reference_outputs, captured_outputs):
            # torch.allclose does not work with fp8 datatype, so cast to fp64.
            # However, casting complex values to real discards the imaginary
            # part, so skip complex dtypes.
            # Similarly, packed fp4 dtype cannot be compared neither, we view
            # it as int8 and run comparison as-is.
            if ref_out.dtype == torch.float4_e2m1fn_x2:
                ref_out = ref_out.view(torch.int8)
            elif not ref_out.dtype.is_complex:
                ref_out = ref_out.to(torch.float64)
            if cap_out.dtype == torch.float4_e2m1fn_x2:
                cap_out = cap_out.view(torch.int8)
            elif not cap_out.dtype.is_complex:
                cap_out = cap_out.to(torch.float64)
            if not torch.allclose(ref_out, cap_out, equal_nan=True):
                return False

        # Check that the stride of all outputs match
        return all(
            [
                ref_out.stride() == cap_out.stride()
                for ref_out, cap_out in zip(reference_outputs, captured_outputs)
            ]
        )
    except Exception as err:
        print("\nException For Printed FusionDefinition:")
        print(
            "(A failure here suggests a mismatch in functionality between the original definition and the printed definition.)"
        )
        if "fd_str" in locals():
            print(fd_str)
        raise err


def verify_stride_order(output_strides, stride_order):
    sorted_stride = list(output_strides)
    rank = len(output_strides)
    for idx, axis in enumerate(stride_order):
        sorted_stride[rank - 1 - axis] = output_strides[idx]
    assert sorted(sorted_stride, reverse=True) == sorted_stride


UPDATED_SDPA = LooseVersion(torch.__version__) >= LooseVersion("2.7.0")


def define_sdpa_rng_state(fd: FusionDefinition) -> tuple[TensorView, TensorView]:
    dtype = DataType.UInt64 if UPDATED_SDPA else DataType.Int
    is_cpu = False if UPDATED_SDPA else True
    philox_shape = [2] if UPDATED_SDPA else []
    philox_seed = fd.define_tensor(
        shape=philox_shape,
        dtype=dtype,
        is_cpu=is_cpu,
    )
    philox_offset = fd.define_tensor(
        shape=[],
        dtype=dtype,
        is_cpu=is_cpu,
    )
    return philox_seed, philox_offset


def create_sdpa_rng_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.uint64 if UPDATED_SDPA else torch.int64
    device = "cuda" if UPDATED_SDPA else "cpu"
    philox_shape = (2,) if UPDATED_SDPA else ()
    philox_seed = torch.testing.make_tensor(philox_shape, device=device, dtype=dtype)
    philox_offset = torch.testing.make_tensor((), device=device, dtype=dtype)
    return philox_seed, philox_offset


def skip_if_global_memory_below_gb(min_gb: int, gpu_id: int = 0):
    device_properties = torch.cuda.get_device_properties(gpu_id)
    total_memory_bytes = device_properties.total_memory
    min_bytes = min_gb * (1024**3)

    if total_memory_bytes < min_bytes:
        pytest.skip(
            f"Insufficient GPU global memory: requires ~{min_bytes} B, "
            f"but only {total_memory_bytes} B available"
        )
