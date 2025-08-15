# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser_direct import FusionDefinition, DataType, TensorView  # noqa: F401
from looseversion import LooseVersion


def is_pre_volta():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_pre_ampere():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 8


def is_pre_hopper():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 9


def is_pre_blackwell():
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 10


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
            if not ref_out.dtype.is_complex:
                ref_out = ref_out.to(torch.float64)
            if not cap_out.dtype.is_complex:
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
