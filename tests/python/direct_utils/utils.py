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
        # Make sure the original and captured definitions match
        # torch.allclose does not work with fp8 datatype, so cast to fp64.
        match_values = all(
            [
                torch.allclose(
                    ref_out.to(torch.float64),
                    captured_outputs[idx].to(torch.float64),
                    equal_nan=True,
                )
                for idx, ref_out in enumerate(reference_outputs)
            ]
        )

        # Check that the stride of all outputs match
        match_stride = all(
            [
                ref_out.stride() == captured_outputs[idx].stride()
                for idx, ref_out in enumerate(reference_outputs)
            ]
        )
        return match_values and match_stride
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
