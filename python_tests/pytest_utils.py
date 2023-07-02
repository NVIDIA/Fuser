# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from torch.testing import make_tensor

# uint8, int8, int16, bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32
# but does not return the original type.
all_dtypes = (
    torch.bool,
    # torch.uint8,
    # torch.int8,
    # torch.int16,
    torch.int32,
    torch.int64,
    # torch.bfloat16,
    # torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

# bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32 but does not return the original type.
int_float_dtypes = (
    torch.int32,
    torch.int64,
    # torch.bfloat16,
    # torch.float16,
    torch.float32,
    torch.float64,
)

float_complex_dtypes = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

complex_dtypes = (torch.complex64, torch.complex128)

map_dtype_to_str = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}


def make_tensor_like(a):
    # type: (torch.Tensor) -> torch.Tensor
    """Returns a tensor with the same properties as the given tensor.

    Args:
        a (torch.Tensor): The tensor to copy properties from.

    Returns:
        torch.Tensor: A tensor with the same properties as :attr:`a`.
    """
    return torch.testing.make_tensor(
        a.shape, device=a.device, dtype=a.dtype, requires_grad=a.requires_grad
    )


def make_number(dtype: torch.dtype):
    """Returns a random number with desired dtype

    Args:
        dtype (torch.dtype): Desired dtype for number.

    Returns:
        (Scalar): The scalar number with specified dtype.
    """
    return make_tensor([1], device="cpu", dtype=dtype).item()


def find_nonmatching_dtype(dtype: torch.dtype):
    if dtype in int_float_dtypes:
        return torch.complex128
    elif dtype in complex_dtypes:
        return torch.double
    elif dtype is torch.bool:
        return torch.float32
    return None
