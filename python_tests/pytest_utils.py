# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch

# uint8, int8, int16, bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32
# but does not return the original type.
all_dtypes = [
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
]

# bf16, fp16 are disables because nvfuser upcasts those dtypes to fp32 but does not return the original type.
int_float_dtypes = [
    torch.int32,
    torch.int64,
    # torch.bfloat16,
    # torch.float16,
    torch.float32,
    torch.float64,
]

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
