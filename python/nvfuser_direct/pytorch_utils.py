# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch

from ._C_DIRECT import DataType

from typing import Type, Union

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
    torch.float8_e8m0fnu: DataType.Float8_e8m0fnu,
    torch.long: DataType.Int,
    torch.int: DataType.Int32,
    torch.bool: DataType.Bool,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}

if hasattr(torch, "float4_e2m1fn_x2"):
    _torch_dtype_to_nvfuser_dtype_map[
        torch.float4_e2m1fn_x2
    ] = DataType.Float4_e2m1fn_x2


def torch_dtype_to_nvfuser_dtype(dtype: Union[torch.dtype, NumberTypeType]):
    """
    Translates from torch.dtype to nvFuser's DataType enum
    """
    return _torch_dtype_to_nvfuser_dtype_map[dtype]
