import os
from typing import Dict, Union

import torch

from nvfuser import DataType


__all__ = [
    "cmake_prefix_path",
    "to_nvfuser_dtype",
]


cmake_prefix_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "nvfuser",
    "share",
    "cmake",
    "nvfuser",
)


_torch_to_nvfuser: Dict[torch.dtype, DataType] = {
    torch.float16: DataType.Half,
    torch.bfloat16: DataType.BFloat16,
    torch.float32: DataType.Float,
    torch.float64: DataType.Double,
    torch.int32: DataType.Int32,
    torch.int64: DataType.Int,
    torch.bool: DataType.Bool,
    torch.complex64: DataType.ComplexFloat,
    torch.complex128: DataType.ComplexDouble,
}


def to_nvfuser_dtype(dtype: Union[torch.dtype, DataType]) -> DataType:
    if isinstance(dtype, DataType):
        return dtype
    if isinstance(dtype, torch.dtype):
        return _torch_to_nvfuser[dtype]
