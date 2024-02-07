from __future__ import annotations
from nvfuser._C import DataType
import torch as torch
import typing
__all__ = ['DataType', 'NumberTypeType', 'patch_codegen_so', 'python_scalar_to_nvfuser_dtype', 'torch', 'torch_dtype_to_nvfuser_dtype']
def patch_codegen_so():
    """
    
        Replace libnvfuser_codegen.so installed along with torch
        
    """
def python_scalar_to_nvfuser_dtype(a: typing.Union[int, float, complex, bool]):
    ...
def torch_dtype_to_nvfuser_dtype(dtype: typing.Union[torch.dtype, typing.Type[bool], typing.Type[int], typing.Type[float], typing.Type[complex]]):
    """
    
        Translates from torch.dtype to nvFuser's DataType enum
        
    """
NumberTypeType: typing._UnionGenericAlias  # value = typing.Union[typing.Type[bool], typing.Type[int], typing.Type[float], typing.Type[complex]]
_torch_dtype_to_nvfuser_dtype_map: dict  # value = {torch.complex128: <DataType.ComplexDouble: 10>, torch.complex64: <DataType.ComplexFloat: 11>, torch.float64: <DataType.Double: 0>, torch.float32: <DataType.Float: 1>, torch.float16: <DataType.Half: 2>, torch.bfloat16: <DataType.BFloat16: 3>, torch.int64: <DataType.Int: 4>, torch.int32: <DataType.Int32: 5>, torch.bool: <DataType.Bool: 9>, complex: <DataType.ComplexDouble: 10>, float: <DataType.Double: 0>, int: <DataType.Int: 4>, bool: <DataType.Bool: 9>}
