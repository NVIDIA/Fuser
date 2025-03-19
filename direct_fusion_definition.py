import torch
from nvfuser import fusion, DataType

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

def python_scalar_to_nvfuser_dtype(a):
    return _torch_dtype_to_nvfuser_dtype_map[type(a)]


def torch_dtype_to_nvfuser_dtype(dtype):
    """
    Translates from torch.dtype to nvFuser's DataType enum
    """
    return _torch_dtype_to_nvfuser_dtype_map[dtype]

class FusionDefinition:
    def __init__(self):
        self.profiled = False
        self.fusion = fusion.Fusion()
        self.fusion_guard = fusion.FusionGuard(self.fusion)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self

    def execute(self, inputs, *, device=None) -> list[torch.Tensor]:
        self.fec = fusion.FusionExecutorCache(self.fusion)
        return self.fec.execute(inputs)

    def from_pytorch(self, tensor, static_sizes=False):
        """
        Defines an nvfuser input tensor from a pytorch tensor and defaults
        to definining a symbolic tensor for dynamic shape usage.

        Args:
            tensor (torch.Tensor): Input tensor to nvFuser
            static_sizes (bool)  : Interprets sizes as static rather than
                                   as symbolic for dynamic shape usage

        Returns:
            nvfuser.Tensor
        """

        if not tensor.is_cuda and len(tensor.size()) != 0:
            raise ValueError("CPU non-scalar tensor is not supported!")

        tv = fusion.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
            is_cpu=tensor.is_cpu,
        )
        self.fusion.add_input(tv)
        return tv
