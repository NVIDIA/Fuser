# CUDA devices:
#  0: NVIDIA GeForce RTX 3090
#  1: NVIDIA GeForce RTX 3090
# torch version: 2.6.0a0+gitd622b49
# cuda version: 12.1
# nvfuser version: 0.2.22+git6912435
import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id28(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[5, 5],
        contiguity=[True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[5, 5],
        contiguity=[True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T2 = fd.define_tensor(
        shape=[5],
        contiguity=[True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[0],
    )
    T3 = fd.ops.linear(T0, T1, T2)
    fd.add_output(T3)


with FusionDefinition() as fd:
    nvfuser_fusion_id28(fd)

inputs = [
    torch.testing.make_tensor((5, 5), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((5, 5), dtype=torch.float32, device="cuda:0"),
    torch.testing.make_tensor((5,), dtype=torch.float32, device="cuda:0"),
]
fd.execute(inputs)
