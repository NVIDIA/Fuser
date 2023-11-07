import torch
from typing import Union, List, Tuple
from nvfuser import DataType
from .core import DEVICE_PROPERTIES


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    # TODO: Add more input sizes.
    for dim in dims:
        if dim == 2:
            range_outer = [2**i for i in range(1, 5)]
            range_inner = [32 * 1024 * 2**i for i in range(11)]
            inputs.extend([(i, j) for i in range_outer for j in range_inner])
            inputs.extend([(j, i) for i in range_outer for j in range_inner])
        elif dim == 3:
            dim_range = [2**i for i in range(1, 10)]
            inputs.extend(
                [(i, j, k) for i in dim_range for j in dim_range for k in dim_range]
            )
        elif dim == 4:
            batch_range = [2**i for i in range(6, 10)]  # {64, 512}
            channel_range = [2**i for i in range(5, 8)]  # {32, 128}
            spatial_range = [2**i for i in range(2, 7)]  # {2, 64}

            inputs.extend(
                [
                    (i, j, k, l)
                    for i in batch_range
                    for j in channel_range
                    for k in spatial_range
                    for l in spatial_range
                ]
            )

            batch_range = [2**i for i in range(1, 7)]  # {2, 64}
            channel_range = [2**i for i in range(1, 6)]  # {2, 32}
            spatial_range = [2**i for i in range(2, 9)]  # {2, 256}

            inputs.extend(
                [
                    (i, j, k, l)
                    for i in batch_range
                    for j in channel_range
                    for k in spatial_range
                    for l in spatial_range
                ]
            )

        # TODO: Add ResNet/ResNext sizes.

        else:
            raise NotImplementedError(
                f"Generating input sizes of dimension {dim} is not implemented"
            )
    return inputs


# Datatypes to benchmark
FLOAT_DTYPES = [torch.float16, torch.float32]
if DEVICE_PROPERTIES["gpu_compute_capability_major"] >= 8:
    FLOAT_DTYPES.append(torch.bfloat16)

# Datatypes that will be promoted to Datatype.Float in Fusion Definitions
PROMOTE_DTYPES = [DataType.BFloat16, DataType.Half]
