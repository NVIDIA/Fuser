import torch
from typing import Union, List, Tuple
from nvfuser import Datatype


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    # TODO: Generate 3D input sizes.
    # TODO: Add more input sizes.
    for dim in dims:
        if dim == 2:
            range_outer = [2**i for i in range(1, 5)]
            range_inner = [32 * 1024 * 2**i for i in range(11)]
            inputs.extend([(i, j) for i in range_outer for j in range_inner])
            inputs.extend([(j, i) for i in range_outer for j in range_inner])
        else:
            raise NotImplementedError(
                f"Generating input sizes of dimension {dim} is not implemented"
            )
    return inputs


# Datatypes to benchmark
FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# Datatypes that will be promoted to Datatype.Float in Fusion Definitions
PROMOTE_DTYPES = [Datatype.Bfloat16, Datatype.Half]
