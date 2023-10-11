import torch
from typing import Union, List

# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[tuple]:
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
            raise NotImplementedError(f'Generating input sizes of dimension {dim} is not implemented')
    return inputs

FLOAT_DTYPES = [torch.float16, torch.float32]
