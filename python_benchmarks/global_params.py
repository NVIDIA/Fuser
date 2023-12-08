import torch
from typing import Union, List, Tuple
from nvfuser import DataType
from .core import DEVICE_PROPERTIES
import numpy as np

# Model Parameters from LLMs (GPT2/3, PaLM, LLama)

# Input sequence length
SEQ_LENGTH_MIN = 1024
SEQ_LENGTH_MAX = 16384
# Maximum sequence lengths in LLMs.
SEQ_LENGTHS = [1024, 2048, 4096, 16384]

# Embedding size: d_model, d_ff = 4 * d_model
D_MODEL_MIN = 768
D_MODEL_MAX = 18432

# Actual d_model sizes seen in models.
D_MODEL_SIZES = [
    768,
    1024,
    1280,
    1536,
    1600,
    2048,
    2560,
    4096,
    5120,
    5140,
    6656,
    8192,
    12288,
    18432,
]

# Number of heads: n_head
N_HEAD_MIN = 12
N_HEAD_MAX = 96


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    for dim in dims:
        if dim == 2:
            batch_range = [2**i for i in range(4, 15)] # {16, 16384}
            # max_hidden_size = 4 * d_model_max (max hidden size in feedforward layers)
            hidden_range = (np.arange(D_MODEL_MIN, 4*D_MODEL_MAX+1, 64)) # (768, 4*18432)
            inputs.extend([(i, j) for i in batch_range for j in hidden_range])  
        elif dim == 3:
            dim_range = [2**i for i in range(1, 10)]
            inputs.extend(
                [(i, j, k) for i in dim_range for j in dim_range for k in dim_range]
            )
        elif dim == 4:
            # TODO: Add spatial_dim = 2.
            batch_range = [2**i for i in range(6, 10)]  # {64, 512}
            channel_range = [2**i for i in range(5, 8)]  # {32, 128}
            spatial_range = [2**i for i in range(2, 7)]  # {4, 64}

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
            spatial_range = [2**i for i in range(2, 9)]  # {4, 256}

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

generate_input_sizes(dims=3)
