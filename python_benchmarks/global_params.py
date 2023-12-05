import torch
from typing import Union, List, Tuple
from nvfuser import DataType
from .core import DEVICE_PROPERTIES
import numpy as np

# Model Parameters from LLMs (GPT2/3, PaLM, LLama)

# Input sequence length
SEQ_LENGTH_MIN = 1024
SEQ_LENGTH_MAX = 16384
SEQ_LENGTH = [1024, 2048, 4096, 16384]
# Embedding size: d_model, d_ff = 4*d_model
D_MODEL_MIN = 768
D_MODEL_MAX = 18432
# Actual d_model sizes seen in models.
D_MODEL = [
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
D_FF = [4 * i for i in D_MODEL]

# Number of heads: n_head
N_HEAD_MIN = 12
N_HEAD_MAX = 96


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    for dim in dims:
        # TODO: Add sizes < 16 and > 1048576.
        if dim == 2:
            range_outer = [2**i for i in range(4, 9)]  # {16, 256}
            range_inner = [32 * 1024 * 2**i for i in range(6)]  # {32768, 1048576}
            inputs.extend([(i, j) for i in range_outer for j in range_inner])
            inputs.extend([(j, i) for i in range_outer for j in range_inner])
        elif dim == 3:
            # Limiting batch size to avoid OOM
            batch_range = [16]
            # Note: The granularity for sequence length and embedding size will vary for weekly vs nightly CI runs.
            embd_range = np.concatenate((D_MODEL, D_FF))

            inputs.extend(
                [(i, j, k) for i in batch_range for j in SEQ_LENGTH for k in embd_range]
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
