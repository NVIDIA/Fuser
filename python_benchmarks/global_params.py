import torch
from typing import Union, List, Tuple
from nvfuser import DataType
from .core import DEVICE_PROPERTIES
import numpy as np

# Model Parameters from LLMs (GPT2/3, PaLM, LLama)

# Embedding size: d_model, d_ff = 4 * d_model
D_MODEL_MIN = 768
D_MODEL_MAX = 18432

# Actual d_model sizes seen in models.
D_MODEL_SIZES = [
    768,  # GPT-2 (124M), GPT-3 (125M)
    1024,  # GPT-2 (350M), GPT-3 (350M)
    1280,  # GPT-2 (774M)
    1536,  # GPT-3 (760M), GPT-3 (760M)
    1600,  # GPT-2 (1558M)
    2048,  # GPT-3 (1.3B)
    2560,  # GPT-3 (2.7B)
    4096,  # LLaMA (7B), PaLM (8B), GPT-3 (6.7B)
    5120,  # LLaMA (13B), GPT-3 (13B)
    6656,  # LLaMA (30B)
    8192,  # LLaMA (65B), PaLM (63B)
    12288,  # GPT-3 (175B)
    18432,  # PaLM (540B)
]


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    for dim in dims:
        if dim == 2:
            batch_range = [2**i for i in range(4, 15)]  # {16, 16384}
            step_size = 256
            # max_hidden_size = 4 * d_model_max (max hidden size in feedforward layers)
            hidden_range = np.arange(
                D_MODEL_MIN, 4 * D_MODEL_MAX + 1, step_size
            )  # (768, 4*18432)
            inputs.extend([(i, j) for i in batch_range for j in hidden_range])
        elif dim == 3:
            dim_range = [2**i for i in range(1, 10)]
            inputs.extend(
                [(i, j, k) for i in dim_range for j in dim_range for k in dim_range]
            )
        elif dim == 4:
            # TODO: Add spatial_dim = 2.
            input_ranges = []

            batch_range = [2**i for i in range(1, 10)]  # {2, 512}
            channel_range = [2**i for i in range(1, 8)]  # {2, 128}
            spatial_range = [2**i for i in range(2, 7)]  # {4, 64}
            input_ranges.append((batch_range, channel_range, spatial_range))

            batch_range = [2**i for i in range(1, 7)]  # {2, 64}
            channel_range = [2**i for i in range(1, 6)]  # {2, 32}
            spatial_range = [128, 256]
            input_ranges.append((batch_range, channel_range, spatial_range))

            # Resnet/ResNext sizes
            batch_range = [2**i for i in range(5, 9)]  # {32, 256}
            channel_range = [2**i for i in range(6, 9)]  # {64, 256}
            spatial_range = [7 * 2**i for i in range(5)]  # {7, 112}
            input_ranges.append((batch_range, channel_range, spatial_range))

            for batch_range, channel_range, spatial_range in input_ranges:
                inputs.extend(
                    [
                        (n, c, hw, hw)
                        for n in batch_range
                        for c in channel_range
                        for hw in spatial_range
                    ]
                )

            inputs.extend(
                [
                    (n, c, hw, hw)
                    for n in [128, 256]
                    for (c, hw) in [
                        (512, 7),
                        (512, 14),
                        (512, 28),
                        (1024, 7),
                        (1024, 14),
                        (2048, 7),
                    ]
                ]
            )

        else:
            raise NotImplementedError(
                f"Generating input sizes of dimension {dim} is not implemented"
            )
    return inputs


# Datatypes to benchmark
FLOAT_DTYPES = [torch.float32]
# Run only one of float16 / bfloat16.
if DEVICE_PROPERTIES["gpu_compute_capability_major"] >= 8:
    FLOAT_DTYPES.append(torch.bfloat16)
else:
    FLOAT_DTYPES.append(torch.float16)

# Datatypes that will be promoted to Datatype.Float in Fusion Definitions
PROMOTE_DTYPES = [DataType.BFloat16, DataType.Half]
