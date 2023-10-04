import pytest
import torch

# Correctness Tests tolerance
RTOL = 1e-3
ATOL = 1e-3


# Common parameters for all tests
def generate_input_sizes():
    range_outer = [2**i for i in range(1, 5)]
    range_inner = [32 * 1024 * 2**i for i in range(11)]

    inputs = [(i, j) for i in range_outer for j in range_inner]
    inputs.extend([(j, i) for i in range_outer for j in range_inner])
    return inputs


INPUT_SIZES = generate_input_sizes()
DTYPES = [torch.float16, torch.float32]
TEST_CORRECTNESS = [True]

pytestmark = [
    pytest.mark.parametrize("size", INPUT_SIZES),
    pytest.mark.parametrize("dtype", DTYPES),
    pytest.mark.parametrize("test_correctness", TEST_CORRECTNESS),
]
