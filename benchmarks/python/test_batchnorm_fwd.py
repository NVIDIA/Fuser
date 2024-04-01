import pytest
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES
from .normalization import norm_fwd_benchmark


@pytest.mark.parametrize("size", generate_input_sizes(dims=4))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_batchnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    channels_last: bool,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    norm_fwd_benchmark(
        benchmark,
        size,
        dtype,
        "batch_norm",
        channels_last,
        disable_validation,
        disable_benchmarking,
    )
