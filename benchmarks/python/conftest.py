# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import BENCHMARK_CONFIG
from nvfuser.pytorch_utils import DEVICE_PROPERTIES


def pytest_addoption(parser):
    parser.addoption(
        "--disable-validation",
        action="store_true",
        help="Disable output validation in benchmarks.",
    )
    parser.addoption(
        "--disable-benchmarking",
        action="store_true",
        help="Disable benchmarking.",
    )
    parser.addoption(
        "--benchmark-eager",
        action="store_true",
        help="Benchmarks torch eager mode.",
    )
    parser.addoption(
        "--benchmark-thunder",
        action="store_true",
        help="Benchmarks thunder jit.",
    )
    parser.addoption(
        "--benchmark-torchcompile",
        action="store_true",
        help="Benchmarks torch.compile mode.",
    )

    # pytest-benchmark does not have CLI options to set rounds/warmup_rounds for benchmark.pedantic.
    # The following two options are used to overwrite the default values through CLI.
    parser.addoption(
        "--benchmark-rounds",
        action="store",
        default=10,
        help="Number of rounds for each benchmark.",
    )

    parser.addoption(
        "--benchmark-warmup-rounds",
        action="store",
        default=1,
        help="Number of warmup rounds for each benchmark.",
    )

    parser.addoption(
        "--benchmark-num-inputs",
        action="store",
        default=None,
        help="Number of inputs to randomly sample for each benchmark.",
    )


@pytest.fixture
def disable_validation(request):
    return request.config.getoption("--disable-validation")


@pytest.fixture
def disable_benchmarking(request):
    return request.config.getoption("--disable-benchmarking")


def pytest_make_parametrize_id(val, argname):
    if isinstance(val, tuple):
        return f'{argname}=[{"_".join(str(v) for v in val)}]'
    return f"{argname}={repr(val)}"


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info.update(DEVICE_PROPERTIES)


def pytest_configure(config):
    BENCHMARK_CONFIG["rounds"] = int(config.getoption("--benchmark-rounds"))
    BENCHMARK_CONFIG["warmup_rounds"] = int(
        config.getoption("--benchmark-warmup-rounds")
    )
    if config.getoption("--benchmark-num-inputs"):
        BENCHMARK_CONFIG["num_inputs"] = int(config.getoption("--benchmark-num-inputs"))
    config.addinivalue_line(
        "markers",
        "inner_outer_persistent: mark tests using inner_outer_persistent scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "inner_persistent: mark tests using inner_persistent scheduler if not being segmented.",
    )


def pytest_collection_modifyitems(session, config, items):
    """
    The executor parameter is used to run 'eager', 'torchcompile', 'thunder'
    based on the given CLI options. They are skipped by default currently.
    """

    skip_eager = pytest.mark.skip(reason="need --benchmark-eager option to run")
    skip_torchcompile = pytest.mark.skip(
            reason="need --benchmark-torchcompile option to run"
        )
    skip_thunder = pytest.mark.skip(reason="need --benchmark-thunder option to run")
    
    markers = []
    for executor in ["eager", "torchcompile", "thunder"]:
        if not config.getoption(f'--benchmark-{executor}'):
            markers.append(pytest.mark.skip(reason=f"need --benchmark-{executor} options to run."))
    
    for item in items:
        if (
            hasattr(item, "callspec")
            and "executor" in item.callspec.params
        ):
            for marker in markers:
                item.add_marker(marker)