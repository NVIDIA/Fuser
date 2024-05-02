# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import DEVICE_PROPERTIES, BENCHMARK_CONFIG


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
    config.addinivalue_line(
        "markers", "inner_outer_persistent: mark tests using inner_outer_persistent scheduler."
    )

def pytest_collection_modifyitems(session, config, items):
    """
    The baseline benchmarks use `compile` parameter:
        compile = false: Eager mode benchmark
        compile = true: torch.compile benchmark
    """
    run_eager = config.getoption("--benchmark-eager")
    run_torchcompile = config.getoption("--benchmark-torchcompile")

    if not run_eager:
        skip_eager = pytest.mark.skip(reason="need --benchmark-eager option to run")
        for item in items:
            # If the benchmark has compile=False parameter (eager mode), skip it.
            if (
                hasattr(item, "callspec")
                and "compile" in item.callspec.params
                and not item.callspec.params["compile"]
            ):
                item.add_marker(skip_eager)

    if not run_torchcompile:
        skip_torchcompile = pytest.mark.skip(
            reason="need --benchmark-torchcompile option to run"
        )
        for item in items:
            # If the benchmark has compile=True parameter (torch.compile mode), skip it.
            if (
                hasattr(item, "callspec")
                and "compile" in item.callspec.params
                and item.callspec.params["compile"]
            ):
                item.add_marker(skip_torchcompile)
