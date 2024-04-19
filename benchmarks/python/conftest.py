# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import DEVICE_PROPERTIES


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


@pytest.fixture
def disable_validation(request):
    return request.config.getoption("--disable-validation")


@pytest.fixture
def disable_benchmarking(request):
    return request.config.getoption("--disable-benchmarking")


def pytest_make_parametrize_id(val):
    return repr(val)


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info.update(DEVICE_PROPERTIES)


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
