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
    The baseline benchmarks use `compile` parameter:
        compile = false: Eager mode benchmark
        compile = true: torch.compile benchmark
    """
    run_eager = config.getoption("--benchmark-eager")
    run_thunder = config.getoption("--benchmark-thunder")
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

    if not run_thunder:
        skip_thunder = pytest.mark.skip(reason="need --benchmark-thunder option to run")
        for item in items:
            if "thunder" in item.nodeid:
                item.add_marker(skip_thunder)


def pytest_runtest_call(item: "Item") -> None:
    """Called to run the test for test item (the call phase).

    The default implementation calls ``item.runtest()``.
    """
    import torch

    retry = False
    try:
        item.runtest()
    except torch.OutOfMemoryError:
        retry = True

    if not retry:
        return

    # We have hit an OOM error, so clear the cache and retry
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    try:
        item.runtest()
    except torch.OutOfMemoryError as e:
        # If we hit an OOM this time, then skip the test
        import pytest

        pytest.skip(f"Test failed due to OutOfMemoryError: {e}")
