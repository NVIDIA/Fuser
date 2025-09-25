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
    parser.addoption(
        "--benchmark-thunder-torchcompile",
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

    parser.addoption(
        "--with-nsys",
        action="store_true",
        default=False,
        help="Run benchmark scripts with nsys. Disable all other profilers.",
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
    BENCHMARK_CONFIG["with_nsys"] = config.getoption("--with-nsys")

    if config.getoption("--benchmark-num-inputs"):
        BENCHMARK_CONFIG["num_inputs"] = int(config.getoption("--benchmark-num-inputs"))

    # Scheduler markers may become stale and are not 100% accurate.
    config.addinivalue_line(
        "markers",
        "inner_outer_persistent: mark tests using inner_outer_persistent scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "inner_persistent: mark tests using inner_persistent scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "outer_persistent: mark tests using outer_persistent scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "reduction: mark tests using reduction scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "matmul: mark tests using matmul scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "resize: mark tests using resize scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "transpose: mark tests using transpose scheduler if not being segmented.",
    )
    config.addinivalue_line(
        "markers",
        "pointwise: mark tests using pointwise scheduler if not being segmented.",
    )


def pytest_collection_modifyitems(session, config, items):
    """
    The baseline benchmarks use `executor` parameter with
    values ["eager", "torchcompile", "thunder", "thunder-torchcompile"] that are optionally
    run using `--benchmark-{executor}` flag. They are skipped by
    default.
    """

    from nvfuser.pytorch_utils import retry_on_oom_or_skip_test

    executors = ["eager", "torchcompile", "thunder", "thunder-torchcompile"]

    def get_test_executor(item) -> str | None:
        if hasattr(item, "callspec") and "executor" in item.callspec.params:
            test_executor = item.callspec.params["executor"]
            assert (
                test_executor in executors
            ), f"Expected executor to be one of 'eager', 'torchcompile', 'thunder', 'thunder-torchcompile', found {test_executor}."
            return test_executor
        return None

    executors_to_skip = []

    for executor in executors:
        if not config.getoption(f"--benchmark-{executor}"):
            executors_to_skip.append(executor)

    for item in items:
        item.obj = retry_on_oom_or_skip_test(item.obj)

        test_executor = get_test_executor(item)

        if test_executor is not None and test_executor in executors_to_skip:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"need --benchmark-{test_executor} option to run."
                )
            )
