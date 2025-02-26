# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .core import BENCHMARK_CONFIG
from nvfuser.pytorch_utils import DEVICE_PROPERTIES
from pytest import hookimpl, TestReport, Item, Parser
import multiprocessing as mp
import subprocess

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

def launch_benchmark(target_file, target_name: str):
    # with open(target_log, "w") as target_log_file:
    subprocess.run(
        [
            "pytest",
            f"{target_file}::{target_name}",
            "-vs"
        ],
        check=True,
        text=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.STDOUT
    )


def run_in_isolation(item) -> TestReport:
    process = mp.Process(
        target=launch_benchmark,
        args=(
            item.location[0],
            item.name,
        ),
    )
    process.start()
    process.join()

    # # Will mark skip as passed because pytest returns error only if there are failed tests.
    outcome = "failed" if process.exitcode != 0 else "passed"
    # target_filename = item.name.replace("/", "_")

    # if outcome == "passed":
    #     test_log = path.join(FAILED_BENCHMARK_LOGS_DIR, f"{target_filename}.log")
    #     os.remove(test_log)

    # benchmark_json = path.join(BENCHMARK_JSON_DIR, f"{target_filename}.json")
    # if outcome == "failed" or path.getsize(benchmark_json) == 0:
    #     os.remove(benchmark_json)

    return TestReport(item.nodeid, item.location, keywords=item.keywords, outcome=outcome, longrepr=None, when="call")


@hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # # If the option was not passed, let pytest manage the run.
    # if not item.config.getoption("--isolate-benchmarks"):
    #     return None

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    test_report = run_in_isolation(item)

    ihook.pytest_runtest_logreport(report=test_report)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True
def pytest_runtestloop(session):
    # global BENCHMARK_JSON_DIR, FAILED_BENCHMARK_LOGS_DIR

    # if not session.config.getoption("--isolate-benchmarks"):
    #     return None

    mp.set_start_method("spawn")

    # from _pytest.terminal import TerminalReporter

    # terminal: TerminalReporter = session.config.pluginmanager.get_plugin("terminalreporter")

    # custom_report_dir = os.getenv("THUNDER_BENCH_DIR")
    # BENCHMARK_JSON_DIR = custom_report_dir if custom_report_dir else BENCHMARK_JSON_DIR

    # os.makedirs(BENCHMARK_JSON_DIR, exist_ok=True)
    # os.makedirs(FAILED_BENCHMARK_LOGS_DIR, exist_ok=True)

    # terminal.write_line(f"Saving failed benchmarks logs in {FAILED_BENCHMARK_LOGS_DIR}")
    # terminal.write_line(f"Saving benchmarks reports in {BENCHMARK_JSON_DIR}")