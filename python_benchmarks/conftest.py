import pytest
from .core import DEVICE_PROPERTIES


def pytest_addoption(parser):
    parser.addoption(
        "--disable-validation",
        action="store_true",
        default=False,
        help="Disable output validation in benchmarks.",
    )
    parser.addoption(
        "--disable-benchmarking",
        action="store_true",
        default=False,
        help="Disable benchmarking.",
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
    for property, value in DEVICE_PROPERTIES.items():
        machine_info[property] = value
