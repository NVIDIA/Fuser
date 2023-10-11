import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--disable-validation", action="store_true", default=False, help="Disable output validation in benchmarks."
    )
    parser.addoption(
        "--disable-benchmarking", action="store_true", default=False, help="Disable benchmarking."
    )

@pytest.fixture
def disable_validation(request):
    return request.config.getoption("--disable-validation")

@pytest.fixture
def disable_benchmarking(request):
    return request.config.getoption("--disable-benchmarking")
