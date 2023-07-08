
# Pytest Opinfo Framework

* Run tests with `pytest python_tests/pytest_def_tests.py`
* Filter tests with `-k` option. e.g., `pytest python_tests/pytest_def_tests.py -k var_mean`

## Dependencies
* pytest
* jax[cuda12_local] OR jax[cuda11_local]
See https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder.
