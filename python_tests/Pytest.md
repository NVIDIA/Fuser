
# Pytest Opinfo Framework

## Usage

* Run tests with `pytest python_tests/pytest_def_tests.py`
* Filter tests with `-k` option. e.g., `pytest python_tests/pytest_def_tests.py -k var_mean`

## Dependencies
* pytest
* jax[cuda12_local] OR jax[cuda11_local]
See https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder.

## Code Organization
### Files modified When Adding a New Op
* `pytest_opinfos.py`: Each operation corresponds to an OpInfo object
* `pytest_input_generators.py`: A set of correctness and error input generators are needed to create test cases for each operation.
* `pytest_fd_functions.py` (Less Frequent): A specific operation might need a unique `FusionDefinition` function in order to test the new operation and that function would be added in this file.

### Structural Code Used By All Tests 
* `pytest_core.py`: Contains the defintion of the `Opinfo` object.
* `pytest_framework.py`: Contains the decorator template to iterate over all ops for a given test case.
* `pytest_def_tests.py`: Defines correctness and error tests for `FusionDefinition` `definition` operations.

### Misc
* `pytest_utils.py`: Common helper functions

