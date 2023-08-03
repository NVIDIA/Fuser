
# Pytest Opinfo Framework

## Usage

* Run tests: `pytest python_tests/pytest_ops.py`
* Filter tests with `-k` option: `pytest python_tests/pytest_ops.py -k var_mean`
* Show all possible tests: `pytest python_tests/pytest_ops.py --collect-only`
* Filter all possible tests with `-k` option: `pytest python_tests/pytest_ops.py --collect-only -k var_mean`

## Dependencies
* `pytest`
* `jax[cuda12_local]` OR `jax[cuda11_local]` See [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder).

## Code Organization
### Files modified When Adding a New Op
* `pytest_opinfos.py`: Each operation corresponds to an OpInfo object
* `pytest_input_generators.py`: A set of correctness and error input generators are needed to create test cases for each operation.
* `pytest_fusion_definitions.py` (Less Frequent): A specific operation might need a unique `FusionDefinition` function in order to test the new operation and that function would be added in this file.

### Structural Code Used By All Tests
* `pytest_core.py`: Contains the defintion of the `Opinfo` object.
* `pytest_framework.py`: Contains the decorator template to iterate over all ops for a given test case.
* `pytest_ops.py`: Defines correctness and error tests for `FusionDefinition` `definition` operations.

### Misc
* `pytest_utils.py`: Common helper functions
