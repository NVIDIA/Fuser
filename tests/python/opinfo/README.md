<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Pytest Opinfo Framework

## Usage

* Run tests: `pytest tests/python/opinfo/test_ops.py`
* Filter tests with `-k` option: `pytest tests/python/opinfo/test_ops.py -k var_mean`
* Show all possible tests: `pytest tests/python/opinfo/test_ops.py --collect-only`
* Filter all possible tests with `-k` option: `pytest tests/python/opinfo/test_ops.py --collect-only -k var_mean`

## Dependencies
* `pytest`
* `jax[cuda12_local]` OR `jax[cuda11_local]` See [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-locally-harder).

## Code Organization
### Files modified When Adding a New Op
* `opinfos.py`: Each operation corresponds to an OpInfo object
* `opinfo_input_generators.py`: A set of correctness and error input generators are needed to create test cases for each operation.
* `opinfo_fusion_definitions.py` (Less Frequent): A specific operation might need a unique `FusionDefinition` function in order to test the new operation and that function would be added in this file.

### Structural Code Used By All Tests
* `opinfo_core.py`: Contains the definition of the `Opinfo` object.
* `opinfo_framework.py`: Contains the decorator template to iterate over all ops for a given test case.
* `test_ops.py`: Defines correctness and error tests for `FusionDefinition` `definition` operations.

### Misc
* `opinfo_utils.py`: Common helper functions
