<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->
# Unit Tests in tests/python/direct

This document provides a comprehensive overview of all unit tests located in the `tests/python/direct` directory of the nvFuser project.

## Test Files Overview

The `tests/python/direct` directory contains the following test files:

1. **test_python_direct.py** - Core nvFuser direct API tests
2. **test_python_frontend.py** - Python frontend functionality tests
3. **test_sdpa.py** - Scaled Dot-Product Attention (SDPA) tests
4. **test_cutlass_nvfp4_gemm.py** - CUTLASS nvFP4 GEMM tests
5. **test_import.py** - Import functionality tests
6. **test_repro.py** - Reproduction tests for specific issues
7. **conftest.py** - Test configuration and fixtures

## Comparison: Main vs Direct Frontend Tests

### Overview
- **Main frontend tests** (`tests/python/test_python_frontend.py`): 136 tests
- **Direct frontend tests** (`tests/python/direct/test_python_frontend.py`, `tests/python/direct/test_repro.py`): 74 tests
- **Tests only in main**: 77 tests
- **Tests only in direct**: 15 tests

#### Issue-Specific Test Migration

A total of **18 issue-specific tests** have been successfully migrated from the main frontend (`tests/python/test_python_frontend.py`) to the direct frontend (`tests/python/direct/test_repro.py`). These tests were selected based on their importance for verifying critical bug fixes and edge cases in nvFuser.

##### Migration Process
Each migrated test underwent the following adaptations:
1. **API Conversion**: Changed from `nvfuser` to `nvfuser_direct` imports
2. **Fixture Adaptation**: Updated to use `nvfuser_direct_test` pytest fixture
3. **Method Updates**: Changed `self.exec_nvfuser()` to `nvfuser_direct_test.exec_nvfuser()`
4. **Assertion Updates**: Changed `self.assertEqual()` to `nvfuser_direct_test.assertEqual()`
5. **Documentation**: Added comprehensive docstrings explaining test purpose and functionality
6. **Code Optimization**: Updated vector definitions to use direct list passing where appropriate

### Tests Only in Main Frontend (Not in Direct)

The following 31 tests exist in `tests/python/test_python_frontend.py` but are **NOT** present in `tests/python/direct/test_python_frontend.py`:

**Note**: The legacy frontend uses class-based tests (`def test_*(self):`) while the direct frontend uses standalone functions (`def test_*(nvfuser_direct_test):`). Only actual pytest test methods (with `self` parameter) are counted here. `test_cat_qwen2_v2` is the only pytest outside of `TestNvFuserFrontend` in `tests/python/test_python_frontend.py`.

#### Advanced Operations & Features

**General Tests -- Legacy-Only**
The following tests only exist in legacy frontend:
- `test_def_op_in_schedule` - Tests operation definition in schedules; scheduling and definition are not separate.
- `test_func_definition` - Tests function definition; Redundant
- `test_fusion_definition_error_cache` - Tests fusion definition error caching; No fusion cache
- `test_fusion_information` - Tests fusion information retrieval; Not used in direct bindings
- `test_debug_output` - Tests debug output functionality; Deprecated
- `test_compute_contiguity` - Tests contiguity computation; Not used in Thunder
- `test_static_tensor_sizes` - Tests static tensor sizes; Not used in Thunder
- `test_import_conflict_nvfuser_then_direct` - Tests import conflict handling; An analogous test already exists
- `test_repro_script_generation` - Tests reproduction script generation (130 lines); An analogous test already exists.
- `test_pad_cache` - Tests padding cache; No fusion cache
- `test_segmentation_reduction_pointwise_epilogue` - Tests segmented reduction; No segmentation support
- `test_fusion_profiler` - Tests fusion profiling; Cuda 13 incompatibility
- `test_fusion_profiler_user_schedule` - Tests user-defined fusion profiling; Cuda 13 incompatibility
- `test_fusion_profiler_with_noncodegen_kernels` - Tests profiling with non-codegen kernels; Cuda 13 incompatibility
- `test_cuda_code_and_scheduled_fusion_ir_strings` - Tests CUDA code generation (101 lines)
- `test_arithmetic_ops` - TODO: Tests __neg__ and __abs__ arithmetic operations

**General tests to add with more Than 50 Lines of Code:**
The following tests are complex and will be moved to tests/python/direct/test_high_complexity.py.
- `test_broadcast_in_dim_with_dynamic_shapes` - Tests broadcasting with dynamic shapes
- `test_cat_symbolic` - Tests symbolic concatenation
- `test_slice_error_checks` - Tests slice error checking
- `test_deterministic_random` - Tests deterministic random number generation
- `test_uniform_range` - Tests uniform range generation

- `test_cat_qwen2_v2` - Tests concatenation for Qwen2 v2 model (201 lines)
- `test_nanogpt_mha_dpa` - Tests NanoGPT multi-head attention
- `test_nanogpt_split_mha_linears` - Tests NanoGPT split MHA linear layers
- `test_prim_layer_norm_fwd` - Tests layer normalization forward pass (127 lines)
- `test_prim_rms_norm_fwd` - Tests RMS normalization forward pass (65 lines)

### Tests Only in Direct Frontend (Not in Main)

The following 12 tests exist in `tests/python/direct/test_python_frontend.py` but are **NOT** present in `tests/python/test_python_frontend.py`:

- `test_broadcast` - Tests basic broadcasting functionality; Maps to legacy `test_ops_broadcast`.
- `test_cast_scalar` - Tests scalar casting operations
- `test_cummax` - Tests cumulative maximum operations
- `test_cummin` - Tests cumulative minimum operations
- `test_cumprod` - Tests cumulative product operations
- `test_cumsum` - Tests cumulative sum operations
- `test_embedding` - Tests embedding operations
- `test_linear_with_bias` - Tests linear layers with bias
- `test_linear_without_bias` - Tests linear layers without bias
- `test_matmul` - Tests matrix multiplication
- `test_slice` - Tests tensor slicing operations; Maps to legacy `test_slice_api`
- `test_where` - Tests where operations; Maps to legacy `test_where_op`

### Shared Tests

Both test files contain these 80 common tests:
- `test_addcmul` - Addcmul operations
- `test_alias_output_to_input` - Output aliasing to input
- `test_all_dim_var_mean` - Tests variance and mean across all dimensions
- `test_bcast_squeeze_replace_aliased_output` - Tests broadcast squeeze with aliased output replacement; Tests issue 3833 with reshape and set operations
- `test_broadcast_and_stride_order` - Tests broadcast operations with specific stride order handling
- `test_broadcast_in_dim_with_dynamic_shapes` - Tests broadcasting with dynamic shapes (79 lines)
- `test_allocation_domain_concretization` - Tests allocation domain handling
- `test_allocation_domain_index_select` - Tests index select with allocation domains
- `test_basic` - Basic fusion operations
- `test_basic_fp16` - Basic operations with FP16
- `test_broadcast_mixing` - Broadcast mixing operations
- `test_cast_double_to_half` - Casting double to half precision
- `test_cast_fp8` - FP8 casting operations
- `test_cat` - Concatenation operations
- `test_complex_constants` - Tests complex number constants
- `test_complex_rsqrt` - Tests complex reciprocal square root
- `test_compute_tensor_descriptor` - Tests tensor descriptor computation
- `test_constant_nans` - Tests constant NaN handling
- `test_dynamic_reshape` - Dynamic reshape operations
- `test_empty_reshape` - Empty tensor reshape
- `test_execute_with_tuple_and_list` - Execution with tuple and list inputs
- `test_expand` - Tensor expansion operations
- `test_expand_to_zero` - Tests expansion to zero dimensions
- `test_expanded_bcast_tensor` - Tests expanded broadcast tensors
- `test_expanded_reduction` - Expanded reduction operations
- `test_explicit_broadcast_input` - Explicit broadcast input handling
- `test_gather` - Gather operations
- `test_gcd` - Tests greatest common divisor
- `test_implicit_broadcast_input` - Implicit broadcast input handling
- `test_index_select` - Index selection operations
- `test_index_select_scalar_indices` - Index selection with scalar indices
- `test_inplace_update_on_non_contiguous_inputs` - Tests in-place updates
- `test_input_scalar` - Tests scalar input handling
- `test_integer_division` - Tests integer division
- `test_iota` - Iota tensor generation
- `test_mark_alias_pass` - Tests alias marking
- `test_misaligned_add` - Tests misaligned addition
- `test_nextafter` - Tests nextafter function
- `test_normal` - Normal distribution generation
- `test_output_stride_order` - Output stride order handling
- `test_output_stride_order_with_reduction` - Output stride order with reduction
- `test_pad` - Padding operations
- `test_pad_dynamic` - Dynamic padding operations
- `test_pad_expanded_empty` - Tests padding with expanded empty tensors
- `test_pad_prior_cat` - Tests padding before concatenation
- `test_prod` - Tests product operations
- `test_promote_to_double` - Type promotion to double
- `test_random_distinct_values` - Tests random distinct value generation
- `test_real_imag` - Tests real and imaginary parts
- `test_reduction_complex_number` - Tests complex number reduction
- `test_replaced_sizes_pr2714` - Tests size replacement
- `test_reshape_dynamic` - Dynamic reshape functionality
- `test_reshape_squeeze_concretization` - Tests reshape squeeze concretization
- `test_returning_aliased_outputs` - Returning aliased outputs
- `test_right_shift_arithmetic` - Tests arithmetic right shift
- `test_right_shift_logical` - Tests logical right shift
- `test_right_shift_logical_sizeof_dtype` - Tests logical right shift with dtype size
- `test_scalar_only_inputs` - Scalar-only input operations
- `test_scatter_output_intermediate` - Scatter output intermediate operations
- `test_scatter_scalar_src` - Scatter scalar source operations
- `test_segment_set` - Tests segment set operations
- `test_select` - Tensor selection operations
- `test_signbit` - Tests sign bit operations
- `test_squeeze` - Tensor squeezing operations
- `test_stride_order_with_explicit_broadcast` - Tests stride order with explicit broadcast
- `test_sum_sliced_reshape_to_broadcast` - Tests sum sliced reshape to broadcast
- `test_take_along_axis` - Take along axis operations
- `test_tensor_ndim` - Tensor dimensionality
- `test_tensor_shape` - Tests tensor shape operations
- `test_tensor_shape_expand_bcast` - Tests tensor shape expansion broadcast
- `test_tensor_shape_nobcast` - Tests tensor shape without broadcast
- `test_tensor_shape_with_output_bcast` - Tests tensor shape with output broadcast
- `test_tensor_size_both_args_bcast` - Tests tensor size with broadcast arguments
- `test_triu` - Upper triangular operations
- `test_uniform` - Uniform distribution generation
- `test_var_correction` - Tests variance correction; Missing `var` operation
- `test_var_mean_correction` - Tests variance mean correction
- `test_welford` - Welford algorithm implementation
- `test_where_dtypes` - Where operations with different data types
- `test_zero_size_dim` - Tests zero size dimensions

### Additional Direct Frontend Test Files

#### test_repro.py (32 tests)
The following 19 issue-specific tests have been migrated from the main frontend to the direct frontend and are now available in `tests/python/direct/test_repro.py`:

- `test_issue1129` - Tests fix for issue 1129 (reshape and index_select with strided tensors)
- `test_issue1246` - Tests fix for issue 1246 (concatenation with empty tensors and strided tensors)
- `test_issue1270` - Tests fix for issue 1270 (empty tensors and dead code removal)
- `test_issue1273` - Tests fix for issue 1273 (squeeze of dynamic input handling)
- `test_issue1277` - Tests fix for issue 1277 (complex operations with strided tensors and slicing)
- `test_issue1279` - Tests fix for issue 1279 (var_mean operations with casting)
- `test_issue1310` - Tests fix for issue 1310 (input forwarding with multiple UnaryOps)
- `test_issue1393` - Tests fix for issue 1393 (complex operations with strided tensors and broadcasting)
- `test_issue1691` - Tests fix for issue 1691 (complex reduction operations with reshape and multiplication)
- `test_issue1706` - Tests fix for issue 1706 (complex operations derived from Llama2 network)
- `test_issue1872` - Tests fix for issue 1872 (full tensor creation with slice operations and casting)
- `test_issue1953` - Tests fix for issue 1953 (complex operations with strided tensors and multiple data types)
- `test_issue2275_repro1` - Tests fix for issue 2275 (unpadded concatenation operations with complex tensor manipulations); Maps to legacy `test_unpadded_catop_issue2275_repro1`
- `test_issue2275_repro2` - Tests fix for issue 2275 (unpadded concatenation operations with trigonometric functions); Maps to legacy `test_unpadded_catop_issue2275_repro2`
- `test_issue2317` - Tests fix for issue 2317 (reduction transpose scheduling); Maps to legacy `test_reduction_transpose_sched_issue2317`
- `test_issue2545` - Tests fix for issue 2545 (complex operations with empty tensors and concatenation); Maps to legacy `test_remove_empty_issue_2545`
- `test_issue2549` - Tests fix for issue 2549 (broadcast_in_dim and division operations); Maps to `test_fix_2549`
- `test_issue2755` - Tests fix for issue 2755 (slice operations with negation)
- `test_issue3292` - Tests fix for issue 3292 (complex tensor operations with manual normalization and padding)

The following 13 tests are from the original `tests/python/test_repro.py`.
- `test_domain_map_hang` - Tests domain mapping hang issue
- `test_issue4444` - Tests fix for issue 4444
- `test_issue4459` - Tests fix for issue 4459
- `test_issue4670` - Tests fix for issue 4670
- `test_loop_promotion_cyclic_war` - Tests loop promotion cyclic WAR
- `test_reduction_reference_missing_input_ids` - Tests reduction reference missing input IDs
- `test_reshape_cancellation` - Tests reshape cancellation
- `test_ws_tma_normalization1` - Tests workspace TMA normalization 1
- `test_ws_tma_normalization2` - Tests workspace TMA normalization 2
- `test_ws_tma_normalization3` - Tests workspace TMA normalization 3
- `test_ws_tma_normalization4` - Tests workspace TMA normalization 4
- `test_ws_tma_normalization5` - Tests workspace TMA normalization 5
- `test_ws_tma_normalization6` - Tests workspace TMA normalization 6

#### test_python_direct.py
Contains direct frontend specific functionality tests:
- `test_python_version_API` - Tests Python version API
- `test_fusion_not_defined` - Tests that `execute` raises exception when `Fusion` is not defined; Maps to legacy `test_no_definition`
- `test_fusion_empty` - Tests that `execute` raise exception when `Fusion` is empty; Maps to legacy `test_no_definition`
- `test_from_pytorch_fails_on_cpu_tensor` - Tests CPU tensor handling
- `test_define_tensor` - Tests tensor definition
- `test_enable_disable_options` - Tests enable/disable options
- `test_execute_with_different_device` - Tests execution with different devices
- `test_fusion_definition_print` - Tests fusion definition printing
- `test_fusion_execution_cache` - Tests fusion execution caching; Maps to legacy `test_selected_device`
- `test_repro_script_for` - Tests reproduction script generation; Maps to legacy `test_repro_script_generation`
- `test_enable_disable_options` - Tests enable/disable options for scheduler selection; Tests matmul scheduler vs expr_eval scheduler options
- `test_mismatched_input_types` - Tests mismatched input type handling

### Test Count Summary

**Important Note**: The legacy frontend uses class-based tests (`def test_*(self):`) while the direct frontend uses standalone functions (`def test_*(nvfuser_direct_test):`). Only actual pytest test methods (with `self` parameter) are counted for the legacy frontend.

- **Legacy Frontend**: 133 actual pytest test methods in `tests/python/test_python_frontend.py` (6 additional functions start with `test_` but are not pytest methods)
- **Direct Frontend**: 119 actual pytest test functions in `tests/python/direct/test_python_frontend.py`
- **Direct Repro**: 32 tests in `tests/python/direct/test_repro.py`
- **Direct Python**: 7 tests in `tests/python/direct/test_python_direct.py`
- **Total Direct Tests**: 153 tests across 3 files
- **Shared Tests**: 75 tests between legacy and direct frontend
- **Legacy-Only Tests**: 22 tests (not yet migrated)
- **Direct-Only Tests**: 12 tests (new functionality)

## Test Functions by File

### test_python_direct.py

Core tests for the nvFuser direct API functionality:

#### `test_fusion_definition_print()`
- **Purpose**: Tests fusion definition printing and string representation
- **Functionality**:
  - Creates a simple fusion with tensor addition
  - Tests fusion math representation output
  - Tests TensorView string representation
  - Tests TensorDomain string representation
  - Tests IterDomain string representation
  - Tests axis extents

#### `test_fusion_execution_cache()`
- **Purpose**: Tests fusion execution caching and compilation
- **Functionality**:
  - Tests fusion execution with tensor inputs
  - Tests compilation status verification
  - Tests scheduled IR representation
  - Tests CUDA kernel generation

#### `test_repro_script_for()`
- **Purpose**: Tests reproduction script generation
- **Functionality**:
  - Creates a fusion with multiple operations (add, mul, sum)
  - Tests generation of reproducible Python scripts
  - Verifies script contains expected fusion definition

#### `test_define_tensor()`
- **Purpose**: Tests tensor definition with dynamic shapes
- **Functionality**:
  - Tests tensor definition with stride order specification
  - Tests fusion execution with dynamic shapes
  - Tests tensor addition with dynamic inputs

#### `test_execute_with_different_device()`
- **Purpose**: Tests execution on different GPU devices
- **Requirements**: Multiple GPUs available
- **Functionality**:
  - Tests fusion execution on specific GPU device (cuda:1)
  - Tests device assignment verification

### test_python_frontend.py

Comprehensive tests for Python frontend operations:

#### `test_basic()`
- **Purpose**: Basic fusion operations test
- **Functionality**: Tests tensor addition, multiplication, and reduction operations

#### `test_basic_fp16()`
- **Purpose**: Tests operations with FP16 precision
- **Functionality**: Tests fusion with half-precision floating point

#### `test_cast_scalar()`
- **Purpose**: Tests scalar casting operations
- **Functionality**: Tests casting between different data types

#### `test_index_select_scalar_indices()`
- **Purpose**: Tests index selection with scalar indices
- **Functionality**: Tests tensor indexing operations

#### `test_select()`
- **Purpose**: Tests tensor selection operations
- **Functionality**: Tests selecting specific dimensions of tensors

#### `test_take_along_axis()`
- **Purpose**: Tests take-along-axis operations
- **Functionality**: Tests advanced tensor indexing

#### `test_addcmul()`
- **Purpose**: Tests addcmul operation
- **Functionality**: Tests fused add and multiply operations

#### `test_slice()`
- **Purpose**: Tests tensor slicing operations
- **Functionality**: Tests dynamic and static slicing with various parameters

#### `test_iota()`
- **Purpose**: Tests iota tensor generation
- **Functionality**: Tests creating tensors with sequential values

#### `test_scalar_only_inputs()`
- **Purpose**: Tests operations with scalar-only inputs
- **Functionality**: Tests fusion behavior with scalar inputs

#### `test_cat()`
- **Purpose**: Tests tensor concatenation
- **Functionality**: Tests concatenating tensors along different dimensions

#### `test_normal()`
- **Purpose**: Tests normal distribution generation
- **Functionality**: Tests random number generation with normal distribution

#### `test_uniform()`
- **Purpose**: Tests uniform distribution generation
- **Functionality**: Tests random number generation with uniform distribution

### test_sdpa.py

Tests for Scaled Dot-Product Attention functionality:

#### `test_softmax_logsumexp()`
- **Purpose**: Tests softmax logsumexp in SDPA
- **Requirements**: Ampere or newer GPU architecture
- **Functionality**: Tests flash attention logsumexp computation

#### `test_sdpa_fwd()`
- **Purpose**: Tests forward pass of SDPA
- **Requirements**: Ampere or newer GPU architecture
- **Functionality**: Tests scaled dot-product attention forward pass with various configurations

#### `test_sdpa_bwd()`
- **Purpose**: Tests backward pass of SDPA
- **Requirements**: Ampere or newer GPU architecture
- **Functionality**: Tests gradient computation for SDPA

#### `test_sdpa_fwd_bwd()`
- **Purpose**: Tests combined forward and backward SDPA
- **Requirements**: Ampere or newer GPU architecture
- **Functionality**: Tests end-to-end SDPA with gradient computation

### test_cutlass_nvfp4_gemm.py

Tests for CUTLASS nvFP4 GEMM operations:

#### `test_nvfp4_gemm()`
- **Purpose**: Tests nvFP4 quantized matrix multiplication
- **Requirements**: Compute capability 10.0 or above
- **Functionality**:
  - Tests quantized matrix multiplication with nvFP4 format
  - Tests various matrix shapes and data types (float16, bfloat16)
  - Tests scale factor handling and quantization

### test_import.py

Tests for import functionality:

#### `test_import_correct()`
- **Purpose**: Tests correct import of nvfuser_direct
- **Functionality**: Verifies successful import without errors

#### `test_import_conflict_direct_then_nvfuser()`
- **Purpose**: Tests import conflict handling
- **Functionality**: Tests warning generation when both nvfuser_direct and nvfuser are imported

### test_repro.py

Reproduction tests for specific issues. Tests primarily use fd.validate instead
of providing torch reference and using assertEqual.

## Test Configuration (conftest.py)

### `NVFuserTest` Class
- **Purpose**: Base test class with helper methods
- **Key Methods**:
  - `exec_nvfuser()`: Executes fusion functions and captures string definitions
  - `assertEqual()`: Custom assertion method for nvFuser outputs

### `nvfuser_direct_test` Fixture
- **Purpose**: Provides test fixture for nvFuser direct tests
- **Requirements**: Volta or newer GPU architecture
- **Functionality**: Provides configured test instance for all tests

## Test Categories

### Core Functionality Tests
- Fusion definition and execution
- Tensor operations and transformations
- Data type handling and casting
- Device management

### Advanced Operations Tests
- Scaled Dot-Product Attention (SDPA)
- Quantized operations (nvFP4)
- Random number generation
- Tensor indexing and slicing

### Integration Tests
- Import functionality
- Error handling and warnings
- Cross-module compatibility

### Performance Tests
- Caching and compilation verification
- CUDA kernel generation
- Memory management

## Requirements and Dependencies

### Hardware Requirements
- **Minimum**: Volta GPU architecture (for basic tests)
- **Recommended**: Ampere or newer (for SDPA tests)
- **Special**: Compute capability 10.0+ (for nvFP4 tests)
- **Optional**: Multiple GPUs (for device selection tests)

### Software Dependencies
- PyTorch with CUDA support
- nvFuser direct bindings
- pytest testing framework
- CUTLASS library (for GEMM tests)

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests in the directory
pytest tests/python/direct/

# Run specific test file
pytest tests/python/direct/test_python_direct.py

# Run specific test function
pytest tests/python/direct/test_python_direct.py::test_fusion_definition_print

# Run tests with specific markers
pytest tests/python/direct/ -m "not slow"
```

## Test Coverage

The test suite covers:
- ✅ Basic fusion operations
- ✅ Advanced tensor operations
- ✅ Data type conversions
- ✅ Device management
- ✅ SDPA functionality
- ✅ Quantized operations
- ✅ Import and compatibility
- ✅ Error handling
- ✅ Performance verification

## Notes

- Tests are designed to be deterministic and reproducible
- GPU architecture requirements are enforced through pytest markers
- Tests include both functional verification and performance validation
- Comprehensive error handling and edge case testing
- Support for both single and multi-GPU configurations
