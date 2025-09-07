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
2. **test_python_frontend.py** - Basic Python frontend functionality tests
3. **test_high_complexity.py** -- Python frontend tests greater than 50 LoC
4. **test_sdpa.py** - Scaled Dot-Product Attention (SDPA) tests
5. **test_cutlass_nvfp4_gemm.py** - CUTLASS nvFP4 GEMM tests
6. **test_import.py** - Import functionality tests
7. **test_repro.py** - Reproduction tests for specific issues
8. **conftest.py** - Test configuration and fixtures

## Test migration from legacy to direct bindings

### Overview
- **Legacy Frontend**: 134 actual pytest test methods in `tests/python/test_python_frontend.py`
- **Direct Frontend**: 122 actual pytest test functions in `tests/python/direct/test_python_frontend.py`; 83 from legacy tests
- **Direct Repro**: 32 tests in `tests/python/direct/test_repro.py`; 19 from legacy tests
- **Direct Python**: 11 tests in `tests/python/direct/test_python_direct.py`; 7 from legacy tests
- **Direct High Complexity**: 10 tests in `tests/python/direct/test_python_direct.py`; All from legacy tests
- **Legacy-Only Tests**: 15 tests (not yet migrated)
- **Total Direct Tests**: 175 tests across 4 files
- **Shared Tests**: 83 tests between legacy and direct frontend
- **Direct-Only Tests**: 39 tests (new functionality)

### Legacy-Only Tests (Not in Direct)
The following 15 tests only exist in legacy frontend:

- `test_def_op_in_schedule` - Tests operation definition in schedules; scheduling and definition are not separate.
- `test_func_definition` - Tests function definition; Redundant
- `test_fusion_definition_error_cache` - Tests fusion definition error caching; No fusion cache
- `test_fusion_information` - Tests fusion information retrieval; Not used in direct bindings
- `test_debug_output` - Tests debug output functionality; Deprecated
- `test_compute_contiguity` - Tests contiguity computation; Not used in Thunder
- `test_static_tensor_sizes` - Tests static tensor sizes; Not used in Thunder
- `test_import_conflict_nvfuser_then_direct` - Tests import conflict handling; An analogous test already exists
- `test_pad_cache` - Tests padding cache; No fusion cache
- `test_segmentation_reduction_pointwise_epilogue` - Tests segmented reduction; No segmentation support
- `test_fusion_profiler` - Tests fusion profiling; Cuda 13 incompatibility
- `test_fusion_profiler_user_schedule` - Tests user-defined fusion profiling; Cuda 13 incompatibility
- `test_fusion_profiler_with_noncodegen_kernels` - Tests profiling with non-codegen kernels; Cuda 13 incompatibility
- `test_cuda_code_and_scheduled_fusion_ir_strings` - Tests CUDA code generation (101 lines)
- `test_arithmetic_ops` - TODO: Tests __neg__ and __abs__ arithmetic operations

### Direct-Only Tests (Not in Legacy)
The following tests exist in `tests/python/direct/test_python_frontend.py` but are **NOT** present in `tests/python/test_python_frontend.py`:

- `test_cast_scalar` - Tests scalar casting operations
- `test_cummax` - Tests cumulative maximum operations
- `test_cummin` - Tests cumulative minimum operations
- `test_cumprod` - Tests cumulative product operations
- `test_embedding` - Tests embedding operations; 32 variants
- `test_linear_with_bias` - Tests linear layers with bias
- `test_linear_without_bias` - Tests linear layers without bias
- `test_matmul` - Tests matrix multiplication

### Shared Tests in `tests/python/direct/test_python_frontend.py` and `tests/python/test_python_frontend.py`

Both test files contain these 83 common tests:
- `test_addcmul` - Addcmul operations
- `test_alias_output_to_input` - Output aliasing to input
- `test_all_dim_var_mean` - Tests variance and mean across all dimensions
- `test_bcast_squeeze_replace_aliased_output` - Tests broadcast squeeze with aliased output replacement; Tests issue 3833 with reshape and set operations
- `test_broadcast` - Tests basic broadcasting functionality; Maps to legacy `test_ops_broadcast`.
- `test_broadcast_and_stride_order` - Tests broadcast operations with specific stride order handling
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
- `test_cumsum` - Tests cumulative sum operations
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
- `test_slice` - Tests tensor slicing operations; Maps to legacy `test_slice_api`
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
- `test_where` - Tests where operations; Maps to legacy `test_where_op`
- `test_where_dtypes` - Where operations with different data types
- `test_zero_size_dim` - Tests zero size dimensions

#### test_high_complexity.py
The following 10 tests are moved from `tests/python/test_python_frontend.py` to `tests/python/direct/test_high_complexity.py`:

- `test_broadcast_in_dim_with_dynamic_shapes` - Tests broadcasting with dynamic shapes
- `test_cat_symbolic` - Tests symbolic concatenation
- `test_slice_error_checks` - Tests slice error checking
- `test_deterministic_random` - Tests deterministic random number generation
- `test_uniform_range` - Tests uniform range generation
- `test_cat_qwen2_v2` - Tests concatenation for Qwen2 v2 model
- `test_nanogpt_mha_dpa` - Tests NanoGPT multi-head attention
- `test_nanogpt_split_mha_linears` - Tests NanoGPT split MHA linear layers
- `test_prim_layer_norm_fwd` - Tests layer normalization forward pass
- `test_prim_rms_norm_fwd` - Tests RMS normalization forward pass

#### test_repro.py
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

The following tests are from the original `tests/python/test_repro.py`.
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
- `test_python_version_API` - Tests Python version API; A legacy test
- `test_fusion_not_defined` - Tests that `execute` raises exception when `Fusion` is not defined; Maps to legacy `test_no_definition`
- `test_fusion_empty` - Tests that `execute` raise exception when `Fusion` is empty; Maps to legacy `test_no_definition`
- `test_from_pytorch_fails_on_cpu_tensor` - Tests CPU tensor handling; A legacy test
- `test_define_tensor` - Tests tensor definition
- `test_execute_with_different_device` - Tests execution with different devices; Maps to legacy `test_selected_device`
- `test_fusion_definition_print` - Tests fusion definition printing
- `test_fusion_execution_cache` - Tests fusion execution caching
- `test_repro_script_for` - Tests reproduction script generation; Maps to legacy `test_repro_script_generation`
- `test_enable_disable_options` - Tests enable/disable options for scheduler selection; A legacy test
- `test_mismatched_input_types` - Tests mismatched input type handling; A legacy Tests
