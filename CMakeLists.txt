# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nvfuser)
enable_language(CUDA)

cmake_policy(SET CMP0063 NEW) # make symbol visibility always apply

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NVFUSER_ROOT ${PROJECT_SOURCE_DIR})
set(NVFUSER_SRCS_DIR "${NVFUSER_ROOT}/csrc")
set(NVFUSER_PYTHON_DIR "${NVFUSER_ROOT}/python")
set(NVFUSER_PYTHON_BINDINGS "${NVFUSER_ROOT}/python/python_frontend")
set(NVFUSER_PYTHON_COMMON "${NVFUSER_ROOT}/python/python_common")
set(NVFUSER_PYTHON_DIRECT_BINDINGS "${NVFUSER_ROOT}/python/python_direct")
set(NVFUSER_CUTLASS "${NVFUSER_ROOT}/cutlass")
set(NVFUSER_THIRD_PARTY_DIR "${NVFUSER_ROOT}/third_party")

option(NVFUSER_STANDALONE_BUILD_WITH_UCC "" OFF)
option(NVFUSER_EXPLICIT_ERROR_CHECK "" OFF)

if(NVFUSER_EXPLICIT_ERROR_CHECK)
  add_compile_definitions(NVFUSER_EXPLICIT_ERROR_CHECK)
endif()

option(NVFUSER_BUILD_WITH_ASAN "Build nvFuser with asan" OFF)

include(CMakeDependentOption)
cmake_dependent_option(NVFUSER_DISTRIBUTED "" ON "USE_DISTRIBUTED" OFF)

if(NVFUSER_DISTRIBUTED)
  add_compile_definitions(NVFUSER_DISTRIBUTED)
endif()

message(STATUS "Setting NVFUSER_DISTRIBUTED=${NVFUSER_DISTRIBUTED}")

# We try to update which C++ standard we use together in lockstep across all
# built libraries, and these variables control which that is. Generally we are
# on C++20, but we still support a version of CUDA (11) that does not recognize
# C++20 and so we drop back to 17 there. Also, we allow all of these to be
# overridden by the user.
# Note we do not use a global set_property on e.g. CXX_STANDARD. CMake globals
# are footguns that should generally be avoided, because they are difficult to
# target where and *only* where they are needed. See e.g.:
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/dodonot.html
set(NVFUSER_C_STANDARD 20 CACHE STRING "C standard to use for C code")
set(NVFUSER_CPP_STANDARD 20 CACHE STRING "C++ standard to use for C++ code")
set(NVFUSER_CUDA_STANDARD 17 CACHE STRING "C++ standard to use for CUDA code")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # TODO: gcc 11.4 has been end of life according to https://gcc.gnu.org/
  # I believe we should bump up the version below to 12.x.
  # However, because gcc 11.4 is well tested and stable, let's defer this
  # rejection until the day that we find a bug in gcc 11.4.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.4)
    message(FATAL_ERROR "GCC < 11.4 has compiler bugs and can not compile nvFuser.")
  endif()
endif()

string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")

find_package(Torch REQUIRED)
find_package(Python REQUIRED Development.Module Interpreter)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# need this since the pytorch execution uses a different name
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# CXX flags is necessary since https://github.com/pytorch/pytorch/issues/98093
string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
include(cmake/FlatBuffers.cmake)
include(cmake/Dependencies.cmake)

# set CUDA_ARCH for cu tests.
if(TORCH_CUDA_ARCH_LIST)
  set(ARCH_FLAGS)
  cuda_select_nvcc_arch_flags(ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib/dynamic_type)

set(CUTLASS_STATUS "N/A")
if(BUILD_CUTLASS)
  enable_language(CUDA)

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(WARNING "Skip building CUTLASS because of incompatible CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
    set(CUTLASS_STATUS "DISABLED")
  else()
    add_compile_definitions(NVFUSER_ENABLE_CUTLASS)
    set(CUTLASS_STATUS "ENABLED")

    find_package(CUDAToolkit REQUIRED)
    include(FetchContent)

    # cutlass
    FetchContent_Declare(
      repo-cutlass
      GIT_REPOSITORY https://github.com/NVIDIA/cutlass
      GIT_TAG        f115c3f85467d5d9619119d1dbeb9c03c3d73864
      GIT_SHALLOW    OFF
    )
    FetchContent_Populate(repo-cutlass)

    include(ProcessorCount)
    ProcessorCount(NPROC)

    set(NVF_CUTLASS_CUDA_FLAGS
      "-DCUTE_USE_PACKED_TUPLE=1"
      "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
      "-DCUTLASS_VERSIONS_GENERATED"
      "-DCUTLASS_TEST_LEVEL=0"
      "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
      "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
      "--expt-relaxed-constexpr"
      "--expt-extended-lambda"
      "--threads=${NPROC}"
      # -----------------
      # Suppress warnings
      # -----------------
      "-Xcompiler=-Wconversion"
      "-Xcompiler=-fno-strict-aliasing"
      # CUDA 13 has deprecated old vector types such as ulong4: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0
      "-Xcompiler=-Wno-deprecated-declarations"
    )

    set(NVFUSER_CUTLASS_SRCS)
    list(APPEND NVFUSER_CUTLASS_SRCS
	${NVFUSER_CUTLASS}/nvfp4_scaled_mm.cu
        ${NVFUSER_CUTLASS}/nvf_cutlass.cpp
        ${NVFUSER_CUTLASS}/cutlass_utils.cpp
    )
    add_library(nvf_cutlass SHARED ${NVFUSER_CUTLASS_SRCS})

    target_include_directories(nvf_cutlass PRIVATE ${repo-cutlass_SOURCE_DIR}/include)
    target_include_directories(nvf_cutlass PRIVATE ${repo-cutlass_SOURCE_DIR}/tools/util/include)
    target_compile_options(nvf_cutlass PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVF_CUTLASS_CUDA_FLAGS}>)
    if(NOT MSVC)
      set(NVF_LIB_SUFFIX ".so")
    else()
      set(NVF_LIB_SUFFIX ".pyd")
    endif()

    target_include_directories(nvf_cutlass PUBLIC
      "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
      "$<BUILD_INTERFACE:${NVFUSER_CUTLASS}>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
    )
    target_link_libraries(nvf_cutlass PRIVATE "${TORCH_LIBRARIES}" c10)
    set_target_properties(nvf_cutlass PROPERTIES
      C_STANDARD ${NVFUSER_C_STANDARD}
      CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
      CXX_STANDARD ${NVFUSER_CPP_STANDARD}
      CXX_STANDARD_REQUIRED ON
      CXX_VISIBILITY_PRESET hidden
      INSTALL_RPATH
      "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib"
      POSITION_INDEPENDENT_CODE Yes
      VISIBILITY_INLINES_HIDDEN Yes
      CUDA_ARCHITECTURES "100a"
    )
    install(TARGETS nvf_cutlass EXPORT NvfuserTargets DESTINATION lib)

  endif()
endif()

# ------------------------------
# build nvfuser_codegen library
# ------------------------------

# nvfuser codegen sources
set(NVFUSER_SRCS)
list(APPEND NVFUSER_SRCS
  ${NVFUSER_SRCS_DIR}/alias_analysis.cpp
  ${NVFUSER_SRCS_DIR}/codegen.cpp
  ${NVFUSER_SRCS_DIR}/compute_at.cpp
  ${NVFUSER_SRCS_DIR}/compute_at_map.cpp
  ${NVFUSER_SRCS_DIR}/contiguity.cpp
  ${NVFUSER_SRCS_DIR}/debug.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/bank_conflict.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/circular_buffer.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/device_version.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/divisible_split.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/fused_reduction.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/fusion_info.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/index_compute.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/non_divisible_split.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/padded_parallel_dimensions.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/predicate_elimination.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/sync_information.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/tensor_memory.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/tensor_producer_aliases.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/thread_predicate.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/tma.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/trivial_broadcast.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/lower2device.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/alias_memory.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/allocation.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/circular_buffer.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/expr_sort.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/fusion_simplifier.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/grid_serialization.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/index.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/inline_ptx.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/inplace_alias.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/insert_syncs.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/instrument.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/loop_rotation.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/loops.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/magic_zero.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/predicate.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/replace_size.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/rng.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/scalar_hoist.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/unroll.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/vectorize_welford.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/warp_reduce.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/utils.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/validation.cpp
  ${NVFUSER_SRCS_DIR}/dispatch.cpp
  ${NVFUSER_SRCS_DIR}/driver_api.cpp
  ${NVFUSER_SRCS_DIR}/dynamic_transform.cpp
  ${NVFUSER_SRCS_DIR}/evaluator_common.cpp
  ${NVFUSER_SRCS_DIR}/exceptions.cpp
  ${NVFUSER_SRCS_DIR}/expr_evaluator.cpp
  ${NVFUSER_SRCS_DIR}/expr_simplifier.cpp
  ${NVFUSER_SRCS_DIR}/fusion.cpp
  ${NVFUSER_SRCS_DIR}/fusion_guard.cpp
  ${NVFUSER_SRCS_DIR}/fusion_segmenter.cpp
  ${NVFUSER_SRCS_DIR}/global_allocator.cpp
  ${NVFUSER_SRCS_DIR}/grouped_reduction.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/container.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/executor.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/host_ir.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/lower.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/lower_to_communication.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/lowering.cpp
  ${NVFUSER_SRCS_DIR}/id_model/circular_buffer_indexing.cpp
  ${NVFUSER_SRCS_DIR}/id_model/contiguity.cpp
  ${NVFUSER_SRCS_DIR}/id_model/id_model.cpp
  ${NVFUSER_SRCS_DIR}/id_model/id_model_index_compute.cpp
  ${NVFUSER_SRCS_DIR}/id_model/indexing.cpp
  ${NVFUSER_SRCS_DIR}/id_model/indexing_traversal.cpp
  ${NVFUSER_SRCS_DIR}/id_model/loop_promotion.cpp
  ${NVFUSER_SRCS_DIR}/id_model/predicate_indexing.cpp
  ${NVFUSER_SRCS_DIR}/id_model/schedule.cpp
  ${NVFUSER_SRCS_DIR}/id_model/to_string.cpp
  ${NVFUSER_SRCS_DIR}/id_model/transform_replay.cpp
  ${NVFUSER_SRCS_DIR}/id_model/validation_utils.cpp
  ${NVFUSER_SRCS_DIR}/index_compute.cpp
  ${NVFUSER_SRCS_DIR}/instrumentation.cpp
  ${NVFUSER_SRCS_DIR}/interval_analysis.cpp
  ${NVFUSER_SRCS_DIR}/ir/allocation_utils.cpp
  ${NVFUSER_SRCS_DIR}/ir/base_nodes.cpp
  ${NVFUSER_SRCS_DIR}/ir/builder.cpp
  ${NVFUSER_SRCS_DIR}/ir/cloner.cpp
  ${NVFUSER_SRCS_DIR}/ir/container.cpp
  ${NVFUSER_SRCS_DIR}/ir/graphviz.cpp
  ${NVFUSER_SRCS_DIR}/ir/iostream.cpp
  ${NVFUSER_SRCS_DIR}/ir/nodes.cpp
  ${NVFUSER_SRCS_DIR}/ir/utils.cpp
  ${NVFUSER_SRCS_DIR}/iter_visitor.cpp
  ${NVFUSER_SRCS_DIR}/kernel.cpp
  ${NVFUSER_SRCS_DIR}/kernel_db/kernel_db.cpp
  ${NVFUSER_SRCS_DIR}/kernel_db/utils.cpp
  ${NVFUSER_SRCS_DIR}/kernel_ir.cpp
  ${NVFUSER_SRCS_DIR}/kernel_ir_dispatch.cpp
  ${NVFUSER_SRCS_DIR}/logical_domain_map.cpp
  ${NVFUSER_SRCS_DIR}/mma_type.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/communication.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/communicator.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/cuda_p2p.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/ipc_handle.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/device_mesh.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/executor.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/utils.cpp
  ${NVFUSER_SRCS_DIR}/mutator.cpp
  ${NVFUSER_SRCS_DIR}/ops/alias.cpp
  ${NVFUSER_SRCS_DIR}/ops/arith.cpp
  ${NVFUSER_SRCS_DIR}/ops/composite.cpp
  ${NVFUSER_SRCS_DIR}/ops/indexing.cpp
  ${NVFUSER_SRCS_DIR}/ops/normalization.cpp
  ${NVFUSER_SRCS_DIR}/ops/utils.cpp
  ${NVFUSER_SRCS_DIR}/options.cpp
  ${NVFUSER_SRCS_DIR}/parallel_dimension_map.cpp
  ${NVFUSER_SRCS_DIR}/parallel_type_bitmap.cpp
  ${NVFUSER_SRCS_DIR}/polymorphic_value.cpp
  ${NVFUSER_SRCS_DIR}/predicate_compute.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/add_axioms.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/allocation_order_inference.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/consecutive_cast.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/exact_mapped_extent_substitution.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/decompose_reshardings.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/finalize_multidevice_domains.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/mark_aliases_prepare.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/move_gather.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/move_pad.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/move_repeat_forward.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/move_split_cat.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/pre_segmenter.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/propagate_shardings.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/remove_bcast_squeeze.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/remove_empty.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/reorder_sharded_axis.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/segment_inplace_update.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/pass/convert_op_to_communication.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/pass/stream_parallel_type.cpp
  ${NVFUSER_SRCS_DIR}/host_ir/pass/insert_deallocations.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/translate_no_reduction_matmul_to_mul_squeeze.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/translate_repeat_to_expand.cpp
  ${NVFUSER_SRCS_DIR}/rng.cpp
  ${NVFUSER_SRCS_DIR}/runtime/allocations.cpp
  ${NVFUSER_SRCS_DIR}/runtime/compiled_kernel.cpp
  ${NVFUSER_SRCS_DIR}/runtime/executor.cpp
  ${NVFUSER_SRCS_DIR}/runtime/executor_dispatch.cpp
  ${NVFUSER_SRCS_DIR}/runtime/executor_kernel_arg.cpp
  ${NVFUSER_SRCS_DIR}/runtime/executor_params.cpp
  ${NVFUSER_SRCS_DIR}/runtime/executor_utils.cpp
  ${NVFUSER_SRCS_DIR}/runtime/fusion_cache_utils.cpp
  ${NVFUSER_SRCS_DIR}/runtime/fusion_executor_cache.cpp
  ${NVFUSER_SRCS_DIR}/runtime/fusion_kernel_runtime.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/cache_policy_refiner.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/heuristic.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/greedy.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/mark_aliases.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_ampere-.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_hopper+.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_heuristic_plugin.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/mma_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/no_op.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/communication.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner_outer.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner_outer_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner_outer_tma_ws.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner_outer_multi_wave.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_outer.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/pointwise.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/pointwise_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/reduction.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/reduction_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/registry.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/registry_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/resize.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/runtime_info.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/scheduler_types.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/domain_map.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/inlining.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/loop_domain_scheduler.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/maxinfo_propagator.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/resize_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/tools/static_repeat.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/transpose.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/vectorize_helper.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/expr_eval_sched.cpp
  ${NVFUSER_SRCS_DIR}/serde/polymorphic_value.cpp
  ${NVFUSER_SRCS_DIR}/serde/utils.cpp
  ${NVFUSER_SRCS_DIR}/statement_guard.cpp
  ${NVFUSER_SRCS_DIR}/swizzle.cpp
  ${NVFUSER_SRCS_DIR}/sys_utils.cpp
  ${NVFUSER_SRCS_DIR}/tensor_metadata.cpp
  ${NVFUSER_SRCS_DIR}/tensor_view.cpp
  ${NVFUSER_SRCS_DIR}/tma.cpp
  ${NVFUSER_SRCS_DIR}/transform_iter.cpp
  ${NVFUSER_SRCS_DIR}/transform_replay.cpp
  ${NVFUSER_SRCS_DIR}/transform_rfactor.cpp
  ${NVFUSER_SRCS_DIR}/transform_view.cpp
  ${NVFUSER_SRCS_DIR}/type.cpp
  ${NVFUSER_SRCS_DIR}/type_promotion.cpp
  ${NVFUSER_SRCS_DIR}/utils.cpp
  ${NVFUSER_SRCS_DIR}/val_graph.cpp
  ${NVFUSER_SRCS_DIR}/val_graph_visitor.cpp
  ${NVFUSER_SRCS_DIR}/validator_utils.cpp
)

cmake_dependent_option(NVFUSER_HOST_IR_JIT "Build nvFuser with LLVM" ON "USE_HOST_IR_JIT" OFF)


message(STATUS "Setting NVFUSER_HOST_IR_JIT=${NVFUSER_HOST_IR_JIT}")

if(NVFUSER_HOST_IR_JIT)
  add_compile_definitions(NVFUSER_HOST_IR_JIT)
  # Add LLVM JIT related dependencies
  find_package(LLVM 18.1 REQUIRED CONFIG)
  llvm_map_components_to_libnames(LLVM_LIBS
    support
    core
    orcjit
    executionengine
    irreader
    nativecodegen
    Target
    Analysis
    JITLink
    Demangle
  )

  add_library(LLVM_JIT INTERFACE)
  target_include_directories(LLVM_JIT INTERFACE ${LLVM_INCLUDE_DIRS})
  target_compile_definitions(LLVM_JIT INTERFACE ${LLVM_DEFINITIONS})
  target_link_libraries(LLVM_JIT INTERFACE ${LLVM_LIBS})

  # Add LLVM JIT related sources
  list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/host_ir/jit.cpp
  )
endif()

# We don't link CUPTI for MSVC
if(NOT MSVC)
  list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/fusion_profiler.cpp
  )
endif()

if(BUILD_PYTHON)
  list(APPEND NVFUSER_SRCS
    ${NVFUSER_PYTHON_BINDINGS}/fusion_cache.cpp
    ${NVFUSER_PYTHON_BINDINGS}/fusion_definition.cpp
    ${NVFUSER_PYTHON_BINDINGS}/fusion_state.cpp
    ${NVFUSER_PYTHON_BINDINGS}/segmentation.cpp
    ${NVFUSER_PYTHON_BINDINGS}/translation.cpp
    ${NVFUSER_PYTHON_BINDINGS}/translation_utils.cpp
    ${NVFUSER_SRCS_DIR}/serde/fusion_record.cpp
    ${NVFUSER_PYTHON_COMMON}/distributed_tensor.cpp
    ${NVFUSER_PYTHON_COMMON}/python_utils.cpp
    ${NVFUSER_PYTHON_COMMON}/translation_names.cpp
  )
endif()

# We create both static and shared libraries.
#
# Shared libraries are what ships, but a large advantage of static libraries is
# that symbols are all visible. This allows us to test internal components
# inside our test or benchmark binaries, even if we do not want said components
# to be visible to the outside. If we used only shared libraries, then any API
# we invoked from test binaries would need to be marked as public, even if we
# did not want to expose it to users.
#
# Note technically we create an "OBJECT" library instead of a "STATIC" library.
# This is just a CMake quirk; an OBJECT library is a better way to implement a
# "private" (not installed) static library.
add_library(codegen_internal OBJECT ${NVFUSER_SRCS})

if(NOT MSVC)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(codegen_internal PRIVATE
      -Wall -Wno-unused-function -Werror

      # These warnings are not treated as errors because of gcc 12.2 used in
      # manylinux image. consider enable this when we upgrade.
      # linking comment:
      # https://github.com/NVIDIA/Fuser/pull/3001#discussion_r1772551266
      -Wno-error=restrict -Wno-error=stringop-overflow -Wno-error=maybe-uninitialized)
  else()
    target_compile_options(codegen_internal PRIVATE
      -Wall -Wno-unused-function -Werror)
  endif()
endif()

target_compile_definitions(codegen_internal PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
target_include_directories(codegen_internal PUBLIC ${NVFUSER_PYTHON_DIR})
target_include_directories(codegen_internal PUBLIC ${NVFUSER_PYTHON_COMMON})
target_include_directories(codegen_internal SYSTEM PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
  PRIVATE
  ${CUDA_INCLUDE_DIRS}
)
target_include_directories(codegen_internal PUBLIC
  "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
)
set_target_properties(codegen_internal PROPERTIES
  C_STANDARD ${NVFUSER_C_STANDARD}
  CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
  CXX_STANDARD ${NVFUSER_CPP_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden

  # this is to find pip installed nvrtc.so
  INSTALL_RPATH
  "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib"
  POSITION_INDEPENDENT_CODE Yes
  VISIBILITY_INLINES_HIDDEN Yes
)

# Ensure we don't link against libcuda; we'll dlopen it ourselves.
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "libcuda\.so")
target_link_libraries(codegen_internal PUBLIC
  dynamic_type
  CUDA::cupti
  ${TORCH_LIBRARIES}
  dl
)
if (BUILD_CUTLASS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  target_link_libraries(codegen_internal PUBLIC nvf_cutlass)
  target_compile_definitions(codegen_internal PRIVATE "-DNVFUSER_CUTLASS_KERNEL_ENABLED")
endif()

if(NVFUSER_HOST_IR_JIT)
  target_link_libraries(codegen_internal PUBLIC LLVM_JIT)
endif()

add_library(nvfuser_codegen SHARED $<TARGET_OBJECTS:codegen_internal>)

if (BUILD_CUTLASS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  target_link_libraries(nvfuser_codegen PUBLIC nvf_cutlass)
endif()

# Conditionally link CUTLASS using generator expression to avoid export issues
target_link_libraries(nvfuser_codegen PRIVATE
  $<$<AND:$<BOOL:${BUILD_CUTLASS}>,$<VERSION_GREATER_EQUAL:${CMAKE_CUDA_COMPILER_VERSION},12.8>>:nvf_cutlass>
)

if(NVFUSER_BUILD_WITH_ASAN)
  target_compile_options(codegen_internal PRIVATE -fsanitize=address)
  target_link_options(codegen_internal PUBLIC -fsanitize=address)
  target_link_options(nvfuser_codegen PUBLIC -fsanitize=address)
endif()

target_include_directories(nvfuser_codegen PUBLIC
  # Core nvfuser
  "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
  # Python-specific
  "$<BUILD_INTERFACE:${NVFUSER_PYTHON_DIR}>"
  "$<BUILD_INTERFACE:${NVFUSER_PYTHON_COMMON}>"
  # External dependencies
  "$<BUILD_INTERFACE:${NVFUSER_ROOT}/lib/dynamic_type/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
)
target_include_directories(nvfuser_codegen SYSTEM PUBLIC
  "$<BUILD_INTERFACE:${NVFUSER_ROOT}/third_party/flatbuffers/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
)
target_link_libraries(nvfuser_codegen
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE dynamic_type flatbuffers ${CUDA_NVRTC_LIB} CUDA::cupti dl
)
set_target_properties(nvfuser_codegen PROPERTIES
  C_STANDARD ${NVFUSER_C_STANDARD}
  CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
  CXX_STANDARD ${NVFUSER_CPP_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  INSTALL_RPATH
  "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib:$ORIGIN"
  POSITION_INDEPENDENT_CODE Yes
  VISIBILITY_INLINES_HIDDEN Yes
)

install(TARGETS nvfuser_codegen EXPORT NvfuserTargets DESTINATION lib)

# We are keeping fusion_cache_generated.h for the submodule build because flatc is unavailable.
add_custom_command(
  OUTPUT
  ${NVFUSER_ROOT}/csrc/serde/fusion_cache_generated.h
  DEPENDS
  ${NVFUSER_ROOT}/csrc/serde/fusion_cache.fbs
  DEPENDS flatc
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/third_party/flatbuffers/flatc --scoped-enums -o ${NVFUSER_ROOT}/csrc/serde/ -c -b ${NVFUSER_ROOT}/csrc/serde/fusion_cache.fbs
  COMMENT "Generating fusion_cache_generated header from fusion_cache.fbs"
  VERBATIM
)
add_custom_target(build_flatbuffer_config ALL
  DEPENDS ${NVFUSER_ROOT}/csrc/serde/fusion_cache_generated.h)

if(NVFUSER_STANDALONE_BUILD_WITH_UCC)
  # User may need to set env vars UCC_DIR, UCX_DIR, UCC_HOME, UCX_HOME for CMake's Find_UCC to work.
  find_package(UCC REQUIRED)
  find_package(UCX REQUIRED)

  add_library(__nvfuser_ucc INTERFACE)
  set_target_properties(__nvfuser_ucc PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE Yes
    VISIBILITY_INLINES_HIDDEN Yes
  )
  target_link_libraries(__nvfuser_ucc INTERFACE ucx::ucs ucx::ucp ucc::ucc)
  target_include_directories(__nvfuser_ucc INTERFACE ${UCC_INCLUDE_DIRS})
  target_link_libraries(codegen_internal PRIVATE __nvfuser_ucc)
  target_compile_definitions(codegen_internal PRIVATE NVFUSER_BUILD_WITH_UCC)
endif()

add_dependencies(codegen_internal flatc build_flatbuffer_config)

# installing nvfuser headers
install(DIRECTORY "${NVFUSER_SRCS_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "csrc/C++20/type_traits"
  PATTERN "csrc/struct.inl")

# TODO guard including flatbuffers headers
# installing flatbuffers headers
install(DIRECTORY "${NVFUSER_THIRD_PARTY_DIR}/flatbuffers/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser")

# installing dynamic_type headers
install(DIRECTORY "${NVFUSER_ROOT}/lib/dynamic_type/src/dynamic_type"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser")

if(BUILD_PYTHON)
  # -----------------------------
  # build nvfuser python library
  # -----------------------------
  # nvfuser python API sources
  set(NVFUSER_PYTHON_SRCS)
  list(APPEND NVFUSER_PYTHON_SRCS
    ${NVFUSER_PYTHON_BINDINGS}/multidevice_bindings.cpp
    ${NVFUSER_PYTHON_BINDINGS}/python_bindings.cpp
    ${NVFUSER_PYTHON_BINDINGS}/python_bindings_extension.cpp
    ${NVFUSER_PYTHON_BINDINGS}/schedule_bindings.cpp
  )

  add_library(nvf_py_internal OBJECT ${NVFUSER_PYTHON_SRCS})
  target_include_directories(nvf_py_internal PUBLIC ${NVFUSER_PYTHON_DIR})
  target_include_directories(nvf_py_internal PUBLIC ${NVFUSER_PYTHON_COMMON})
  target_include_directories(nvf_py_internal PUBLIC ${NVFUSER_CUTLASS})
  target_include_directories(nvf_py_internal SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
  )

  # setup python API version
  add_custom_command(
    OUTPUT ${NVFUSER_PYTHON_DIR}/nvfuser/version.py
    COMMAND
    "${PYTHON_EXECUTABLE}" -c \"from pathlib import Path\; Path('${NVFUSER_PYTHON_DIR}/tools/gen_nvfuser_version.py') .touch() \"
    COMMAND
    "${PYTHON_EXECUTABLE}" ${NVFUSER_PYTHON_DIR}/tools/gen_nvfuser_version.py
    DEPENDS ${NVFUSER_PYTHON_DIR}/tools/gen_nvfuser_version.py
    DEPENDS ${NVFUSER_PYTHON_DIR}/version.txt
    WORKING_DIRECTORY ${NVFUSER_ROOT}/tools/
  )
  add_custom_target(
    gen_nvfuser_version ALL
    DEPENDS ${NVFUSER_PYTHON_DIR}/nvfuser/version.py
  )
  add_dependencies(nvf_py_internal gen_nvfuser_version)

  target_compile_definitions(nvf_py_internal PRIVATE
    "-DTORCH_CUDA_BUILD_MAIN_LIB"
    "-DC10_BUILD_MAIN_LIB=1"
    EXTENSION_NAME=_C
  )

  add_library(nvfuser MODULE $<TARGET_OBJECTS:nvf_py_internal>)
  target_compile_definitions(nvfuser PRIVATE
    "-DTORCH_CUDA_BUILD_MAIN_LIB"
    "-DC10_BUILD_MAIN_LIB=1"
    EXTENSION_NAME=_C
  )

  if(NOT MSVC)
    target_compile_options(nvf_py_internal PRIVATE -Wall -Wno-unused-function)
    target_compile_options(nvf_py_internal PRIVATE -Werror)
    set(NVF_LIB_SUFFIX ".so")
  else()
    set(NVF_LIB_SUFFIX ".pyd")
  endif()

  set_target_properties(nvfuser PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH
    "$ORIGIN/lib:$ORIGIN/../nvfuser_common/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
    POSITION_INDEPENDENT_CODE Yes
    SUFFIX ${NVF_LIB_SUFFIX}
    VISIBILITY_INLINES_HIDDEN Yes
  )
  set_target_properties(nvf_py_internal PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH
    "$ORIGIN/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
    POSITION_INDEPENDENT_CODE Yes
    VISIBILITY_INLINES_HIDDEN Yes
  )

  if (BUILD_CUTLASS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_link_libraries(nvf_py_internal PRIVATE nvf_cutlass)
  endif()

  target_link_libraries(nvf_py_internal PRIVATE
    nvfuser_codegen
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    pybind11::pybind11 pybind11::headers
  )

  target_link_libraries(nvfuser PRIVATE
    nvf_py_internal
    Python::Module
  )
  set_target_properties(nvfuser PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../build:$ORIGIN/../nvfuser_common/lib"
  )
  install(TARGETS nvfuser DESTINATION lib)

  # ------------------------------------------------
  # build nvfuser next python library
  # ------------------------------------------------
  # nvfuser next bindings API sources
  set(NVFUSER_PYTHON_DIRECT_SRCS)
  list(APPEND NVFUSER_PYTHON_DIRECT_SRCS
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/extension.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/bindings.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/enum.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/ir.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/multidevice.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/ops.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/cutlass.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/runtime.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/direct_utils.cpp
    ${NVFUSER_PYTHON_DIRECT_BINDINGS}/python_translate.cpp
  )
  add_library(nvf_py_direct_internal OBJECT ${NVFUSER_PYTHON_DIRECT_SRCS})

  # NOTE: For any future extension, change PYTHON_DIRECT_EXTENSION to another
  # name other than EXTENSION_NAME.
  target_compile_definitions(nvf_py_direct_internal PRIVATE
    "-DTORCH_CUDA_BUILD_MAIN_LIB"
    "-DC10_BUILD_MAIN_LIB=1"
    PYTHON_DIRECT_EXTENSION=_C_DIRECT
  )

  add_library(nvfuser_direct MODULE $<TARGET_OBJECTS:nvf_py_direct_internal>)
  target_compile_definitions(nvfuser_direct PRIVATE
    "-DTORCH_CUDA_BUILD_MAIN_LIB"
    "-DC10_BUILD_MAIN_LIB=1"
    PYTHON_DIRECT_EXTENSION=_C_DIRECT
  )

  if(NOT MSVC)
    target_compile_options(nvf_py_direct_internal PRIVATE -Wall -Wno-unused-function)
    target_compile_options(nvf_py_direct_internal PRIVATE -Werror)
    set(NVF_LIB_SUFFIX ".so")
  else()
    set(NVF_LIB_SUFFIX ".pyd")
  endif()

  set_target_properties(nvf_py_direct_internal PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH
    "$ORIGIN/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
    POSITION_INDEPENDENT_CODE Yes
    VISIBILITY_INLINES_HIDDEN Yes
  )
  set_target_properties(nvfuser_direct PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    INSTALL_RPATH
    "$ORIGIN/lib:$ORIGIN/../nvfuser_common/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
    POSITION_INDEPENDENT_CODE Yes
    SUFFIX ${NVF_LIB_SUFFIX}
    VISIBILITY_INLINES_HIDDEN Yes
  )

  target_include_directories(nvf_py_direct_internal PUBLIC ${NVFUSER_PYTHON_DIRECT_BINDINGS})
  target_include_directories(nvf_py_direct_internal PUBLIC ${NVFUSER_PYTHON_COMMON})
  if (BUILD_CUTLASS AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_link_libraries(nvf_py_direct_internal PRIVATE
      nvfuser_codegen
      nvf_cutlass
      "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
      pybind11::pybind11 pybind11::headers
    )
  else()
    target_link_libraries(nvf_py_direct_internal PRIVATE
      nvfuser_codegen
      "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
      pybind11::pybind11 pybind11::headers
    )
  endif()

  target_link_libraries(nvfuser_direct PRIVATE
    nvf_py_direct_internal
    Python::Module
  )
  set_target_properties(nvfuser_direct PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../build:$ORIGIN/../nvfuser_common/lib"
  )
  install(TARGETS nvfuser_direct DESTINATION lib)
endif()

set(JIT_TEST_SRCS)
list(APPEND JIT_TEST_SRCS
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_open.cpp
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_query.cpp
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_write.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_abstract_tensor.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_alias.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_alias_analysis.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_allocation_domain.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_allocation_order_inference.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_bfs.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_ca_root_domain_map.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_circular_buffering.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_circular_buffering_ping_pong.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_combined_inner_outer_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_compute_at_map.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_compute_with.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_contiguity_id_model.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_driver_api.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_dynamic_transform.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_embedding_node.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_evaluator.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_exceptions.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_expr_simplifier.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_expr_sort.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gather.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu1.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu2.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu3.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_id_model.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_index_put.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_index_select.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_indexing.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_indexing_advanced.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_inlining.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_interval_analysis.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_iter_visitor.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_linked_hash_map.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_loop_domain_scheduling.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_loop_rotation.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_low_precision_recipe.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_mbarrier.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_memory.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_move_pad.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_move_repeat_forward.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_move_split_cat.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_mutator.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_no_op.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_outer_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_overlap.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_persistent_buffer.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_pointwise.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_polymorphic_value.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_predicate_elimination.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_preseg_passes.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_reduction_pointwise.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_remove_bcast_squeeze.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_remove_trivial_ops.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_replay.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_resharding.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_resize.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_rope.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_runtime.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_scalar_hoisting.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_scatter.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_sdpa_node.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_segmentation.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_select.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_serial_gridreduce.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_sharding.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_smem_reuse.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_statement_guard.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_swizzle.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_tensor_factories.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_tmem.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_transpose.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_unary.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_utils.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_vectorization.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_welford.cpp
)

if(BUILD_TEST)
  set(RNG_TEST_KERNELS "rng_test_kernels")
  add_library(${RNG_TEST_KERNELS} SHARED ${NVFUSER_ROOT}/tests/cpp/rng_kernels.cu)

  # CUDA 11 does not support C++20, so hard code C++17 here
  set_property(TARGET ${RNG_TEST_KERNELS} PROPERTY CXX_STANDARD 17)
  target_link_libraries(${RNG_TEST_KERNELS} PRIVATE torch ${TORCH_LIBRARIES})
  target_include_directories(${RNG_TEST_KERNELS} PRIVATE "${NVFUSER_ROOT}")

  set(ARGSORT_TEST_KERNELS "argsort_test_kernels")
  add_library(${ARGSORT_TEST_KERNELS} SHARED ${NVFUSER_ROOT}/tests/cpp/argsort_test_kernels.cu)

  # CUDA 11 does not support C++20, so hard code C++17 here
  set_property(TARGET ${ARGSORT_TEST_KERNELS} PROPERTY CXX_STANDARD 17)
  target_link_libraries(${ARGSORT_TEST_KERNELS} PRIVATE torch ${TORCH_LIBRARIES})
  target_include_directories(${ARGSORT_TEST_KERNELS} PRIVATE "${NVFUSER_ROOT}")
  target_include_directories(${ARGSORT_TEST_KERNELS} SYSTEM PRIVATE
    ${NVFUSER_ROOT}/third_party/googletest/googletest/include
    ${NVFUSER_ROOT}/third_party/googletest/googlemock/include
  )

  set(TOPK_TEST_KERNELS "topk_test_kernels")
  add_library(${TOPK_TEST_KERNELS} SHARED
    ${NVFUSER_ROOT}/tests/cpp/topk_test_kernels.cu
    ${NVFUSER_ROOT}/tests/cpp/topk_test_helper.cpp
  )

  # CUDA 11 does not support C++20, so hard code C++17 here
  set_property(TARGET ${TOPK_TEST_KERNELS} PROPERTY CXX_STANDARD 17)
  target_link_libraries(${TOPK_TEST_KERNELS} PRIVATE torch ${TORCH_LIBRARIES})
  target_include_directories(${TOPK_TEST_KERNELS} PRIVATE "${NVFUSER_ROOT}")
  target_include_directories(${TOPK_TEST_KERNELS} PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
  )
  target_include_directories(${TOPK_TEST_KERNELS} SYSTEM PRIVATE
    ${NVFUSER_ROOT}/third_party/googletest/googletest/include
    ${NVFUSER_ROOT}/third_party/googletest/googlemock/include
  )

  set(SCAN_TEST_KERNELS "scan_test_kernels")
  add_library(${SCAN_TEST_KERNELS} SHARED
    ${NVFUSER_ROOT}/tests/cpp/scan_test_kernels.cu
    ${NVFUSER_ROOT}/tests/cpp/scan_test_helper.cpp
  )

  # CUDA 11 does not support C++20, so hard code C++17 here
  set_property(TARGET ${SCAN_TEST_KERNELS} PROPERTY CXX_STANDARD 17)
  target_link_libraries(${SCAN_TEST_KERNELS} PRIVATE torch ${TORCH_LIBRARIES})
  target_include_directories(${SCAN_TEST_KERNELS} PRIVATE "${NVFUSER_ROOT}")
  target_include_directories(${SCAN_TEST_KERNELS} PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
  )
  target_include_directories(${SCAN_TEST_KERNELS} SYSTEM PRIVATE
    ${NVFUSER_ROOT}/third_party/googletest/googletest/include
    ${NVFUSER_ROOT}/third_party/googletest/googlemock/include
  )
endif()

function(add_test_without_main TEST_NAME TEST_SRC ADDITIONAL_LINK)
  list(APPEND TEST_SRC
    ${NVFUSER_ROOT}/tests/cpp/utils.cpp
    ${NVFUSER_ROOT}/tests/cpp/validator.cpp
  )
  add_executable(${TEST_NAME} ${TEST_SRC})
  set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD ${NVFUSER_CPP_STANDARD})
  target_compile_definitions(${TEST_NAME} PRIVATE USE_GTEST)
  target_include_directories(${TEST_NAME} PRIVATE "${NVFUSER_ROOT}")
  target_include_directories(${TEST_NAME} SYSTEM PRIVATE
    ${NVFUSER_ROOT}/third_party/googletest/googletest/include
    ${NVFUSER_ROOT}/third_party/googletest/googlemock/include
  )
  target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
  )
  target_link_libraries(${TEST_NAME} PRIVATE
    codegen_internal
    ${ADDITIONAL_LINK}
    dynamic_type
    GTest::gtest
    GTest::gmock
    flatbuffers
    ${TORCH_LIBRARIES}
  )

  if(NOT MSVC)
    target_compile_options(${TEST_NAME} PRIVATE
      -Wall -Wno-unused-function -Werror
    )
  endif()
endfunction()

function(add_test TEST_NAME TEST_SRC ADDITIONAL_LINK)
  list(APPEND ADDITIONAL_LINK "GTest::gtest_main")
  add_test_without_main("${TEST_NAME}" "${TEST_SRC}" "${ADDITIONAL_LINK}")
endfunction()

if(BUILD_TEST)
  set(TEST_BINARIES)
  add_test(test_nvfuser "${JIT_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_nvfuser)

  add_test(test_rng ${NVFUSER_ROOT}/tests/cpp/test_rng.cpp ${RNG_TEST_KERNELS})
  list(APPEND TEST_BINARIES test_rng)

  set(ARGSORT_TEST_SRCS)
  list(APPEND ARGSORT_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_argsort.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_argsort_device_func.cpp
  )
  add_test(test_argsort "${ARGSORT_TEST_SRCS}" ${ARGSORT_TEST_KERNELS})
  list(APPEND TEST_BINARIES test_argsort)

  set(SCAN_TEST_SRCS)
  list(APPEND SCAN_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_scan.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_scan_device_func.cpp
  )
  add_test(test_scan "${SCAN_TEST_SRCS}" ${SCAN_TEST_KERNELS})
  list(APPEND TEST_BINARIES test_scan)

  set(TOPK_TEST_SRCS)
  list(APPEND TOPK_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_topk.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_topk_device_func.cpp
  )
  add_test(test_topk "${TOPK_TEST_SRCS}" ${TOPK_TEST_KERNELS})
  list(APPEND TEST_BINARIES test_topk)

  set(MOE_TEST_SRCS)
  list(APPEND MOE_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_moe.cpp
  )
  add_test(test_moe "${MOE_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_moe)

  set(MULTIDEVICE_TEST_SRCS)
  list(APPEND MULTIDEVICE_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/multidevice.cpp
    ${NVFUSER_ROOT}/tests/cpp/multidevice_transformer.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_host_ir_overlap.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_communications.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_communicator.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_host_ir.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_lower_communication.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_matmul.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_pipeline.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_sharding.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_stream_parallel_type.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_transformer.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_ipc.cpp
  )
  add_test_without_main(test_multidevice "${MULTIDEVICE_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_multidevice)

  set(MULTIDEVICE_TUTORIAL_SRCS)
  list(APPEND MULTIDEVICE_TUTORIAL_SRCS
    ${NVFUSER_ROOT}/tests/cpp/multidevice.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_tutorial.cpp
  )
  add_test_without_main(tutorial_multidevice "${MULTIDEVICE_TUTORIAL_SRCS}" "")
  list(APPEND TEST_BINARIES tutorial_multidevice)

  add_test(test_reshape "${NVFUSER_ROOT}/tests/cpp/test_reshape.cpp" "")
  list(APPEND TEST_BINARIES test_reshape)

  set(MATMUL_TEST_SRCS)
  list(APPEND MATMUL_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_translate_mma.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul_aten_evaluation.cpp
    # ${NVFUSER_ROOT}/tests/cpp/test_matmul_sass.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul_scheduler.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_mma.cpp
  )
  add_test(test_matmul "${MATMUL_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_matmul)

  add_test(test_external_src "${NVFUSER_ROOT}/tests/cpp/test_external_src.cpp" "")
  list(APPEND TEST_BINARIES test_external_src)

  set(TUTORIAL_SRCS)
  list(APPEND TUTORIAL_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_tutorial.cpp
    ${NVFUSER_ROOT}/tests/cpp/tutorial_tmem.cpp
    ${NVFUSER_ROOT}/tests/cpp/tutorial_ldmatrix_stmatrix.cpp)
  add_test(test_tutorial "${TUTORIAL_SRCS}" "")
  list(APPEND TEST_BINARIES test_tutorial)

  set(HOSTIR_TEST_SRCS)
  list(APPEND HOSTIR_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_host_irs.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_host_ir_integration.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_host_ir_stream_lowering.cpp
  )
  add_test(test_host_ir "${HOSTIR_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_host_ir)

  if(NVFUSER_HOST_IR_JIT)
    set(LLVM_COMPILE_TEST_SRCS)
    list(APPEND LLVM_COMPILE_TEST_SRCS
      ${NVFUSER_ROOT}/tests/cpp/test_host_ir_jit.cpp
    )
    add_test(test_host_ir_jit "${LLVM_COMPILE_TEST_SRCS}" "")
    target_link_libraries(test_host_ir_jit PUBLIC LLVM_JIT)
    list(APPEND TEST_BINARIES test_host_ir_jit)
  endif()


  if(BUILD_PYTHON)
    set(PY_FRONTEND_TEST_SRCS)
    list(APPEND PY_FRONTEND_TEST_SRCS
      ${NVFUSER_PYTHON_DIR}/tests/python_frontend/test_nvfuser_fusion_cache.cpp
      ${NVFUSER_PYTHON_DIR}/tests/python_frontend/test_nvfuser_fusion_definition.cpp
      ${NVFUSER_PYTHON_DIR}/tests/python_frontend/test_nvfuser_fusion_record.cpp
    )
    add_test(test_python_frontend "${PY_FRONTEND_TEST_SRCS}" "")
    list(APPEND TEST_BINARIES test_python_frontend)
  endif()

  # We don't link CUPTI for MSVC
  if(NOT MSVC)
    add_test(test_profiler "${NVFUSER_ROOT}/tests/cpp/test_fusion_profiler.cpp" "")
    list(APPEND TEST_BINARIES test_profiler)
  endif()

  add_custom_target(tests DEPENDS ${TEST_BINARIES})
endif()

# -- build benchmark
if(BUILD_NVFUSER_BENCHMARK)
  # nvfuser benchmark sources
  set(BENCHMARK_SRCS)
  list(APPEND BENCHMARK_SRCS
    ${NVFUSER_ROOT}/benchmarks/cpp/batch_norm_channels_first.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/batch_norm_channels_first_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/batch_norm_channels_last.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/batch_norm_channels_last_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/bert.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/broadcast.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/gelu_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/gelu_backward_reduction.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/heuristic_cache.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/heuristic_lookup.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/indexselect.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/instance_norm.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/layer_norm.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/layer_norm_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/layer_norm_fused.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/lstm_cell.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/main.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/many_pointwise_ops.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/matmul.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/reduction.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/rms_norm.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/rms_norm_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/scale_bias_relu.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/shape_inference.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/softmax.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/softmax_backward.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/softmax_dropout.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/timm.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/transpose.cpp
    ${NVFUSER_ROOT}/benchmarks/cpp/utils.cpp
    ${NVFUSER_ROOT}/tests/cpp/utils.cpp
  )

  add_executable(nvfuser_bench ${BENCHMARK_SRCS})
  set_target_properties(nvfuser_bench PROPERTIES
    C_STANDARD ${NVFUSER_C_STANDARD}
    CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
    CXX_STANDARD ${NVFUSER_CPP_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE Yes
    VISIBILITY_INLINES_HIDDEN Yes
  )

  target_include_directories(nvfuser_bench SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/benchmark/include
    ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
    ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
  )
  target_include_directories(nvfuser_bench PUBLIC ${NVFUSER_ROOT})
  target_link_libraries(nvfuser_bench PRIVATE
    GTest::gtest
    benchmark::benchmark
    codegen_internal
  )
  add_dependencies(nvfuser_bench flatc build_flatbuffer_config)

  if(NOT MSVC)
    target_compile_options(nvfuser_bench PRIVATE
      -Wall -Wno-unused-function
      -Werror -Wno-deprecated-copy
    )
  endif()

  # multidevice transformer benchmark
  if(NVFUSER_DISTRIBUTED)
    set(MULTIDEVICE_BENCHMARK_SRCS)
    list(APPEND MULTIDEVICE_BENCHMARK_SRCS
      ${NVFUSER_ROOT}/benchmarks/cpp/transformer.cpp
      ${NVFUSER_ROOT}/tests/cpp/multidevice_transformer.cpp
      ${NVFUSER_ROOT}/tests/cpp/utils.cpp
    )
    add_executable(nvfuser_multidevice_bench ${MULTIDEVICE_BENCHMARK_SRCS})
    set_target_properties(nvfuser_multidevice_bench PROPERTIES
      C_STANDARD ${NVFUSER_C_STANDARD}
      CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
      CXX_STANDARD ${NVFUSER_CPP_STANDARD}
      CXX_STANDARD_REQUIRED ON
      CXX_VISIBILITY_PRESET hidden
      POSITION_INDEPENDENT_CODE Yes
      VISIBILITY_INLINES_HIDDEN Yes
    )
    target_include_directories(nvfuser_multidevice_bench SYSTEM PRIVATE
      ${CMAKE_SOURCE_DIR}/third_party/benchmark/include
      ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
      ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
    )
    target_include_directories(nvfuser_multidevice_bench PUBLIC ${NVFUSER_ROOT})
    target_link_libraries(nvfuser_multidevice_bench PRIVATE
      GTest::gtest
      benchmark::benchmark
      codegen_internal
    )
    add_dependencies(nvfuser_multidevice_bench flatc build_flatbuffer_config)

    if(NOT MSVC)
      target_compile_options(nvfuser_bench PRIVATE
        -Wall -Wno-unused-function
        -Werror -Wno-deprecated-copy
      )
    endif()
  endif()
endif()

# --- generate runtime files
# nvfuser runtime files
set(NVFUSER_RUNTIME_FILES)
list(APPEND NVFUSER_RUNTIME_FILES
  ${NVFUSER_ROOT}/runtime/argsort.cu
  ${NVFUSER_ROOT}/runtime/array.cu
  ${NVFUSER_ROOT}/runtime/basic_type_traits.cu
  ${NVFUSER_ROOT}/runtime/bf16_support.cu
  ${NVFUSER_ROOT}/runtime/bit.cu
  ${NVFUSER_ROOT}/runtime/block_reduction.cu
  ${NVFUSER_ROOT}/runtime/block_sync_atomic.cu
  ${NVFUSER_ROOT}/runtime/block_sync_default.cu
  ${NVFUSER_ROOT}/runtime/block_welford_outer.cu
  ${NVFUSER_ROOT}/runtime/broadcast.cu
  ${NVFUSER_ROOT}/runtime/casts.cu
  ${NVFUSER_ROOT}/runtime/cluster.cu
  ${NVFUSER_ROOT}/runtime/complex_number.cu
  ${NVFUSER_ROOT}/runtime/cub_utils.cu
  ${NVFUSER_ROOT}/runtime/fp16_support.cu
  ${NVFUSER_ROOT}/runtime/fp8_support.cu
  ${NVFUSER_ROOT}/runtime/fp4_support.cu
  ${NVFUSER_ROOT}/runtime/fused_reduction.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_helper.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_impl.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_impl_outer.cu
  ${NVFUSER_ROOT}/runtime/grid_broadcast.cu
  ${NVFUSER_ROOT}/runtime/grid_reduction.cu
  ${NVFUSER_ROOT}/runtime/grid_sync.cu
  ${NVFUSER_ROOT}/runtime/helpers.cu
  ${NVFUSER_ROOT}/runtime/index_utils.cu
  ${NVFUSER_ROOT}/runtime/mbarrier.cu
  ${NVFUSER_ROOT}/runtime/memory.cu
  ${NVFUSER_ROOT}/runtime/random_numbers.cu
  ${NVFUSER_ROOT}/runtime/tensor_memory.cu
  ${NVFUSER_ROOT}/runtime/tensor.cu
  ${NVFUSER_ROOT}/runtime/topk.cu
  ${NVFUSER_ROOT}/runtime/scan.cu
  ${NVFUSER_ROOT}/runtime/tuple.cu
  ${NVFUSER_ROOT}/runtime/type_traits.cu
  ${NVFUSER_ROOT}/runtime/warp.cu
  ${NVFUSER_ROOT}/runtime/welford.cu
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvfuser_resources")

# "stringify" NVFUSER runtime sources
# (generate C++ header files embedding the original input as a string literal)
set(NVFUSER_STRINGIFY_TOOL "${NVFUSER_ROOT}/tools/stringify_file.py")

foreach(src ${NVFUSER_RUNTIME_FILES})
  get_filename_component(filename ${src} NAME_WE)
  set(dst "${CMAKE_BINARY_DIR}/include/nvfuser_resources/${filename}.h")
  add_custom_command(
    COMMENT "Stringify NVFUSER runtime source file ${src}"
    OUTPUT ${dst}
    DEPENDS ${src} "${NVFUSER_STRINGIFY_TOOL}"
    COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst}
  )
  add_custom_target(nvfuser_rt_${filename} DEPENDS ${dst})
  add_dependencies(codegen_internal nvfuser_rt_${filename})

  # Do not overwrite resource header if it already exists. This avoids unnecessary rebuilds.
  # If ${dst} doesn't exist, this `if` is also true, so header will be generated.
  if(${src} IS_NEWER_THAN ${dst})
    # also generate the resource headers during the configuration step
    # (so tools like clang-tidy can run w/o requiring a real build)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
  endif()
endforeach()

target_include_directories(codegen_internal PRIVATE "${CMAKE_BINARY_DIR}/include")

# -- install nvfuser cmake config files and symlink to build binaries
install(EXPORT NvfuserTargets FILE NvfuserConfig.cmake DESTINATION share/cmake/nvfuser)

file(CREATE_LINK "${CMAKE_BINARY_DIR}" "${NVFUSER_ROOT}/bin" SYMBOLIC)
# These symbolic links help IDEs like Cursor resolve symbols in nvfuser and
# nvfuser_direct.
file(CREATE_LINK "${NVFUSER_ROOT}/python/nvfuser" "${NVFUSER_ROOT}/nvfuser" SYMBOLIC)
file(CREATE_LINK "${NVFUSER_ROOT}/python/nvfuser_direct" "${NVFUSER_ROOT}/nvfuser_direct" SYMBOLIC)

message(STATUS "")
message(STATUS "******** Nvfuser configuration summary ********")
message(STATUS "  BUILD_CUTLASS: ${CUTLASS_STATUS}")
message(STATUS "  UCC_FOUND: ${UCC_FOUND}")
message(STATUS "  NVFUSER_STANDALONE_BUILD_WITH_UCC  : ${NVFUSER_STANDALONE_BUILD_WITH_UCC}")
message(STATUS "  NVFUSER_BUILD_WITH_ASAN            : ${NVFUSER_BUILD_WITH_ASAN}")
message(STATUS "  NVFUSER_DISTRIBUTED                : ${NVFUSER_DISTRIBUTED}")
message(STATUS "  NVFUSER_HOST_IR_JIT                : ${NVFUSER_HOST_IR_JIT}")
message(STATUS "  NVFUSER_CPP_STANDARD               : ${NVFUSER_CPP_STANDARD}")

if(NVFUSER_STANDALONE_BUILD_WITH_UCC)
  message(STATUS "    UCC_HOME: $ENV{UCC_HOME}")
  message(STATUS "    UCC_DIR : $ENV{UCC_DIR}")
  message(STATUS "    UCX_HOME: $ENV{UCX_HOME}")
  message(STATUS "    UCX_DIR : $ENV{UCX_DIR}")
endif()

message(STATUS "******** End of Nvfuser configuration summary ********")
