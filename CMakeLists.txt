# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nvfuser)

cmake_policy(SET CMP0063 NEW) # make symbol visibility always apply

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NVFUSER_ROOT ${PROJECT_SOURCE_DIR})
set(NVFUSER_SRCS_DIR "${NVFUSER_ROOT}/csrc")
set(NVFUSER_THIRD_PARTY_DIR "${NVFUSER_ROOT}/third_party")

option(NVFUSER_STANDALONE_BUILD_WITH_UCC "" OFF)
option(NVFUSER_BUILD_WITH_ASAN "Build nvFuser with asan" OFF)

# We try to update which C++ standard we use together in lockstep across all
# built libraries, and these variables control which that is. Generally we are
# on C++20, but we still support a version of CUDA (11) that does not recognize
# C++20 and so we drop back to 17 there. Also, we allow all of these to be
# overridden by the user.
# Note we do not use a global set_property on e.g. CXX_STANDARD. CMake globals
# are footguns that should generally be avoided, because they are difficult to
# target where and *only* where they are needed. See e.g.:
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/dodonot.html
set(NVFUSER_C_STANDARD 17 CACHE STRING "C standard to use for C code")
set(NVFUSER_CPP_STANDARD 20 CACHE STRING "C++ standard to use for C++ code")
set(NVFUSER_CUDA_STANDARD 17 CACHE STRING "C++ standard to use for CUDA code")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.4)
    message(FATAL_ERROR "GCC < 11.4 has compiler bugs and can not compile nvFuser.")
  endif()
endif()

string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")

find_package(Torch REQUIRED)
find_package(Python REQUIRED Development.Module Interpreter)
find_package(pybind11 REQUIRED)

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

# TODO: fix MSVC
if(NOT MSVC)
  find_library(LIBCUPTI libcupti.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/ ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
  find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
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
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/device_version.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/divisible_split.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/fused_reduction.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/index_compute.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/predicate_elimination.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/shift.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/sync_information.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/thread_predicate.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/analysis/trivial_broadcast.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/lower2device.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/alias_memory.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/allocation.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/double_buffer.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/expr_sort.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/fusion_simplifier.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/grid_serialization.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/index.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/inline_ptx.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/insert_syncs.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/instrument.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/loop_rotation.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/loops.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/magic_zero.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/misaligned_vectorization.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/predicate.cpp
  ${NVFUSER_SRCS_DIR}/device_lower/pass/replace_size.cpp
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
  ${NVFUSER_SRCS_DIR}/executor.cpp
  ${NVFUSER_SRCS_DIR}/executor_kernel_arg.cpp
  ${NVFUSER_SRCS_DIR}/executor_params.cpp
  ${NVFUSER_SRCS_DIR}/executor_utils.cpp
  ${NVFUSER_SRCS_DIR}/expr_evaluator.cpp
  ${NVFUSER_SRCS_DIR}/expr_simplifier.cpp
  ${NVFUSER_SRCS_DIR}/fusion.cpp
  ${NVFUSER_SRCS_DIR}/fusion_segmenter.cpp
  ${NVFUSER_SRCS_DIR}/global_allocator.cpp
  ${NVFUSER_SRCS_DIR}/grouped_reduction.cpp
  ${NVFUSER_SRCS_DIR}/id_model/id_model.cpp
  ${NVFUSER_SRCS_DIR}/id_model/to_string.cpp
  ${NVFUSER_SRCS_DIR}/id_model/transform_replay.cpp
  ${NVFUSER_SRCS_DIR}/id_model/validation_utils.cpp
  ${NVFUSER_SRCS_DIR}/index_compute.cpp
  ${NVFUSER_SRCS_DIR}/inlining.cpp
  ${NVFUSER_SRCS_DIR}/instrumentation.cpp
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
  ${NVFUSER_SRCS_DIR}/kernel_cache.cpp
  ${NVFUSER_SRCS_DIR}/kernel_db/kernel_db.cpp
  ${NVFUSER_SRCS_DIR}/kernel_db/utils.cpp
  ${NVFUSER_SRCS_DIR}/kernel_ir.cpp
  ${NVFUSER_SRCS_DIR}/kernel_ir_dispatch.cpp
  ${NVFUSER_SRCS_DIR}/maxinfo_propagator.cpp
  ${NVFUSER_SRCS_DIR}/mma_type.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/communication.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/communicator.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/device_mesh.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/executor.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/lower_communication.cpp
  ${NVFUSER_SRCS_DIR}/multidevice/utils.cpp
  ${NVFUSER_SRCS_DIR}/mutator.cpp
  ${NVFUSER_SRCS_DIR}/non_divisible_split.cpp
  ${NVFUSER_SRCS_DIR}/ops/alias.cpp
  ${NVFUSER_SRCS_DIR}/ops/arith.cpp
  ${NVFUSER_SRCS_DIR}/ops/composite.cpp
  ${NVFUSER_SRCS_DIR}/ops/indexing.cpp
  ${NVFUSER_SRCS_DIR}/ops/normalization.cpp
  ${NVFUSER_SRCS_DIR}/ops/utils.cpp
  ${NVFUSER_SRCS_DIR}/options.cpp
  ${NVFUSER_SRCS_DIR}/parallel_dimension_map.cpp
  ${NVFUSER_SRCS_DIR}/parallel_type_bitmap.cpp
  ${NVFUSER_SRCS_DIR}/partial_split_map.cpp
  ${NVFUSER_SRCS_DIR}/predicate_compute.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/add_axioms.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/allocation_order_inference.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/consecutive_cast.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/exact_mapped_extent_substitution.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/mark_aliases_prepare.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/move_split_cat.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/pre_segmenter.cpp
  ${NVFUSER_SRCS_DIR}/preseg_passes/remove_empty.cpp
  ${NVFUSER_SRCS_DIR}/rng.cpp
  ${NVFUSER_SRCS_DIR}/root_domain_map.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/cache_policy_refiner.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/heuristic_types.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/mark_aliases.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_heuristic_plugin.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/matmul_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/mma_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/no_op.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_inner_outer.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_outer.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/normalization_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/pointwise.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/pointwise_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/reduction.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/reduction_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/registry.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/registry_utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/transpose.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/utils.cpp
  ${NVFUSER_SRCS_DIR}/scheduler/vectorize_helper.cpp
  ${NVFUSER_SRCS_DIR}/serde/polymorphic_value.cpp
  ${NVFUSER_SRCS_DIR}/serde/utils.cpp
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
)

# We don't link CUPTI for MSVC
if(NOT MSVC)
  list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/fusion_profiler.cpp
  )
endif()

if(BUILD_PYTHON)
  list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_cache.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_definition.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_state.cpp
    ${NVFUSER_SRCS_DIR}/serde/fusion_record.cpp
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
  target_compile_options(codegen_internal PRIVATE
    -Wall -Wno-unused-function
    -Werror
  )
endif()

target_compile_definitions(codegen_internal PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
target_include_directories(codegen_internal SYSTEM PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/gloo # TODO: guard this on usage
  ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
  PRIVATE
  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
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
  # this is to find pip installed nvrtc/nvtx .so
  INSTALL_RPATH
  "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/nvtx/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib"
  POSITION_INDEPENDENT_CODE Yes
  VISIBILITY_INLINES_HIDDEN Yes
)

# Ensure we don't link against libcuda; we'll dlopen it ourselves.
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "libcuda\.so")
target_link_libraries(codegen_internal PUBLIC
  dynamic_type
  ${LIBCUPTI}
  ${TORCH_LIBRARIES}
  dl
)

add_library(nvfuser_codegen SHARED $<TARGET_OBJECTS:codegen_internal>)
if(NVFUSER_BUILD_WITH_ASAN)
  target_compile_options(codegen_internal PRIVATE -fsanitize=address)
  target_link_options(codegen_internal PRIVATE -fsanitize=address)
  target_link_options(nvfuser_codegen PRIVATE -fsanitize=address)
endif()

target_include_directories(nvfuser_codegen PUBLIC
  "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
)
target_link_libraries(nvfuser_codegen PRIVATE
  flatbuffers
  ${CUDA_NVRTC_LIB}
  ${LIBNVTOOLSEXT}
  ${LIBCUPTI}
  ${TORCH_LIBRARIES}
  dl
)
set_target_properties(nvfuser_codegen PROPERTIES
  C_STANDARD ${NVFUSER_C_STANDARD}
  CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
  CXX_STANDARD ${NVFUSER_CPP_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  INSTALL_RPATH
  "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/nvtx/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib"
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
install(DIRECTORY "${NVFUSER_THIRD_PARTY_DIR}/flatbuffers/include/flatbuffers/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser/flatbuffers")

# installing dynamic_type headers
install(DIRECTORY "${NVFUSER_ROOT}/lib/dynamic_type/src/dynamic_type"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser")

# -----------------------------
# build nvfuser python library
# -----------------------------
if(BUILD_PYTHON)
  # nvfuser python API sources
  set(NVFUSER_PYTHON_SRCS)
  list(APPEND NVFUSER_PYTHON_SRCS
    ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings_extension.cpp
  )

  add_library(nvf_py_internal OBJECT ${NVFUSER_PYTHON_SRCS})
  target_include_directories(nvf_py_internal SYSTEM INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
  )

  # setup python API version
  add_custom_command(
    OUTPUT ${NVFUSER_ROOT}/nvfuser/version.py
    COMMAND
    "${PYTHON_EXECUTABLE}" -c \"from pathlib import Path\; Path('${NVFUSER_ROOT}/tools/gen_nvfuser_version.py') .touch() \"
    COMMAND
    "${PYTHON_EXECUTABLE}" ${NVFUSER_ROOT}/tools/gen_nvfuser_version.py
    DEPENDS ${NVFUSER_ROOT}/tools/gen_nvfuser_version.py
    DEPENDS ${NVFUSER_ROOT}/version.txt
    WORKING_DIRECTORY ${NVFUSER_ROOT}/tools/
  )
  add_custom_target(
    gen_nvfuser_version ALL
    DEPENDS ${NVFUSER_ROOT}/nvfuser/version.py
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
    "$ORIGIN/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../nvidia/nvtx/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
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
    "$ORIGIN/lib:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_nvrtc/lib:$ORIGIN/../nvidia/nvtx/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../torch/lib"
    POSITION_INDEPENDENT_CODE Yes
    VISIBILITY_INLINES_HIDDEN Yes
  )

  target_link_libraries(nvf_py_internal PRIVATE
    codegen_internal
    ${LIBNVTOOLSEXT}
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    pybind11::pybind11 pybind11::headers
  )

  target_link_libraries(nvfuser PRIVATE
    nvf_py_internal
    codegen_internal
    Python::Module
  )
  install(TARGETS nvfuser DESTINATION lib)
endif()

set(JIT_TEST_SRCS)
list(APPEND JIT_TEST_SRCS
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_open.cpp
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_query.cpp
  ${NVFUSER_ROOT}/tests/cpp/kernel_db/test_nvfuser_kernel_db_write.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_alias.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_allocation_domain.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_allocation_order_inference.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_ca_root_domain_map.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_combined_inner_outer_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_double_buffering.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_dynamic_transform.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_evaluator.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_exceptions.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_expr_simplifier.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_expr_sort.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gather.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu1.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu2.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu3.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_compute_with.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_fused_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_indexing.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_indexing_ops.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_outer_reduction.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_transpose.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_gpu_utils.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_id_model.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_iter_visitor.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_linked_hash_map.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_loop_rotation.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_mbarrier.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_memory.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_move_split_cat.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_no_op.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_persistent_buffer.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_pointwise.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_polymorphic_value.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_predicate_elimination.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_preseg_passes.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_replay.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_resharding.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_resize.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_reduction_pointwise.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_scalar_hoisting.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_segmentation.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_serial_gridreduce.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_sharding.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_smem_reuse.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_swizzle.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_tensor_factories.cpp
  ${NVFUSER_ROOT}/tests/cpp/test_unary.cpp
)

if(BUILD_TEST)
  set(RNG_TEST_KERNELS "${NVFUSER_TESTS}_kernels")
  add_library(${RNG_TEST_KERNELS} SHARED ${NVFUSER_ROOT}/tests/cpp/rng_kernels.cu)

  # CUDA 11 does not support C++20, so hard code C++17 here
  set_property(TARGET ${RNG_TEST_KERNELS} PROPERTY CXX_STANDARD 17)
  target_link_libraries(${RNG_TEST_KERNELS} PRIVATE torch ${TORCH_LIBRARIES} codegen_internal)
  target_include_directories(${RNG_TEST_KERNELS} PRIVATE "${NVFUSER_ROOT}")
endif()

function(add_test TEST_NAME TEST_SRC ADDITIONAL_LINK)
  list(APPEND TEST_SRC ${NVFUSER_ROOT}/tests/cpp/utils.cpp)
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
    GTest::gtest_main GTest::gmock_main
    flatbuffers
    ${TORCH_LIBRARIES}
  )

  if(NOT MSVC)
    target_compile_options(${TEST_NAME} PRIVATE
      -Wall -Wno-unused-function -Werror
    )
  endif()
endfunction()

if(BUILD_TEST)
  set(TEST_BINARIES)
  add_test(nvfuser_tests "${JIT_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES nvfuser_tests)

  add_test(test_rng ${NVFUSER_ROOT}/tests/cpp/test_rng.cpp ${RNG_TEST_KERNELS})
  list(APPEND TEST_BINARIES test_rng)

  set(MULTIDEVICE_TEST_SRCS)
  list(APPEND MULTIDEVICE_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/multidevice.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_communications.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_pipeline.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_multidevice_sharding.cpp
  )
  add_test(test_multidevice "${MULTIDEVICE_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_multidevice)

  add_test(test_view "${NVFUSER_ROOT}/tests/cpp/test_gpu_view.cpp" "")
  list(APPEND TEST_BINARIES test_view)

  set(MATMUL_TEST_SRCS)
  list(APPEND MATMUL_TEST_SRCS
    ${NVFUSER_ROOT}/tests/cpp/test_combine_mul_sum.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_gpu_tensorcore.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul_aten_evaluation.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul_sass.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_matmul_scheduler.cpp
    ${NVFUSER_ROOT}/tests/cpp/test_mma.cpp
  )
  add_test(test_matmul "${MATMUL_TEST_SRCS}" "")
  list(APPEND TEST_BINARIES test_matmul)

  add_test(test_external_src "${NVFUSER_ROOT}/tests/cpp/test_external_src.cpp" "")
  list(APPEND TEST_BINARIES test_external_src)

  add_test(test_shift "${NVFUSER_ROOT}/tests/cpp/test_shift.cpp" "")
  list(APPEND TEST_BINARIES test_shift)

  add_test(tutorial "${NVFUSER_ROOT}/tests/cpp/test_tutorial.cpp" "")
  list(APPEND TEST_BINARIES tutorial)

  if(BUILD_PYTHON)
    set(PY_FRONTEND_TEST_SRCS)
    list(APPEND PY_FRONTEND_TEST_SRCS
      ${NVFUSER_ROOT}/tests/cpp/python_frontend/test_nvfuser_fusion_cache.cpp
      ${NVFUSER_ROOT}/tests/cpp/python_frontend/test_nvfuser_fusion_definition.cpp
      ${NVFUSER_ROOT}/tests/cpp/python_frontend/test_nvfuser_fusion_record.cpp
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
endif()

# --- generate runtime files
# nvfuser runtime files
set(NVFUSER_RUNTIME_FILES)
list(APPEND NVFUSER_RUNTIME_FILES
  ${NVFUSER_ROOT}/runtime/array.cu
  ${NVFUSER_ROOT}/runtime/basic_type_traits.cu
  ${NVFUSER_ROOT}/runtime/bf16_support.cu
  ${NVFUSER_ROOT}/runtime/bit.cu
  ${NVFUSER_ROOT}/runtime/block_reduction.cu
  ${NVFUSER_ROOT}/runtime/block_sync_atomic.cu
  ${NVFUSER_ROOT}/runtime/block_sync_default.cu
  ${NVFUSER_ROOT}/runtime/block_welford_outer.cu
  ${NVFUSER_ROOT}/runtime/broadcast.cu
  ${NVFUSER_ROOT}/runtime/complex_number.cu
  ${NVFUSER_ROOT}/runtime/fp16_support.cu
  ${NVFUSER_ROOT}/runtime/fp8_support.cu
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
  ${NVFUSER_ROOT}/runtime/tensor.cu
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
  if (${src} IS_NEWER_THAN ${dst})
    # also generate the resource headers during the configuration step
    # (so tools like clang-tidy can run w/o requiring a real build)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
  endif()
endforeach()

target_include_directories(codegen_internal PRIVATE "${CMAKE_BINARY_DIR}/include")

# -- install nvfuser cmake config files and symlink to build binaries
install(EXPORT NvfuserTargets FILE NvfuserConfig.cmake DESTINATION share/cmake/nvfuser)

file(CREATE_LINK "${CMAKE_BINARY_DIR}" "${NVFUSER_ROOT}/bin" SYMBOLIC)

message(STATUS "")
message(STATUS "******** Nvfuser configuration summary ********")
message(STATUS "  UCC_FOUND: ${UCC_FOUND}")
message(STATUS "  NVFUSER_STANDALONE_BUILD_WITH_UCC  : ${NVFUSER_STANDALONE_BUILD_WITH_UCC}")
message(STATUS "  NVFUSER_BUILD_WITH_ASAN            : ${NVFUSER_BUILD_WITH_ASAN}")
message(STATUS "  NVFUSER_CPP_STANDARD               : ${NVFUSER_CPP_STANDARD}")

if(NVFUSER_STANDALONE_BUILD_WITH_UCC)
  message(STATUS "    UCC_HOME: $ENV{UCC_HOME}")
  message(STATUS "    UCC_DIR : $ENV{UCC_DIR}")
  message(STATUS "    UCX_HOME: $ENV{UCX_HOME}")
  message(STATUS "    UCX_DIR : $ENV{UCX_DIR}")
endif()

message(STATUS "******** End of Nvfuser configuration summary ********")
