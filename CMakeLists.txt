# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nvfuser)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# compensating the lack of PROJECT_IS_TOP_LEVEL for older cmake version
if (CMAKE_VERSION VERSION_LESS 3.21)
    if ("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
        set (PROJECT_IS_TOP_LEVEL ON)
    endif ()
endif ()

set(NVFUSER_ROOT ${PROJECT_SOURCE_DIR})
set(NVFUSER_SRCS_DIR "${NVFUSER_ROOT}/csrc")
if (PROJECT_IS_TOP_LEVEL)
  find_package(Torch REQUIRED)
  find_package(Python REQUIRED Development Interpreter)
  find_package(pybind11 REQUIRED)
  # need this since the pytorch execution uses a different name
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  set(ATEN_CUDA_ROOT "${TORCH_INSTALL_PREFIX}/include/ATen")
  # CXX flags is necessary since https://github.com/pytorch/pytorch/issues/98093
  string(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
  include(cmake/FlatBuffers.cmake)
  if(BUILD_NVFUSER_BENCHMARK)
    include(cmake/Dependencies.cmake)
  endif()
  # set CUDA_ARCH for cu tests.
  if(EXISTS ${TORCH_CUDA_ARCH_LIST})
    set(ARCH_FLAGS)
    cuda_select_nvcc_arch_flags(ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  endif()
else() # PROJECT_IS_TOP_LEVEL
  # TODO: move this to the top~
  # if we are included by other project and `BUILD_NVFUSER` is not set, we skip
  if (NOT BUILD_NVFUSER)
    return()
  endif()

  if(NOT USE_ROCM)
    set(TORCHLIB_FLAVOR torch_cuda)
  else()
    set(TORCHLIB_FLAVOR torch_hip)
  endif()

  # TODO: have TORCH_ROOT setup as a variable instead
  # currently we are expecting nvfuser to be added from the pytorch root cmake file.
  set(TORCH_ROOT "${CMAKE_SOURCE_DIR}")
  set(ATEN_CUDA_ROOT "${TORCH_ROOT}/aten/src/ATen")
  set(TORCH_INSTALL_LIB_DIR ${TORCH_ROOT}/torch/lib)
  find_package(CUDAToolkit REQUIRED)
endif() # PROJECT_IS_TOP_LEVEL

# TODO: fix MSVC
if(NOT MSVC)
  find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
endif()

#------------------------------
# build nvfuser_codegen library
#------------------------------

# nvfuser codegen sources
set(NVFUSER_SRCS)
list(APPEND NVFUSER_SRCS
    ${NVFUSER_SRCS_DIR}/assume.cpp
    ${NVFUSER_SRCS_DIR}/compute_at.cpp
    ${NVFUSER_SRCS_DIR}/inlining.cpp
    ${NVFUSER_SRCS_DIR}/compute_at_map.cpp
    ${NVFUSER_SRCS_DIR}/codegen.cpp
    ${NVFUSER_SRCS_DIR}/contiguity.cpp
    ${NVFUSER_SRCS_DIR}/dispatch.cpp
    ${NVFUSER_SRCS_DIR}/dynamic_transform.cpp
    ${NVFUSER_SRCS_DIR}/expr_evaluator.cpp
    ${NVFUSER_SRCS_DIR}/expr_simplifier.cpp
    ${NVFUSER_SRCS_DIR}/executor.cpp
    ${NVFUSER_SRCS_DIR}/executor_kernel_arg.cpp
    ${NVFUSER_SRCS_DIR}/executor_params.cpp
    ${NVFUSER_SRCS_DIR}/evaluator_common.cpp
    ${NVFUSER_SRCS_DIR}/executor_utils.cpp
    ${NVFUSER_SRCS_DIR}/fusion.cpp
    ${NVFUSER_SRCS_DIR}/graph_fuser.cpp
    ${NVFUSER_SRCS_DIR}/grouped_reduction.cpp
    ${NVFUSER_SRCS_DIR}/index_compute.cpp
    ${NVFUSER_SRCS_DIR}/instrumentation.cpp
    ${NVFUSER_SRCS_DIR}/ir/base_nodes.cpp
    ${NVFUSER_SRCS_DIR}/ir/builder.cpp
    ${NVFUSER_SRCS_DIR}/ir/cloner.cpp
    ${NVFUSER_SRCS_DIR}/ir/container.cpp
    ${NVFUSER_SRCS_DIR}/ir/graphviz.cpp
    ${NVFUSER_SRCS_DIR}/ir/iostream.cpp
    ${NVFUSER_SRCS_DIR}/ir/utils.cpp
    ${NVFUSER_SRCS_DIR}/ir/nodes.cpp
    ${NVFUSER_SRCS_DIR}/iter_visitor.cpp
    ${NVFUSER_SRCS_DIR}/kernel.cpp
    ${NVFUSER_SRCS_DIR}/kernel_cache.cpp
    ${NVFUSER_SRCS_DIR}/kernel_db/kernel_db.cpp
    ${NVFUSER_SRCS_DIR}/kernel_db/utils.cpp
    ${NVFUSER_SRCS_DIR}/kernel_ir.cpp
    ${NVFUSER_SRCS_DIR}/kernel_ir_dispatch.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/index_compute.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/divisible_split.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/fused_reduction.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/predicate_elimination.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/shift.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/sync_information.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/thread_predicate.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/trivial_broadcast.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/analysis/bank_conflict.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/alias_memory.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/allocation.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/double_buffer.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/expr_sort.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/fusion_simplifier.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/index.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/scalar_hoist.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/insert_syncs.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/instrument.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/loop_rotation.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/loops.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/magic_zero.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/misaligned_vectorization.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/predicate.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/replace_size.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/unroll.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/vectorize_welford.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/pass/warp_reduce.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/utils.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/validation.cpp
    ${NVFUSER_SRCS_DIR}/device_lower/lower2device.cpp
    ${NVFUSER_SRCS_DIR}/manager.cpp
    ${NVFUSER_SRCS_DIR}/maxinfo_propagator.cpp
    ${NVFUSER_SRCS_DIR}/multidevice/aggregate_dag.cpp
    ${NVFUSER_SRCS_DIR}/multidevice/multidevice_runtime.cpp
    ${NVFUSER_SRCS_DIR}/multidevice/multicluster_fusion.cpp
    ${NVFUSER_SRCS_DIR}/multidevice/ProcessGroupBuilder.cpp
    ${NVFUSER_SRCS_DIR}/mutator.cpp
    ${NVFUSER_SRCS_DIR}/non_divisible_split.cpp
    ${NVFUSER_SRCS_DIR}/ops/alias.cpp
    ${NVFUSER_SRCS_DIR}/ops/arith.cpp
    ${NVFUSER_SRCS_DIR}/ops/composite.cpp
    ${NVFUSER_SRCS_DIR}/ops/indexing.cpp
    ${NVFUSER_SRCS_DIR}/ops/normalization.cpp
    ${NVFUSER_SRCS_DIR}/ops/utils.cpp
    ${NVFUSER_SRCS_DIR}/parallel_dimension_map.cpp
    ${NVFUSER_SRCS_DIR}/parallel_type_bitmap.cpp
    ${NVFUSER_SRCS_DIR}/parser.cpp
    ${NVFUSER_SRCS_DIR}/partial_split_map.cpp
    ${NVFUSER_SRCS_DIR}/partition.cpp
    ${NVFUSER_SRCS_DIR}/predicate_compute.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_cache.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_definition.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/fusion_state.cpp
    ${NVFUSER_SRCS_DIR}/register_interface.cpp
    ${NVFUSER_SRCS_DIR}/root_domain_map.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/pointwise.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/pointwise_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/transpose.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/matmul.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/matmul_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/normalization.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/normalization_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/reduction.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/reduction_utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/registry.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/utils.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/vectorize_helper.cpp
    ${NVFUSER_SRCS_DIR}/serde/fusion_record_serde.cpp
    ${NVFUSER_SRCS_DIR}/swizzle.cpp
    ${NVFUSER_SRCS_DIR}/sys_utils.cpp
    ${NVFUSER_SRCS_DIR}/type_inference.cpp
    ${NVFUSER_SRCS_DIR}/type_promotion.cpp
    ${NVFUSER_SRCS_DIR}/fusion_segmenter.cpp
    ${NVFUSER_SRCS_DIR}/tensor_view.cpp
    ${NVFUSER_SRCS_DIR}/transform_iter.cpp
    ${NVFUSER_SRCS_DIR}/transform_replay.cpp
    ${NVFUSER_SRCS_DIR}/transform_rfactor.cpp
    ${NVFUSER_SRCS_DIR}/transform_view.cpp
    ${NVFUSER_SRCS_DIR}/type.cpp
    ${NVFUSER_SRCS_DIR}/utils.cpp
    ${NVFUSER_SRCS_DIR}/mma_type.cpp
    ${NVFUSER_SRCS_DIR}/scheduler/mma_utils.cpp
)

set(NVFUSER_CODEGEN ${PROJECT_NAME}_codegen)
add_library(${NVFUSER_CODEGEN} SHARED ${NVFUSER_SRCS})
target_compile_options(${NVFUSER_CODEGEN} PRIVATE -Wall -Wno-unused-function)
target_compile_options(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
if(NOT MSVC)
  target_compile_options(${NVFUSER_CODEGEN} PRIVATE -Werror)
endif()
# NB: This must be target_compile_definitions, not target_compile_options,
# as the latter is not respected by nvcc
target_compile_definitions(${NVFUSER_CODEGEN} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")

# Link flatbuffers for serialization support
target_link_libraries(${NVFUSER_CODEGEN} PRIVATE flatbuffers)
# For kernel_db, linking STL Filesystem Library for backward compatability with C++14
target_link_libraries(${NVFUSER_CODEGEN} PRIVATE stdc++fs)
target_link_libraries(${NVFUSER_CODEGEN} PRIVATE ${CUDA_NVRTC_LIB} ${LIBNVTOOLSEXT})
target_include_directories(${NVFUSER_CODEGEN} PRIVATE ${CUDA_INCLUDE_DIRS})
# TODO: we should guard the include of gloo
target_include_directories(${NVFUSER_CODEGEN} PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gloo)
target_include_directories(${NVFUSER_CODEGEN} PUBLIC
                          "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
                          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
                          )
set_property(TARGET ${NVFUSER_CODEGEN} PROPERTY CXX_STANDARD 17)

if (PROJECT_IS_TOP_LEVEL)
  target_link_libraries(${NVFUSER_CODEGEN} PRIVATE torch ${TORCH_LIBRARIES})
  # this is to find pip installed nvrtc .so
  set_target_properties(${NVFUSER_CODEGEN} PROPERTIES INSTALL_RPATH "$ORIGIN/../../nvidia/cuda_nvrtc/lib")
  install(TARGETS ${NVFUSER_CODEGEN} EXPORT NvfuserTargets DESTINATION lib)

  # We are keeping fusion_cache_generated.h for the submodule build because flatc is unavailable.
  add_custom_command(
    OUTPUT
      ${NVFUSER_ROOT}/csrc/serde/fusion_cache_generated.h
    DEPENDS
      ${NVFUSER_ROOT}/csrc/serde/fusion_cache.fbs
    DEPENDS flatc
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/third_party/flatbuffers/flatc -o ${NVFUSER_ROOT}/csrc/serde/ -c -b ${NVFUSER_ROOT}/csrc/serde/fusion_cache.fbs
    COMMENT "Generating fusion_cache_generated header from fusion_cache.fbs"
    VERBATIM
    )
  add_custom_target(build_flatbuffer_config ALL
      DEPENDS ${NVFUSER_ROOT}/csrc/serde/fusion_cache_generated.h)

  add_dependencies(${NVFUSER_CODEGEN} flatc build_flatbuffer_config)
else()
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/nvfuser")
  target_link_libraries(${NVFUSER_CODEGEN} PRIVATE torch ${TORCHLIB_FLAVOR})
  # NOTE: our CI build somehow fails with find driver lib.
  target_link_libraries(${NVFUSER_CODEGEN} PUBLIC CUDA::cuda_driver)
  # installing nvfuser python tests
  install(DIRECTORY "${NVFUSER_ROOT}/python_tests/"
  	DESTINATION "${TORCH_ROOT}/test/_nvfuser"
          FILES_MATCHING PATTERN "*.py" )
  # installing nvfuser lib
  install(TARGETS ${NVFUSER_CODEGEN} EXPORT NvfuserTargets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

# installing nvfuser headers
install(DIRECTORY "${NVFUSER_SRCS_DIR}/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvfuser"
      FILES_MATCHING PATTERN "*.h" )

#-----------------------------
# build nvfuser python library
#-----------------------------

if(BUILD_PYTHON)
  # nvfuser python API sources
  set(NVFUSER_PYTHON_SRCS)
  list(APPEND NVFUSER_PYTHON_SRCS
      ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings.cpp
      ${NVFUSER_SRCS_DIR}/python_frontend/python_bindings_extension.cpp
  )

    set(NVFUSER "${PROJECT_NAME}")
    add_library(${NVFUSER} MODULE ${NVFUSER_PYTHON_SRCS})
    set_property(TARGET ${NVFUSER} PROPERTY CXX_STANDARD 17)

    # setup python API version
    add_custom_command(
      OUTPUT ${NVFUSER_ROOT}/nvfuser/version.py
      COMMAND
        "${PYTHON_EXECUTABLE}" -c \"from pathlib import Path\; Path('${NVFUSER_ROOT}/tools/gen_nvfuser_version.py').touch()\"
      COMMAND
        "${PYTHON_EXECUTABLE}" ${NVFUSER_ROOT}/tools/gen_nvfuser_version.py
      DEPENDS ${NVFUSER_ROOT}/tools/gen_nvfuser_version.py
      WORKING_DIRECTORY ${NVFUSER_ROOT}/tools/
    )
    add_custom_target(
      gen_nvfuser_version ALL
      DEPENDS ${NVFUSER_ROOT}/nvfuser/version.py
    )
    add_dependencies(${NVFUSER} gen_nvfuser_version)

    target_compile_options(${NVFUSER} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
    # NB: This must be target_compile_definitions, not target_compile_options,
    # as the latter is not respected by nvcc
    target_compile_definitions(${NVFUSER} PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
    if(NOT MSVC)
      target_compile_options(${NVFUSER} PRIVATE -Werror)
      set_target_properties(${NVFUSER} PROPERTIES SUFFIX ".so")
    else()
      set_target_properties(${NVFUSER} PROPERTIES SUFFIX ".pyd")
    endif()

    target_link_libraries(${NVFUSER} PRIVATE ${LIBNVTOOLSEXT})
    target_link_libraries(${NVFUSER} PRIVATE ${NVFUSER_CODEGEN})

    target_compile_definitions(${NVFUSER} PRIVATE EXTENSION_NAME=_C)

  if (PROJECT_IS_TOP_LEVEL)
    target_compile_options(${NVFUSER} PRIVATE -Wall -Wno-unused-function)
    target_link_libraries(${NVFUSER} PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(${NVFUSER} PRIVATE "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")
    target_link_libraries(${NVFUSER} PRIVATE pybind11::pybind11)
    set_target_properties(${NVFUSER} PROPERTIES INSTALL_RPATH "$ORIGIN/../torch/lib")
    install(TARGETS ${NVFUSER} DESTINATION lib)
  else() # PROJECT_IS_TOP_LEVEL
    torch_compile_options(${NVFUSER})

    target_link_libraries(${NVFUSER} PRIVATE torch torch_python ${TORCHLIB_FLAVOR})
    target_include_directories(${NVFUSER} PRIVATE ${TORCH_ROOT})


    target_link_libraries(${NVFUSER} PRIVATE pybind::pybind11)
    target_compile_options(${NVFUSER} PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

    # avoid using Python3_add_library, copied from functorch
    set_target_properties(${NVFUSER} PROPERTIES PREFIX "" DEBUG_POSTFIX "")
    set_target_properties(${NVFUSER} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
          ${CMAKE_BINARY_DIR}/nvfuser)
    set_target_properties(${NVFUSER} PROPERTIES INSTALL_RPATH "${_rpath_portable_origin}/../torch/lib")
    if(TORCH_PYTHON_LINK_FLAGS AND NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
      set_target_properties(${NVFUSER} PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
    endif()
    install(TARGETS ${NVFUSER} EXPORT NvfuserTargets DESTINATION ${TORCH_ROOT}/nvfuser/)

    # install nvfuser python files
    install(DIRECTORY "${NVFUSER_ROOT}/nvfuser/"
            DESTINATION "${TORCH_ROOT}/nvfuser"
            FILES_MATCHING PATTERN "*.py" )
    file(WRITE "${TORCH_ROOT}/nvfuser/.gitignore" "*")
  endif()
endif()


# -- build tests

if(BUILD_TEST)
  # nvfuser test sources
  set(JIT_TEST_SRCS)
  set(JIT_TEST_CU_SRCS)
  list(APPEND JIT_TEST_SRCS
    ${NVFUSER_SRCS_DIR}/kernel_db/test/test_nvfuser_kernel_db_open.cpp
    ${NVFUSER_SRCS_DIR}/kernel_db/test/test_nvfuser_kernel_db_query.cpp
    ${NVFUSER_SRCS_DIR}/kernel_db/test/test_nvfuser_kernel_db_write.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_definition.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_cache.cpp
    ${NVFUSER_SRCS_DIR}/python_frontend/test/test_nvfuser_fusion_record.cpp
    ${NVFUSER_ROOT}/test/main.cpp
    ${NVFUSER_ROOT}/test/utils.cpp
    ${NVFUSER_ROOT}/test/test_allocation_domain.cpp
    ${NVFUSER_ROOT}/test/test_dynamic_transform.cpp
    ${NVFUSER_ROOT}/test/test_gather.cpp
    ${NVFUSER_ROOT}/test/test_gpu1.cpp
    ${NVFUSER_ROOT}/test/test_gpu2.cpp
    ${NVFUSER_ROOT}/test/test_gpu3.cpp
    ${NVFUSER_ROOT}/test/test_gpu_compute_with.cpp
    ${NVFUSER_ROOT}/test/test_expr_simplifier.cpp
    ${NVFUSER_ROOT}/test/test_external_src.cpp
    ${NVFUSER_ROOT}/test/test_swizzle.cpp
    ${NVFUSER_ROOT}/test/test_gpu_tensor_factories.cpp
    ${NVFUSER_ROOT}/test/test_gpu_fused_reduction.cpp
    ${NVFUSER_ROOT}/test/test_gpu_outer_reduction.cpp
    ${NVFUSER_ROOT}/test/test_loop_rotation.cpp
    ${NVFUSER_ROOT}/test/test_gpu_shift.cpp
    ${NVFUSER_ROOT}/test/test_resize.cpp
    ${NVFUSER_ROOT}/test/test_gpu_tensorcore.cpp
    ${NVFUSER_ROOT}/test/test_matmul_sass.cpp
    ${NVFUSER_ROOT}/test/test_gpu_view.cpp
    ${NVFUSER_ROOT}/test/test_gpu_transpose.cpp
    ${NVFUSER_ROOT}/test/test_gpu_utils.cpp
    ${NVFUSER_ROOT}/test/test_gpu_indexing_ops.cpp
    ${NVFUSER_ROOT}/test/test_gpu_indexing.cpp
    ${NVFUSER_ROOT}/test/test_gpu_multidevice.cpp
    ${NVFUSER_ROOT}/test/test_multicluster_fusion.cpp
    ${NVFUSER_ROOT}/test/test_combined_inner_outer_reduction.cpp
  )
  list(APPEND JIT_TEST_CU_SRCS ${NVFUSER_ROOT}/test/test_gpu_rng.cu)

  set(NVFUSER_TESTS "${PROJECT_NAME}_tests")

  add_executable(${NVFUSER_TESTS}
             ${JIT_TEST_SRCS}
             ${JIT_TEST_CU_SRCS})
  set_property(TARGET ${NVFUSER_TESTS} PROPERTY CXX_STANDARD 17)
  target_compile_definitions(${NVFUSER_TESTS} PRIVATE USE_GTEST)
  target_link_libraries(${NVFUSER_TESTS} PRIVATE ${NVFUSER_CODEGEN} gtest_main gmock_main flatbuffers)
  target_include_directories(${NVFUSER_TESTS} PRIVATE "${NVFUSER_ROOT}")

  if (PROJECT_IS_TOP_LEVEL)
    target_compile_options(${NVFUSER_TESTS} PRIVATE -Wall -Wno-unused-function)
    target_link_libraries(${NVFUSER_TESTS} PRIVATE ${TORCH_LIBRARIES})
    # only install nvfuser_tests with submodule build
  else()
    torch_compile_options(${NVFUSER_TESTS})
    target_include_directories(${NVFUSER_TESTS} PRIVATE "${TORCH_ROOT}/torch/csrc/api/include/")
    target_link_libraries(${NVFUSER_TESTS} PRIVATE torch ${TORCHLIB_FLAVOR})
    install(TARGETS ${NVFUSER_TESTS} DESTINATION bin)
  endif() # PROJECT_IS_TOP_LEVEL

  if(NOT MSVC)
    set_property(SOURCE ${JIT_TEST_SRCS} APPEND PROPERTY COMPILE_OPTIONS "-Werror")
  endif()
endif()

# -- build benchmark
if(BUILD_NVFUSER_BENCHMARK)
  # nvfuser benchmark sources
  set(BENCHMARK_SRCS)
  list(APPEND BENCHMARK_SRCS
    ${NVFUSER_ROOT}/benchmark/batch_norm_channels_first.cpp
    ${NVFUSER_ROOT}/benchmark/batch_norm_channels_first_backward.cpp
    ${NVFUSER_ROOT}/benchmark/batch_norm_channels_last.cpp
    ${NVFUSER_ROOT}/benchmark/batch_norm_channels_last_backward.cpp
    ${NVFUSER_ROOT}/benchmark/bert.cpp
    ${NVFUSER_ROOT}/benchmark/broadcast.cpp
    ${NVFUSER_ROOT}/benchmark/gelu_backward.cpp
    ${NVFUSER_ROOT}/benchmark/gelu_backward_reduction.cpp
    ${NVFUSER_ROOT}/benchmark/heuristic_lookup.cpp
    ${NVFUSER_ROOT}/benchmark/shape_inference.cpp
    ${NVFUSER_ROOT}/benchmark/instance_norm.cpp
    ${NVFUSER_ROOT}/benchmark/many_pointwise_ops.cpp
    ${NVFUSER_ROOT}/benchmark/layer_norm.cpp
    ${NVFUSER_ROOT}/benchmark/layer_norm_backward.cpp
    ${NVFUSER_ROOT}/benchmark/layer_norm_fused.cpp
    ${NVFUSER_ROOT}/benchmark/rms_norm.cpp
    ${NVFUSER_ROOT}/benchmark/rms_norm_backward.cpp
    ${NVFUSER_ROOT}/benchmark/lstm_cell.cpp
    ${NVFUSER_ROOT}/benchmark/reduction.cpp
    ${NVFUSER_ROOT}/benchmark/softmax.cpp
    ${NVFUSER_ROOT}/benchmark/softmax_backward.cpp
    ${NVFUSER_ROOT}/benchmark/scale_bias_relu.cpp
    ${NVFUSER_ROOT}/benchmark/transpose.cpp
    ${NVFUSER_ROOT}/benchmark/matmul.cpp
    ${NVFUSER_ROOT}/benchmark/timm.cpp
    ${NVFUSER_ROOT}/benchmark/indexselect.cpp
    ${NVFUSER_ROOT}/benchmark/utils.cpp
    ${NVFUSER_ROOT}/benchmark/main.cpp
    ${NVFUSER_ROOT}/test/utils.cpp
  )

  set(NVFUSER_BENCHMARK "${PROJECT_NAME}_bench")
  add_executable(${NVFUSER_BENCHMARK} ${BENCHMARK_SRCS})
  set_property(TARGET ${NVFUSER_BENCHMARK} PROPERTY CXX_STANDARD 17)
  if(PROJECT_IS_TOP_LEVEL)
    target_compile_options(${NVFUSER_BENCHMARK} PRIVATE -Wall -Wno-unused-function)
    target_link_libraries(${NVFUSER_BENCHMARK} PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(${NVFUSER_BENCHMARK} PRIVATE benchmark::benchmark)
    # only install nvfuser_bench with submodule build
  else()
    torch_compile_options(${NVFUSER_BENCHMARK})
    target_link_libraries(${NVFUSER_BENCHMARK} PRIVATE torch_library benchmark)
    install(TARGETS ${NVFUSER_BENCHMARK} DESTINATION bin)
  endif()
  if(NOT MSVC)
    target_compile_options(${NVFUSER_BENCHMARK} PRIVATE -Werror -Wno-deprecated-copy)
  endif()
  target_link_libraries(${NVFUSER_BENCHMARK} PRIVATE ${NVFUSER_CODEGEN})
  target_include_directories(${NVFUSER_BENCHMARK} PRIVATE ${NVFUSER_ROOT})

endif()

# --- generate runtime files
# nvfuser runtime files
set(NVFUSER_RUNTIME_FILES)
list(APPEND NVFUSER_RUNTIME_FILES
  ${NVFUSER_ROOT}/runtime/array.cu
  ${NVFUSER_ROOT}/runtime/basic_type_traits.cu
  ${NVFUSER_ROOT}/runtime/bf16_support.cu
  ${NVFUSER_ROOT}/runtime/block_reduction.cu
  ${NVFUSER_ROOT}/runtime/block_sync_atomic.cu
  ${NVFUSER_ROOT}/runtime/block_sync_default.cu
  ${NVFUSER_ROOT}/runtime/block_welford_outer.cu
  ${NVFUSER_ROOT}/runtime/broadcast.cu
  ${NVFUSER_ROOT}/runtime/complex_number.cu
  ${NVFUSER_ROOT}/runtime/fp16_support.cu
  ${NVFUSER_ROOT}/runtime/fused_reduction.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_helper.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_impl.cu
  ${NVFUSER_ROOT}/runtime/fused_welford_impl_outer.cu
  ${NVFUSER_ROOT}/runtime/grid_broadcast.cu
  ${NVFUSER_ROOT}/runtime/grid_reduction.cu
  ${NVFUSER_ROOT}/runtime/grid_sync.cu
  ${NVFUSER_ROOT}/runtime/helpers.cu
  ${NVFUSER_ROOT}/runtime/index_utils.cu
  ${NVFUSER_ROOT}/runtime/memory.cu
  ${NVFUSER_ROOT}/runtime/random_numbers.cu
  ${NVFUSER_ROOT}/runtime/tensor.cu
  ${NVFUSER_ROOT}/runtime/tensorcore.cu
  ${NVFUSER_ROOT}/runtime/tuple.cu
  ${NVFUSER_ROOT}/runtime/type_traits.cu
  ${NVFUSER_ROOT}/runtime/warp.cu
  ${NVFUSER_ROOT}/runtime/welford.cu
  ${ATEN_CUDA_ROOT}/cuda/detail/PhiloxCudaStateRaw.cuh
  ${ATEN_CUDA_ROOT}/cuda/detail/UnpackRaw.cuh
)


file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/nvfuser_resources")

# "stringify" NVFUSER runtime sources
# (generate C++ header files embedding the original input as a string literal)
set(NVFUSER_STRINGIFY_TOOL "${NVFUSER_ROOT}/tools/stringify_file.py")
foreach(src ${NVFUSER_RUNTIME_FILES})
  get_filename_component(filename ${src} NAME_WE)
  set(dst "${CMAKE_BINARY_DIR}/include/nvfuser_resources/${filename}.h")
  add_custom_command(
    COMMENT "Stringify NVFUSER runtime source file"
    OUTPUT ${dst}
    DEPENDS ${src} "${NVFUSER_STRINGIFY_TOOL}"
    COMMAND ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst}
  )
  add_custom_target(nvfuser_rt_${filename} DEPENDS ${dst})
  add_dependencies(${NVFUSER_CODEGEN} nvfuser_rt_${filename})

  # also generate the resource headers during the configuration step
  # (so tools like clang-tidy can run w/o requiring a real build)
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} ${NVFUSER_STRINGIFY_TOOL} -i ${src} -o ${dst})
endforeach()

target_include_directories(${NVFUSER_CODEGEN} PRIVATE "${CMAKE_BINARY_DIR}/include")

# -- install nvfuser cmake config files and symlink to build binaries
install(EXPORT NvfuserTargets FILE NvfuserConfig.cmake DESTINATION share/cmake/nvfuser)
if (PROJECT_IS_TOP_LEVEL)
  file(CREATE_LINK "${CMAKE_BINARY_DIR}" "${NVFUSER_ROOT}/bin" SYMBOLIC)
else()
  file(CREATE_LINK "${CMAKE_BINARY_DIR}/bin" "${NVFUSER_ROOT}/bin" SYMBOLIC)
  file(CREATE_LINK "${CMAKE_BINARY_DIR}" "${NVFUSER_ROOT}/build" SYMBOLIC)
endif()
