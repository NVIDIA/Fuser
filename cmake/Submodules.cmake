# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------
# gtest
# -----------------------------------------------------------------------------

message(STATUS "Configuring Google Test submodule ...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
# For gtest, we will simply embed it into our test binaries, so we will not need to install it.
set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)
set(gtest_hide_internal_symbols ON CACHE BOOL "Use symbol visibility" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "End of Google Test configuration.")
message()

# -----------------------------------------------------------------------------
# benchmark
# -----------------------------------------------------------------------------

message(STATUS "Setting up Google Benchmark submodule ...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
# We will not need to test benchmark lib itself.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
# We will not need to install benchmark since we link it statically.
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")

if(NOT USE_SYSTEM_BENCHMARK)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
else()
  add_library(benchmark SHARED IMPORTED)
  find_library(BENCHMARK_LIBRARY benchmark)
  if(NOT BENCHMARK_LIBRARY)
    message(FATAL_ERROR "Cannot find google benchmark library")
  endif()
  message("-- Found benchmark: ${BENCHMARK_LIBRARY}")
  set_property(TARGET benchmark PROPERTY IMPORTED_LOCATION ${BENCHMARK_LIBRARY})
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "End of Google Benchmark configuration.")
message()

# -----------------------------------------------------------------------------
# FlatBuffer
# -----------------------------------------------------------------------------

message(STATUS "Setting up FlatBuffer submodule ...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
set(FlatBuffers_Src_Dir ${PROJECT_SOURCE_DIR}/third_party/flatbuffers)

option(FLATBUFFERS_BUILD_TESTS "Enable the build of tests and samples." OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" ON)
option(FLATBUFFERS_STATIC_FLATC "Build flatbuffers compiler with -static flag" OFF)
option(FLATBUFFERS_BUILD_FLATHASH "Enable the build of flathash" OFF)

# Add FlatBuffers directly to our build. This defines the `flatbuffers` target.
add_subdirectory(${FlatBuffers_Src_Dir})
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "End of FlatBuffer configuration.")
message()

# -----------------------------------------------------------------------------
# CUTLASS
# -----------------------------------------------------------------------------

message(STATUS "Setting up CUTLASS submodule ...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
if(NVFUSER_DISABLE_CUTLASS)
  message(STATUS "CUTLASS Support DISABLED.")
  set(NVFUSER_USE_CUTLASS FALSE)
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(WARNING "CUTLASS Support DISABLED: Requires CUDA 12.8+")
  set(NVFUSER_USE_CUTLASS FALSE)
else()
  message(STATUS "CUTLASS Support ENABLED.")
  set(NVFUSER_USE_CUTLASS TRUE)
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "End of CUTLASS configuration.")
message()
