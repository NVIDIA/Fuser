# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------
# gtest
# -----------------------------------------------------------------------------

# For gtest, we will simply embed it into our test binaries, so we will not need to install it.
set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)
set(gtest_hide_internal_symbols ON CACHE BOOL "Use symbol visibility" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)

# -----------------------------------------------------------------------------
# benchmark
# -----------------------------------------------------------------------------

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

