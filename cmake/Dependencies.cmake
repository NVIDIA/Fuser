# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Preserve build options.
set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

# We will build gtest as static libs and embed it directly into the binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

# For gtest, we will simply embed it into our test binaries, so we will not need to install it.
set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)

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

# Recover build options.
set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

# Cacheing variables to enable incremental build.
# Without this is cross compiling we end up having to blow build directory
# and rebuild from scratch.
if(CMAKE_CROSSCOMPILING)
  if(COMPILE_HAVE_STD_REGEX)
    set(RUN_HAVE_STD_REGEX 0 CACHE INTERNAL "Cache RUN_HAVE_STD_REGEX output for cross-compile.")
  endif()
endif()
