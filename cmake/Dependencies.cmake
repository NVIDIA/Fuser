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

# Add googletest subdirectory but make sure our INCLUDE_DIRECTORIES do not bleed into it.
# This is because libraries installed into the root conda env (e.g. MKL) add a global /opt/conda/include directory,
# and if there is gtest installed in conda, the third_party/googletest/**.cc source files would try to include headers
# from /opt/conda/include/gtest/**.h instead of its own. Once we have proper target-based include directories,
# this shouldn't be necessary anymore.
get_property(INC_DIR_temp DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest)
set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${INC_DIR_temp})

include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/include)
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googlemock/include)

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
include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark/include)

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
