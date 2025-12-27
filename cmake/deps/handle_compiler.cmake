# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# C++ Compiler Handler
# ------------------------------------------------------------------------------

macro(handle_compiler)
  message("")
  message("Finding Compiler...")

  # Always found (we're already running CMake)
  set(NVFUSER_Compiler_FOUND TRUE)

  # Version check based on compiler type
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC requires 13+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE")
      set(NVFUSER_DEPENDENCIES_OK FALSE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang requires 19+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19")
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE")
      set(NVFUSER_DEPENDENCIES_OK FALSE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS")
    endif()
  else()
    # Unknown compiler - allow but this will show as success
    set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS")
  endif()
endmacro()
