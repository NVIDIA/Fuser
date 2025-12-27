# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# C++ Compiler Handler
# ------------------------------------------------------------------------------

macro(handle_compiler)
  # Always found (we're already running CMake)
  set(NVFUSER_Compiler_FOUND TRUE)

  # Version check based on compiler type
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC requires 13+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS NVFUSER_REQUIREMENT_GNU_VERSION_MIN)
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE")
      set(NVFUSER_DEPENDENCIES_OK FALSE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang requires 19+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS NVFUSER_REQUIREMENT_Clang_VERSION_MIN)
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE")
      set(NVFUSER_DEPENDENCIES_OK FALSE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS")
    endif()
  else()
    # Unknown compiler - allow but this will show as success
    set(NVFUSER_REQUIREMENT_Compiler_STATUS "NOT FOUND")
    message(WARNING "nvFuser requires gcc or clang c++ compiler.")
    set(NVFUSER_DEPENDENCIES_OK FALSE)
  endif()
endmacro()
