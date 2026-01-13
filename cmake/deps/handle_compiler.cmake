# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# C++ Compiler Handler
# ------------------------------------------------------------------------------

macro(handle_compiler)
  # Always found (we're already running CMake)
  set(Compiler_FOUND TRUE)
  set(Compiler_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

  # Mark compiler as optional to allow builds with any compiler.
  # Only GNU and Clang have version requirements - other compilers get SUCCESS status.
  # Python report will show warnings for unknown/old compilers without failing the build.
  set(NVFUSER_REQUIREMENT_Compiler_OPTIONAL TRUE)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(NVFUSER_REQUIREMENT_Compiler_VERSION_MIN ${NVFUSER_REQUIREMENT_GNU_VERSION_MIN})
    set(NVFUSER_REQUIREMENT_Compiler_OPTIONAL FALSE) # Not optional - we have defined version constraints
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(NVFUSER_REQUIREMENT_Compiler_VERSION_MIN ${NVFUSER_REQUIREMENT_Clang_VERSION_MIN})
    set(NVFUSER_REQUIREMENT_Compiler_OPTIONAL FALSE) # Not optional - we have defined version constraints
  else()
    message(WARNING "Unknown compiler '${CMAKE_CXX_COMPILER_ID}' - cannot validate")
  endif()

  set_dependency_report_status(Compiler)

  # Caching variables to enable incremental build.
  # Without this is cross compiling we end up having to blow build directory
  # and rebuild from scratch.
  if(CMAKE_CROSSCOMPILING)
    if(COMPILE_HAVE_STD_REGEX)
      set(RUN_HAVE_STD_REGEX 0 CACHE INTERNAL "Cache RUN_HAVE_STD_REGEX output for cross-compile.")
    endif()
  endif()

endmacro()
