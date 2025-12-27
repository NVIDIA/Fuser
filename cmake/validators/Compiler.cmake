# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Compiler Validator (GCC/Clang)
# ------------------------------------------------------------------------------

# Validates that the C++ compiler is GCC 13+ or Clang 19+
function(validate_compiler)
  # Get compiler info
  set(compiler_path "${CMAKE_CXX_COMPILER}")

  # Set found and version for reporting
  set(NVFUSER_Compiler_FOUND TRUE PARENT_SCOPE)

  # Check version based on compiler type
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC requires 13+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE" PARENT_SCOPE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang requires 19+
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19")
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "INCOMPATIBLE" PARENT_SCOPE)
    else()
      set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
  else()
    # Unknown compiler
    set(NVFUSER_REQUIREMENT_Compiler_STATUS "SUCCESS" PARENT_SCOPE)
  endif()
endfunction()
