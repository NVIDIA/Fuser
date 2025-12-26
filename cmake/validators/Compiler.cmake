# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Compiler Validator (GCC/Clang)
# ------------------------------------------------------------------------------

# Validates that the C++ compiler is GCC 13+ or Clang 19+
function(validate_compiler)
  # Get compiler info
  set(compiler_id "${CMAKE_CXX_COMPILER_ID}")
  set(compiler_version "${CMAKE_CXX_COMPILER_VERSION}")
  set(compiler_path "${CMAKE_CXX_COMPILER}")

  # Set found and version for reporting
  set(Compiler_FOUND TRUE PARENT_SCOPE)
  set(Compiler_VERSION "${compiler_version}" PARENT_SCOPE)

  # Check version based on compiler type
  if(compiler_id STREQUAL "GNU")
    # GCC requires 13+
    if(compiler_version VERSION_LESS "13")
      set(Compiler_STATUS "INCOMPATIBLE" PARENT_SCOPE)
    else()
      set(Compiler_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
    set(Compiler_NAME "GCC" PARENT_SCOPE)
  elseif(compiler_id STREQUAL "Clang")
    # Clang requires 19+
    if(compiler_version VERSION_LESS "19")
      set(Compiler_STATUS "INCOMPATIBLE" PARENT_SCOPE)
    else()
      set(Compiler_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
    set(Compiler_NAME "Clang" PARENT_SCOPE)
  else()
    # Unknown compiler
    set(Compiler_STATUS "SUCCESS" PARENT_SCOPE)
    set(Compiler_NAME "${compiler_id}" PARENT_SCOPE)
  endif()
endfunction()
