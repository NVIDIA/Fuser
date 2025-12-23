# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# nvFuser Dependency Validators
# ==============================================================================
#
# This file contains validation logic for dependency checking, including:
# - Pre-find hooks: Execute before find_package (e.g., Torch needs Python setup)
# - Post-find hooks: Execute after successful find_package (e.g., LLVM needs library mapping)
# - Main validator: Orchestrates hooks and find_package calls
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Pre-find hook for Torch
# ------------------------------------------------------------------------------
# Torch requires Python to be found first, and needs special setup via Python
# to locate the torch.utils.cmake_prefix_path
function(torch_pre_find_hook)
  if(NOT Python_FOUND)
    set(Torch_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
  )

  if(NOT TORCH_FIND_RESULT EQUAL 0)
    set(Torch_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  # Setup environment for Torch find_package
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE} PARENT_SCOPE)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

  # Set CUDA_ARCH for cu tests
  if(TORCH_CUDA_ARCH_LIST)
    set(ARCH_FLAGS)
    cuda_select_nvcc_arch_flags(ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" PARENT_SCOPE)
  endif()

  # CXX flags necessary for https://github.com/pytorch/pytorch/issues/98093
  string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# Post-find hook for Torch
# ------------------------------------------------------------------------------
# Validates that Torch's CUDA version matches CUDAToolkit version (major.minor)
# Sets variables for the Torch_CUDA pseudo-dependency report
function(torch_post_find_hook)
  # Check if both CUDAToolkit and Torch were found
  if(NOT CUDAToolkit_FOUND)
    # Can't validate if CUDAToolkit wasn't found
    set(Torch_CUDA_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  # Query torch Python package for CUDA version
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.version.cuda if torch.version.cuda else 'N/A')"
    OUTPUT_VARIABLE torch_cuda_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE torch_cuda_result
  )

  if(NOT torch_cuda_result EQUAL 0 OR torch_cuda_version STREQUAL "N/A" OR torch_cuda_version STREQUAL "None")
    # Torch might not have CUDA support or query failed
    set(Torch_CUDA_FOUND FALSE PARENT_SCOPE)
    set(Torch_CUDA_VERSION "N/A" PARENT_SCOPE)
    return()
  endif()

  # Get CUDAToolkit version (major.minor only for comparison)
  set(cuda_toolkit_version "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")

  # Extract major.minor from Torch CUDA version
  string(REGEX MATCH "^([0-9]+\\.[0-9]+)" torch_cuda_major_minor "${torch_cuda_version}")

  # Set the expected version (what CUDAToolkit has)
  set(NVFUSER_REQUIREMENT_Torch_CUDA_VERSION_MIN "${cuda_toolkit_version}" PARENT_SCOPE)

  # Compare major.minor versions (use = symbol for exact match)
  if(NOT torch_cuda_major_minor STREQUAL cuda_toolkit_version)
    # Version mismatch
    set(Torch_CUDA_FOUND FALSE PARENT_SCOPE)
    set(Torch_CUDA_VERSION "${torch_cuda_major_minor}" PARENT_SCOPE)
  else()
    # Versions match!
    set(Torch_CUDA_FOUND TRUE PARENT_SCOPE)
    set(Torch_CUDA_VERSION "${torch_cuda_major_minor}" PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------
# Post-find hook for LLVM
# ------------------------------------------------------------------------------
# LLVM requires mapping component names to library names after find_package
function(llvm_post_find_hook)
  llvm_map_components_to_libnames(LLVM_LIBS
    support
    core
    orcjit
    executionengine
    irreader
    nativecodegen
    Target
    Analysis
    JITLink
    Demangle
  )
  set(LLVM_LIBS "${LLVM_LIBS}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------
# Main validation macro
# ------------------------------------------------------------------------------
# Validates a single dependency by:
# 1. Executing pre-find hook if specified
# 2. Calling find_package (without REQUIRED)
# 3. Executing post-find hook if specified and package was found
#
# Usage: validate_dependency(NAME <package_name>)
# Note: This must be a macro (not function) to properly propagate find_package results
macro(validate_dependency)
  # Parse named arguments
  cmake_parse_arguments(
    ARG                    # prefix for parsed variables
    ""                     # options (boolean flags)
    "NAME"                 # one-value keywords
    ""                     # multi-value keywords
    ${ARGN}
  )

  if(NOT ARG_NAME)
    _message(FATAL_ERROR "validate_dependency: NAME argument required")
  endif()

  # Check if this is a constraint (pseudo-dependency for reporting only)
  set(is_constraint "${NVFUSER_REQUIREMENT_${ARG_NAME}_IS_CONSTRAINT}")

  if(NOT is_constraint)
    # Only validate if NOT a constraint

    message("")
    message("Finding ${ARG_NAME}...")

    # Get metadata from requirements
    set(version_min "${NVFUSER_REQUIREMENT_${ARG_NAME}_VERSION_MIN}")
    set(components_list "${NVFUSER_REQUIREMENT_${ARG_NAME}_COMPONENTS}")
    set(pre_find_hook "${NVFUSER_REQUIREMENT_${ARG_NAME}_PRE_FIND_HOOK}")
    set(post_find_hook "${NVFUSER_REQUIREMENT_${ARG_NAME}_POST_FIND_HOOK}")

    # Execute pre-find hook if specified
    if(pre_find_hook)
      cmake_language(CALL ${pre_find_hook})
    endif()

    # Only proceed with find_package if hook didn't abort
    if(NOT DEFINED ${ARG_NAME}_FOUND OR ${ARG_NAME}_FOUND)
      # Standard find_package call (without REQUIRED to allow reporting all failures)
      if(components_list)
        # Use cmake_language(EVAL) to properly expand the components list
        cmake_language(EVAL CODE "find_package(${ARG_NAME} COMPONENTS ${components_list})")
      else()
        find_package(${ARG_NAME})
      endif()

      # Execute post-find hook if specified and package was found
      if(post_find_hook AND ${ARG_NAME}_FOUND)
        cmake_language(CALL ${post_find_hook})
      endif()
    endif()
  endif()
endmacro()
