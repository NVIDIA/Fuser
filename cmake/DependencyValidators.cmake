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
    # Export constraint info for JSON
    set(Torch_EXTRA_JSON "{\"constraint_cuda_status\": \"not_available\"}" PARENT_SCOPE)
    return()
  endif()

  # Get CUDAToolkit version (major.minor only for comparison)
  set(cuda_toolkit_version "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")

  # Extract major.minor from Torch CUDA version
  string(REGEX MATCH "^([0-9]+\\.[0-9]+)" torch_cuda_major_minor "${torch_cuda_version}")

  # Set the expected version (what CUDAToolkit has)
  set(NVFUSER_REQUIREMENT_Torch_CUDA_VERSION_MIN "${cuda_toolkit_version}" PARENT_SCOPE)

  # Compare major.minor versions
  if(NOT torch_cuda_major_minor STREQUAL cuda_toolkit_version)
    # Version mismatch
    set(Torch_CUDA_FOUND FALSE PARENT_SCOPE)
    set(Torch_CUDA_VERSION "${torch_cuda_major_minor}" PARENT_SCOPE)
    # Export constraint info for JSON
    set(Torch_EXTRA_JSON "{\"constraint_cuda_status\": \"mismatch\", \"constraint_cuda_found\": \"${torch_cuda_major_minor}\", \"constraint_cuda_required\": \"${cuda_toolkit_version}\"}" PARENT_SCOPE)
  else()
    # Versions match!
    set(Torch_CUDA_FOUND TRUE PARENT_SCOPE)
    set(Torch_CUDA_VERSION "${torch_cuda_major_minor}" PARENT_SCOPE)
    # Export constraint info for JSON
    set(Torch_EXTRA_JSON "{\"constraint_cuda_status\": \"match\", \"constraint_cuda_version\": \"${torch_cuda_major_minor}\"}" PARENT_SCOPE)
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
# Validator for Compiler (GCC or Clang)
# ------------------------------------------------------------------------------
# Validates C++ compiler version (GCC 13+ or Clang 19+)
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

# ------------------------------------------------------------------------------
# Validator for Ninja
# ------------------------------------------------------------------------------
# Validates that Ninja build system is available
function(validate_ninja)
  # Check if Ninja is the current generator
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(Ninja_FOUND TRUE PARENT_SCOPE)
    set(Ninja_VERSION "" PARENT_SCOPE)
    set(Ninja_STATUS "SUCCESS" PARENT_SCOPE)
    return()
  endif()

  # Check if ninja executable exists
  find_program(NINJA_EXECUTABLE ninja)

  if(NINJA_EXECUTABLE)
    set(Ninja_FOUND TRUE PARENT_SCOPE)
    set(Ninja_VERSION "" PARENT_SCOPE)
    set(Ninja_STATUS "SUCCESS" PARENT_SCOPE)
  else()
    set(Ninja_FOUND FALSE PARENT_SCOPE)
    set(Ninja_VERSION "" PARENT_SCOPE)
    set(Ninja_STATUS "NOT_FOUND" PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------
# Validator for Git Submodules
# ------------------------------------------------------------------------------
# Validates that Git submodules are initialized
function(validate_git_submodules)
  # Check if we're in a git repository
  find_package(Git QUIET)

  if(NOT Git_FOUND)
    # Not in a git repo or git not available - assume OK (pip install from tarball)
    set(GitSubmodules_FOUND TRUE PARENT_SCOPE)
    set(GitSubmodules_VERSION "" PARENT_SCOPE)
    set(GitSubmodules_STATUS "SUCCESS" PARENT_SCOPE)
    return()
  endif()

  # Check if .git exists
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Not a git repo - assume OK (pip install from tarball)
    set(GitSubmodules_FOUND TRUE PARENT_SCOPE)
    set(GitSubmodules_VERSION "" PARENT_SCOPE)
    set(GitSubmodules_STATUS "SUCCESS" PARENT_SCOPE)
    return()
  endif()

  # Check if submodules are initialized
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule status
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE submodule_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE git_result
  )

  if(NOT git_result EQUAL 0)
    # Git command failed - assume OK
    set(GitSubmodules_FOUND TRUE PARENT_SCOPE)
    set(GitSubmodules_VERSION "" PARENT_SCOPE)
    set(GitSubmodules_STATUS "SUCCESS" PARENT_SCOPE)
    return()
  endif()

  # Check if any submodules are uninitialized (line starts with '-')
  # Git submodule status format:
  #   -<commit> <path> (description) = uninitialized
  #    <commit> <path> (description) = initialized
  #   +<commit> <path> (description) = modified
  # We need to check if any line starts with '-' (not just contains it)
  string(REGEX MATCH "(^|\n)-" uninit_match "${submodule_status}")

  if(uninit_match)
    # At least one submodule is uninitialized
    set(GitSubmodules_FOUND FALSE PARENT_SCOPE)
    set(GitSubmodules_VERSION "" PARENT_SCOPE)
    set(GitSubmodules_STATUS "NOT_FOUND" PARENT_SCOPE)
  else()
    # All submodules initialized
    set(GitSubmodules_FOUND TRUE PARENT_SCOPE)
    set(GitSubmodules_VERSION "" PARENT_SCOPE)
    set(GitSubmodules_STATUS "SUCCESS" PARENT_SCOPE)
  endif()
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
  set(dep_type "${NVFUSER_REQUIREMENT_${ARG_NAME}_TYPE}")

  if(NOT is_constraint)
    # Only validate if NOT a constraint

    message("")
    message("Finding ${ARG_NAME}...")

    # Check for special dependency types that don't use find_package
    if(dep_type STREQUAL "compiler")
      validate_compiler()
    elseif(dep_type STREQUAL "build_tool")
      validate_ninja()
    elseif(dep_type STREQUAL "git")
      validate_git_submodules()
    else()
      # Standard find_package dependency
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

      # Set status based on found state and version check
      # This will be used for JSON export
      # Note: Must call as function to get proper version checking
      set_dependency_status(${ARG_NAME})
    endif()
  endif()
endmacro()
