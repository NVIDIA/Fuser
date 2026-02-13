# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# LLVM Handler with Component Mapping
# ------------------------------------------------------------------------------

macro(handle_llvm)
  message("")
  message("Finding LLVM...")

  # LLVM's LLVMConfigVersion.cmake requires an exact major.minor match,
  # so find_package(LLVM 18.1) would reject LLVM 19+. Instead, try each
  # major version from highest to lowest until we find one that meets our
  # minimum requirement.
  string(REGEX MATCH "^([0-9]+)" _llvm_min_major "${NVFUSER_REQUIREMENT_LLVM_VERSION_MIN}")
  foreach(_llvm_try_ver RANGE 25 ${_llvm_min_major} -1)
    find_package(LLVM ${_llvm_try_ver} QUIET)
    if(LLVM_FOUND)
      break()
    endif()
  endforeach()

  if(NOT LLVM_FOUND AND MAYBE_REQUIRED)
    message(FATAL_ERROR "Could not find LLVM >= ${NVFUSER_REQUIREMENT_LLVM_VERSION_MIN}")
  endif()

  # Use common status function
  set_dependency_report_status(LLVM)

  # Additional validation: Map LLVM components to library names
  if(LLVM_FOUND)
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
  endif()
endmacro()
