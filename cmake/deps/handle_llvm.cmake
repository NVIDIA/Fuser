# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# LLVM Handler with Component Mapping
# ------------------------------------------------------------------------------

macro(handle_llvm)
  message("")
  message("Finding LLVM...")

  # LLVM's LLVMConfigVersion.cmake requires an exact major.minor match, so
  # find_package(LLVM 18.1) rejects 19.x/20.x. We iterate major.minor
  # combinations (highest first) to find the best compatible version.
  # If LLVM_DIR is explicitly set, skip the search and use it directly.
  if(LLVM_DIR)
    find_package(LLVM ${MAYBE_REQUIRED})
  else()
    string(REGEX MATCH "^([0-9]+)" _llvm_min_major "${NVFUSER_REQUIREMENT_LLVM_VERSION_MIN}")
    set(_llvm_search_done FALSE)
    foreach(_major RANGE 25 ${_llvm_min_major} -1)
      foreach(_minor RANGE 9 0 -1)
        unset(LLVM_DIR CACHE)
        find_package(LLVM ${_major}.${_minor} QUIET)
        if(LLVM_FOUND)
          set(_llvm_search_done TRUE)
          break()
        endif()
      endforeach()
      if(_llvm_search_done)
        break()
      endif()
    endforeach()

    if(NOT LLVM_FOUND)
      # Fall back: bare search so user sees diagnostic output
      unset(LLVM_DIR CACHE)
      find_package(LLVM ${MAYBE_REQUIRED})
    endif()
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
