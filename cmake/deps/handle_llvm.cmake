# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# LLVM Handler with Component Mapping
# ------------------------------------------------------------------------------

macro(handle_llvm)
  message("")
  message("Finding LLVM...")

  # Direct find_package call
  find_package(LLVM ${MAYBE_REQUIRED})

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
