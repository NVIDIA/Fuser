# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Git Submodules Handler
# ------------------------------------------------------------------------------

macro(handle_git_submodules)
  message("")
  message("Checking Git Submodules...")

  # Find git executable
  find_package(Git QUIET)

  if(GIT_FOUND)
    # Use 'git submodule status' which only reads state, never modifies
    # This command shows:
    # - (no prefix) = submodule is initialized and up to date
    # - '-' prefix = submodule is not initialized
    # - '+' prefix = submodule is initialized but checked out to different commit than expected
    # - 'U' prefix = submodule has merge conflicts
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" submodule status
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE SUBMODULE_STATUS
      ERROR_VARIABLE SUBMODULE_ERROR
      RESULT_VARIABLE SUBMODULE_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(SUBMODULE_RESULT EQUAL 0)
      # Parse the output to check for uninitialized submodules (lines starting with '-')
      string(REGEX MATCH "^-" HAS_UNINITIALIZED_SUBMODULES "${SUBMODULE_STATUS}")
      string(REGEX MATCH "\n-" HAS_UNINITIALIZED_SUBMODULES_MULTILINE "${SUBMODULE_STATUS}")

      if(HAS_UNINITIALIZED_SUBMODULES OR HAS_UNINITIALIZED_SUBMODULES_MULTILINE)
        set(GitSubmodules_FOUND FALSE)
        message(STATUS "Git submodules: NOT initialized")
        message(STATUS "  Run: git submodule update --init --recursive")
      else()
        set(GitSubmodules_FOUND TRUE)
        message(STATUS "Git submodules: initialized")
      endif()
    else()
      message(WARNING "Failed to check git submodule status: ${SUBMODULE_ERROR}")
      set(GitSubmodules_FOUND FALSE)
      set(NVFUSER_REQUIREMENT_GitSubmodules_STATUS "ERROR")
    endif()
  else()
    message(WARNING "Git not found - cannot check submodule status")
    set(GitSubmodules_FOUND FALSE)
    set(NVFUSER_REQUIREMENT_GitSubmodules_STATUS "UNKNOWN")
  endif()

  # Use common status function
  set_dependency_report_status(GitSubmodules)
endmacro()
