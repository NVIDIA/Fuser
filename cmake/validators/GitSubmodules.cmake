# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Git Submodules Validator
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
