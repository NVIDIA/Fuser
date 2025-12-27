# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Git Submodules Handler
# ------------------------------------------------------------------------------

macro(handle_git_submodules)
  message("")
  message("Finding GitSubmodules...")

  # Check if third_party/googletest exists and is populated
  set(test_file "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")

  if(EXISTS "${test_file}")
    set(GitSubmodules_FOUND TRUE)
    set(NVFUSER_REQUIREMENT_GitSubmodules_STATUS "SUCCESS")
  else()
    set(GitSubmodules_FOUND FALSE)
    set(NVFUSER_REQUIREMENT_GitSubmodules_STATUS "NOT_FOUND")
    set(NVFUSER_DEPENDENCIES_OK FALSE)
  endif()
endmacro()
