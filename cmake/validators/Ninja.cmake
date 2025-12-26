# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Ninja Build System Validator
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
