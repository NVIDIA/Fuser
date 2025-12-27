# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Ninja Build Tool Handler
# ------------------------------------------------------------------------------

macro(handle_ninja)
  message("")
  message("Finding Ninja...")

  # Check if using Ninja generator
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(Ninja_FOUND TRUE)
    set(NVFUSER_REQUIREMENT_Ninja_STATUS "SUCCESS")
  else()
    set(Ninja_FOUND FALSE)
    set(NVFUSER_REQUIREMENT_Ninja_STATUS "NOT_FOUND")
  endif()
endmacro()
