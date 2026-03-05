# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Ninja Build Tool Handler
# ------------------------------------------------------------------------------
#
# Checks if the Ninja build system is being used as the CMake generator.
# Note: This check happens after generator selection, so it only reports status.
# To use Ninja, specify it when running CMake: cmake -G Ninja ..

macro(handle_ninja)
  message("")
  message("Finding Ninja...")

  # Check if using Ninja generator (CMAKE_GENERATOR is already set by this point)
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(Ninja_FOUND TRUE)
  else()
    set(Ninja_FOUND FALSE)
  endif()

  set_dependency_report_status(Ninja)
endmacro()
