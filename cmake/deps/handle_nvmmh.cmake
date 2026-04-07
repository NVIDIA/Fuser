# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# nvidia-matmul-heuristics Handler
# ------------------------------------------------------------------------------

macro(handle_nvmmh)
  message("")
  message("Finding nvidia-matmul-heuristics...")

  set(NVMMH_INCLUDE_DIR "NVMMH_INCLUDE_DIR-NOTFOUND" CACHE PATH "Directory containing nvMatmulHeuristics.h")

  if (NOT IS_DIRECTORY "${NVMMH_INCLUDE_DIR}")
    # Search in Python's site-packages first, then fall back to common locations
    find_path(NVMMH_INCLUDE_DIR nvMatmulHeuristics.h
      PATHS
      "${Python_SITELIB}/nvidia/nvMatmulHeuristics/include"
      NO_DEFAULT_PATH
    )
  endif()

  if(IS_DIRECTORY "${NVMMH_INCLUDE_DIR}")
    set(NVMMH_FOUND TRUE)
    string(APPEND CMAKE_CXX_FLAGS " -DHAS_NVMMH=1")
    message(STATUS "Found nvidia-matmul-heuristics: ${NVMMH_INCLUDE_DIR}")
  else()
    set(NVMMH_FOUND FALSE)
    message(WARNING "nvidia-matmul-heuristics headers not found – building without nvMatmulHeuristics support")
  endif()

  # Use common status function
  set_dependency_report_status(NVMMH)
endmacro()
