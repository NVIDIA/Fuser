# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Python Handler
# ------------------------------------------------------------------------------

macro(handle_python)
  message("")
  message("Finding Python...")

  # Direct find_package call
  find_package(Python ${MAYBE_REQUIRED} COMPONENTS ${NVFUSER_REQUIREMENT_Python_COMPONENTS})

  # Use common status function
  set_dependency_report_status(Python)
endmacro()
