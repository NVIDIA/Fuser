# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# pybind11 Handler
# ------------------------------------------------------------------------------

macro(handle_pybind11)
  message("")
  message("Finding pybind11...")

  # Direct find_package call
  find_package(pybind11)

  # Use common status function
  set_dependency_report_status(pybind11)
endmacro()
