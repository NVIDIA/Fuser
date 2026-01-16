# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# CUDA Toolkit Handler
# ------------------------------------------------------------------------------

macro(handle_cuda_toolkit)
  message("")
  message("Finding CUDAToolkit...")

  # Direct find_package call with components
  find_package(CUDAToolkit ${MAYBE_REQUIRED} COMPONENTS ${NVFUSER_REQUIREMENT_CUDAToolkit_COMPONENTS})

  # Use common status function
  set_dependency_report_status(CUDAToolkit)
endmacro()
