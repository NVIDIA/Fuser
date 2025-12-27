# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# nvFuser Dependency Requirements
# ==============================================================================
#
# This file centralizes all dependency requirement metadata for nvFuser.
# Each requirement entry contains:
# - VERSION_MIN: Minimum version required (can be empty for "any version")
# - OPTIONAL: TRUE/FALSE (default FALSE)
# - COMPONENTS: Components required (for find_package, semicolon-separated)
# - INSTALL_HELP: Text to show on failure (content TBD from #5609)
# - LOCATION_VAR: Which variable to use for display path
# - PRE_FIND_HOOK: Function name to call before find_package (optional)
# - POST_FIND_HOOK: Function name to call after successful find_package (optional)
#
# ==============================================================================

# Compiler (GCC or Clang)
set(NVFUSER_REQUIREMENT_Compiler_VERSION_MIN "13")  # GCC 13+ or Clang 19+

# Python
set(NVFUSER_REQUIREMENT_Python_VERSION_MIN "3.8")
set(NVFUSER_REQUIREMENT_Python_COMPONENTS "Interpreter;Development")

# Torch
set(NVFUSER_REQUIREMENT_Torch_VERSION_MIN "2.0")

# Torch_CUDA (constraint check - not a real find_package)
# This is a pseudo-dependency that reports the CUDA version constraint
# Note: This is displayed as part of Torch output, not as separate requirement

# pybind11
set(NVFUSER_REQUIREMENT_pybind11_VERSION_MIN "2.0")

# CUDAToolkit
set(NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN "12.6")
set(NVFUSER_REQUIREMENT_CUDAToolkit_COMPONENTS "Cupti;cuda_driver")

# LLVM
set(NVFUSER_REQUIREMENT_LLVM_VERSION_MIN "18.1")

# Ninja
set(NVFUSER_REQUIREMENT_Ninja_LOCATION_VAR "CMAKE_MAKE_PROGRAM")
set(NVFUSER_REQUIREMENT_Ninja_OPTIONAL "True")
