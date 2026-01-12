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
#
# ==============================================================================

# Ninja
set(NVFUSER_REQUIREMENT_Ninja_OPTIONAL "TRUE")

# Compiler (GCC or Clang)
set(NVFUSER_REQUIREMENT_GNU_VERSION_MIN "13.1")
set(NVFUSER_REQUIREMENT_Clang_VERSION_MIN "19")

# Python
set(NVFUSER_REQUIREMENT_Python_VERSION_MIN "3.10")
set(NVFUSER_REQUIREMENT_Python_COMPONENTS "Interpreter;Development")

# Torch
set(NVFUSER_REQUIREMENT_Torch_VERSION_MIN "2.9")

# pybind11
set(NVFUSER_REQUIREMENT_pybind11_VERSION_MIN "2.0")

# CUDAToolkit
set(NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN "12.6")
set(NVFUSER_REQUIREMENT_CUDAToolkit_COMPONENTS "Cupti;cuda_driver")

# LLVM
set(NVFUSER_REQUIREMENT_LLVM_VERSION_MIN "18.1")
