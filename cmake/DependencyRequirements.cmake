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

# Python
set(NVFUSER_REQUIREMENT_Python_VERSION_MIN "3.8")
set(NVFUSER_REQUIREMENT_Python_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_Python_COMPONENTS "Interpreter;Development")
set(NVFUSER_REQUIREMENT_Python_LOCATION_VAR "Python_EXECUTABLE")
set(NVFUSER_REQUIREMENT_Python_CLASS_TYPE "version")

# Torch
set(NVFUSER_REQUIREMENT_Torch_VERSION_MIN "2.0")
set(NVFUSER_REQUIREMENT_Torch_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_Torch_LOCATION_VAR "Torch_DIR")
set(NVFUSER_REQUIREMENT_Torch_PRE_FIND_HOOK "torch_pre_find_hook")  # Setup Python path
set(NVFUSER_REQUIREMENT_Torch_POST_FIND_HOOK "torch_post_find_hook")  # Validate CUDA constraint
set(NVFUSER_REQUIREMENT_Torch_CLASS_TYPE "torch")  # Special torch requirement with CUDA constraint

# Torch_CUDA (constraint check - not a real find_package)
# This is a pseudo-dependency that reports the CUDA version constraint
# Note: This is displayed as part of Torch output, not as separate requirement
set(NVFUSER_REQUIREMENT_Torch_CUDA_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_Torch_CUDA_IS_CONSTRAINT TRUE)  # Skip validation, just report

# pybind11
set(NVFUSER_REQUIREMENT_pybind11_VERSION_MIN "2.0")
set(NVFUSER_REQUIREMENT_pybind11_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_pybind11_COMPONENTS "")
set(NVFUSER_REQUIREMENT_pybind11_LOCATION_VAR "pybind11_DIR")
set(NVFUSER_REQUIREMENT_pybind11_CLASS_TYPE "version")

# CUDAToolkit
set(NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN "12.6")
set(NVFUSER_REQUIREMENT_CUDAToolkit_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_CUDAToolkit_COMPONENTS "Cupti;cuda_driver")
set(NVFUSER_REQUIREMENT_CUDAToolkit_LOCATION_VAR "CUDAToolkit_LIBRARY_ROOT")
set(NVFUSER_REQUIREMENT_CUDAToolkit_CLASS_TYPE "version")

# LLVM
set(NVFUSER_REQUIREMENT_LLVM_VERSION_MIN "18.1")
set(NVFUSER_REQUIREMENT_LLVM_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_LLVM_COMPONENTS "")
set(NVFUSER_REQUIREMENT_LLVM_LOCATION_VAR "LLVM_DIR")
set(NVFUSER_REQUIREMENT_LLVM_POST_FIND_HOOK "llvm_post_find_hook")  # Map components
set(NVFUSER_REQUIREMENT_LLVM_CLASS_TYPE "version")

# Compiler (GCC or Clang)
set(NVFUSER_REQUIREMENT_Compiler_VERSION_MIN "13")  # GCC 13+ or Clang 19+
set(NVFUSER_REQUIREMENT_Compiler_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_Compiler_TYPE "compiler")
set(NVFUSER_REQUIREMENT_Compiler_LOCATION_VAR "CMAKE_CXX_COMPILER")
set(NVFUSER_REQUIREMENT_Compiler_CLASS_TYPE "version")

# Ninja
set(NVFUSER_REQUIREMENT_Ninja_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_Ninja_TYPE "build_tool")
set(NVFUSER_REQUIREMENT_Ninja_LOCATION_VAR "CMAKE_MAKE_PROGRAM")
set(NVFUSER_REQUIREMENT_Ninja_CLASS_TYPE "boolean")

# Git Submodules
set(NVFUSER_REQUIREMENT_GitSubmodules_OPTIONAL FALSE)
set(NVFUSER_REQUIREMENT_GitSubmodules_TYPE "git")
set(NVFUSER_REQUIREMENT_GitSubmodules_CLASS_TYPE "boolean")

# ==============================================================================
# Master list of all requirements (checked in order)
# ==============================================================================
# Note: CUDAToolkit must come before Torch for CUDA version constraint validation
set(NVFUSER_ALL_REQUIREMENTS
  GitSubmodules
  Ninja
  Compiler
  Python
  CUDAToolkit
  Torch
  Torch_CUDA
  pybind11
  LLVM
)

# Note: Future dependency types (compiler, header_only, submodule, constraint)
# can be added by defining their metadata above and adding appropriate
# validation logic in DependencyValidators.cmake
