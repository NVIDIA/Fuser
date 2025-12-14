# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
nvFuser Prerequisite Validation Package

This package provides utilities for validating build prerequisites before
attempting to build nvFuser from source. It helps provide clear, actionable
error messages when prerequisites are missing or have incorrect versions.

Version requirements are centralized in requirements.py. See PYTHON, CMAKE,
NINJA, PYTORCH, CUDA, PYBIND11, GCC, LLVM constants for current versions.

Key Components:
    - PrerequisiteMissingError: Exception raised when prerequisites are missing
    - detect_platform(): Detect OS, architecture, and Linux distribution
    - format_platform_info(): Format platform information as readable string
    - check_python_version(): Validate Python meets minimum version
    - check_cmake_version(): Validate CMake meets minimum version
    - check_ninja_installed(): Validate Ninja build system (any version)
    - check_pybind11_installed(): Validate pybind11 with CMake support
    - check_torch_installed(): Validate PyTorch with CUDA support
    - check_git_submodules_initialized(): Validate git submodules are initialized
    - validate_compiler(): Validate C++ compiler (GCC 13+ or Clang 19+)
    - check_nccl_available(): Validate NCCL headers/library for distributed builds
    - check_llvm_installed(): Validate LLVM for build-time linking

Usage:
    from tools.prereqs import PrerequisiteMissingError, detect_platform

    platform_info = detect_platform()
    if platform_info['os'] != 'Linux':
        raise PrerequisiteMissingError("nvFuser requires Linux")
"""

from .exceptions import PrerequisiteMissingError
from .platform import detect_platform, format_platform_info
from .python_version import check_python_version
from .build_tools import check_cmake_version, check_ninja_installed
from .python_packages import check_pybind11_installed, check_torch_installed
from .git import check_git_submodules_initialized
from .compiler import validate_compiler
from .nccl import check_nccl_available
from .llvm import check_llvm_installed
from .validate import validate_prerequisites
from .requirements import (
    Requirement,
    parse_version,
    format_version,
    PYTHON,
    CMAKE,
    NINJA,
    PYTORCH,
    CUDA,
    PYBIND11,
    GCC,
    CLANG,
    LLVM,
    CUDA_AVAILABLE,
    pytorch_index_url,
    llvm_download_url,
    pytorch_install_instructions,
)

__all__ = [
    "PrerequisiteMissingError",
    "detect_platform",
    "format_platform_info",
    "check_python_version",
    "check_cmake_version",
    "check_ninja_installed",
    "check_pybind11_installed",
    "check_torch_installed",
    "check_git_submodules_initialized",
    "validate_compiler",
    "check_nccl_available",
    "check_llvm_installed",
    "validate_prerequisites",
    # Central requirements
    "Requirement",
    "parse_version",
    "format_version",
    "PYTHON",
    "CMAKE",
    "NINJA",
    "PYTORCH",
    "CUDA",
    "PYBIND11",
    "GCC",
    "CLANG",
    "LLVM",
    "CUDA_AVAILABLE",
    "pytorch_index_url",
    "llvm_download_url",
    "pytorch_install_instructions",
]
