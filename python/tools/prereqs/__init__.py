# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
nvFuser Prerequisite Validation Package

This package provides utilities for validating build prerequisites before
attempting to build nvFuser from source. It helps provide clear, actionable
error messages when prerequisites are missing or have incorrect versions.

Key Components:
    - PrerequisiteMissingError: Exception raised when prerequisites are missing
    - detect_platform(): Detect OS, architecture, and Linux distribution
    - format_platform_info(): Format platform information as readable string
    - check_python_version(): Validate Python version (3.8+)
    - check_cmake_version(): Validate CMake version (3.18+)
    - check_ninja_installed(): Validate Ninja build system
    - check_pybind11_installed(): Validate pybind11 with CMake support
    - check_torch_installed(): Validate PyTorch 2.0+ with CUDA 12.8+
    - check_git_submodules_initialized(): Validate git submodules are initialized
    - validate_gcc(): Validate GCC 13+ with C++20 <format> header support
    - check_nccl_available(): Validate NCCL headers/library for distributed builds
    - check_llvm_installed(): Validate LLVM 18.1+ for build-time linking

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
from .gcc import validate_gcc
from .nccl import check_nccl_available
from .llvm import check_llvm_installed
from .validate import validate_prerequisites

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
    "validate_gcc",
    "check_nccl_available",
    "check_llvm_installed",
    "validate_prerequisites",
]

