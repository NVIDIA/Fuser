# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Prerequisite validation orchestrator for nvFuser build.

This module coordinates all prerequisite checks in the correct order and
provides a summary of the validation results. It is designed to be called
from setup.py before the build begins.

The validation follows a fail-fast approach: if any prerequisite check fails,
it raises PrerequisiteMissingError immediately with actionable error messages.
"""

import sys
from typing import Any, Dict, List, Tuple

from .platform import detect_platform, format_platform_info
from .python_version import check_python_version
from .build_tools import check_cmake_version, check_ninja_installed
from .python_packages import check_pybind11_installed, check_torch_installed
from .git import check_git_submodules_initialized
from .gcc import validate_gcc
from .nccl import check_nccl_available
from .llvm import check_llvm_installed
from .requirements import (
    PYTHON, CMAKE, NINJA, PYTORCH, CUDA, PYBIND11, GCC, LLVM,
    format_version,
)


def validate_prerequisites() -> Dict[str, Any]:
    """
    Validate all nvFuser build prerequisites in the correct order.
    
    This function runs all prerequisite checks sequentially and collects
    metadata about the system. If any check fails, it raises PrerequisiteMissingError
    with detailed instructions on how to fix the issue.
    
    Check order (fail-fast after platform detection):
    1. Platform detection (informational only)
    2. Python
    3. CMake
    4. Ninja
    5. PyTorch with CUDA (includes system CUDA validation)
    6. pybind11
    7. Git submodules initialized
    8. GCC with C++20 <format> header
    9. NCCL headers/library (if distributed enabled)
    10. LLVM
    
    Returns:
        Dict[str, Any]: Dictionary containing metadata about all detected prerequisites
        
    Raises:
        PrerequisiteMissingError: If any prerequisite is missing or has wrong version
        
    Example:
        >>> metadata = validate_prerequisites()
        [nvFuser] Platform: Linux x86_64, Ubuntu 22.04
        [nvFuser] ✓ Python X.Y.Z >= {PYTHON.min_str}
        [nvFuser] ✓ CMake X.Y.Z >= {CMAKE.min_str}
        [nvFuser] ✓ Ninja X.Y.Z (any version)
        [nvFuser] ✓ PyTorch X.Y with CUDA X.Y >= {PYTORCH.min_str} with CUDA {CUDA.min_str}
        [nvFuser] ✓ pybind11 X.Y.Z >= {PYBIND11.min_str} with CMake support
        [nvFuser] ✓ Git submodules: N initialized
        [nvFuser] ✓ GCC X.Y.Z >= {GCC.min_str} with <format> header
        [nvFuser] ✓ NCCL found (headers: /path/to/nccl/include)
        [nvFuser] ✓ LLVM X.Y.Z >= {LLVM.min_str}
        
        ✓✓✓ All prerequisites validated ✓✓✓
        
        Note: Version requirements are defined in requirements.py.
        
        >>> metadata.keys()
        dict_keys(['platform', 'python', 'cmake', 'ninja', 'torch', 'cuda', 
                   'pybind11', 'git_submodules', 'gcc', 'nccl', 'llvm'])
    """
    # Prominent banner - start of validation
    print("\n" + "=" * 60)
    print("[nvFuser] Validating build prerequisites...")
    print("=" * 60)
    sys.stdout.flush()
    
    # Platform detection (informational only - doesn't fail)
    platform_info = detect_platform()
    platform_str = format_platform_info(platform_info)
    print(f"[nvFuser] Platform: {platform_str}")
    
    # Python version check
    python_ver = check_python_version()
    print(f"[nvFuser] ✓ {PYTHON.name} {format_version(python_ver)} >= {PYTHON.min_str}")
    
    # Build tools checks
    cmake_ver = check_cmake_version()
    print(f"[nvFuser] ✓ {CMAKE.name} {format_version(cmake_ver)} >= {CMAKE.min_str}")
    
    ninja_ver = check_ninja_installed()
    ninja_display = f">= {NINJA.min_str}" if NINJA.min_version else "(any version)"
    print(f"[nvFuser] ✓ {NINJA.name} {ninja_ver} {ninja_display}")
    
    # PyTorch and CUDA check (includes system CUDA validation)
    torch_ver, cuda_ver = check_torch_installed()
    print(f"[nvFuser] ✓ {PYTORCH.name} {torch_ver} with {CUDA.name} {cuda_ver} >= {PYTORCH.min_str} with {CUDA.name} {CUDA.min_str}")
    # System CUDA validation messages are printed by check_torch_installed()
    
    # pybind11 check
    pybind11_ver = check_pybind11_installed()
    print(f"[nvFuser] ✓ {PYBIND11.name} {pybind11_ver} >= {PYBIND11.min_str} with CMake support")
    
    # Git submodules check
    submodules = check_git_submodules_initialized()
    if submodules:
        print(f"[nvFuser] ✓ Git submodules: {len(submodules)} initialized")
    else:
        print(f"[nvFuser] ✓ Git submodules: N/A (not a git repository)")
    
    # GCC validation
    gcc_ver = validate_gcc()
    print(f"[nvFuser] ✓ {GCC.name} {format_version(gcc_ver)} >= {GCC.min_str} with <format> header")
    
    # NCCL check (only when distributed is enabled)
    nccl_result = check_nccl_available()
    if nccl_result:
        nccl_inc, nccl_lib = nccl_result
        print(f"[nvFuser] ✓ NCCL found (headers: {nccl_inc})")
    else:
        print("[nvFuser] ✓ NCCL: skipped (distributed disabled)")
    
    # LLVM check
    llvm_ver = check_llvm_installed()
    print(f"[nvFuser] ✓ {LLVM.name} {llvm_ver} >= {LLVM.min_str}")
    
    # Success summary with prominent banner
    print("\n" + "=" * 60)
    print("✓✓✓ All prerequisites validated ✓✓✓")
    print("=" * 60 + "\n")
    sys.stdout.flush()
    
    # Return collected metadata
    return {
        'platform': platform_info,
        'python': python_ver,
        'cmake': cmake_ver,
        'ninja': ninja_ver,
        'torch': torch_ver,
        'cuda': cuda_ver,
        'pybind11': pybind11_ver,
        'git_submodules': submodules,
        'gcc': gcc_ver,
        'nccl': nccl_result,
        'llvm': llvm_ver,
    }

