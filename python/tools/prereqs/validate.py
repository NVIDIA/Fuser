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


def validate_prerequisites() -> Dict[str, Any]:
    """
    Validate all nvFuser build prerequisites in the correct order.
    
    This function runs all prerequisite checks sequentially and collects
    metadata about the system. If any check fails, it raises PrerequisiteMissingError
    with detailed instructions on how to fix the issue.
    
    Check order (fail-fast after platform detection):
    1. Platform detection (informational only)
    2. Python 3.8+
    3. CMake 3.18+
    4. Ninja 1.10+
    5. PyTorch 2.0+ with CUDA 12.8+ (includes system CUDA validation)
    6. pybind11 2.0+
    7. Git submodules initialized
    8. GCC 13+ with C++20 <format> header
    9. NCCL headers/library (if distributed enabled)
    10. LLVM 18.1+
    
    Returns:
        Dict[str, Any]: Dictionary containing metadata about all detected prerequisites
        
    Raises:
        PrerequisiteMissingError: If any prerequisite is missing or has wrong version
        
    Example:
        >>> metadata = validate_prerequisites()
        [nvFuser] Platform: Linux x86_64, Ubuntu 22.04
        [nvFuser] ✓ Python 3.10.12 >= 3.8
        [nvFuser] ✓ CMake 3.22.1 >= 3.18
        [nvFuser] ✓ Ninja 1.11.1 >= 1.10
        [nvFuser] ✓ PyTorch 2.1.0 with CUDA 12.8 >= 2.0 with CUDA 12.8
        [nvFuser] ✓ System CUDA 12.8 (matches PyTorch CUDA 12.8)
        [nvFuser] ✓ pybind11 2.11.1 >= 2.0 with CMake support
        [nvFuser] ✓ Git submodules: 15 initialized
        [nvFuser] ✓ GCC 13.2.0 >= 13.0 with <format> header
        [nvFuser] ✓ NCCL found (headers: /path/to/nvidia/nccl/include)
        [nvFuser] ✓ LLVM 18.1.8 >= 18.1
        
        ✓✓✓ All prerequisites validated ✓✓✓
        
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
    print(f"[nvFuser] ✓ Python {'.'.join(map(str, python_ver))} >= 3.8")
    
    # Build tools checks
    cmake_ver = check_cmake_version()
    print(f"[nvFuser] ✓ CMake {'.'.join(map(str, cmake_ver))} >= 3.18")
    
    ninja_ver = check_ninja_installed()
    print(f"[nvFuser] ✓ Ninja {ninja_ver} >= 1.10")
    
    # PyTorch and CUDA check (includes system CUDA validation)
    torch_ver, cuda_ver = check_torch_installed()
    print(f"[nvFuser] ✓ PyTorch {torch_ver} with CUDA {cuda_ver} >= 2.0 with CUDA 12.8")
    # System CUDA validation messages are printed by check_torch_installed()
    
    # pybind11 check
    pybind11_ver = check_pybind11_installed()
    print(f"[nvFuser] ✓ pybind11 {pybind11_ver} >= 2.0 with CMake support")
    
    # Git submodules check
    submodules = check_git_submodules_initialized()
    if submodules:
        print(f"[nvFuser] ✓ Git submodules: {len(submodules)} initialized")
    else:
        print(f"[nvFuser] ✓ Git submodules: N/A (not a git repository)")
    
    # GCC validation
    gcc_ver = validate_gcc()
    print(f"[nvFuser] ✓ GCC {'.'.join(map(str, gcc_ver))} >= 13.0 with <format> header")
    
    # NCCL check (only when distributed is enabled)
    nccl_result = check_nccl_available()
    if nccl_result:
        nccl_inc, nccl_lib = nccl_result
        print(f"[nvFuser] ✓ NCCL found (headers: {nccl_inc})")
    else:
        print("[nvFuser] ✓ NCCL: skipped (distributed disabled)")
    
    # LLVM check
    llvm_ver = check_llvm_installed()
    print(f"[nvFuser] ✓ LLVM {llvm_ver} >= 18.1")
    
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

