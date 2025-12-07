# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
NCCL header and library detection for nvFuser distributed build.

nvFuser's distributed/multi-GPU support requires NCCL headers to compile.
When NVFUSER_DISTRIBUTED is enabled (default), nvFuser includes PyTorch's
ProcessGroupNCCL.hpp which in turn includes <nccl.h> as a system header.

Include chain:
    nvFuser: csrc/multidevice/communication.cpp
        └── #include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
                └── #include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
                        └── #include <nccl.h>  (system include)

NCCL can be found in several locations:
1. Bundled with PyTorch pip package: nvidia-nccl-cu* provides headers and libs
   at {site-packages}/nvidia/nccl/include and {site-packages}/nvidia/nccl/lib
2. System installation: apt install libnccl-dev
3. CUDA toolkit: sometimes bundled with CUDA
4. Custom installation: via NCCL_ROOT or NCCL_INCLUDE_DIR env vars

This module checks these locations to detect NCCL before the build starts.
Detection is skipped if NVFUSER_BUILD_WITHOUT_DISTRIBUTED is set.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .exceptions import PrerequisiteMissingError


def _get_pip_nccl_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find NCCL headers and library from pip-installed nvidia-nccl-cu* package.
    
    PyTorch's pip package depends on nvidia-nccl-cu* which bundles:
    - {site-packages}/nvidia/nccl/include/nccl.h
    - {site-packages}/nvidia/nccl/lib/libnccl.so.2
    
    Note: Similar logic exists in utils.py::get_pip_nccl_include_dir() for the
    build system. This function returns both include AND lib paths for complete
    validation, while utils.py only needs the include path for CMake. The
    duplication is intentional to keep validation and build logic independent.
    
    Returns:
        Tuple of (include_path, lib_path) or (None, None) if not found
        
    Example:
        >>> inc, lib = _get_pip_nccl_paths()
        >>> inc
        PosixPath('/path/to/site-packages/nvidia/nccl/include')
    """
    # Search all site-packages directories
    for site_path in sys.path:
        if not site_path:
            continue
        nccl_include = Path(site_path) / 'nvidia' / 'nccl' / 'include'
        nccl_lib = Path(site_path) / 'nvidia' / 'nccl' / 'lib'
        
        header = nccl_include / 'nccl.h'
        # Check for versioned library (libnccl.so.2) or unversioned
        lib_exists = (nccl_lib / 'libnccl.so.2').exists() or (nccl_lib / 'libnccl.so').exists()
        
        if header.exists() and lib_exists:
            return nccl_include, nccl_lib
    
    return None, None


def _get_nccl_search_paths() -> Tuple[List[Path], List[Path]]:
    """
    Get NCCL header and library search paths matching compiler/CMake logic.
    
    Search order:
    1. Pip-installed nvidia-nccl-cu* package (highest priority - bundled with PyTorch)
    2. Explicit environment variable overrides (NCCL_INCLUDE_DIR, NCCL_LIB_DIR)
    3. NCCL_ROOT based paths
    4. CUDA toolkit paths (NCCL sometimes bundled with CUDA)
    5. Standard system paths
    
    Returns:
        Tuple[List[Path], List[Path]]: (include_paths, library_paths)
        
    Example:
        >>> inc_paths, lib_paths = _get_nccl_search_paths()
        >>> inc_paths[0]
        PosixPath('/path/to/site-packages/nvidia/nccl/include')
    """
    include_paths: List[Path] = []
    library_paths: List[Path] = []
    
    # 1. Pip-installed nvidia-nccl-cu* (bundled with PyTorch)
    # This is the most common case for pip-installed PyTorch users
    pip_inc, pip_lib = _get_pip_nccl_paths()
    if pip_inc:
        include_paths.append(pip_inc)
    if pip_lib:
        library_paths.append(pip_lib)
    
    # 2. Explicit NCCL_INCLUDE_DIR and NCCL_LIB_DIR (user override)
    # These match PyTorch's FindNCCL.cmake behavior
    if nccl_include := os.environ.get('NCCL_INCLUDE_DIR'):
        include_paths.append(Path(nccl_include))
    
    if nccl_lib := os.environ.get('NCCL_LIB_DIR'):
        library_paths.append(Path(nccl_lib))
    
    # 3. NCCL_ROOT based paths (PyTorch convention)
    for env_var in ['NCCL_ROOT', 'NCCL_ROOT_DIR']:
        if nccl_root := os.environ.get(env_var):
            root = Path(nccl_root)
            include_paths.append(root / 'include')
            library_paths.append(root / 'lib')
            library_paths.append(root / 'lib64')
    
    # 4. CUDA toolkit paths (some install NCCL alongside CUDA)
    # PyTorch's FindNCCL.cmake adds CUDA_TOOLKIT_ROOT_DIR to NCCL_ROOT
    for cuda_env in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_TOOLKIT_ROOT_DIR']:
        if cuda_root := os.environ.get(cuda_env):
            root = Path(cuda_root)
            include_paths.append(root / 'include')
            library_paths.append(root / 'lib64')
            library_paths.append(root / 'lib')
    
    # 5. Standard system paths (compiler defaults)
    # These are where apt install libnccl-dev places files
    system_include_paths = [
        Path('/usr/include'),
        Path('/usr/local/include'),
        Path('/usr/local/cuda/include'),
    ]
    include_paths.extend(system_include_paths)
    
    system_library_paths = [
        Path('/usr/lib/x86_64-linux-gnu'),  # Debian/Ubuntu multiarch
        Path('/usr/lib64'),                  # RHEL/CentOS
        Path('/usr/lib'),
        Path('/usr/local/lib'),
        Path('/usr/local/cuda/lib64'),
    ]
    library_paths.extend(system_library_paths)
    
    return include_paths, library_paths


def _find_nccl_header(search_paths: List[Path]) -> Optional[Path]:
    """
    Search for nccl.h in the given paths.
    
    Args:
        search_paths: List of directories to search
        
    Returns:
        Path to directory containing nccl.h, or None if not found
    """
    for path in search_paths:
        header = path / 'nccl.h'
        if header.exists() and header.is_file():
            return path
    return None


def _find_nccl_library(search_paths: List[Path]) -> Optional[Path]:
    """
    Search for NCCL shared library in the given paths.
    
    Looks for libnccl.so or libnccl.so.2 (versioned).
    
    Args:
        search_paths: List of directories to search
        
    Returns:
        Path to directory containing libnccl.so, or None if not found
    """
    library_names = ['libnccl.so', 'libnccl.so.2']
    
    for path in search_paths:
        for lib_name in library_names:
            lib = path / lib_name
            if lib.exists():
                return path
    return None


def check_nccl_available() -> Optional[Tuple[str, str]]:
    """
    Check if NCCL headers and library are available when distributed is enabled.
    
    This function replicates the compiler's header search and CMake's library
    detection to ensure validation accurately predicts build success.
    
    When NVFUSER_BUILD_WITHOUT_DISTRIBUTED is set, this check is skipped
    and returns None (distributed support disabled).
    
    Returns:
        Optional[Tuple[str, str]]: (header_path, library_path) if found,
                                    None if distributed is disabled
        
    Raises:
        PrerequisiteMissingError: If NCCL not found and distributed is enabled
        
    Example:
        >>> result = check_nccl_available()
        >>> result
        ('/usr/include', '/usr/lib/x86_64-linux-gnu')
        
        >>> # With distributed disabled:
        >>> os.environ['NVFUSER_BUILD_WITHOUT_DISTRIBUTED'] = '1'
        >>> check_nccl_available()
        None
    """
    # Check if distributed is disabled
    if os.environ.get('NVFUSER_BUILD_WITHOUT_DISTRIBUTED'):
        return None
    
    # Get search paths
    include_paths, library_paths = _get_nccl_search_paths()
    
    # Search for header
    header_dir = _find_nccl_header(include_paths)
    
    # Search for library
    library_dir = _find_nccl_library(library_paths)
    
    # Both must be found for distributed build to succeed
    if header_dir is None or library_dir is None:
        # Build descriptive error message
        missing_parts = []
        if header_dir is None:
            missing_parts.append("nccl.h header")
        if library_dir is None:
            missing_parts.append("libnccl.so library")
        
        missing_str = " and ".join(missing_parts)
        
        # Format searched paths for error message (limit to first 5 for readability)
        inc_paths_str = "\n".join(f"  - {p}" for p in include_paths[:5])
        lib_paths_str = "\n".join(f"  - {p}" for p in library_paths[:5])
        
        # Check if pip NCCL was expected but missing
        pip_note = ""
        pip_inc, pip_lib = _get_pip_nccl_paths()
        if pip_inc is None:
            pip_note = (
                "Note: NCCL is usually bundled with PyTorch's pip package (nvidia-nccl-cu*).\n"
                "If you installed PyTorch via pip, try reinstalling it:\n"
                "  pip install --force-reinstall torch\n\n"
            )
        
        raise PrerequisiteMissingError(
            f"ERROR: NCCL {missing_str} not found.\n\n"
            "nvFuser's distributed/multi-GPU support requires NCCL.\n"
            "The build will fail because PyTorch headers include <nccl.h>.\n\n"
            f"{pip_note}"
            "Options:\n\n"
            "Option 1: Install NCCL system-wide:\n"
            "  sudo apt install libnccl-dev\n\n"
            "Option 2: Build without distributed support:\n"
            "  export NVFUSER_BUILD_WITHOUT_DISTRIBUTED=1\n"
            "  pip install --no-build-isolation -e . -v\n\n"
            "Searched include paths:\n"
            f"{inc_paths_str}\n\n"
            "Searched library paths:\n"
            f"{lib_paths_str}"
        )
    
    return (str(header_dir), str(library_dir))

