# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Python package validation for nvFuser build system.

Validates that required Python packages (pybind11, PyTorch) are installed with
correct versions and features. This module handles pybind11 and PyTorch validation.
"""

from typing import Tuple

from .exceptions import PrerequisiteMissingError


def _get_torch_install_instructions(upgrade: bool = False, force_reinstall: bool = False) -> str:
    """
    Generate PyTorch installation instructions with appropriate pip flags.
    
    Args:
        upgrade: If True, adds --upgrade flag
        force_reinstall: If True, adds --force-reinstall flag
        
    Returns:
        Formatted installation instructions for CUDA 13.0 and 12.8
    """
    cmd = "pip install"
    if force_reinstall:
        cmd += " --force-reinstall"
    elif upgrade:
        cmd += " --upgrade"
    
    return (
        f"  # For CUDA 13.0:\n"
        f"  {cmd} torch --index-url https://download.pytorch.org/whl/cu130\n"
        f"  # For CUDA 12.8:\n"
        f"  {cmd} torch --index-url https://download.pytorch.org/whl/cu128"
    )


def check_pybind11_installed() -> str:
    """
    Check that pybind11 is installed with CMake support (2.0+).
    
    pybind11 2.0+ with CMake support is required for building nvFuser's Python bindings.
    The [global] extra is recommended as it provides CMake integration files.
    
    In pybind11 3.0+, CMake configuration files are included in the base package,
    so [global] is optional but still recommended.
    
    Returns:
        str: pybind11 version string
        
    Raises:
        PrerequisiteMissingError: If pybind11 is not installed, version is too old,
                                 or CMake support is missing
        
    Example:
        >>> version = check_pybind11_installed()
        [nvFuser] pybind11: 2.13.6 with CMake support ✓
        >>> version
        '2.13.6'
    """
    # Check if pybind11 is installed
    try:
        import pybind11
    except ImportError:
        raise PrerequisiteMissingError(
            "ERROR: pybind11 is not installed.\n\n"
            "pybind11 2.0+ is required to build nvFuser's Python bindings.\n"
            "The [global] extra provides CMake integration.\n\n"
            "Install pybind11 with CMake support:\n"
            "  pip install 'pybind11[global]'"
        )
    
    # Check version
    version = pybind11.__version__
    
    # Parse major version
    try:
        major = int(version.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse pybind11 version: {version}\n\n"
            f"Please reinstall pybind11:\n"
            f"  pip install --force-reinstall 'pybind11[global]'"
        )
    
    # Check minimum version requirement (2.0+)
    if major < 2:
        raise PrerequisiteMissingError(
            f"ERROR: pybind11 2.0+ is required to build nvFuser.\n"
            f"Found: pybind11 {version}\n\n"
            f"Upgrade pybind11:\n"
            f"  pip install --upgrade 'pybind11[global]'"
        )
    
    # Check for CMake support
    # In pybind11 2.x and earlier, CMake support may not be available
    # In pybind11 3.0+, CMake files are included in the base package
    # The [global] extra is recommended but optional in 3.0+
    
    # Check if get_cmake_dir() exists and returns a valid path
    try:
        cmake_dir = pybind11.get_cmake_dir()
    except AttributeError:
        # Very old pybind11 without CMake support at all
        raise PrerequisiteMissingError(
            "ERROR: pybind11 is installed without CMake support.\n\n"
            f"Found: pybind11 {version} (too old)\n\n"
            "Install pybind11 2.0+ with CMake support:\n"
            "  pip install --upgrade 'pybind11[global]'"
        )
    
    # Verify the cmake directory exists and contains config files
    import os
    if not os.path.exists(cmake_dir) or not os.path.exists(os.path.join(cmake_dir, 'pybind11Config.cmake')):
        raise PrerequisiteMissingError(
            "ERROR: pybind11 CMake configuration is missing or invalid.\n\n"
            f"Found: pybind11 {version} (CMake dir: {cmake_dir})\n\n"
            "Reinstall pybind11 with CMake support:\n"
            "  pip install --force-reinstall 'pybind11[global]'"
        )
    
    # Success: print confirmation
    print(f"[nvFuser] pybind11: {version} with CMake support ✓")
    
    return version


def check_torch_installed() -> Tuple[str, str]:
    """
    Check that PyTorch 2.0+ with CUDA 12+ support is installed.
    
    nvFuser requires PyTorch 2.0+ compiled with CUDA 12+ support. CPU-only PyTorch
    builds are not supported. The CUDA version must match the system CUDA toolkit
    that will be used to build nvFuser.
    
    Returns:
        Tuple[str, str]: (torch_version, cuda_version_str)
        
    Raises:
        PrerequisiteMissingError: If PyTorch is not installed, version is too old,
                                 is CPU-only, or has CUDA < 12
        
    Example:
        >>> version, cuda = check_torch_installed()
        [nvFuser] PyTorch: 2.5.1 with CUDA 12.1 ✓
        >>> version, cuda
        ('2.5.1', '12.1')
    """
    # Check if PyTorch is installed
    try:
        import torch
    except ImportError:
        raise PrerequisiteMissingError(
            "ERROR: PyTorch is not installed.\n\n"
            "nvFuser requires PyTorch 2.0+ with CUDA 12+ support.\n"
            "The CUDA version must match your system CUDA toolkit.\n"
            "Check your system CUDA version: nvcc --version\n\n"
            "Install PyTorch with CUDA support:\n"
            f"{_get_torch_install_instructions()}\n\n"
            "Visit https://pytorch.org for more installation options."
        )
    
    # Get PyTorch version (remove any +cu130 suffix)
    torch_version = torch.__version__.split('+')[0]
    
    # Parse major version
    try:
        major = int(torch_version.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse PyTorch version: {torch.__version__}\n\n"
            f"Please reinstall PyTorch:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Check minimum version requirement (2.0+)
    if major < 2:
        raise PrerequisiteMissingError(
            f"ERROR: PyTorch 2.0+ is required to build nvFuser.\n"
            f"Found: PyTorch {torch_version}\n\n"
            f"Upgrade PyTorch (match your system CUDA version):\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Check if PyTorch has CUDA support (not CPU-only)
    cuda_version_str = torch.version.cuda
    if cuda_version_str is None:
        raise PrerequisiteMissingError(
            "ERROR: PyTorch is CPU-only. nvFuser requires CUDA-enabled PyTorch.\n\n"
            "You have installed PyTorch without CUDA support. This is a common mistake.\n"
            "nvFuser needs PyTorch compiled with CUDA 12+ to build and run correctly.\n"
            "The CUDA version must match your system CUDA toolkit.\n"
            "Check your system CUDA version: nvcc --version\n\n"
            "Install PyTorch with CUDA support:\n"
            f"{_get_torch_install_instructions()}"
        )
    
    # Parse CUDA version and validate >= 12
    try:
        cuda_major = int(cuda_version_str.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse CUDA version from PyTorch: {cuda_version_str}\n\n"
            f"Please reinstall PyTorch with CUDA 12+:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Check CUDA version requirement (12+)
    if cuda_major < 12:
        raise PrerequisiteMissingError(
            f"ERROR: PyTorch with CUDA 12+ is required to build nvFuser.\n"
            f"Found: PyTorch {torch_version} with CUDA {cuda_version_str}\n\n"
            f"nvFuser requires CUDA 12 or newer. Please upgrade PyTorch:\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Success: print confirmation
    print(f"[nvFuser] PyTorch: {torch_version} with CUDA {cuda_version_str} ✓")
    
    return torch_version, cuda_version_str

