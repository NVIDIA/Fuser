# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Python package validation for nvFuser build system.

Validates that required Python packages (pybind11, PyTorch) are installed with
correct versions and features. This module handles pybind11 validation; PyTorch
validation will be added in Task 4.
"""

from .exceptions import PrerequisiteMissingError


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

