# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Build tool validation for nvFuser build system.

Validates that CMake and Ninja build tools are installed with minimum required
versions. These tools are essential for configuring and building nvFuser.
"""

import re
import shutil
import subprocess
from typing import Tuple

from .exceptions import PrerequisiteMissingError


def check_cmake_version() -> Tuple[int, int, int]:
    """
    Check that CMake meets nvFuser's minimum requirement (3.18+).
    
    CMake 3.18+ is required for modern CUDA support features used by nvFuser.
    
    Returns:
        Tuple[int, int, int]: CMake version as (major, minor, patch) tuple
        
    Raises:
        PrerequisiteMissingError: If CMake is not installed or version is below 3.18
        
    Example:
        >>> version = check_cmake_version()
        [nvFuser] CMake: 3.22.1 ✓
        >>> version
        (3, 22, 1)
    """
    # Check if cmake exists in PATH
    if not shutil.which('cmake'):
        raise PrerequisiteMissingError(
            "ERROR: CMake is not installed.\n\n"
            "CMake 3.18+ is required to configure the nvFuser build.\n"
            "CMake 3.18+ provides modern CUDA support features.\n\n"
            "Install all build dependencies:\n"
            "  pip install -r requirements.txt\n\n"
            "Or install CMake individually:\n"
            "  pip install 'cmake>=3.18'"
        )
    
    # Get CMake version
    try:
        result = subprocess.run(
            ['cmake', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise PrerequisiteMissingError(
            f"ERROR: Failed to check CMake version: {e}\n\n"
            f"Install CMake:\n"
            f"  pip install cmake"
        )
    
    # Parse version string
    # Expected format: "cmake version 3.22.1" (first line)
    version_line = result.stdout.strip().split('\n')[0]
    
    # Extract version numbers using regex
    version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_line)
    if not version_match:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse CMake version from: {version_line}\n\n"
            f"Please ensure CMake is installed correctly:\n"
            f"  pip install cmake"
        )
    
    major, minor, patch = map(int, version_match.groups())
    
    # Check minimum version requirement (3.18+)
    if (major, minor) < (3, 18):
        raise PrerequisiteMissingError(
            f"ERROR: CMake 3.18+ is required to build nvFuser.\n"
            f"Found: CMake {major}.{minor}.{patch}\n\n"
            f"CMake 3.18+ is required for modern CUDA support features.\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or upgrade CMake individually:\n"
            f"  pip install --upgrade 'cmake>=3.18'"
        )
    
    # Success: print confirmation
    print(f"[nvFuser] CMake: {major}.{minor}.{patch} ✓")
    
    return (major, minor, patch)


def check_ninja_installed() -> str:
    """
    Check that Ninja build system is installed.
    
    Ninja provides fast parallel builds and is recommended for nvFuser.
    
    Returns:
        str: Ninja version string
        
    Raises:
        PrerequisiteMissingError: If Ninja is not installed
        
    Example:
        >>> version = check_ninja_installed()
        [nvFuser] Ninja: 1.11.1 ✓
        >>> version
        '1.11.1'
    """
    # Check if ninja exists in PATH
    if not shutil.which('ninja'):
        raise PrerequisiteMissingError(
            "ERROR: Ninja build system is not installed.\n\n"
            "Ninja is required for fast parallel builds of nvFuser.\n\n"
            "Install all build dependencies:\n"
            "  pip install -r requirements.txt\n\n"
            "Or install Ninja individually:\n"
            "  pip install ninja"
        )
    
    # Get Ninja version
    try:
        result = subprocess.run(
            ['ninja', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise PrerequisiteMissingError(
            f"ERROR: Failed to check Ninja version: {e}\n\n"
            f"Install Ninja:\n"
            f"  pip install ninja"
        )
    
    # Parse version string
    # Expected format: "1.11.1" (just the version number)
    version_str = result.stdout.strip()
    
    # Success: print confirmation
    print(f"[nvFuser] Ninja: {version_str} ✓")
    
    return version_str

