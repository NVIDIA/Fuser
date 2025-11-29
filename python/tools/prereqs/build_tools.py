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
            "Install CMake:\n"
            "  pip install cmake\n\n"
            "Alternative (system install):\n"
            "  sudo apt install cmake  # Ubuntu/Debian\n"
            "  brew install cmake      # macOS"
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
            f"Upgrade CMake:\n"
            f"  pip install --upgrade cmake"
        )
    
    # Success: print confirmation
    print(f"[nvFuser] CMake: {major}.{minor}.{patch} ✓")
    
    return (major, minor, patch)


def check_ninja_installed() -> str:
    """
    Check that Ninja build system is installed with minimum requirement (1.10+).
    
    Ninja provides fast parallel builds and is recommended for nvFuser.
    
    Returns:
        str: Ninja version string
        
    Raises:
        PrerequisiteMissingError: If Ninja is not installed or version is below 1.10
        
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
            "Ninja 1.10+ is required for fast parallel builds of nvFuser.\n\n"
            "Install Ninja:\n"
            "  pip install ninja\n\n"
            "Alternative (system install):\n"
            "  sudo apt install ninja-build  # Ubuntu/Debian\n"
            "  brew install ninja            # macOS"
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
    
    # Extract version numbers using regex
    version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if not version_match:
        # Try simpler pattern (major.minor)
        version_match = re.search(r'(\d+)\.(\d+)', version_str)
        if not version_match:
            raise PrerequisiteMissingError(
                f"ERROR: Could not parse Ninja version from: {version_str}\n\n"
                f"Please ensure Ninja is installed correctly:\n"
                f"  pip install ninja"
            )
    
    version_parts = version_match.groups()
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    # Check minimum version requirement (1.10+)
    if (major, minor) < (1, 10):
        raise PrerequisiteMissingError(
            f"ERROR: Ninja 1.10+ is required to build nvFuser.\n"
            f"Found: Ninja {version_str}\n\n"
            f"Upgrade Ninja:\n"
            f"  pip install --upgrade ninja"
        )
    
    # Success: print confirmation
    print(f"[nvFuser] Ninja: {version_str} ✓")
    
    return version_str

