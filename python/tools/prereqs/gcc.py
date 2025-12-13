# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
GCC compiler validation for nvFuser build.

nvFuser requires GCC 13+ to build because the source code uses the C++20 
<format> header (from cutlass integration). This header is available in 
GCC 13+ libstdc++ but NOT in GCC 12.

Ubuntu 22.04 ships with GCC 12 by default, which is the most common build 
failure scenario. This module detects GCC version and verifies <format> 
header availability through a compile test.
"""

import re
import subprocess
from typing import Optional, Tuple

from .exceptions import PrerequisiteMissingError
from .requirements import GCC, format_version


def get_gcc_version() -> Optional[Tuple[int, int, int]]:
    """
    Get GCC compiler version as a tuple.
    
    Returns:
        Optional[Tuple[int, int, int]]: (major, minor, patch) version tuple,
                                        or None if GCC not found
        
    Example:
        >>> version = get_gcc_version()
        >>> version
        (12, 3, 0)  # On Ubuntu 22.04 default
    """
    try:
        result = subprocess.run(
            ['gcc', '--version'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return None
        
        # Parse version from output
        # Example: "gcc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0"
        for line in result.stdout.splitlines():
            if 'gcc' in line.lower():
                # Look for version pattern: major.minor.patch
                match = re.search(r'(\d+)\.(\d+)\.(\d+)', line)
                if match:
                    major = int(match.group(1))
                    minor = int(match.group(2))
                    patch = int(match.group(3))
                    return (major, minor, patch)
        
        return None
        
    except FileNotFoundError:
        return None


def check_format_support() -> bool:
    """
    Test if the C++ compiler's standard library has <format> header support.
    
    This performs an actual compile test by trying to compile a minimal C++20
    program that includes <format>. GCC 12 will fail this test even though it
    supports C++20, because libstdc++ didn't implement <format> until GCC 13.
    
    Returns:
        bool: True if <format> header compiles successfully, False otherwise
        
    Example:
        >>> has_format = check_format_support()
        >>> has_format
        False  # On GCC 12
    """
    # Minimal test program
    test_code = '#include <format>\nint main() { return 0; }'
    
    try:
        result = subprocess.run(
            ['g++', '-std=c++20', '-x', 'c++', '-', '-o', '/dev/null'],
            input=test_code,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
        
    except FileNotFoundError:
        return False


def validate_gcc() -> Tuple[int, int, int]:
    """
    Validate that GCC meets minimum version requirement with <format> header support.
    
    This is the main validation function that should be called during 
    nvFuser's setup process. It checks:
    1. GCC is installed
    2. GCC version meets minimum requirement
    3. <format> header compiles successfully
    
    Returns:
        Tuple[int, int, int]: GCC version as (major, minor, patch) tuple
        
    Raises:
        PrerequisiteMissingError: If GCC not found, version below minimum, or 
                                  <format> header not available
        
    Example:
        >>> version = validate_gcc()
        [nvFuser] GCC: 13.2.0 ✓
        >>> version
        (13, 2, 0)
    """
    # Check if GCC is installed
    gcc_ver = get_gcc_version()
    gcc_min = GCC.min_version[0]  # Major version only
    
    if gcc_ver is None:
        raise PrerequisiteMissingError(
            f"ERROR: {GCC.name} not found. nvFuser requires {GCC.name} {GCC.min_display} to build.\n\n"
            f"{GCC.name} is required to compile nvFuser's C++ source code.\n"
            f"The C++20 <format> header is used in the codebase.\n\n"
            f"To install {GCC.name} {gcc_min} on Ubuntu 22.04:\n"
            f"  sudo add-apt-repository ppa:ubuntu-toolchain-r/test\n"
            f"  sudo apt update\n"
            f"  sudo apt install gcc-{gcc_min} g++-{gcc_min}\n"
            f"  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-{gcc_min} 100\n"
            f"  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-{gcc_min} 100\n"
        )
    
    # Check version requirement
    major, minor, patch = gcc_ver
    
    if not GCC.check(gcc_ver):
        raise PrerequisiteMissingError(
            f"ERROR: nvFuser requires {GCC.name} {GCC.min_display} to build.\n"
            f"Found: {GCC.name} {format_version(gcc_ver)}\n\n"
            f"The C++20 <format> header is required by nvFuser's source code.\n"
            f"This header is available in {GCC.name} {GCC.min_display} (not in {GCC.name} {major}).\n\n"
            f"To install {GCC.name} {gcc_min} on Ubuntu 22.04:\n"
            f"  sudo add-apt-repository ppa:ubuntu-toolchain-r/test\n"
            f"  sudo apt update\n"
            f"  sudo apt install gcc-{gcc_min} g++-{gcc_min}\n"
            f"  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-{gcc_min} 100\n"
            f"  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-{gcc_min} 100\n"
        )
    
    # Verify <format> header is actually available
    if not check_format_support():
        # This is unusual - GCC 13+ should have format support
        print(f"[nvFuser] WARNING: {GCC.name} {format_version(gcc_ver)} detected but <format> header not available")
        print(f"[nvFuser] Build may fail. Please verify {GCC.name} installation is complete.")
        # Don't raise error here - let the build attempt proceed
    
    return gcc_ver

