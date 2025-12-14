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
from .requirements import CMAKE, NINJA, format_version


def check_cmake_version() -> Tuple[int, int, int]:
    """
    Check that CMake meets nvFuser's minimum requirement.

    CMake is required for modern CUDA support features used by nvFuser.

    Returns:
        Tuple[int, int, int]: CMake version as (major, minor, patch) tuple

    Raises:
        PrerequisiteMissingError: If CMake is not installed or version is below minimum

    Example:
        >>> version = check_cmake_version()
        [nvFuser] CMake: 3.22.1 ✓
        >>> version
        (3, 22, 1)
    """
    # Check if cmake exists in PATH
    if not shutil.which("cmake"):
        raise PrerequisiteMissingError(
            f"ERROR: {CMAKE.name} is not installed.\n\n"
            f"{CMAKE.name} {CMAKE.min_display} is required to configure the nvFuser build.\n"
            f"{CMAKE.name} {CMAKE.min_display} provides modern CUDA support features.\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or install {CMAKE.name} individually:\n"
            f"  pip install 'cmake>={CMAKE.min_str}'"
        )

    # Get CMake version
    try:
        result = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise PrerequisiteMissingError(
            f"ERROR: Failed to check {CMAKE.name} version: {e}\n\n"
            f"Install {CMAKE.name}:\n"
            f"  pip install cmake"
        )

    # Parse version string
    # Expected format: "cmake version 3.22.1" (first line)
    version_line = result.stdout.strip().split("\n")[0]

    # Extract version numbers using regex
    version_match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_line)
    if not version_match:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse {CMAKE.name} version from: {version_line}\n\n"
            f"Please ensure {CMAKE.name} is installed correctly:\n"
            f"  pip install cmake"
        )

    major, minor, patch = map(int, version_match.groups())
    detected = (major, minor, patch)

    # Check minimum version requirement
    if not CMAKE.check(detected):
        raise PrerequisiteMissingError(
            f"ERROR: {CMAKE.name} {CMAKE.min_display} is required to build nvFuser.\n"
            f"Found: {CMAKE.name} {format_version(detected)}\n\n"
            f"{CMAKE.name} {CMAKE.min_display} is required for modern CUDA support features.\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or upgrade {CMAKE.name} individually:\n"
            f"  pip install --upgrade 'cmake>={CMAKE.min_str}'"
        )

    return (major, minor, patch)


def check_ninja_installed() -> str:
    """
    Check that Ninja build system is installed.

    Ninja provides fast parallel builds and is recommended for nvFuser.
    Any version is accepted.

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
    if not shutil.which("ninja"):
        raise PrerequisiteMissingError(
            f"ERROR: {NINJA.name} build system is not installed.\n\n"
            f"{NINJA.name} is required for fast parallel builds of nvFuser.\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or install {NINJA.name} individually:\n"
            f"  pip install ninja"
        )

    # Get Ninja version
    try:
        result = subprocess.run(
            ["ninja", "--version"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise PrerequisiteMissingError(
            f"ERROR: Failed to check {NINJA.name} version: {e}\n\n"
            f"Install {NINJA.name}:\n"
            f"  pip install ninja"
        )

    # Parse version string
    # Expected format: "1.11.1" (just the version number)
    version_str = result.stdout.strip()

    # Note: NINJA.min_version is None, so any version is accepted
    return version_str
