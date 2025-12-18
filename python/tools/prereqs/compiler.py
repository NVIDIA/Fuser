# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
C++ compiler validation for nvFuser build.

nvFuser requires GCC 13+ or Clang 19+ to build because the source code uses
the C++20 <format> header. This module detects the compiler type and version,
and verifies <format> header availability through a compile test.

Note: CI may use Clang via update-alternatives (gcc -> clang), so we detect
the actual compiler from version output, not just the command name.
"""

import re
import subprocess
from typing import Optional, Tuple

from .exceptions import PrerequisiteMissingError
from .requirements import GCC, CLANG, format_version


def get_compiler_info() -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Get C++ compiler type and version.

    Runs 'gcc --version' and detects whether it's actually GCC or Clang
    (CI uses update-alternatives to make gcc point to clang).

    Returns:
        Optional[Tuple[str, Tuple[int, int, int]]]:
            ("gcc", (major, minor, patch)) or ("clang", (major, minor, patch)),
            or None if no compiler found
    """
    try:
        result = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            return None

        output = result.stdout.lower()
        first_line = result.stdout.splitlines()[0] if result.stdout else ""

        # Check if it's actually Clang (via update-alternatives)
        if "clang" in output:
            match = re.search(r"clang version\s+(\d+)\.(\d+)\.(\d+)", output)
            if match:
                return (
                    "clang",
                    (int(match.group(1)), int(match.group(2)), int(match.group(3))),
                )
            return None

        # Parse as GCC
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", first_line)
        if match:
            return (
                "gcc",
                (int(match.group(1)), int(match.group(2)), int(match.group(3))),
            )

        return None

    except FileNotFoundError:
        return None


def check_format_support(compiler_type: str = "gcc") -> bool:
    """
    Test if the C++ compiler's standard library has <format> header support.

    Args:
        compiler_type: "gcc" or "clang" to select appropriate compiler command

    Returns:
        bool: True if <format> header compiles successfully, False otherwise
    """
    test_code = "#include <format>\nint main() { return 0; }"
    cxx = "clang++" if compiler_type == "clang" else "g++"

    try:
        result = subprocess.run(
            [cxx, "-std=c++20", "-x", "c++", "-", "-o", "/dev/null"],
            input=test_code,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    except FileNotFoundError:
        return False


def validate_compiler() -> Tuple[str, Tuple[int, int, int]]:
    """
    Validate that C++ compiler meets requirements (GCC 13+ or Clang 19+).

    Returns:
        Tuple[str, Tuple[int, int, int]]: (compiler_type, version_tuple)

    Raises:
        PrerequisiteMissingError: If compiler not found or version too low
    """
    info = get_compiler_info()

    if info is None:
        raise PrerequisiteMissingError(
            f"ERROR: No C++ compiler found. nvFuser requires GCC {GCC.min_display} or Clang {CLANG.min_display}.\n\n"
            f"To install GCC {GCC.min_version[0]} on Ubuntu:\n"
            f"  sudo apt install gcc-{GCC.min_version[0]} g++-{GCC.min_version[0]}\n"
        )

    compiler_type, version = info

    # Check version based on compiler type
    if compiler_type == "clang":
        req = CLANG
    else:
        req = GCC

    if not req.check(version):
        raise PrerequisiteMissingError(
            f"ERROR: nvFuser requires {req.name} {req.min_display} to build.\n"
            f"Found: {req.name} {format_version(version)}\n\n"
        )

    # Verify <format> header is actually available
    if not check_format_support(compiler_type):
        raise PrerequisiteMissingError(
            f"ERROR: {req.name} {format_version(version)} detected but <format> header not available.\n\n"
            f"The C++20 <format> header is required by nvFuser's source code.\n"
        )

    return (compiler_type, version)
