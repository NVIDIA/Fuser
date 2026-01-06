# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Central source of truth for nvFuser build requirements.

UPDATE VERSIONS HERE when requirements change. All validation modules
import from this file, so changes propagate automatically.

Example:
    from .requirements import CUDA, LLVM, parse_version

    detected = parse_version("18.1.8")
    if not LLVM.check(detected):
        raise PrerequisiteMissingError(f"{LLVM.name} {LLVM.min_display} required")
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# VERSION CONVERSION UTILITIES
# =============================================================================


def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse version string to tuple.

    Args:
        version_str: Version string like "3.8", "18.1.8", "13", "18.1.8git"

    Returns:
        Tuple of integers: (3, 8), (18, 1, 8), (13,), (18, 1, 8)

    Examples:
        >>> parse_version("3.8")
        (3, 8)
        >>> parse_version("18.1.8")
        (18, 1, 8)
        >>> parse_version("13")
        (13,)
        >>> parse_version("18.1.8git")  # strips non-numeric suffix
        (18, 1, 8)

    Raises:
        ValueError: If version string cannot be parsed
    """
    # Strip common suffixes like "git", "rc1", "+cu128", etc.
    clean = re.match(r"^[\d.]+", version_str.strip())
    if not clean:
        raise ValueError(f"Cannot parse version: {version_str}")

    parts = clean.group().rstrip(".").split(".")
    return tuple(int(p) for p in parts if p)


def format_version(version: Tuple[int, ...]) -> str:
    """
    Format version tuple to string.

    Args:
        version: Tuple of integers like (3, 8), (18, 1, 8), (13,)

    Returns:
        Version string: "3.8", "18.1.8", "13"

    Examples:
        >>> format_version((3, 8))
        '3.8'
        >>> format_version((18, 1, 8))
        '18.1.8'
        >>> format_version((13,))
        '13'
    """
    return ".".join(map(str, version))


# =============================================================================
# REQUIREMENT DATACLASS
# =============================================================================


@dataclass
class Requirement:
    """
    A version requirement with optional recommended version for downloads.

    Attributes:
        name: Human-readable name (e.g., "CMake", "LLVM")
        min_version: Minimum required version tuple, or None for "any version"
        recommended: Recommended version tuple for download URLs (optional)

    Examples:
        >>> CMAKE = Requirement("CMake", (3, 18))
        >>> CMAKE.min_str
        '3.18'
        >>> CMAKE.min_display
        '3.18+'
        >>> CMAKE.check((3, 22, 1))
        True

        >>> NINJA = Requirement("Ninja", None)  # Any version
        >>> NINJA.min_display
        'any version'
        >>> NINJA.check((1, 0, 0))
        True

        >>> LLVM = Requirement("LLVM", (18, 1), recommended=(18, 1, 8))
        >>> LLVM.min_str
        '18.1'
        >>> LLVM.recommended_str
        '18.1.8'
    """

    name: str
    min_version: Optional[Tuple[int, ...]]
    recommended: Optional[Tuple[int, ...]] = None

    @property
    def min_str(self) -> str:
        """Minimum version as string: '3.18' or 'any'"""
        if self.min_version is None:
            return "any"
        return format_version(self.min_version)

    @property
    def min_display(self) -> str:
        """Minimum version for display: '3.18+' or 'any version'"""
        if self.min_version is None:
            return "any version"
        return f"{self.min_str}+"

    @property
    def recommended_str(self) -> str:
        """Recommended version as string, falls back to min_str"""
        if self.recommended is None:
            return self.min_str
        return format_version(self.recommended)

    def check(self, detected: Tuple[int, ...]) -> bool:
        """
        Check if detected version meets minimum requirement.

        Args:
            detected: Detected version tuple (e.g., from parse_version)

        Returns:
            True if detected >= min_version (or min_version is None)

        Note:
            Compares only as many parts as min_version specifies.
            So (3, 22, 1) >= (3, 18) compares (3, 22) >= (3, 18) -> True
        """
        if self.min_version is None:
            return True
        # Compare only as many parts as min_version specifies
        return detected[: len(self.min_version)] >= self.min_version


# =============================================================================
# VERSION REQUIREMENTS - UPDATE THESE WHEN VERSIONS CHANGE
# =============================================================================

PYTHON = Requirement("Python", (3, 8), recommended=(3, 10))
CMAKE = Requirement("CMake", (3, 18))
NINJA = Requirement("Ninja", None)  # Any version accepted
PYTORCH = Requirement("PyTorch", (2, 0))
CUDA = Requirement("CUDA", (12, 8))  # Minimum PyTorch CUDA version
PYBIND11 = Requirement("pybind11", (2, 0))
GCC = Requirement("GCC", (13,))  # Major version only; requires <format> header
CLANG = Requirement("Clang", (19,))  # Major version only; Clang 19 has <format> support
LLVM = Requirement("LLVM", (18, 1), recommended=(18, 1, 8))


# =============================================================================
# AVAILABLE CUDA VERSIONS - For install instructions
# =============================================================================

# PyTorch wheel CUDA versions currently available (newest first)
CUDA_AVAILABLE = [(13, 0), (12, 8)]


# =============================================================================
# URL GENERATORS
# =============================================================================


def cuda_wheel_suffix(cuda: Tuple[int, int]) -> str:
    """
    Convert CUDA version tuple to PyTorch wheel suffix.

    Examples:
        >>> cuda_wheel_suffix((12, 8))
        'cu128'
        >>> cuda_wheel_suffix((13, 0))
        'cu130'
    """
    return f"cu{cuda[0]}{cuda[1]}"


def pytorch_index_url(cuda: Tuple[int, int]) -> str:
    """
    Generate PyTorch wheel index URL for a CUDA version.

    Examples:
        >>> pytorch_index_url((12, 8))
        'https://download.pytorch.org/whl/cu128'
        >>> pytorch_index_url((13, 0))
        'https://download.pytorch.org/whl/cu130'
    """
    return f"https://download.pytorch.org/whl/{cuda_wheel_suffix(cuda)}"


def llvm_download_url(version: Tuple[int, ...] = None) -> str:
    """
    Generate LLVM prebuilt binary download URL.

    Args:
        version: LLVM version tuple, defaults to LLVM.recommended

    Returns:
        GitHub release URL for Ubuntu 18.04 x86_64 binary

    Example:
        >>> llvm_download_url()
        'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz'
    """
    v = format_version(version) if version else LLVM.recommended_str
    return (
        f"https://github.com/llvm/llvm-project/releases/download/"
        f"llvmorg-{v}/clang+llvm-{v}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
    )


def pytorch_install_instructions(upgrade: bool = False) -> str:
    """
    Generate PyTorch installation instructions for all available CUDA versions.

    Args:
        upgrade: If True, adds --upgrade flag

    Returns:
        Multi-line string with pip install commands
    """
    flag = " --upgrade" if upgrade else ""
    lines = []
    for cuda in CUDA_AVAILABLE:
        lines.append(f"  # For CUDA {format_version(cuda)}:")
        lines.append(f"  pip install{flag} torch --index-url {pytorch_index_url(cuda)}")
    return "\n".join(lines)
