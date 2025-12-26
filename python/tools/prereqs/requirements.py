# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Utility functions for nvFuser dependency reporting.

IMPORTANT: This module provides ONLY utility functions for formatting and URL generation.
Version requirements and dependency validation are handled by CMake.

CMake defines requirements in: cmake/DependencyRequirements.cmake
CMake exports status to: build/nvfuser_dependencies.json
Python reads JSON and uses these utilities to format help text.
"""

import platform
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
# REQUIREMENT DATACLASS (Utility Only)
# =============================================================================


@dataclass
class Requirement:
    """
    A version requirement utility class (NOT the source of truth).

    NOTE: This is for utility methods only. Actual version requirements come from
    CMake's DependencyRequirements.cmake and are exported via JSON.

    Attributes:
        name: Human-readable name (e.g., "CMake", "LLVM")
        min_version: Minimum required version tuple, or None for "any version"
        recommended: Recommended version tuple for download URLs (optional)
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
# CUDA VERSIONS - For PyTorch wheel URLs
# =============================================================================

# PyTorch wheel CUDA versions currently available (newest first)
CUDA_AVAILABLE = [(13, 1), (13, 0), (12, 8)]


# =============================================================================
# URL GENERATORS
# =============================================================================


def cuda_wheel_suffix(cuda: Tuple[int, int]) -> str:
    """
    Convert CUDA version tuple to PyTorch wheel suffix.

    Examples:
        >>> cuda_wheel_suffix((12, 8))
        'cu128'
        >>> cuda_wheel_suffix((13, 1))
        'cu131'
    """
    return f"cu{cuda[0]}{cuda[1]}"


def pytorch_index_url(cuda: Tuple[int, int]) -> str:
    """
    Generate PyTorch wheel index URL for a CUDA version.

    Examples:
        >>> pytorch_index_url((12, 8))
        'https://download.pytorch.org/whl/cu128'
        >>> pytorch_index_url((13, 1))
        'https://download.pytorch.org/whl/cu131'
    """
    return f"https://download.pytorch.org/whl/{cuda_wheel_suffix(cuda)}"


def pytorch_install_instructions(cuda_major: Optional[int] = None) -> str:
    """
    Generate PyTorch installation instructions.

    Args:
        cuda_major: If specified, only show instructions for this CUDA major version.
                   Otherwise show all available versions.

    Returns:
        Multi-line string with pip install commands
    """
    if cuda_major is not None:
        # Filter to matching CUDA major version
        matching = [cuda for cuda in CUDA_AVAILABLE if cuda[0] == cuda_major]
        versions_to_show = matching if matching else CUDA_AVAILABLE
    else:
        versions_to_show = CUDA_AVAILABLE

    lines = []
    for cuda in versions_to_show:
        lines.append(f"  # For CUDA {format_version(cuda)}:")
        lines.append(f"  pip install torch --index-url {pytorch_index_url(cuda)}")
    return "\n".join(lines)


def llvm_download_url(version: Optional[Tuple[int, ...]] = None) -> str:
    """
    Generate LLVM prebuilt binary download URL.

    Args:
        version: LLVM version tuple. If None, uses (18, 1, 8) as default.

    Returns:
        GitHub release URL for prebuilt binary matching current platform

    Raises:
        NotImplementedError: If platform is not supported

    Example:
        >>> llvm_download_url((18, 1, 8))  # doctest: +SKIP
        'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz'
    """
    if version is None:
        version = (18, 1, 8)

    v = format_version(version)
    machine = platform.machine()

    if machine == "x86_64":
        return (
            f"https://github.com/llvm/llvm-project/releases/download/"
            f"llvmorg-{v}/clang+llvm-{v}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
        )
    elif machine == "aarch64":
        return (
            f"https://github.com/llvm/llvm-project/releases/download/"
            f"llvmorg-{v}/clang+llvm-{v}-aarch64-linux-gnu.tar.xz"
        )
    elif machine.startswith("arm64"):
        # 64-bit ARM (macOS)
        return (
            f"https://github.com/llvm/llvm-project/releases/download/"
            f"llvmorg-{v}/clang+llvm-{v}-arm64-apple-macos11.tar.xz"
        )
    else:
        raise NotImplementedError(f"LLVM prebuilt binaries not available for: {machine}")


def cuda_toolkit_download_url() -> str:
    """
    Return NVIDIA CUDA Toolkit download page URL.

    Returns:
        URL to CUDA downloads page
    """
    return "https://developer.nvidia.com/cuda-downloads"
