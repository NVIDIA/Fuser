# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compiler dependency requirement (GCC/Clang)."""

from typing import Optional
from .base import VersionRequirement


class CompilerRequirement(VersionRequirement):
    """
    C++ compiler requirement with name mapping.

    CMake variables used:
    - CMAKE_CXX_COMPILER: Path to compiler binary
    - Compiler_NAME: Mapped compiler name ("GCC" or "Clang")
    - Compiler_VERSION: Detected version
    - Compiler_FOUND: Whether compiler was found
    - Compiler_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Compiler_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_Compiler_OPTIONAL: Whether compiler is optional

    Note: CMake exports "GCC" or "Clang" as the name, but variables are prefixed with "Compiler_"

    Minimum versions:
    - GCC: 13+ (C++20 with <format>)
    - Clang: 19+ (C++20 with <format>)
    """

    def __init__(
        self,
        name: str,
        found: str,
        status: str,
        optional: str,
        version_found: Optional[str] = None,
        version_required: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize compiler requirement.

        Note: Name will be "GCC" or "Clang", but all CMake variables use "Compiler_" prefix.

        Args:
            name: Dependency name ("GCC" or "Clang" from Compiler_NAME)
            found: Compiler_FOUND CMake variable
            status: Compiler_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_Compiler_OPTIONAL CMake variable
            version_found: Compiler_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_Compiler_VERSION_MIN CMake variable
            location: CMAKE_CXX_COMPILER (from NVFUSER_REQUIREMENT_Compiler_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)

    def format_status_line(self, colors) -> str:
        """Format with compiler path."""
        # Get base version line
        main_line = super().format_status_line(colors)

        # Add compiler path if available and successful
        if self.status == "SUCCESS" and self.location:
            return main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")
        else:
            return main_line

