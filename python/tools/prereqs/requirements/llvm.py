# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""LLVM dependency requirement."""

from typing import Optional
from .base import VersionRequirement


class LLVMRequirement(VersionRequirement):
    """
    LLVM requirement for Host IR JIT compilation.

    CMake variables used:
    - LLVM_DIR: Path to LLVM CMake config
    - LLVM_VERSION: Detected version (e.g., "18.1.3")
    - LLVM_FOUND: Whether LLVM was found
    - LLVM_STATUS: Validation status
    - NVFUSER_REQUIREMENT_LLVM_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_LLVM_OPTIONAL: Whether LLVM is optional

    Minimum version: 18.1+ (defined in CMake)
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
        Initialize LLVM requirement.

        Args:
            name: Dependency name ("LLVM")
            found: LLVM_FOUND CMake variable
            status: LLVM_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_LLVM_OPTIONAL CMake variable
            version_found: LLVM_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_LLVM_VERSION_MIN CMake variable
            location: LLVM_DIR (from NVFUSER_REQUIREMENT_LLVM_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)

    def format_status_line(self, colors) -> str:
        """Format with LLVM directory path."""
        # Get base version line
        main_line = super().format_status_line(colors)

        # Add LLVM directory if available and successful
        if self.status == "SUCCESS" and self.location:
            return main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")
        else:
            return main_line

