# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compiler dependency requirement (GCC/Clang)."""

from typing import Dict
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

    Note: CMake exports "GCC" or "Clang" as the name, but variables are prefixed with "Compiler_"

    Minimum versions:
    - GCC: 13+ (C++20 with <format>)
    - Clang: 19+ (C++20 with <format>)
    """

    def __init__(self, data: Dict):
        # Special handling: name is "GCC" or "Clang" but variables are "Compiler_*"
        # Override base class to use "Compiler" prefix for variable lookups
        self.name = data["name"]  # GCC or Clang
        self.type = data["type"]
        cmake_vars = data.get("cmake_vars", {})
        metadata = data.get("metadata", {})

        # Use "Compiler" prefix for CMake variables (not self.name)
        self.found = cmake_vars.get("Compiler_FOUND", False)
        self.status = cmake_vars.get("Compiler_STATUS", "UNKNOWN")
        self.version_found = cmake_vars.get("Compiler_VERSION")
        self.version_required = metadata.get("NVFUSER_REQUIREMENT_Compiler_VERSION_MIN")

        # Get location
        location_var = metadata.get("NVFUSER_REQUIREMENT_Compiler_LOCATION_VAR")
        if location_var:
            self.location = cmake_vars.get(location_var)
        else:
            self.location = None

        self.optional = metadata.get("NVFUSER_REQUIREMENT_Compiler_OPTIONAL", False)

        self._data = data
        self._cmake_vars = cmake_vars
        self._metadata = metadata

    # Inherits format_status_line from VersionRequirement
    # Name (GCC/Clang) is already set correctly for display
