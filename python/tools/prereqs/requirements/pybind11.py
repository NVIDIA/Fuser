# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""pybind11 dependency requirement."""

from typing import Optional, Dict
from .base import VersionRequirement


class Pybind11Requirement(VersionRequirement):
    """
    pybind11 requirement for Python bindings.

    CMake variables used:
    - pybind11_DIR: Path to pybind11 CMake config
    - pybind11_VERSION: Detected version (e.g., "3.0.1")
    - pybind11_FOUND: Whether pybind11 was found
    - pybind11_STATUS: Validation status
    - NVFUSER_REQUIREMENT_pybind11_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_pybind11_OPTIONAL: Whether pybind11 is optional

    Minimum version: 2.0+ (defined in CMake)
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize pybind11 requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "pybind11"
        found_var = "pybind11_FOUND"
        status_var = "NVFUSER_REQUIREMENT_pybind11_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_pybind11_OPTIONAL"
        version_found_var = "pybind11_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_pybind11_VERSION_MIN"
        location_var = "pybind11_DIR"

        super().__init__(name, cmake_vars, found_var, status_var, optional_var, version_found_var, version_required_var, location_var)

    def generate_help(self, platform_info):
        """
        Generate pybind11 installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        version_min = self.version_required or "2.0"

        print(f"pybind11 {version_min}+ Required")
        print()
        print("Why: pybind11 provides Python bindings for nvFuser's C++ code.")
        print()
        print(f"Install pybind11 {version_min} or higher:")
        print()
        print("  pip install 'pybind11[global]>=2.0'")
        print()
        print("  Note: The [global] extra provides CMake integration.")
        print()
