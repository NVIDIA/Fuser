# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""pybind11 dependency requirement."""

from typing import Optional
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
        Initialize pybind11 requirement.

        Args:
            name: Dependency name ("pybind11")
            found: pybind11_FOUND CMake variable
            status: pybind11_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_pybind11_OPTIONAL CMake variable
            version_found: pybind11_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_pybind11_VERSION_MIN CMake variable
            location: pybind11_DIR (from NVFUSER_REQUIREMENT_pybind11_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)
