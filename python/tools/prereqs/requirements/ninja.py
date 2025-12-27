# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ninja build system dependency requirement."""

from typing import Optional
from .base import BooleanRequirement


class NinjaRequirement(BooleanRequirement):
    """
    Ninja build system check.

    CMake variables used:
    - Ninja_FOUND: Whether Ninja is available
    - Ninja_STATUS: Validation status
    - CMAKE_MAKE_PROGRAM: Path to build tool
    - NVFUSER_REQUIREMENT_Ninja_OPTIONAL: Whether Ninja is optional

    No version checking - just verifies Ninja is available.
    """

    def __init__(
        self,
        name: str,
        found: str,
        status: str,
        optional: str,
        location: Optional[str] = None,
    ):
        """
        Initialize Ninja requirement.

        Args:
            name: Dependency name ("Ninja")
            found: Ninja_FOUND CMake variable
            status: Ninja_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_Ninja_OPTIONAL CMake variable
            location: CMAKE_MAKE_PROGRAM (from NVFUSER_REQUIREMENT_Ninja_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, location)
