# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Git submodules dependency requirement."""

from typing import Dict
from .base import BooleanRequirement


class GitSubmodulesRequirement(BooleanRequirement):
    """
    Git submodules initialization check.

    CMake variables used:
    - GitSubmodules_FOUND: Whether submodules are initialized
    - GitSubmodules_STATUS: Validation status
    - NVFUSER_REQUIREMENT_GitSubmodules_OPTIONAL: Whether submodules are optional

    No version checking - simple pass/fail.
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize Git submodules requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "GitSubmodules"
        found_var = "GitSubmodules_FOUND"
        status_var = "NVFUSER_REQUIREMENT_GitSubmodules_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_GitSubmodules_OPTIONAL"
        location_var = "NVFUSER_REQUIREMENT_GitSubmodules_LOCATION_VAR"

        super().__init__(name, cmake_vars, found_var, status_var, optional_var, location_var)

    def generate_help(self, platform_info):
        """
        Generate Git submodules help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        print("Git Submodules Not Initialized")
        print()
        print("Why: nvFuser depends on third-party libraries included as Git submodules.")
        print()
        print("Initialize and update Git submodules:")
        print()
        print("  # From the repository root:")
        print("  git submodule update --init --recursive")
        print()
        print("  # Or if you just cloned:")
        print("  git clone --recursive <repository-url>")
        print()
