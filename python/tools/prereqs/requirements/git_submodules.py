# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Git submodules dependency requirement."""

from typing import Optional
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

    def __init__(
        self,
        name: str,
        found: str,
        status: str,
        optional: str,
        location: Optional[str] = None,
    ):
        """
        Initialize Git submodules requirement.

        Args:
            name: Dependency name ("GitSubmodules")
            found: GitSubmodules_FOUND CMake variable
            status: GitSubmodules_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_GitSubmodules_OPTIONAL CMake variable
            location: Not used for submodules
        """
        super().__init__(name, found, status, optional, location)

    def generate_help(self, platform_info):
        """Generate Git submodules help."""
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
