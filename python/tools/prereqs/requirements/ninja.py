# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ninja build system dependency requirement."""

from typing import Optional, Dict
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

    def __init__(self, cmake_vars: Dict):
        """
        Initialize Ninja requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "Ninja"
        found_var = f"{name}_FOUND"
        status_var = f"{name}_STATUS"
        optional_var = f"NVFUSER_REQUIREMENT_{name}_OPTIONAL"
        location_var = f"NVFUSER_REQUIREMENT_{name}_LOCATION_VAR"

        super().__init__(name, cmake_vars, found_var, status_var, optional_var, location_var)

    def generate_help(self, platform_info):
        """Generate Ninja installation help."""
        print("Ninja Build System Required")
        print()
        print("Why: Ninja is a fast build system used by nvFuser for faster compilation.")
        print()
        print("Install Ninja:")
        print()

        os_type = platform_info["os"]

        if os_type == "Linux":
            if platform_info.get("ubuntu_based"):
                print("  Option 1: Ubuntu/Debian:")
                print()
                print("    sudo apt update")
                print("    sudo apt install ninja-build")
                print()
            else:
                print("  Option 1: System package manager:")
                print()
                print("    # Example for RHEL/CentOS:")
                print("    # sudo yum install ninja-build")
                print()

        elif os_type == "Darwin":
            print("  Option 1: Homebrew:")
            print()
            print("    brew install ninja")
            print()

        print("  Option 2: pip:")
        print()
        print("    pip install ninja")
        print()
