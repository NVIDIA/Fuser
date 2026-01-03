# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Python dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class PythonRequirement(VersionRequirement):
    """
    Python interpreter requirement.

    CMake variables used:
    - Python_FOUND: Whether Python was found
    - Python_VERSION: Detected version (e.g., "3.12.3")
    - Python_EXECUTABLE: Path to python binary
    - NVFUSER_REQUIREMENT_Python_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Python_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_Python_OPTIONAL: Whether Python is optional

    Minimum version: 3.8+ (defined in CMake)
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize Python requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "Python"
        found_var = "Python_FOUND"
        status_var = "NVFUSER_REQUIREMENT_Python_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_Python_OPTIONAL"
        version_found_var = "Python_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_Python_VERSION_MIN"
        location_var = "Python_EXECUTABLE"

        super().__init__(
            name,
            cmake_vars,
            found_var,
            status_var,
            optional_var,
            version_found_var,
            version_required_var,
            location_var,
        )

    def generate_help(self, platform_info):
        """
        Generate Python installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        version_min = self.version_required or "3.8"

        print(f"Python {version_min}+ Required")
        print()
        print(
            "Why: nvFuser requires modern Python with type hints and language features."
        )
        print()
        print(f"Install Python {version_min} or higher:")
        print()

        os_type = platform_info["os"]

        if os_type == "Linux":
            if platform_info.get("ubuntu_based"):
                print("  Option 1: Ubuntu/Debian system package:")
                print()
                print("    sudo apt update")
                print(f"    sudo apt install python{version_min}")
                print()
            else:
                print("  Option 1: System package manager:")
                print()
                print(f"    # Example for RHEL/CentOS:")
                print(f"    # sudo yum install python{version_min}")
                print()

        elif os_type == "Darwin":
            print("  Option 1: Homebrew:")
            print()
            print(f"    brew install python@{version_min}")
            print()

        print("  Option 2: Conda:")
        print()
        print(f"    conda create -n nvfuser python={version_min}")
        print("    conda activate nvfuser")
        print()
