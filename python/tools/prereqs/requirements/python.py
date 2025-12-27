# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Python dependency requirement."""

from typing import Optional
from .base import VersionRequirement


class PythonRequirement(VersionRequirement):
    """
    Python interpreter requirement.

    CMake variables used:
    - Python_EXECUTABLE: Path to python binary
    - Python_VERSION: Detected version (e.g., "3.12.3")
    - Python_FOUND: Whether Python was found
    - Python_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Python_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_Python_OPTIONAL: Whether Python is optional

    Minimum version: 3.8+ (defined in CMake)
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
        Initialize Python requirement.

        Args:
            name: Dependency name ("Python")
            found: Python_FOUND CMake variable
            status: Python_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_Python_OPTIONAL CMake variable
            version_found: Python_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_Python_VERSION_MIN CMake variable
            location: Python_EXECUTABLE (from NVFUSER_REQUIREMENT_Python_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)

    def format_status_line(self, colors) -> str:
        """Format with Python executable path."""
        # Get base version line
        main_line = super().format_status_line(colors)

        # Add executable path if available and successful
        if self.status == "SUCCESS" and self.location:
            return main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")
        else:
            return main_line

    def generate_help(self, platform_info):
        """Generate Python installation help."""
        version_min = self.version_required or "3.8"

        print(f"Python {version_min}+ Required")
        print()
        print("Why: nvFuser requires modern Python with type hints and language features.")
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

