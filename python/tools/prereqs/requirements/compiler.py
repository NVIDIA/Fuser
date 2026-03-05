# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compiler dependency requirement (GNU/Clang)."""

from typing import Dict
from .base import VersionRequirement


class CompilerRequirement(VersionRequirement):
    """
    C++ compiler requirement with name mapping.

    CMake variables used:
    - CMAKE_CXX_COMPILER_ID: Compiler name (GNU or Clang)
    - Compiler_FOUND: Whether compiler is available (always TRUE)
    - CMAKE_CXX_COMPILER_VERSION: Detected compiler version
    - CMAKE_CXX_COMPILER: Path to compiler executable
    - NVFUSER_REQUIREMENT_Compiler_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Compiler_VERSION_MIN: Minimum required version (set based on compiler ID)
    - NVFUSER_REQUIREMENT_Compiler_OPTIONAL: Whether compiler is optional
    - NVFUSER_REQUIREMENT_GNU_VERSION_MIN: Minimum GNU version (from DependencyRequirements.cmake)
    - NVFUSER_REQUIREMENT_Clang_VERSION_MIN: Minimum Clang version (from DependencyRequirements.cmake)

    Note: CMake exports "GNU" or "Clang" as the name, but variables are prefixed with "Compiler_"

    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize compiler requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Extract compiler name from CMake (GNU or Clang)
        name = cmake_vars.get("CMAKE_CXX_COMPILER_ID", "Unknown")

        # Compiler uses "Compiler_" prefix for all variables, regardless of actual name
        found_var = "Compiler_FOUND"
        status_var = "NVFUSER_REQUIREMENT_Compiler_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_Compiler_OPTIONAL"
        version_found_var = "CMAKE_CXX_COMPILER_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_Compiler_VERSION_MIN"
        location_var = "CMAKE_CXX_COMPILER"

        self.gnu_min_version = cmake_vars.get("NVFUSER_REQUIREMENT_GNU_VERSION_MIN")
        self.clang_min_version = cmake_vars.get("NVFUSER_REQUIREMENT_Clang_VERSION_MIN")

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
        Generate compiler installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        # Use the version requirement that was set during initialization
        version_min = self.version_required

        print(f"{self.name} {version_min}+ Required")
        print()
        print("Why: nvFuser requires a modern C++ compiler with C++20 support,")
        print("     including the <format> header.")
        print()
        print(f"Install {self.name} {version_min} or higher:")
        print()

        os_type = platform_info["os"]

        if self.name == "GNU":
            if os_type == "Linux":
                if platform_info.get("ubuntu_based"):
                    print("  Option 1: Ubuntu PPA (recommended):")
                    print()
                    print("    sudo add-apt-repository ppa:ubuntu-toolchain-r/test")
                    print("    sudo apt update")
                    print(f"    sudo apt install gcc-{version_min} g++-{version_min}")
                    print()
                    print("    # Set as default:")
                    print(
                        f"    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-{version_min} 100"
                    )
                    print(
                        f"    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-{version_min} 100"
                    )
                    print()
                else:
                    print("  Option 1: System package manager:")
                    print()
                    print("    # Example for RHEL/CentOS:")
                    print(f"    # sudo yum install gcc-toolset-{version_min}")
                    print()

            elif os_type == "Darwin":
                print("  On macOS, use Clang instead:")
                print()
                print("    # Xcode Command Line Tools (includes Clang):")
                print("    xcode-select --install")
                print()

        elif self.name == "Clang":
            if os_type == "Linux":
                if platform_info.get("ubuntu_based"):
                    print("  Option 1: LLVM APT repository:")
                    print()
                    print("    wget https://apt.llvm.org/llvm.sh")
                    print("    chmod +x llvm.sh")
                    print(f"    sudo ./llvm.sh {version_min}")
                    print()
                else:
                    print("  Option 1: System package manager:")
                    print()
                    print(f"    # Check your distribution for clang-{version_min}")
                    print()

            elif os_type == "Darwin":
                print("  Option 1: Xcode Command Line Tools:")
                print()
                print("    xcode-select --install")
                print()

        print("  Option 2: Build from source:")
        print()
        print("    # See compiler documentation for build instructions")
        print()
