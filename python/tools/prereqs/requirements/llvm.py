# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""LLVM dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class LLVMRequirement(VersionRequirement):
    """
    LLVM requirement for Host IR JIT compilation.

    CMake variables used:
    - LLVM_FOUND: Whether LLVM was found
    - LLVM_VERSION: Detected version (e.g., "18.1.3")
    - LLVM_DIR: Path to LLVM CMake config
    - NVFUSER_REQUIREMENT_LLVM_STATUS: Validation status
    - NVFUSER_REQUIREMENT_LLVM_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_LLVM_OPTIONAL: Whether LLVM is optional
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize LLVM requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "LLVM"
        found_var = "LLVM_FOUND"
        status_var = "NVFUSER_REQUIREMENT_LLVM_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_LLVM_OPTIONAL"
        version_found_var = "LLVM_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_LLVM_VERSION_MIN"
        location_var = "LLVM_DIR"

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
        Generate LLVM installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        import re

        version_min = self.version_required

        # Parse version to recommend a specific patch version
        try:
            clean = re.match(r"^[\d.]+", version_min.strip())
            if clean:
                parts = clean.group().rstrip(".").split(".")
                version_parts = tuple(int(p) for p in parts if p)
                if len(version_parts) >= 2:
                    recommended = f"{version_parts[0]}.{version_parts[1]}.8"
                    major_version = version_parts[0]
                else:
                    recommended = f"{version_parts[0]}.1.8"
                    major_version = version_parts[0]
            else:
                recommended = "18.1.8"
                major_version = 18
        except Exception:
            recommended = "18.1.8"
            major_version = 18

        print(f"LLVM {version_min}+ Required")
        print()
        print("Why: nvFuser uses LLVM for runtime Host IR JIT compilation.")
        print()
        print(f"Install LLVM {recommended} (recommended):")
        print()

        print("  Option 1: Prebuilt binaries (recommended, no sudo needed):")
        print()
        print(
            f"    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-{recommended}/clang+llvm-{recommended}-x86_64-linux-gnu-ubuntu-18.04.tar.xz"
        )
        print(f"    tar -xf clang+llvm-{recommended}-*.tar.xz")
        print(f"    mv clang+llvm-{recommended}-* ~/.llvm/{recommended}")
        print()
        print("    # Add to PATH:")
        print(f"    export PATH=$HOME/.llvm/{recommended}/bin:$PATH")
        print()

        print("  Option 2: System package manager:")
        print()

        os_type = platform_info["os"]

        if os_type == "Linux":
            if platform_info.get("ubuntu_based"):
                print("    # Ubuntu/Debian (LLVM APT repository):")
                print("    wget https://apt.llvm.org/llvm.sh")
                print("    chmod +x llvm.sh")
                print(f"    sudo ./llvm.sh {major_version}")
                print()
            else:
                print("    # Check your distribution's package manager")
                print()
        elif os_type == "Darwin":
            print(f"    brew install llvm@{major_version}")
            print()
            print("    # Add to PATH:")
            print(f"    export PATH=/opt/homebrew/opt/llvm@{major_version}/bin:$PATH")
            print()
