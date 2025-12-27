# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""LLVM dependency requirement."""

from typing import Optional
from .base import VersionRequirement


class LLVMRequirement(VersionRequirement):
    """
    LLVM requirement for Host IR JIT compilation.

    CMake variables used:
    - LLVM_DIR: Path to LLVM CMake config
    - LLVM_VERSION: Detected version (e.g., "18.1.3")
    - LLVM_FOUND: Whether LLVM was found
    - LLVM_STATUS: Validation status
    - NVFUSER_REQUIREMENT_LLVM_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_LLVM_OPTIONAL: Whether LLVM is optional

    Minimum version: 18.1+ (defined in CMake)
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
        Initialize LLVM requirement.

        Args:
            name: Dependency name ("LLVM")
            found: LLVM_FOUND CMake variable
            status: LLVM_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_LLVM_OPTIONAL CMake variable
            version_found: LLVM_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_LLVM_VERSION_MIN CMake variable
            location: LLVM_DIR (from NVFUSER_REQUIREMENT_LLVM_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)

    def format_status_line(self, colors) -> str:
        """Format with LLVM directory path."""
        # Get base version line
        main_line = super().format_status_line(colors)

        # Add LLVM directory if available and successful
        if self.status == "SUCCESS" and self.location:
            return main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")
        else:
            return main_line

    def generate_help(self, platform_info):
        """Generate LLVM installation help."""
        import re

        version_min = self.version_required or "18.1"

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
        except:
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
        print(f"    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-{recommended}/clang+llvm-{recommended}-x86_64-linux-gnu-ubuntu-18.04.tar.xz")
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

