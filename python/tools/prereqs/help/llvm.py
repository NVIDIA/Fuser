# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""LLVM installation help."""

from typing import Dict
from .base import HelpProvider

try:
    from prereqs import parse_version, llvm_download_url
except ImportError:
    def parse_version(version_str: str):
        import re
        clean = re.match(r"^[\d.]+", version_str.strip())
        if not clean:
            raise ValueError(f"Cannot parse version: {version_str}")
        parts = clean.group().rstrip(".").split(".")
        return tuple(int(p) for p in parts if p)

    def llvm_download_url(version_tuple):
        raise NotImplementedError("llvm_download_url not available")


class LLVMHelp(HelpProvider):
    """Provides installation help for LLVM."""

    def generate_help(self, failure: Dict) -> None:
        """Generate LLVM installation help."""
        version_min = failure.get("version_required", "18.1")

        # Parse version to get components and build recommended version
        # For minimum version like "19.1", recommend "19.1.8" (latest patch)
        # For minimum version like "18", recommend "18.1.8"
        try:
            version_parts = parse_version(version_min)
            if len(version_parts) >= 2:
                # Has major.minor, use latest patch
                recommended = f"{version_parts[0]}.{version_parts[1]}.8"
                recommended_tuple = (version_parts[0], version_parts[1], 8)
            else:
                # Only major version, add minor and patch
                recommended = f"{version_parts[0]}.1.8"
                recommended_tuple = (version_parts[0], 1, 8)
        except:
            # Fallback if parsing fails
            recommended = "18.1.8"
            recommended_tuple = (18, 1, 8)

        print(f"LLVM {version_min}+ Required")
        print()
        print("Why: nvFuser uses LLVM for runtime Host IR JIT compilation.")
        print()
        print(f"Install LLVM {recommended} (recommended):")
        print()

        print("  Option 1: Prebuilt binaries (recommended, no sudo needed):")
        print()

        try:
            url = llvm_download_url(recommended_tuple)
            print(f"    wget {url}")
            print(f"    tar -xf clang+llvm-{recommended}-*.tar.xz")
            print(f"    mv clang+llvm-{recommended}-* ~/.llvm/{recommended}")
            print()
            print("    # Add to PATH:")
            print(f"    export PATH=$HOME/.llvm/{recommended}/bin:$PATH")
            print()
        except NotImplementedError:
            print("    # Prebuilt binaries not available for your platform")
            print("    # See Option 2 below")
            print()

        print("  Option 2: System package manager:")
        print()

        os_type = self.platform_info["os"]
        major_version = recommended_tuple[0]

        if os_type == "Linux":
            if self.platform_info.get("ubuntu_based"):
                print("    # Ubuntu/Debian (LLVM APT repository):")
                print("    wget https://apt.llvm.org/llvm.sh")
                print("    chmod +x llvm.sh")
                print(f"    sudo ./llvm.sh {major_version}")
                print()
            else:
                print("    # Check your distribution's package manager")
                print("    # Example for RHEL/CentOS:")
                print(f"    # sudo yum install llvm{major_version}")
                print()

        elif os_type == "Darwin":
            print(f"    brew install llvm@{major_version}")
            print()
            print("    # Add to PATH:")
            print(f"    export PATH=/opt/homebrew/opt/llvm@{major_version}/bin:$PATH")
            print()
