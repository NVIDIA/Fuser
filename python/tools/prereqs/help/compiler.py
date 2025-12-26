# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compiler (GCC/Clang) installation help."""

from typing import Dict
from .base import HelpProvider


class CompilerHelp(HelpProvider):
    """Provides installation help for GCC/Clang compilers."""

    def generate_help(self, failure: Dict) -> None:
        """Generate compiler installation help."""
        name = failure["name"]
        version_min = failure.get("version_required", "13" if name == "GCC" else "19")

        print(f"{name} {version_min}+ Required")
        print()
        print("Why: nvFuser requires a modern C++ compiler with C++20 support,")
        print("     including the <format> header.")
        print()
        print(f"Install {name} {version_min} or higher:")
        print()

        os_type = self.platform_info["os"]

        if name == "GCC":
            if os_type == "Linux":
                if self.platform_info.get("ubuntu_based"):
                    print("  Option 1: Ubuntu PPA (recommended):")
                    print()
                    print("    sudo add-apt-repository ppa:ubuntu-toolchain-r/test")
                    print("    sudo apt update")
                    print(f"    sudo apt install gcc-{version_min} g++-{version_min}")
                    print()
                    print("    # Set as default:")
                    print(f"    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-{version_min} 100")
                    print(f"    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-{version_min} 100")
                    print()
                else:
                    print("  Option 1: System package manager:")
                    print()
                    print(f"    # Example for RHEL/CentOS:")
                    print(f"    # sudo yum install gcc-toolset-{version_min}")
                    print()

            elif os_type == "Darwin":
                print("  On macOS, use Clang instead:")
                print()
                print("    # Xcode Command Line Tools (includes Clang):")
                print("    xcode-select --install")
                print()

        elif name == "Clang":
            if os_type == "Linux":
                if self.platform_info.get("ubuntu_based"):
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
