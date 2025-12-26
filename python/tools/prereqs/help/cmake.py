# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CMake installation help."""

from typing import Dict
from .base import HelpProvider


class CMakeHelp(HelpProvider):
    """Provides installation help for CMake."""

    def generate_help(self, failure: Dict) -> None:
        """Generate CMake installation help."""
        version_min = failure.get("version_required", "3.18")

        print(f"CMake {version_min}+ Required")
        print()
        print("Why: CMake is the build system generator used by nvFuser.")
        print()
        print(f"Install CMake {version_min} or higher:")
        print()

        os_type = self.platform_info["os"]

        if os_type == "Linux":
            if self.platform_info.get("ubuntu_based"):
                print("  Option 1: Ubuntu/Debian (latest):")
                print()
                print("    # Remove old version if installed:")
                print("    sudo apt remove cmake")
                print()
                print("    # Install from Kitware APT repository:")
                print("    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -")
                print("    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'")
                print("    sudo apt update")
                print("    sudo apt install cmake")
                print()
            else:
                print("  Option 1: Download binary:")
                print()
                print("    wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh")
                print("    sudo sh cmake-3.27.0-linux-x86_64.sh --prefix=/usr/local --skip-license")
                print()

        elif os_type == "Darwin":
            print("  Option 1: Homebrew:")
            print()
            print("    brew install cmake")
            print()

        print("  Option 2: pip:")
        print()
        print(f"    pip install 'cmake>={version_min}'")
        print()
