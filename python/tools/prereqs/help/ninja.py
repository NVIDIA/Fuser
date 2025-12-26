# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ninja build system installation help."""

from typing import Dict
from .base import HelpProvider


class NinjaHelp(HelpProvider):
    """Provides installation help for Ninja build system."""

    def generate_help(self, failure: Dict) -> None:
        """Generate Ninja installation help."""
        print("Ninja Build System Required")
        print()
        print("Why: Ninja is a fast build system used by nvFuser for faster compilation.")
        print()
        print("Install Ninja:")
        print()

        os_type = self.platform_info["os"]

        if os_type == "Linux":
            if self.platform_info.get("ubuntu_based"):
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
