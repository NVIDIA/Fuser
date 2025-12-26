# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Python installation help."""

from typing import Dict
from .base import HelpProvider


class PythonHelp(HelpProvider):
    """Provides installation help for Python."""

    def generate_help(self, failure: Dict) -> None:
        """Generate Python installation help."""
        version_min = failure.get("version_required", "3.8")

        print(f"Python {version_min}+ Required")
        print()
        print("Why: nvFuser requires modern Python with type hints and language features.")
        print()
        print(f"Install Python {version_min} or higher:")
        print()

        os_type = self.platform_info["os"]

        if os_type == "Linux":
            if self.platform_info.get("ubuntu_based"):
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
