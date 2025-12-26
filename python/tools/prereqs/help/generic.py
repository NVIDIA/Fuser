# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Generic fallback installation help."""

from typing import Dict
from .base import HelpProvider


class GenericHelp(HelpProvider):
    """Provides generic fallback help for unknown dependencies."""

    def generate_help(self, failure: Dict) -> None:
        """Generate generic installation help."""
        name = failure["name"]
        version_min = failure.get("version_required", "")

        if version_min:
            print(f"{name} {version_min}+ Required")
        else:
            print(f"{name} Required")
        print()
        print(f"Please install {name} to continue.")
        print()
        print("Check the dependency's official documentation for installation instructions.")
        print()
