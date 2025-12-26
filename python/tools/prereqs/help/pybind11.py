# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""pybind11 installation help."""

from typing import Dict
from .base import HelpProvider


class Pybind11Help(HelpProvider):
    """Provides installation help for pybind11."""

    def generate_help(self, failure: Dict) -> None:
        """Generate pybind11 installation help."""
        version_min = failure.get("version_required", "2.0")

        print(f"pybind11 {version_min}+ Required")
        print()
        print("Why: pybind11 provides Python bindings for nvFuser's C++ code.")
        print()
        print(f"Install pybind11 {version_min} or higher:")
        print()
        print("  pip install 'pybind11[global]>=2.0'")
        print()
        print("  Note: The [global] extra provides CMake integration.")
        print()
