# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NIXL dependency requirement with CUDA constraint validation."""

from typing import Dict
from .base import BooleanRequirement
from ..colors import colorize


class NIXLRequirement(BooleanRequirement):
    """
    NIXL check with CUDA major version constraint.

    CMake variables used:
    - NIXL_FOUND: Whether NIXL is available
    - NVFUSER_REQUIREMENT_NIXL_STATUS: Validation status
    - NVFUSER_REQUIREMENT_NIXL_OPTIONAL: Whether NIXL is optional
    - NIXL_CUDA_constraint_status: CUDA constraint validation result
        - "match": NIXL CUDA major == CUDAToolkit major
        - "mismatch": Versions don't match (WARNING)
        - "not_available": Unable to determine NIXL CUDA version
    - NIXL_CUDA_constraint_version: CUDA major version if match
    - NIXL_CUDA_constraint_found: NIXL's CUDA major version (if mismatch)
    - NIXL_CUDA_constraint_required: System's CUDA major version (if mismatch)
    """

    def __init__(self, cmake_vars: Dict):
        name = "NIXL"
        found_var = "NIXL_FOUND"
        status_var = "NVFUSER_REQUIREMENT_NIXL_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_NIXL_OPTIONAL"
        location_var = "NIXL_LIBRARY"

        super().__init__(
            name, cmake_vars, found_var, status_var, optional_var, location_var
        )

        self.constraint_status = cmake_vars.get("NIXL_CUDA_constraint_status")
        self.constraint_version = cmake_vars.get("NIXL_CUDA_constraint_version")
        self.constraint_found = cmake_vars.get("NIXL_CUDA_constraint_found")
        self.constraint_required = cmake_vars.get("NIXL_CUDA_constraint_required")

    def format_status_line(self, colors) -> str:
        main_line = super().format_status_line(colors)

        constraint_line = self._format_cuda_constraint(colors)
        if constraint_line:
            return main_line + "\n" + constraint_line
        return main_line

    def _format_cuda_constraint(self, colors) -> str:
        if not self.constraint_status or self.constraint_status == "not_available":
            return ""

        name_padded = f"{'NIXL_CUDA':<15}"

        if self.constraint_status == "match":
            cuda_version = self.constraint_version or "unknown"
            status_part = colorize(colors.GREEN, "[nvFuser] ✓") + " " + name_padded
            version_part = colorize(
                colors.CYAN, f"CUDA {cuda_version} (NIXL.CUDA == CUDAToolkit major)"
            )
            return f"{status_part} {version_part}"
        elif self.constraint_status == "mismatch":
            nixl_cuda = self.constraint_found or "unknown"
            toolkit_cuda = self.constraint_required or "unknown"
            status_part = colorize(colors.BOLD_RED, "[nvFuser] ✗") + " " + name_padded
            error_part = colorize(
                colors.BOLD_RED,
                f"mismatch (NIXL: CUDA {nixl_cuda}, CUDAToolkit: CUDA {toolkit_cuda})",
            )
            return f"{status_part} {error_part}"
        return ""

    def generate_help(self, platform_info):
        print("NIXL")
        print()
        print(
            "Why: NIXL provides high-performance data transfer for multi-device nvFuser."
        )
        print()
        print("  nvFuser links against the NIXL C++ API (nixl.h / libnixl.so).")
        print("  'pip install nixl' provides the shared library but does NOT install")
        print("  the C++ headers. You need both headers and libraries for the build.")
        print()
        print("Install NIXL:")
        print()
        print(
            "  Option 1 (recommended for CI): Run the helper script that pip-installs"
        )
        print("  nixl for the .so and clones the repo for headers:")
        print()
        print("    bash tools/install-nixl.sh")
        print("    export NIXL_PREFIX=/tmp/nixl-prefix  # or your chosen path")
        print()
        print("  Option 2: Build NIXL from source and set NIXL_PREFIX to the install")
        print("  directory (must contain include/nixl.h and lib/libnixl.so).")
        print()
        print("  Note: This is an optional dependency. nvFuser will build without it,")
        print("        but multi-device NIXL-based transfers will not be available.")
        print()

        if self.constraint_status == "mismatch":
            print()
            print("IMPORTANT: NIXL CUDA Version Mismatch Detected")
            print()
            print("  NIXL was built for a different CUDA major version than your")
            print("  system's CUDA Toolkit. This will cause linking or runtime errors.")
            print()
            print("  Resolution: Install the NIXL package matching your CUDA version.")
            print("  Check system CUDA major version: nvcc --version")
            print()
