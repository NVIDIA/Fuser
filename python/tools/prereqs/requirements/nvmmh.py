# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""nvidia-matmul-heuristics dependency requirement."""

from typing import Dict
from .base import BooleanRequirement


class NVMMHRequirement(BooleanRequirement):
    """
    nvidia-matmul-heuristics check.

    CMake variables used:
    - NVMMH_FOUND: Whether nvidia-matmul-heuristics is available
    - NVFUSER_REQUIREMENT_NVMMH_STATUS: Validation status
    - NVFUSER_REQUIREMENT_NVMMH_OPTIONAL: Whether NVMMH is optional

    No version checking - just verifies nvidia-matmul-heuristics headers are available.
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize nvidia-matmul-heuristics requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "NVMMH"
        found_var = "NVMMH_FOUND"
        status_var = "NVFUSER_REQUIREMENT_NVMMH_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_NVMMH_OPTIONAL"
        location_var = "NVMMH_INCLUDE_DIR"

        super().__init__(
            name, cmake_vars, found_var, status_var, optional_var, location_var
        )

    def generate_help(self, platform_info):
        """
        Generate nvidia-matmul-heuristics installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        print("nvidia-matmul-heuristics (NVMMH)")
        print()
        print(
            "Why: nvidia-matmul-heuristics provides optimized matrix multiplication heuristics for nvFuser."
        )
        print()
        print("Install nvidia-matmul-heuristics:")
        print()
        print("  Recommended: pip installation:")
        print()
        print("    pip install nvidia-matmul-heuristics")
        print()
        print("  Note: This is an optional dependency. nvFuser will build without it,")
        print(
            "        but matmul operations may not have access to optimized heuristics."
        )
        print()
