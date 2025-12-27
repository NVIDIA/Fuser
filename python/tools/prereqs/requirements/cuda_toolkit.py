# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA Toolkit dependency requirement."""

from typing import Optional, Dict
from .base import VersionRequirement


class CUDAToolkitRequirement(VersionRequirement):
    """
    NVIDIA CUDA Toolkit requirement.

    CMake variables used:
    - CUDAToolkit_LIBRARY_ROOT: Path to CUDA installation
    - CUDAToolkit_VERSION: Detected version (e.g., "13.1.80")
    - CUDAToolkit_FOUND: Whether CUDA was found
    - CUDAToolkit_STATUS: Validation status
    - NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_CUDAToolkit_OPTIONAL: Whether CUDA is optional

    Minimum version: 12.6+ (defined in CMake)
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize CUDAToolkit requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "CUDAToolkit"
        found_var = "CUDAToolkit_FOUND"
        status_var = "NVFUSER_REQUIREMENT_CUDAToolkit_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_CUDAToolkit_OPTIONAL"
        version_found_var = "CUDAToolkit_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN"
        location_var = "CUDAToolkit_ROOT"

        super().__init__(name, cmake_vars, found_var, status_var, optional_var, version_found_var, version_required_var, location_var)

    def generate_help(self, platform_info):
        """Generate CUDA Toolkit installation help."""
        version_min = self.version_required or "12.6"

        print(f"CUDA Toolkit {version_min}+ Required")
        print()
        print("Why: nvFuser needs the CUDA compiler (nvcc) for GPU kernel generation.")
        print()
        print(f"Install CUDA Toolkit {version_min} or higher:")
        print()
        print("  Download from NVIDIA:")
        print()
        print("    https://developer.nvidia.com/cuda-downloads")
        print()
        print("  After installation, ensure CUDA is in your PATH:")
        print()
        print("    export PATH=/usr/local/cuda/bin:$PATH")
        print("    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print()

