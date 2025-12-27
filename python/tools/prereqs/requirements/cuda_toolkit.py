# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA Toolkit dependency requirement."""

from typing import Optional
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

    def __init__(
        self,
        name: str,
        found: str,
        status: str,
        optional: str,
        version_found: Optional[str] = None,
        version_required: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize CUDAToolkit requirement.

        Args:
            name: Dependency name ("CUDAToolkit")
            found: CUDAToolkit_FOUND CMake variable
            status: CUDAToolkit_STATUS CMake variable
            optional: NVFUSER_REQUIREMENT_CUDAToolkit_OPTIONAL CMake variable
            version_found: CUDAToolkit_VERSION CMake variable
            version_required: NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN CMake variable
            location: CUDAToolkit_LIBRARY_ROOT (from NVFUSER_REQUIREMENT_CUDAToolkit_LOCATION_VAR)
        """
        super().__init__(name, found, status, optional, version_found, version_required, location)

    def format_status_line(self, colors) -> str:
        """Format with CUDA Toolkit path."""
        # Get base version line
        main_line = super().format_status_line(colors)

        # Add CUDA path if available and successful
        if self.status == "SUCCESS" and self.location:
            return main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")
        else:
            return main_line

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

