# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch dependency requirement with CUDA constraint validation."""

from typing import Optional, Dict
from .base import VersionRequirement, RequirementStatus


class TorchRequirement(VersionRequirement):
    """
    PyTorch requirement with CUDA version constraint checking.

    CMake variables used:
    - Torch_DIR: Path to Torch CMake config
    - Torch_VERSION: Detected PyTorch version
    - Torch_FOUND: Whether Torch was found
    - Torch_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Torch_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_Torch_OPTIONAL: Whether Torch is optional
    - Torch_CUDA_constraint_status: CUDA constraint validation result
        - "match": Torch CUDA == CUDAToolkit version
        - "mismatch": Versions don't match (FAILURE)
        - "not_available": Torch built without CUDA
    - Torch_CUDA_constraint_version: CUDA version if match
    - Torch_CUDA_constraint_found: Torch's CUDA version (if mismatch)
    - Torch_CUDA_constraint_required: System's CUDA Toolkit version (if mismatch)

    Minimum version: 2.0+ (defined in CMake)
    Special: Also validates CUDA version constraint
    """

    def __init__(self, cmake_vars: Dict):
        """
        Initialize Torch requirement.

        Args:
            cmake_vars: Dictionary of all CMake variables
        """
        # Define dependency name and CMake variable names for this requirement
        name = "Torch"
        found_var = f"{name}_FOUND"
        status_var = f"{name}_STATUS"
        optional_var = f"NVFUSER_REQUIREMENT_{name}_OPTIONAL"
        version_found_var = f"{name}_VERSION"
        version_required_var = f"NVFUSER_REQUIREMENT_{name}_VERSION_MIN"
        location_var = f"NVFUSER_REQUIREMENT_{name}_LOCATION_VAR"

        super().__init__(name, cmake_vars, found_var, status_var, optional_var, version_found_var, version_required_var, location_var)

        # Extract Torch CUDA constraint variables from cmake_vars
        self.constraint_status = cmake_vars.get(f"{name}_CUDA_constraint_status")
        self.constraint_version = cmake_vars.get(f"{name}_CUDA_constraint_version")
        self.constraint_found = cmake_vars.get(f"{name}_CUDA_constraint_found")
        self.constraint_required = cmake_vars.get(f"{name}_CUDA_constraint_required")

    def format_status_line(self, colors) -> str:
        """Format with both Torch version and CUDA constraint."""
        # Main Torch version line
        main_line = super().format_status_line(colors)

        # Add Torch path if available and successful
        if self.status == "SUCCESS" and self.location:
            main_line = main_line.replace(colors.RESET, f" ({self.location}){colors.RESET}")

        # Add CUDA constraint line
        constraint_line = self._format_cuda_constraint(colors)

        if constraint_line:
            return main_line + "\n" + constraint_line
        else:
            return main_line

    def _format_cuda_constraint(self, colors) -> str:
        """Format CUDA constraint validation line."""
        if not self.constraint_status:
            return ""

        if self.constraint_status == "match":
            cuda_version = self.constraint_version or "unknown"
            return f"{colors.GREEN}[nvFuser] ✓ Torch_CUDA {cuda_version} (Torch.CUDA == CUDAToolkit){colors.RESET}"
        elif self.constraint_status == "mismatch":
            torch_cuda = self.constraint_found or "unknown"
            toolkit_cuda = self.constraint_required or "unknown"
            return f"{colors.BOLD_RED}[nvFuser] ✗ Torch_CUDA mismatch (Torch: {torch_cuda}, CUDAToolkit: {toolkit_cuda}){colors.RESET}"
        elif self.constraint_status == "not_available":
            return f"{colors.YELLOW}[nvFuser] ○ Torch_CUDA Torch built without CUDA{colors.RESET}"
        else:
            return ""

    def is_failure(self) -> bool:
        """Check for both version failure and CUDA constraint failure."""
        # Check base version requirement
        if super().is_failure():
            return True

        # Check CUDA constraint
        if self.constraint_status == "mismatch":
            return True

        return False

    def generate_help(self, platform_info):
        """Generate PyTorch installation help."""
        version_min = self.version_required or "2.0"

        print(f"PyTorch {version_min}+ Required")
        print()
        print("Why: nvFuser is a PyTorch extension and requires PyTorch with CUDA support.")
        print()
        print(f"Install PyTorch {version_min} or higher with CUDA:")
        print()

        # Show common CUDA versions
        print("  # For CUDA 13.1:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu131")
        print("  # For CUDA 13.0:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu130")
        print("  # For CUDA 12.8:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print()

        # If CUDA constraint mismatch, add additional help
        if self.constraint_status == "mismatch":
            print()
            print("IMPORTANT: Torch CUDA Version Mismatch Detected")
            print()
            print("Why: PyTorch was built with a different CUDA version than your system's")
            print("     CUDA Toolkit. This will cause runtime errors.")
            print()
            print("Resolution:")
            print()
            print("  You have two options:")
            print()
            print("  Option 1: Install matching CUDA Toolkit (recommended)")
            print()
            print("    Install the CUDA Toolkit version that matches your PyTorch build.")
            print("    Check PyTorch CUDA version: python -c 'import torch; print(torch.version.cuda)'")
            print()
            print("  Option 2: Reinstall PyTorch for your CUDA version")
            print()
            print("    Reinstall PyTorch built for your system's CUDA Toolkit version.")
            print("    Check system CUDA version: nvcc --version")
            print()
