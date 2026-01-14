# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch dependency requirement with CUDA constraint validation."""

from typing import Dict
from .base import VersionRequirement


class TorchRequirement(VersionRequirement):
    """
    PyTorch requirement with CUDA version constraint checking.

    CMake variables used:
    - Torch_FOUND: Whether Torch was found
    - Torch_VERSION: Detected PyTorch version
    - Torch_DIR: Path to Torch CMake config
    - NVFUSER_REQUIREMENT_Torch_STATUS: Validation status
    - NVFUSER_REQUIREMENT_Torch_VERSION_MIN: Minimum required version
    - NVFUSER_REQUIREMENT_Torch_OPTIONAL: Whether Torch is optional
    - Torch_CUDA_constraint_status: CUDA constraint validation result
        - "match": Torch CUDA == CUDAToolkit version
        - "mismatch": Versions don't match (FAILURE)
        - "not_available": Torch built without CUDA
    - Torch_CUDA_constraint_version: CUDA version if match
    - Torch_CUDA_constraint_found: Torch's CUDA version (if mismatch)
    - Torch_CUDA_constraint_required: System's CUDA Toolkit version (if mismatch)

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
        found_var = "Torch_FOUND"
        status_var = "NVFUSER_REQUIREMENT_Torch_STATUS"
        optional_var = "NVFUSER_REQUIREMENT_Torch_OPTIONAL"
        version_found_var = "Torch_VERSION"
        version_required_var = "NVFUSER_REQUIREMENT_Torch_VERSION_MIN"
        location_var = "Torch_DIR"

        super().__init__(
            name,
            cmake_vars,
            found_var,
            status_var,
            optional_var,
            version_found_var,
            version_required_var,
            location_var,
        )

        # Extract Torch CUDA constraint variables from cmake_vars
        self.constraint_status = cmake_vars.get(f"{name}_CUDA_constraint_status")
        self.constraint_version = cmake_vars.get(f"{name}_CUDA_constraint_version")
        self.constraint_found = cmake_vars.get(f"{name}_CUDA_constraint_found")
        self.constraint_required = cmake_vars.get(f"{name}_CUDA_constraint_required")

    def format_status_line(self, colors) -> str:
        """Format with both Torch version and CUDA constraint."""
        # Main Torch version line (base class handles location)
        main_line = super().format_status_line(colors)

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

        # Use same padding as main dependency name
        name_padded = f"{'Torch_CUDA':<15}"

        if self.constraint_status == "match":
            cuda_version = self.constraint_version or "unknown"
            status_part = f"{colors.GREEN}[nvFuser] ✓{colors.RESET} {name_padded}"
            # Use cyan for the CUDA version/result, matching location color
            version_part = (
                f"{colors.CYAN}{cuda_version} (Torch.CUDA == CUDAToolkit){colors.RESET}"
            )
            return f"{status_part} {version_part}"
        elif self.constraint_status == "mismatch":
            torch_cuda = self.constraint_found or "unknown"
            toolkit_cuda = self.constraint_required or "unknown"
            status_part = f"{colors.BOLD_RED}[nvFuser] ✗{colors.RESET} {name_padded}"
            error_part = f"{colors.BOLD_RED}mismatch (Torch: {torch_cuda}, CUDAToolkit: {toolkit_cuda}){colors.RESET}"
            return f"{status_part} {error_part}"
        elif self.constraint_status == "not_available":
            status_part = f"{colors.YELLOW}[nvFuser] ○{colors.RESET} {name_padded}"
            message_part = f"{colors.YELLOW}Torch built without CUDA{colors.RESET}"
            return f"{status_part} {message_part}"
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
        """
        Generate PyTorch installation help.

        Args:
            platform_info: Platform detection dict from detect_platform()
        """
        version_min = self.version_required

        print(f"PyTorch {version_min}+ Required")
        print()
        print(
            "Why: nvFuser is a PyTorch extension and requires PyTorch with CUDA support."
        )
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
            print(
                "Why: PyTorch was built with a different CUDA version than your system's"
            )
            print("     CUDA Toolkit. This will cause runtime errors.")
            print()
            print("Resolution:")
            print()
            print("  You have two options:")
            print()
            print("  Option 1: Install matching CUDA Toolkit (recommended)")
            print()
            print(
                "    Install the CUDA Toolkit version that matches your PyTorch build."
            )
            print(
                "    Check PyTorch CUDA version: python -c 'import torch; print(torch.version.cuda)'"
            )
            print()
            print("  Option 2: Reinstall PyTorch for your CUDA version")
            print()
            print("    Reinstall PyTorch built for your system's CUDA Toolkit version.")
            print("    Check system CUDA version: nvcc --version")
            print()
