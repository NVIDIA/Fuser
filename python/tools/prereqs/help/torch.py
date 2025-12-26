# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch installation help and requirement class."""

from typing import Dict

try:
    from prereqs import pytorch_install_instructions
    from prereqs.requirements_classes import VersionRequirement, RequirementStatus
except ImportError:
    pytorch_install_instructions = lambda: "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    VersionRequirement = None
    RequirementStatus = None


# Requirement class specific to Torch
if VersionRequirement is not None:
    class TorchRequirement(VersionRequirement):
        """
        Torch-specific requirement with CUDA constraint checking.

        Extends VersionRequirement to also display Torch CUDA constraint validation.
        """

        def __init__(self, data: Dict):
            super().__init__(data)
            self.extra = data.get("extra", {})

        def format_status_line(self, colors) -> str:
            """Format status line with version and CUDA constraint."""
            # First, format the main Torch dependency line
            main_line = super().format_status_line(colors)

            # Then add constraint line if present
            constraint_line = self._format_cuda_constraint(colors)

            if constraint_line:
                return main_line + "\n" + constraint_line
            else:
                return main_line

        def _format_cuda_constraint(self, colors) -> str:
            """Format CUDA constraint validation line for Torch."""
            constraint_status = self.extra.get("constraint_cuda_status")
            if not constraint_status:
                return ""

            if constraint_status == "match":
                # Success: versions match
                cuda_version = self.extra.get("constraint_cuda_version", "unknown")
                return f"{colors.GREEN}[nvFuser] ✓ Torch_CUDA {cuda_version} (Torch.CUDA == CUDAToolkit){colors.RESET}"
            elif constraint_status == "mismatch":
                # Failure: version mismatch
                torch_cuda = self.extra.get("constraint_cuda_found", "unknown")
                toolkit_cuda = self.extra.get("constraint_cuda_required", "unknown")
                return f"{colors.BOLD_RED}[nvFuser] ✗ Torch_CUDA mismatch (Torch: {torch_cuda}, CUDAToolkit: {toolkit_cuda}){colors.RESET}"
            elif constraint_status == "not_available":
                # Torch doesn't have CUDA support
                return f"{colors.YELLOW}[nvFuser] ○ Torch_CUDA Torch built without CUDA{colors.RESET}"
            else:
                return ""

        def is_failure(self) -> bool:
            """
            Check if this requirement represents a failure.
            Also checks CUDA constraint failures.
            """
            # Main dependency failure
            if super().is_failure():
                return True

            # CUDA constraint failure
            constraint_status = self.extra.get("constraint_cuda_status")
            if constraint_status == "mismatch":
                return True

            return False


# Help provider class
from .base import HelpProvider


class TorchHelp(HelpProvider):
    """Provides installation help for PyTorch."""

    def generate_help(self, failure: Dict) -> None:
        """Generate PyTorch installation help."""
        version_min = failure.get("version_required", "2.0")

        print(f"PyTorch {version_min}+ Required")
        print()
        print("Why: nvFuser is a PyTorch extension and requires PyTorch with CUDA support.")
        print()
        print(f"Install PyTorch {version_min} or higher with CUDA:")
        print()

        # Show install instructions for available CUDA versions
        print(pytorch_install_instructions())
        print()


class TorchCUDAConstraintHelp(HelpProvider):
    """Provides help for Torch CUDA version mismatch (shown by TorchRequirement)."""

    def generate_help(self, failure: Dict) -> None:
        """Generate Torch CUDA constraint help."""
        print("Torch CUDA Version Mismatch")
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
        print("    Install PyTorch with matching CUDA:")
        print()
        print(pytorch_install_instructions())
        print()
