# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch dependency requirement with CUDA constraint validation."""

from typing import Dict
from .base import VersionRequirement, RequirementStatus


class TorchRequirement(VersionRequirement):
    """
    PyTorch requirement with CUDA version constraint checking.

    CMake variables used:
    - Torch_DIR: Path to Torch CMake config
    - Torch_VERSION: Detected PyTorch version
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

    def __init__(self, data: Dict):
        super().__init__(data)

        # Extract Torch CUDA constraint variables
        self.constraint_status = self._cmake_vars.get("Torch_CUDA_constraint_status")
        self.constraint_version = self._cmake_vars.get("Torch_CUDA_constraint_version")
        self.constraint_found = self._cmake_vars.get("Torch_CUDA_constraint_found")
        self.constraint_required = self._cmake_vars.get("Torch_CUDA_constraint_required")

    def format_status_line(self, colors) -> str:
        """Format with both Torch version and CUDA constraint."""
        # Main Torch version line
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
