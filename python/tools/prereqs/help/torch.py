# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch installation help."""

from typing import Dict

try:
    from prereqs import pytorch_install_instructions
except ImportError:
    pytorch_install_instructions = lambda: "pip install torch --index-url https://download.pytorch.org/whl/cu121"

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
