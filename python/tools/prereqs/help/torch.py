# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch installation help."""

from typing import Dict
from .base import HelpProvider

try:
    from prereqs import pytorch_install_instructions
except ImportError:
    pytorch_install_instructions = lambda: "pip install torch --index-url https://download.pytorch.org/whl/cu121"


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
