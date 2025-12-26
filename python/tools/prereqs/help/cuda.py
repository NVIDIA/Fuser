# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA Toolkit and Torch CUDA constraint installation help."""

from typing import Dict
from .base import HelpProvider

try:
    from prereqs import cuda_toolkit_download_url
except ImportError:
    cuda_toolkit_download_url = lambda: "https://developer.nvidia.com/cuda-downloads"


class CUDAToolkitHelp(HelpProvider):
    """Provides installation help for CUDA Toolkit."""

    def generate_help(self, failure: Dict) -> None:
        """Generate CUDA Toolkit installation help."""
        version_min = failure.get("version_required", "12.6")

        print(f"CUDA Toolkit {version_min}+ Required")
        print()
        print("Why: nvFuser needs the CUDA compiler (nvcc) for GPU kernel generation.")
        print()
        print(f"Install CUDA Toolkit {version_min} or higher:")
        print()
        print("  Download from NVIDIA:")
        print()
        print(f"    {cuda_toolkit_download_url()}")
        print()
        print("  After installation, ensure CUDA is in your PATH:")
        print()
        print("    export PATH=/usr/local/cuda/bin:$PATH")
        print("    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print()


class TorchCUDAConstraintHelp(HelpProvider):
    """Provides help for Torch CUDA version mismatch."""

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

try:
    from prereqs import pytorch_install_instructions
    TorchCUDAConstraintHelp._original_generate = TorchCUDAConstraintHelp.generate_help

    def _enhanced_generate(self, failure: Dict) -> None:
        self._original_generate(failure)
        print("    Install PyTorch with matching CUDA:")
        print()
        print(pytorch_install_instructions())
        print()

    TorchCUDAConstraintHelp.generate_help = _enhanced_generate
except ImportError:
    pass
