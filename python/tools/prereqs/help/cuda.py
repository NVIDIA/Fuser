# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA Toolkit installation help."""

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
