# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA Toolkit dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class CUDAToolkitRequirement(VersionRequirement):
    """
    NVIDIA CUDA Toolkit requirement.

    CMake variables used:
    - CUDAToolkit_LIBRARY_ROOT: Path to CUDA installation
    - CUDAToolkit_VERSION: Detected version (e.g., "13.1.80")

    Minimum version: 12.6+ (defined in CMake)
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # CUDA-specific: could add custom fields here if needed

    # Inherits format_status_line from VersionRequirement
    # Pure version checking
