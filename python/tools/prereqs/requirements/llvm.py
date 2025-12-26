# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""LLVM dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class LLVMRequirement(VersionRequirement):
    """
    LLVM requirement for Host IR JIT compilation.

    CMake variables used:
    - LLVM_DIR: Path to LLVM CMake config
    - LLVM_VERSION: Detected version (e.g., "18.1.3")

    Minimum version: 18.1+ (defined in CMake)
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # LLVM-specific: could add custom fields here if needed
        # For now, inherits all behavior from VersionRequirement

    # Inherits format_status_line from VersionRequirement
    # Pure version checking, no custom behavior needed
