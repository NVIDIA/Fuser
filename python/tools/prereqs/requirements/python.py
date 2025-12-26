# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Python dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class PythonRequirement(VersionRequirement):
    """
    Python interpreter requirement.

    CMake variables used:
    - Python_EXECUTABLE: Path to python binary
    - Python_VERSION: Detected version (e.g., "3.12.3")

    Minimum version: 3.8+ (defined in CMake)
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # Python-specific: could add custom fields here if needed
        # For now, inherits all behavior from VersionRequirement

    # Inherits format_status_line from VersionRequirement
    # Inherits is_failure from Requirement
    # No custom behavior needed - pure version checking
