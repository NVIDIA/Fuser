# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""pybind11 dependency requirement."""

from typing import Dict
from .base import VersionRequirement


class Pybind11Requirement(VersionRequirement):
    """
    pybind11 requirement for Python bindings.

    CMake variables used:
    - pybind11_DIR: Path to pybind11 CMake config
    - pybind11_VERSION: Detected version (e.g., "3.0.1")

    Minimum version: 2.0+ (defined in CMake)
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # Pybind11-specific: could add custom fields here if needed

    # Inherits format_status_line from VersionRequirement
    # Pure version checking
