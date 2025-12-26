# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ninja build system dependency requirement."""

from typing import Dict
from .base import BooleanRequirement


class NinjaRequirement(BooleanRequirement):
    """
    Ninja build system check.

    CMake variables used:
    - Ninja_FOUND: Whether Ninja is available
    - Ninja_STATUS: Validation status
    - CMAKE_MAKE_PROGRAM: Path to build tool

    No version checking - just verifies Ninja is available.
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # Simple boolean check, no additional fields needed

    # Inherits format_status_line from BooleanRequirement
    # Pure boolean validation
