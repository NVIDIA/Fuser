# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Git submodules dependency requirement."""

from typing import Dict
from .base import BooleanRequirement


class GitSubmodulesRequirement(BooleanRequirement):
    """
    Git submodules initialization check.

    CMake variables used:
    - GitSubmodules_FOUND: Whether submodules are initialized
    - GitSubmodules_STATUS: Validation status

    No version checking - simple pass/fail.
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        # Simple boolean check, no additional fields needed

    # Inherits format_status_line from BooleanRequirement
    # Pure boolean validation
