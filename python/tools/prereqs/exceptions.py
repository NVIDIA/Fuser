# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Custom exceptions for nvFuser prerequisite validation.

These exceptions provide structured error handling for build prerequisite
checks, enabling clear and actionable error messages.
"""


class PrerequisiteMissingError(Exception):
    """
    Raised when a prerequisite for building nvFuser is missing or has an incorrect version.

    This exception should include:
    - What prerequisite is missing or incorrect
    - Why it's required
    - Exact commands to install or fix it
    - Platform-specific guidance when applicable
    """

    pass
