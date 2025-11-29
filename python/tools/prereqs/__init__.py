# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
nvFuser Prerequisite Validation Package

This package provides utilities for validating build prerequisites before
attempting to build nvFuser from source. It helps provide clear, actionable
error messages when prerequisites are missing or have incorrect versions.

Key Components:
    - PrerequisiteMissingError: Exception raised when prerequisites are missing
    - detect_platform(): Detect OS, architecture, and Linux distribution
    - format_platform_info(): Format platform information as readable string

Usage:
    from tools.prereqs import PrerequisiteMissingError, detect_platform
    
    platform_info = detect_platform()
    if platform_info['os'] != 'Linux':
        raise PrerequisiteMissingError("nvFuser requires Linux")
"""

from .exceptions import PrerequisiteMissingError
from .platform import detect_platform, format_platform_info

__all__ = [
    "PrerequisiteMissingError",
    "detect_platform",
    "format_platform_info",
]

