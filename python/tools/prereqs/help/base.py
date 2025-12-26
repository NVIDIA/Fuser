# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Base class for dependency help providers.
"""

from typing import Dict, Optional


class HelpProvider:
    """Base class for providing installation help for dependencies."""

    def __init__(self, platform_info: Dict[str, Optional[str]]):
        """
        Initialize help provider with platform information.

        Args:
            platform_info: Platform detection info with keys:
                - os: Operating system (Linux, Darwin, Windows)
                - arch: Architecture (x86_64, aarch64, arm64)
                - distro: Linux distribution (ubuntu, debian, rhel, etc.)
                - distro_version: Distribution version
                - ubuntu_based: Boolean indicating Ubuntu-based distro
        """
        self.platform_info = platform_info

    def generate_help(self, failure: Dict) -> None:
        """
        Generate and print installation help for a failed dependency.

        Args:
            failure: Dictionary containing:
                - name: Dependency name
                - status: Status (NOT_FOUND, INCOMPATIBLE)
                - version_found: Found version (if any)
                - version_required: Required version (if any)
                - optional: Whether dependency is optional
        """
        raise NotImplementedError("Subclasses must implement generate_help()")
