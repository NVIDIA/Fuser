# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Git submodules installation help."""

from typing import Dict
from .base import HelpProvider


class GitSubmodulesHelp(HelpProvider):
    """Provides installation help for Git submodules."""

    def generate_help(self, failure: Dict) -> None:
        """Generate Git submodules help."""
        print("Git Submodules Not Initialized")
        print()
        print("Why: nvFuser depends on third-party libraries included as Git submodules.")
        print()
        print("Initialize and update Git submodules:")
        print()
        print("  # From the repository root:")
        print("  git submodule update --init --recursive")
        print()
        print("  # Or if you just cloned:")
        print("  git clone --recursive <repository-url>")
        print()
