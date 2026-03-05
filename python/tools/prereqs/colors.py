# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Color utilities for terminal output.
"""

import os


class Colors:
    """ANSI color codes for terminal output"""

    _codes = {
        "RESET": "\033[m",
        "BOLD": "\033[1m",
        # Regular colors
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "CYAN": "\033[36m",
        "WHITE": "\033[37m",
        # Bold colors
        "BOLD_RED": "\033[1;31m",
        "BOLD_GREEN": "\033[1;32m",
        "BOLD_WHITE": "\033[1;37m",
    }

    def __init__(self):
        use_colors = os.environ.get("NVFUSER_BUILD_DISABLE_COLOR") is None

        for name, code in self._codes.items():
            setattr(self, name, code if use_colors else "")


def colorize(color: str, text: str) -> str:
    """Helper to wrap text with color and reset codes.

    Args:
        color: The color code (e.g., colors.GREEN, colors.BOLD_RED)
        text: The text to colorize

    Returns:
        Text wrapped with color codes: <color>text<reset>

    Example:
        >>> colors = Colors()
        >>> print(colorize(colors.GREEN, "Success") + " - operation completed")
        # Prints "Success" in green, followed by plain text
    """
    RESET = "\033[m"
    return f"{color}{text}{RESET}"
