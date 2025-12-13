# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Python version validation for nvFuser build.

nvFuser requires Python 3.8+ for modern type hints and language features.
This module detects the Python version and provides actionable error messages
when the version is too old.
"""

import sys
from typing import Tuple

from .exceptions import PrerequisiteMissingError
from .platform import detect_platform
from .requirements import PYTHON, format_version


def check_python_version() -> Tuple[int, int, int]:
    """
    Check that Python version meets nvFuser's minimum requirement.
    
    Returns:
        Tuple[int, int, int]: Python version as (major, minor, patch) tuple
        
    Raises:
        PrerequisiteMissingError: If Python version is below minimum
        
    Example:
        >>> version = check_python_version()
        [nvFuser] Python: 3.10.12 ✓
        >>> version
        (3, 10, 12)
    """
    version = sys.version_info
    major, minor, patch = version.major, version.minor, version.micro
    detected = (major, minor, patch)
    
    # Check minimum version requirement
    if not PYTHON.check(detected):
        platform_info = detect_platform()
        recommended = PYTHON.recommended_str
        error_msg = (
            f"ERROR: {PYTHON.name} {PYTHON.min_display} is required to build nvFuser.\n"
            f"Found: {PYTHON.name} {format_version(detected)}\n\n"
            f"nvFuser uses modern Python features including:\n"
            f"  - Type hints (PEP 484, 585, 604)\n"
            f"  - Assignment expressions (PEP 572)\n"
            f"  - Positional-only parameters (PEP 570)\n\n"
            f"{PYTHON.name} {PYTHON.min_display} is required; {PYTHON.name} {recommended} is recommended and used in the commands below.\n\n"
            f"To install {PYTHON.name} {recommended}:\n"
        )
        
        # Add platform-specific installation guidance
        if platform_info['os'] == 'Linux':
            if platform_info.get('ubuntu_based', False):
                error_msg += (
                    f"\n"
                    f"On Ubuntu or Ubuntu-based distros:\n"
                    f"  # Step 1: Install {PYTHON.name} {recommended} and venv support\n"
                    f"  sudo apt update\n"
                    f"  sudo apt install python{recommended} python3-venv python{recommended}-venv python{recommended}-dev\n"
                    f"  # Note: Some packages may not exist on all releases; install what's available\n"
                    f"\n"
                    f"  # Step 2: Create virtual environment\n"
                    f"  python{recommended} -m venv nvfuser_env\n"
                    f"  source nvfuser_env/bin/activate\n"
                    f"  python -m pip install --upgrade pip\n"
                    f"\n"
                    f"  If python{recommended}-venv is not available, install the generic python3-venv package\n"
                    f"  or follow your distribution's Python setup guide.\n"
                )
            else:
                error_msg += (
                    f"\n"
                    f"On other Linux distributions:\n"
                    f"  # Step 1: Install {PYTHON.name} {recommended}+ and development headers using your package manager\n"
                    f"  # Example (RHEL/CentOS/Fedora):\n"
                    f"  #   sudo yum install python{recommended} python{recommended}-devel\n"
                    f"\n"
                    f"  # Step 2: Create virtual environment\n"
                    f"  python{recommended} -m venv nvfuser_env\n"
                    f"  source nvfuser_env/bin/activate\n"
                    f"  python -m pip install --upgrade pip\n"
                    f"\n"
                    f"  If your distro does not package {PYTHON.name} {recommended}, consider using pyenv, Conda,\n"
                    f"  or your distro's documented method to install a newer Python.\n"
                )
        elif platform_info['os'] == 'Darwin':
            error_msg += (
                f"\n"
                f"On macOS:\n"
                f"  # Step 1: Install {PYTHON.name} {recommended} via Homebrew\n"
                f"  brew install python@{recommended}\n"
                f"\n"
                f"  # Step 2: Create virtual environment\n"
                f"  python{recommended} -m venv nvfuser_env\n"
                f"  source nvfuser_env/bin/activate\n"
                f"  python -m pip install --upgrade pip\n"
            )
        
        # Add conda as alternative (common in ML/PyTorch community)
        error_msg += (
            f"\n"
            f"Alternative - using conda/miniconda:\n"
            f"  conda create -n nvfuser python={recommended}\n"
            f"  conda activate nvfuser\n"
            f"  python -m pip install --upgrade pip\n"
        )
        
        raise PrerequisiteMissingError(error_msg)
    
    return (major, minor, patch)

