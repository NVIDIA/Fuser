# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Git submodules validation for nvFuser build.

nvFuser has third-party dependencies managed as git submodules (cutlass, 
flatbuffers, googletest, benchmark). Uninitialized submodules cause cryptic 
"file not found" CMake errors during configuration.

This module detects uninitialized git submodules and provides actionable 
instructions to initialize them.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .exceptions import PrerequisiteMissingError


def _find_repo_root() -> Optional[Path]:
    """
    Find git repository root using git's own detection.

    This uses 'git rev-parse --show-toplevel' which is more reliable than
    walking directories looking for .git (handles git worktrees correctly).

    Returns:
        Optional[Path]: Repository root path, or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repository, or git not installed
        return None


def check_git_submodules_initialized() -> List[Tuple[str, str]]:
    """
    Check that all git submodules in the repository are initialized.

    Returns:
        List[Tuple[str, str]]: List of (submodule_path, commit_hash) for
                               initialized submodules

    Raises:
        PrerequisiteMissingError: If any submodules are uninitialized

    Example:
        >>> submodules = check_git_submodules_initialized()
        [nvFuser] Git submodules: 4 initialized ✓
        >>> submodules
        [('third_party/benchmark', '0d98dba29d66...'), ...]
    """
    # Find the git repository root
    repo_root = _find_repo_root()

    if repo_root is None:
        # Not in a git repository - this is acceptable (e.g., pip installed package)
        # Don't raise error, just skip the check silently
        return []

    try:
        # Run git submodule status from repository root
        result = subprocess.run(
            ["git", "submodule", "status"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # git command failed - probably not a git repo or git not installed
            # This is acceptable in some scenarios (pip install from tarball)
            return []

        # Parse submodule status output
        # Format: " <commit> <path> (<description>)" for initialized
        #         "-<commit> <path> (<description>)" for uninitialized
        lines = result.stdout.strip().splitlines()

        if not lines:
            # No submodules defined
            return []

        initialized = []
        uninitialized = []

        for line in lines:
            if not line:
                continue

            # Check first character
            status_char = line[0]
            # Parse rest of line: <commit> <path> ...
            parts = line[1:].split(maxsplit=2)
            if len(parts) >= 2:
                commit, path = parts[0], parts[1]

                if status_char == "-":
                    uninitialized.append(path)
                else:
                    initialized.append((path, commit))

        # If any uninitialized, raise detailed error
        if uninitialized:
            error_msg = (
                "ERROR: Git submodules are not initialized.\n\n"
                "nvFuser requires the following third-party dependencies as git submodules:\n"
            )
            for path in uninitialized:
                error_msg += f"  - {path}\n"

            error_msg += (
                f"\nUninitialized submodules cause CMake configuration errors like:\n"
                f"  'CMake Error: Cannot find source file: third_party/.../file.h'\n\n"
                f"To initialize all submodules:\n"
                f"  cd {repo_root}\n"
                f"  git submodule update --init --recursive\n\n"
                f"This will download and initialize all required dependencies.\n"
            )

            raise PrerequisiteMissingError(error_msg)

        return initialized

    except FileNotFoundError:
        # git command not found - acceptable in some scenarios
        return []
