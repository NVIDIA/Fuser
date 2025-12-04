# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
LLVM prerequisite validation for nvFuser build.

nvFuser requires LLVM 18.1+ to build because it links against LLVM libraries
during compilation for runtime Host IR JIT compilation. The build will fail
during CMake configuration or linking if LLVM is missing or too old.

Ubuntu 22.04 ships with LLVM 14 by default, which is too old. Users must
install LLVM 18.1+ either from prebuilt binaries (no sudo required) or from
the LLVM APT repository.

CMakeLists.txt specifies: LLVM_MINIMUM_VERSION "18.1"
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .exceptions import PrerequisiteMissingError


def _find_llvm_config() -> Optional[str]:
    """
    Locate llvm-config binary in order of priority.
    
    Priority:
    1. LLVM_CONFIG environment variable
    2. LLVM_DIR/bin/llvm-config environment variable (CMake convention)
    3. LLVM_ROOT/bin/llvm-config environment variable
    4. llvm-config on PATH
    5. System known locations
    6. Project-local locations (scanning for 18.* versions)
    
    Returns:
        Optional[str]: Path to llvm-config if found, None otherwise
        
    Example:
        >>> llvm_config = _find_llvm_config()
        >>> llvm_config
        '/home/user/nvfuser/.llvm/18.1.8/bin/llvm-config'
    """
    candidates = []
    
    # 1. Explicit LLVM_CONFIG env var
    if llvm_config_env := os.environ.get('LLVM_CONFIG'):
        candidates.append(llvm_config_env)
    
    # 2. LLVM_DIR (CMake convention)
    # CMake typically sets LLVM_DIR to lib/cmake/llvm or similar
    # Try multiple navigation patterns for robustness
    if llvm_dir := os.environ.get('LLVM_DIR'):
        llvm_dir_path = Path(llvm_dir)
        candidates.append(llvm_dir_path / '..' / '..' / '..' / 'bin' / 'llvm-config')  # lib/cmake/llvm -> root/bin
        candidates.append(llvm_dir_path / '..' / '..' / 'bin' / 'llvm-config')  # cmake/llvm -> root/bin
        candidates.append(llvm_dir_path / 'bin' / 'llvm-config')  # if LLVM_DIR points to root
    
    # 3. LLVM_ROOT (alternative convention)
    if llvm_root := os.environ.get('LLVM_ROOT'):
        candidates.append(os.path.join(llvm_root, 'bin', 'llvm-config'))
    
    # 4. PATH lookup
    if llvm_in_path := shutil.which('llvm-config'):
        candidates.append(llvm_in_path)
    
    # 5. System known locations
    system_paths = [
        '/usr/lib/llvm-18/bin/llvm-config',
        '/usr/local/llvm-18/bin/llvm-config',
        '/opt/llvm/bin/llvm-config',
    ]
    candidates.extend(system_paths)
    
    # 6. Project-local locations (wildcards for minor version variations)
    # Navigate from python/tools/prereqs to repo root (3 levels up)
    repo_root = Path(__file__).resolve().parents[3]
    project_paths = []
    
    # Check for any 18.* or 19.* version in project locations
    for parent in [repo_root / '.llvm', repo_root / 'third_party' / 'llvm']:
        if parent.exists():
            # Scan for any 18.x or newer versions
            for pattern in ['18.*', '19.*', '20.*']:
                for child in parent.glob(pattern):
                    if child.is_dir():
                        project_paths.append(child / 'bin' / 'llvm-config')
    
    candidates.extend([str(p) for p in project_paths])
    
    # Try each candidate
    for candidate in candidates:
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists() and os.access(candidate_path, os.X_OK):
                return str(candidate_path)
    
    return None


def _parse_llvm_version(version_str: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse LLVM version string into tuple.
    
    LLVM version format examples:
    - "18.1.8"
    - "18.1.8git"
    - "19.0.0"
    
    Args:
        version_str: Version string from llvm-config --version
        
    Returns:
        Optional[Tuple[int, int, int]]: (major, minor, patch) or None if parse fails
        
    Example:
        >>> _parse_llvm_version("18.1.8")
        (18, 1, 8)
        >>> _parse_llvm_version("18.1.8git")
        (18, 1, 8)
    """
    # Remove 'git' suffix if present
    version_clean = version_str.strip().replace('git', '')
    
    # Match version pattern: major.minor.patch
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_clean)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        return (major, minor, patch)
    
    return None


def check_llvm_installed() -> str:
    """
    Validate that LLVM 18.1+ is available for building nvFuser.
    
    This is the main validation function that should be called during
    nvFuser's setup process. It checks:
    1. llvm-config is available (in PATH or common locations)
    2. LLVM version >= 18.1
    
    Returns:
        str: LLVM version string (e.g., "18.1.8")
        
    Raises:
        PrerequisiteMissingError: If LLVM not found or version < 18.1
        
    Example:
        >>> version = check_llvm_installed()
        [nvFuser] LLVM: 18.1.8 ✓
        >>> version
        '18.1.8'
    """
    # Calculate repo root for error messages
    repo_root = Path(__file__).resolve().parents[3]
    
    # Find llvm-config
    llvm_config = _find_llvm_config()
    
    if not llvm_config:
        raise PrerequisiteMissingError(
            "ERROR: LLVM not found.\n\n"
            "nvFuser requires LLVM 18.1+ to build (for runtime Host IR JIT).\n"
            "llvm-config must be in PATH or at a known location.\n\n"
            "Installation options:\n\n"
            "Option 1: Download prebuilt binaries (recommended, no sudo needed, project-local):\n"
            f"  cd {repo_root}  # your nvfuser repo root\n"
            "  mkdir -p .llvm\n"
            "  cd .llvm\n"
            "  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz\n"
            "  tar -xf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz\n"
            "  mv clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04 18.1.8\n"
            "  # Then set environment variable:\n"
            "  export LLVM_CONFIG=$(pwd)/18.1.8/bin/llvm-config\n\n"
            "Option 2: Install from LLVM APT repository (requires sudo):\n"
            "  wget https://apt.llvm.org/llvm.sh\n"
            "  chmod +x llvm.sh\n"
            "  sudo ./llvm.sh 18\n"
            "  # llvm-config-18 will be installed at /usr/lib/llvm-18/bin/llvm-config\n"
            "  export LLVM_CONFIG=/usr/lib/llvm-18/bin/llvm-config\n"
        )
    
    # Get version
    try:
        result = subprocess.run(
            [llvm_config, '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        version_str = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise PrerequisiteMissingError(
            f"ERROR: Failed to get LLVM version from: {llvm_config}\n\n"
            f"Command: {llvm_config} --version\n"
            f"Error: {e.stderr}\n"
        )
    except FileNotFoundError:
        raise PrerequisiteMissingError(
            f"ERROR: llvm-config found at {llvm_config} but cannot be executed.\n\n"
            "The file may not have execute permissions or may be corrupted.\n"
        )
    
    # Parse version
    version_tuple = _parse_llvm_version(version_str)
    
    if version_tuple is None:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse LLVM version from: {version_str}\n\n"
            f"llvm-config location: {llvm_config}\n"
            "Expected version format: 18.1.8 or similar\n"
        )
    
    major, minor, patch = version_tuple
    
    # Check version requirement: >= 18.1
    if major < 18 or (major == 18 and minor < 1):
        raise PrerequisiteMissingError(
            f"ERROR: nvFuser requires LLVM 18.1+.\n"
            f"Found: LLVM {major}.{minor}.{patch} at {llvm_config}\n\n"
            f"Ubuntu 22.04 ships with LLVM 14, which is too old.\n\n"
            f"Install LLVM 18.1+ using one of these options:\n\n"
            "Option 1: Download prebuilt binaries (recommended, no sudo needed):\n"
            f"  cd {repo_root}  # your nvfuser repo root\n"
            "  mkdir -p .llvm\n"
            "  cd .llvm\n"
            "  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz\n"
            "  tar -xf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz\n"
            "  mv clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04 18.1.8\n"
            "  export LLVM_CONFIG=$(pwd)/18.1.8/bin/llvm-config\n\n"
            "Option 2: Install from LLVM APT repository (requires sudo):\n"
            "  wget https://apt.llvm.org/llvm.sh\n"
            "  chmod +x llvm.sh\n"
            "  sudo ./llvm.sh 18\n"
            "  export LLVM_CONFIG=/usr/lib/llvm-18/bin/llvm-config\n"
        )
    
    return version_str

