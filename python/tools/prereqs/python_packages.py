# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Python package validation for nvFuser build system.

Validates that required Python packages (pybind11, PyTorch) are installed with
correct versions and features. This module handles pybind11 and PyTorch validation.
"""

import re
import shutil
import subprocess
from typing import Optional, Tuple

from .exceptions import PrerequisiteMissingError


def _detect_system_cuda() -> Optional[str]:
    """
    Detect system CUDA toolkit version via nvcc.
    
    Returns:
        Optional[str]: CUDA version string (e.g., "12.5", "13.0"), or None if not found
        
    Example:
        >>> version = _detect_system_cuda()
        >>> version
        '12.5'
    """
    # Check if nvcc exists
    if not shutil.which('nvcc'):
        return None
    
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse version from output
        # Example: "Cuda compilation tools, release 12.5, V12.5.40"
        # or: "release 13.0, V13.0.76"
        for line in result.stdout.splitlines():
            match = re.search(r'release (\d+\.\d+)', line.lower())
            if match:
                return match.group(1)
        
        return None
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_torch_install_instructions(upgrade: bool = False, force_reinstall: bool = False) -> str:
    """
    Generate PyTorch installation instructions with appropriate pip flags.
    
    Args:
        upgrade: If True, adds --upgrade flag
        force_reinstall: If True, adds --force-reinstall flag
        
    Returns:
        Formatted installation instructions for CUDA 13.0 and 12.8
    """
    cmd = "pip install"
    if force_reinstall:
        cmd += " --force-reinstall"
    elif upgrade:
        cmd += " --upgrade"
    
    return (
        f"  # For CUDA 13.0:\n"
        f"  {cmd} torch --index-url https://download.pytorch.org/whl/cu130\n"
        f"  # For CUDA 12.8:\n"
        f"  {cmd} torch --index-url https://download.pytorch.org/whl/cu128"
    )


def check_pybind11_installed() -> str:
    """
    Check that pybind11 is installed with CMake support (2.0+).
    
    pybind11 2.0+ with CMake support is required for building nvFuser's Python bindings.
    The [global] extra is recommended as it provides CMake integration files.
    
    In pybind11 3.0+, CMake configuration files are included in the base package,
    so [global] is optional but still recommended.
    
    Returns:
        str: pybind11 version string
        
    Raises:
        PrerequisiteMissingError: If pybind11 is not installed, version is too old,
                                 or CMake support is missing
        
    Example:
        >>> version = check_pybind11_installed()
        [nvFuser] pybind11: 2.13.6 with CMake support ✓
        >>> version
        '2.13.6'
    """
    # Check if pybind11 is installed
    try:
        import pybind11
    except ImportError:
        raise PrerequisiteMissingError(
            "ERROR: pybind11 is not installed.\n\n"
            "pybind11 2.0+ is required to build nvFuser's Python bindings.\n"
            "The [global] extra provides CMake integration.\n\n"
            "Install all build dependencies:\n"
            "  pip install -r requirements.txt\n\n"
            "Or install pybind11 individually:\n"
            "  pip install 'pybind11[global]>=2.0'"
        )
    
    # Check version
    version = pybind11.__version__
    
    # Parse major version
    try:
        major = int(version.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse pybind11 version: {version}\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or reinstall pybind11 individually:\n"
            f"  pip install --force-reinstall 'pybind11[global]>=2.0'"
        )
    
    # Check minimum version requirement (2.0+)
    if major < 2:
        raise PrerequisiteMissingError(
            f"ERROR: pybind11 2.0+ is required to build nvFuser.\n"
            f"Found: pybind11 {version}\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or upgrade pybind11 individually:\n"
            f"  pip install --upgrade 'pybind11[global]>=2.0'"
        )
    
    # Check for CMake support
    # In pybind11 2.x and earlier, CMake support may not be available
    # In pybind11 3.0+, CMake files are included in the base package
    # The [global] extra is recommended but optional in 3.0+
    
    # Check if get_cmake_dir() exists and returns a valid path
    try:
        cmake_dir = pybind11.get_cmake_dir()
    except AttributeError:
        # Very old pybind11 without CMake support at all
        raise PrerequisiteMissingError(
            "ERROR: pybind11 is installed without CMake support.\n\n"
            f"Found: pybind11 {version} (too old)\n\n"
            "Install all build dependencies:\n"
            "  pip install -r requirements.txt\n\n"
            "Or upgrade pybind11 individually:\n"
            "  pip install --upgrade 'pybind11[global]>=2.0'"
        )
    
    # Verify the cmake directory exists and contains config files
    import os
    if not os.path.exists(cmake_dir) or not os.path.exists(os.path.join(cmake_dir, 'pybind11Config.cmake')):
        raise PrerequisiteMissingError(
            "ERROR: pybind11 CMake configuration is missing or invalid.\n\n"
            f"Found: pybind11 {version} (CMake dir: {cmake_dir})\n\n"
            "Install all build dependencies:\n"
            "  pip install -r requirements.txt\n\n"
            "Or reinstall pybind11 individually:\n"
            "  pip install --force-reinstall 'pybind11[global]>=2.0'"
        )
    
    return version


def check_torch_installed() -> Tuple[str, str]:
    """
    Check that PyTorch 2.0+ with CUDA 12.8+ support is installed.
    
    Note: CUDA 12.6 and earlier have known compatibility issues with nvFuser
    (missing Float8_e8m0fnu type). Use CUDA 12.8 or 13.0+.
    
    nvFuser requires PyTorch 2.0+ compiled with CUDA 12.8+ support. CPU-only PyTorch
    builds are not supported. The CUDA version must match the system CUDA toolkit
    that will be used to build nvFuser.
    
    Returns:
        Tuple[str, str]: (torch_version, cuda_version_str)
        
    Raises:
        PrerequisiteMissingError: If PyTorch is not installed, version is too old,
                                 is CPU-only, or has CUDA < 12
        
    Example:
        >>> version, cuda = check_torch_installed()
        [nvFuser] PyTorch: 2.5.1 with CUDA 12.1 ✓
        >>> version, cuda
        ('2.5.1', '12.1')
    """
    # Check if PyTorch is installed
    try:
        import torch
    except ImportError:
        raise PrerequisiteMissingError(
            "ERROR: PyTorch is not installed.\n\n"
            "nvFuser requires PyTorch 2.0+ with CUDA 12.8+ support.\n"
            "The CUDA version must match your system CUDA toolkit.\n"
            "Check your system CUDA version: nvcc --version\n\n"
            "Install PyTorch with CUDA support:\n"
            f"{_get_torch_install_instructions()}\n\n"
            "Visit https://pytorch.org for more installation options."
        )
    
    # Get PyTorch version (remove any +cu130 suffix)
    torch_version = torch.__version__.split('+')[0]
    
    # Parse major version
    try:
        major = int(torch_version.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse PyTorch version: {torch.__version__}\n\n"
            f"Please reinstall PyTorch:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Check minimum version requirement (2.0+)
    if major < 2:
        raise PrerequisiteMissingError(
            f"ERROR: PyTorch 2.0+ is required to build nvFuser.\n"
            f"Found: PyTorch {torch_version}\n\n"
            f"Upgrade PyTorch (match your system CUDA version):\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Check if PyTorch has CUDA support (not CPU-only)
    cuda_version_str = torch.version.cuda
    if cuda_version_str is None:
        raise PrerequisiteMissingError(
            "ERROR: PyTorch is CPU-only. nvFuser requires CUDA-enabled PyTorch.\n\n"
            "You have installed PyTorch without CUDA support. This is a common mistake.\n"
            "nvFuser needs PyTorch compiled with CUDA 12.8+ to build and run correctly.\n"
            "The CUDA version must match your system CUDA toolkit.\n"
            "Check your system CUDA version: nvcc --version\n\n"
            "Install PyTorch with CUDA support:\n"
            f"{_get_torch_install_instructions()}"
        )
    
    # Parse CUDA version and validate >= 12
    try:
        cuda_major = int(cuda_version_str.split('.')[0])
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse CUDA version from PyTorch: {cuda_version_str}\n\n"
            f"Please reinstall PyTorch with CUDA 12.8+:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Parse CUDA minor version for 12.8+ check
    try:
        cuda_minor = int(cuda_version_str.split('.')[1]) if '.' in cuda_version_str else 0
    except (ValueError, IndexError):
        cuda_minor = 0
    
    # Check CUDA version requirement (12.8+)
    # CUDA 12.6 and earlier have known issues (missing Float8_e8m0fnu type)
    if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 8):
        raise PrerequisiteMissingError(
            f"ERROR: PyTorch with CUDA 12.8+ is required to build nvFuser.\n"
            f"Found: PyTorch {torch_version} with CUDA {cuda_version_str}\n\n"
            f"CUDA 12.6 and earlier have known compatibility issues with nvFuser\n"
            f"(missing Float8 types cause build errors).\n\n"
            f"Please upgrade PyTorch to CUDA 12.8 or 13.0:\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Detect and validate system CUDA toolkit
    system_cuda = _detect_system_cuda()
    
    if system_cuda is None:
        # System CUDA not found - this is a problem
        raise PrerequisiteMissingError(
            f"ERROR: System CUDA toolkit not found.\n\n"
            f"PyTorch has CUDA {cuda_version_str} support, but nvcc is not in PATH.\n"
            f"nvFuser needs the CUDA toolkit (nvcc compiler) to build.\n\n"
            f"Install CUDA toolkit {cuda_major}.x (major version must match PyTorch):\n"
            f"  # Check available versions:\n"
            f"  # https://developer.nvidia.com/cuda-downloads\n"
            f"  # For Ubuntu 22.04 with CUDA {cuda_major}.x:\n"
            f"  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb\n"
            f"  sudo dpkg -i cuda-keyring_1.0-1_all.deb\n"
            f"  sudo apt update\n"
            f"  sudo apt install cuda-toolkit-{cuda_major}-8  # Match PyTorch CUDA major version\n"
        )
    
    # Parse system CUDA version
    try:
        torch_cuda_major = int(cuda_version_str.split('.')[0])
        torch_cuda_minor = int(cuda_version_str.split('.')[1]) if '.' in cuda_version_str else 0
        
        system_cuda_major = int(system_cuda.split('.')[0])
        system_cuda_minor = int(system_cuda.split('.')[1]) if '.' in system_cuda else 0
    except (ValueError, IndexError):
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse CUDA versions.\n"
            f"PyTorch CUDA: {cuda_version_str}\n"
            f"System CUDA: {system_cuda}\n\n"
            f"Please verify your CUDA installations are correct."
        )
    
    # Validate major version match (REQUIRED)
    if torch_cuda_major != system_cuda_major:
        raise PrerequisiteMissingError(
            f"ERROR: CUDA version mismatch between PyTorch and system.\n\n"
            f"PyTorch CUDA: {cuda_version_str} (major: {torch_cuda_major})\n"
            f"System CUDA: {system_cuda} (major: {system_cuda_major})\n\n"
            f"nvFuser requires the CUDA major versions to match.\n"
            f"Code compiled with CUDA {system_cuda_major} cannot link with PyTorch built for CUDA {torch_cuda_major}.\n\n"
            f"Solutions:\n"
            f"  1. Install PyTorch matching your system CUDA {system_cuda_major}:\n"
            f"     pip install torch --index-url https://download.pytorch.org/whl/cu{system_cuda_major}{'8' if system_cuda_major == 12 else '0'}\n"
            f"  OR\n"
            f"  2. Install system CUDA toolkit matching PyTorch CUDA {torch_cuda_major}:\n"
            f"     See: https://developer.nvidia.com/cuda-downloads\n"
        )
    
    # Check minor version (WARNING only, not error)
    if torch_cuda_minor != system_cuda_minor:
        print(f"[nvFuser] WARNING: CUDA minor version mismatch")
        print(f"  PyTorch CUDA: {cuda_version_str}")
        print(f"  System CUDA: {system_cuda}")
        print(f"  Major versions match ({torch_cuda_major}), but minor versions differ.")
        print(f"  Build should work, but consider matching minor versions for best compatibility.")
    
    return torch_version, cuda_version_str

