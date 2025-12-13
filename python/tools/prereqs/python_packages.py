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
from .requirements import (
    PYTORCH, CUDA, PYBIND11,
    format_version, parse_version,
    pytorch_install_instructions,
    CUDA_AVAILABLE,
)


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
        Formatted installation instructions for available CUDA versions
    """
    # Use centralized pytorch_install_instructions with force_reinstall handling
    if force_reinstall:
        # Custom handling for force_reinstall since the centralized function
        # only supports upgrade flag
        from .requirements import pytorch_index_url
        lines = []
        for cuda in CUDA_AVAILABLE:
            lines.append(f"  # For CUDA {format_version(cuda)}:")
            lines.append(f"  pip install --force-reinstall torch --index-url {pytorch_index_url(cuda)}")
        return '\n'.join(lines)
    
    return pytorch_install_instructions(upgrade=upgrade)


def _get_torch_install_for_cuda_major(cuda_major: int) -> str:
    """
    Generate PyTorch install command for a specific CUDA major version.
    
    Finds the best matching CUDA version from CUDA_AVAILABLE.
    
    Args:
        cuda_major: CUDA major version (e.g., 12, 13)
        
    Returns:
        Formatted pip install command string
    """
    from .requirements import pytorch_index_url
    
    # Find matching CUDA version from available versions
    matching = [cuda for cuda in CUDA_AVAILABLE if cuda[0] == cuda_major]
    if matching:
        # Use the first matching version (they're sorted newest first)
        cuda = matching[0]
        return f"     pip install torch --index-url {pytorch_index_url(cuda)}\n"
    else:
        # Fallback: suggest checking PyTorch website
        return f"     # Check https://pytorch.org for {CUDA.name} {cuda_major} wheels\n"


def check_pybind11_installed() -> str:
    """
    Check that pybind11 is installed with CMake support.
    
    pybind11 with CMake support is required for building nvFuser's Python bindings.
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
            f"ERROR: {PYBIND11.name} is not installed.\n\n"
            f"{PYBIND11.name} {PYBIND11.min_display} is required to build nvFuser's Python bindings.\n"
            f"The [global] extra provides CMake integration.\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or install {PYBIND11.name} individually:\n"
            f"  pip install 'pybind11[global]>={PYBIND11.min_str}'"
        )
    
    # Check version
    version = pybind11.__version__
    
    # Parse version using centralized parser
    try:
        detected = parse_version(version)
    except ValueError:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse {PYBIND11.name} version: {version}\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or reinstall {PYBIND11.name} individually:\n"
            f"  pip install --force-reinstall 'pybind11[global]>={PYBIND11.min_str}'"
        )
    
    # Check minimum version requirement
    if not PYBIND11.check(detected):
        raise PrerequisiteMissingError(
            f"ERROR: {PYBIND11.name} {PYBIND11.min_display} is required to build nvFuser.\n"
            f"Found: {PYBIND11.name} {version}\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or upgrade {PYBIND11.name} individually:\n"
            f"  pip install --upgrade 'pybind11[global]>={PYBIND11.min_str}'"
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
            f"ERROR: {PYBIND11.name} is installed without CMake support.\n\n"
            f"Found: {PYBIND11.name} {version} (too old)\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or upgrade {PYBIND11.name} individually:\n"
            f"  pip install --upgrade 'pybind11[global]>={PYBIND11.min_str}'"
        )
    
    # Verify the cmake directory exists and contains config files
    import os
    if not os.path.exists(cmake_dir) or not os.path.exists(os.path.join(cmake_dir, 'pybind11Config.cmake')):
        raise PrerequisiteMissingError(
            f"ERROR: {PYBIND11.name} CMake configuration is missing or invalid.\n\n"
            f"Found: {PYBIND11.name} {version} (CMake dir: {cmake_dir})\n\n"
            f"Install all build dependencies:\n"
            f"  pip install -r requirements.txt\n\n"
            f"Or reinstall {PYBIND11.name} individually:\n"
            f"  pip install --force-reinstall 'pybind11[global]>={PYBIND11.min_str}'"
        )
    
    return version


def check_torch_installed() -> Tuple[str, str]:
    """
    Check that PyTorch with CUDA support is installed.
    
    Note: CUDA versions earlier than the minimum have known compatibility issues 
    with nvFuser (missing Float8 types). Use the minimum CUDA version or newer.
    
    nvFuser requires PyTorch compiled with CUDA support. CPU-only PyTorch
    builds are not supported. The CUDA version must match the system CUDA toolkit
    that will be used to build nvFuser.
    
    Returns:
        Tuple[str, str]: (torch_version, cuda_version_str)
        
    Raises:
        PrerequisiteMissingError: If PyTorch is not installed, version is too old,
                                 is CPU-only, or has CUDA below minimum
        
    Example:
        >>> version, cuda = check_torch_installed()
        [nvFuser] PyTorch: X.Y.Z with CUDA X.Y ✓
        >>> version, cuda
        ('X.Y.Z', 'X.Y')  # Actual versions detected at runtime
    """
    # Check if PyTorch is installed
    try:
        import torch
    except ImportError:
        raise PrerequisiteMissingError(
            f"ERROR: {PYTORCH.name} is not installed.\n\n"
            f"nvFuser requires {PYTORCH.name} {PYTORCH.min_display} with {CUDA.name} {CUDA.min_display} support.\n"
            f"The {CUDA.name} version must match your system CUDA toolkit.\n"
            f"Check your system {CUDA.name} version: nvcc --version\n\n"
            f"Install {PYTORCH.name} with {CUDA.name} support:\n"
            f"{_get_torch_install_instructions()}\n\n"
            f"Visit https://pytorch.org for more installation options."
        )
    
    # Get PyTorch version (remove any +cu130 suffix)
    torch_version = torch.__version__.split('+')[0]
    
    # Parse version using centralized parser
    try:
        torch_detected = parse_version(torch_version)
    except ValueError:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse {PYTORCH.name} version: {torch.__version__}\n\n"
            f"Please reinstall {PYTORCH.name}:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Check minimum version requirement
    if not PYTORCH.check(torch_detected):
        raise PrerequisiteMissingError(
            f"ERROR: {PYTORCH.name} {PYTORCH.min_display} is required to build nvFuser.\n"
            f"Found: {PYTORCH.name} {torch_version}\n\n"
            f"Upgrade {PYTORCH.name} (match your system {CUDA.name} version):\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Check if PyTorch has CUDA support (not CPU-only)
    cuda_version_str = torch.version.cuda
    if cuda_version_str is None:
        raise PrerequisiteMissingError(
            f"ERROR: {PYTORCH.name} is CPU-only. nvFuser requires {CUDA.name}-enabled {PYTORCH.name}.\n\n"
            f"You have installed {PYTORCH.name} without {CUDA.name} support. This is a common mistake.\n"
            f"nvFuser needs {PYTORCH.name} compiled with {CUDA.name} {CUDA.min_display} to build and run correctly.\n"
            f"The {CUDA.name} version must match your system CUDA toolkit.\n"
            f"Check your system {CUDA.name} version: nvcc --version\n\n"
            f"Install {PYTORCH.name} with {CUDA.name} support:\n"
            f"{_get_torch_install_instructions()}"
        )
    
    # Parse CUDA version using centralized parser
    try:
        cuda_detected = parse_version(cuda_version_str)
    except ValueError:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse {CUDA.name} version from {PYTORCH.name}: {cuda_version_str}\n\n"
            f"Please reinstall {PYTORCH.name} with {CUDA.name} {CUDA.min_display}:\n"
            f"{_get_torch_install_instructions(force_reinstall=True)}"
        )
    
    # Check CUDA version requirement
    # CUDA versions earlier than minimum have known issues (missing Float8 types)
    if not CUDA.check(cuda_detected):
        raise PrerequisiteMissingError(
            f"ERROR: {PYTORCH.name} with {CUDA.name} {CUDA.min_display} is required to build nvFuser.\n"
            f"Found: {PYTORCH.name} {torch_version} with {CUDA.name} {cuda_version_str}\n\n"
            f"{CUDA.name} versions earlier than {CUDA.min_str} have known compatibility issues with nvFuser\n"
            f"(missing Float8 types cause build errors).\n\n"
            f"Please upgrade {PYTORCH.name} to {CUDA.name} {CUDA.min_display}:\n"
            f"{_get_torch_install_instructions(upgrade=True)}"
        )
    
    # Detect and validate system CUDA toolkit
    system_cuda = _detect_system_cuda()
    cuda_major = cuda_detected[0]
    cuda_minor = cuda_detected[1] if len(cuda_detected) > 1 else 0
    
    if system_cuda is None:
        # System CUDA not found - this is a problem
        raise PrerequisiteMissingError(
            f"ERROR: System {CUDA.name} toolkit not found.\n\n"
            f"{PYTORCH.name} has {CUDA.name} {cuda_version_str} support, but nvcc is not in PATH.\n"
            f"nvFuser needs the {CUDA.name} toolkit (nvcc compiler) to build.\n\n"
            f"Install {CUDA.name} toolkit {cuda_major}.x (major version must match {PYTORCH.name}):\n"
            f"  # Check available versions:\n"
            f"  # https://developer.nvidia.com/cuda-downloads\n"
            f"  # For Ubuntu 22.04 with {CUDA.name} {cuda_major}.x:\n"
            f"  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb\n"
            f"  sudo dpkg -i cuda-keyring_1.0-1_all.deb\n"
            f"  sudo apt update\n"
            f"  sudo apt install cuda-toolkit-{cuda_major}-8  # Match {PYTORCH.name} {CUDA.name} major version\n"
        )
    
    # Parse system CUDA version
    try:
        system_cuda_detected = parse_version(system_cuda)
        system_cuda_major = system_cuda_detected[0]
        system_cuda_minor = system_cuda_detected[1] if len(system_cuda_detected) > 1 else 0
    except ValueError:
        raise PrerequisiteMissingError(
            f"ERROR: Could not parse {CUDA.name} versions.\n"
            f"{PYTORCH.name} {CUDA.name}: {cuda_version_str}\n"
            f"System {CUDA.name}: {system_cuda}\n\n"
            f"Please verify your {CUDA.name} installations are correct."
        )
    
    # Validate major version match (REQUIRED)
    if cuda_major != system_cuda_major:
        raise PrerequisiteMissingError(
            f"ERROR: {CUDA.name} version mismatch between {PYTORCH.name} and system.\n\n"
            f"{PYTORCH.name} {CUDA.name}: {cuda_version_str} (major: {cuda_major})\n"
            f"System {CUDA.name}: {system_cuda} (major: {system_cuda_major})\n\n"
            f"nvFuser requires the {CUDA.name} major versions to match.\n"
            f"Code compiled with {CUDA.name} {system_cuda_major} cannot link with {PYTORCH.name} built for {CUDA.name} {cuda_major}.\n\n"
            f"Solutions:\n"
            f"  1. Install {PYTORCH.name} matching your system {CUDA.name} {system_cuda_major}:\n"
            f"{_get_torch_install_for_cuda_major(system_cuda_major)}"
            f"  OR\n"
            f"  2. Install system {CUDA.name} toolkit matching {PYTORCH.name} {CUDA.name} {cuda_major}:\n"
            f"     See: https://developer.nvidia.com/cuda-downloads\n"
        )
    
    # Check minor version (WARNING only, not error)
    if cuda_minor != system_cuda_minor:
        print(f"[nvFuser] WARNING: {CUDA.name} minor version mismatch")
        print(f"  {PYTORCH.name} {CUDA.name}: {cuda_version_str}")
        print(f"  System {CUDA.name}: {system_cuda}")
        print(f"  Major versions match ({cuda_major}), but minor versions differ.")
        print(f"  Build should work, but consider matching minor versions for best compatibility.")
    
    return torch_version, cuda_version_str

