# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
nvFuser prerequisite utilities package.

IMPORTANT DESIGN PRINCIPLE:
===========================
CMake is the source of truth for ALL dependency requirements and validation.

- CMake defines version requirements in: cmake/DependencyRequirements.cmake
- CMake finds dependencies and validates versions
- CMake exports all data to JSON: build/nvfuser_dependencies.json
- Python reads JSON and formats output with helpful instructions

This package provides ONLY:
- Platform detection utilities
- Version parsing/formatting utilities
- URL generators for downloads
- Utilities for formatting help text

This package does NOT:
- Define version requirements (CMake does)
- Find or validate dependencies (CMake does)
- Determine build success/failure (CMake does)

Usage:
    from prereqs import detect_platform, pytorch_index_url, llvm_download_url

    platform_info = detect_platform()
    if platform_info["ubuntu_based"]:
        print("Use apt for installation")

    url = pytorch_index_url((13, 1))
    print(f"Install PyTorch: pip install torch --index-url {url}")
"""

# Platform detection
from .platform import detect_platform, format_platform_info

# Version utilities
from .requirements import (
    Requirement,
    parse_version,
    format_version,
    CUDA_AVAILABLE,
)

# URL generators
from .requirements import (
    pytorch_index_url,
    pytorch_install_instructions,
    llvm_download_url,
    cuda_toolkit_download_url,
)

# Exception (included but not used in reporting)
from .exceptions import PrerequisiteMissingError

# Requirement classes (OOP abstraction)
from .requirements_classes import (
    Requirement as BaseRequirement,
    VersionRequirement,
    BooleanRequirement,
    ConstraintRequirement,
    RequirementStatus,
    create_requirement,
)

__all__ = [
    # Platform
    "detect_platform",
    "format_platform_info",
    # Requirements (legacy utility class)
    "Requirement",
    "parse_version",
    "format_version",
    "CUDA_AVAILABLE",
    # URL generators
    "pytorch_index_url",
    "pytorch_install_instructions",
    "llvm_download_url",
    "cuda_toolkit_download_url",
    # Exception
    "PrerequisiteMissingError",
    # Requirement classes (OOP)
    "BaseRequirement",
    "VersionRequirement",
    "BooleanRequirement",
    "ConstraintRequirement",
    "RequirementStatus",
    "create_requirement",
    # Note: TorchRequirement is in prereqs.help.torch to keep torch-specific code modular
]
