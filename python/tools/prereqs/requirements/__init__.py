# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Requirement class registry and factory."""

from typing import Dict
from .base import Requirement, VersionRequirement, BooleanRequirement, RequirementStatus
from .python import PythonRequirement
from .torch import TorchRequirement
from .llvm import LLVMRequirement
from .cuda_toolkit import CUDAToolkitRequirement
from .pybind11 import Pybind11Requirement
from .compiler import CompilerRequirement
from .git_submodules import GitSubmodulesRequirement
from .ninja import NinjaRequirement

# Registry mapping dependency names to their requirement classes
REQUIREMENT_REGISTRY = {
    "Python": PythonRequirement,
    "Torch": TorchRequirement,
    "LLVM": LLVMRequirement,
    "CUDAToolkit": CUDAToolkitRequirement,
    "pybind11": Pybind11Requirement,
    "GCC": CompilerRequirement,
    "Clang": CompilerRequirement,
    "Compiler": CompilerRequirement,
    "GitSubmodules": GitSubmodulesRequirement,
    "Ninja": NinjaRequirement,
}


def create_requirement(data: Dict) -> Requirement:
    """
    Factory function to create appropriate requirement class from JSON data.

    Uses explicit registry mapping dependency name to requirement class.
    Falls back to generic classes if specific class not found.

    Args:
        data: Dictionary from nvfuser_dependencies.json with keys:
            - name: Dependency name (used for registry lookup)
            - cmake_vars: Dict of all CMake variables
            - metadata: Dict of requirement metadata

    Returns:
        Appropriate Requirement subclass instance
    """
    name = data.get("name", "Unknown")

    # First, try explicit registry lookup
    if name in REQUIREMENT_REGISTRY:
        return REQUIREMENT_REGISTRY[name](data)

    # Fallback: default to version requirement for unknown dependencies
    return VersionRequirement(data)


__all__ = [
    # Base classes
    "Requirement",
    "VersionRequirement",
    "BooleanRequirement",
    "RequirementStatus",
    # Specific requirement classes
    "PythonRequirement",
    "TorchRequirement",
    "LLVMRequirement",
    "CUDAToolkitRequirement",
    "Pybind11Requirement",
    "CompilerRequirement",
    "GitSubmodulesRequirement",
    "NinjaRequirement",
    # Factory
    "create_requirement",
    # Registry (for advanced use)
    "REQUIREMENT_REGISTRY",
]
