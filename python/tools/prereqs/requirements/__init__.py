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


def create_requirement(name: str, cmake_vars: Dict) -> Requirement:
    """
    Factory function to create appropriate requirement class.

    Extracts individual CMake variables and passes them explicitly to constructors.
    This makes it clear which CMake variables are used for each field.

    Args:
        name: Dependency name (e.g., "Python", "GCC", "Torch")
        cmake_vars: Flat dict of ALL CMake variables

    Returns:
        Appropriate Requirement subclass instance
    """
    # Explicit instantiation for each dependency
    if name == "Python":
        return PythonRequirement(
            name=name,
            found=cmake_vars.get("Python_FOUND", "FALSE"),
            status=cmake_vars.get("Python_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_Python_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("Python_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_Python_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_Python_LOCATION_VAR", ""), ""),
        )

    elif name == "Torch":
        return TorchRequirement(
            name=name,
            found=cmake_vars.get("Torch_FOUND", "FALSE"),
            status=cmake_vars.get("Torch_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_Torch_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("Torch_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_Torch_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_Torch_LOCATION_VAR", ""), ""),
            torch_cuda_constraint_status=cmake_vars.get("Torch_CUDA_constraint_status"),
            torch_cuda_constraint_version=cmake_vars.get("Torch_CUDA_constraint_version"),
            torch_cuda_constraint_found=cmake_vars.get("Torch_CUDA_constraint_found"),
            torch_cuda_constraint_required=cmake_vars.get("Torch_CUDA_constraint_required"),
        )

    elif name == "LLVM":
        return LLVMRequirement(
            name=name,
            found=cmake_vars.get("LLVM_FOUND", "FALSE"),
            status=cmake_vars.get("LLVM_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_LLVM_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("LLVM_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_LLVM_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_LLVM_LOCATION_VAR", ""), ""),
        )

    elif name == "CUDAToolkit":
        return CUDAToolkitRequirement(
            name=name,
            found=cmake_vars.get("CUDAToolkit_FOUND", "FALSE"),
            status=cmake_vars.get("CUDAToolkit_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_CUDAToolkit_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("CUDAToolkit_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_CUDAToolkit_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_CUDAToolkit_LOCATION_VAR", ""), ""),
        )

    elif name == "pybind11":
        return Pybind11Requirement(
            name=name,
            found=cmake_vars.get("pybind11_FOUND", "FALSE"),
            status=cmake_vars.get("pybind11_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_pybind11_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("pybind11_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_pybind11_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_pybind11_LOCATION_VAR", ""), ""),
        )

    elif name in ("GCC", "Clang"):
        # Compiler: name is "GCC" or "Clang", but CMake variables use "Compiler_" prefix
        return CompilerRequirement(
            name=name,
            found=cmake_vars.get("Compiler_FOUND", "FALSE"),
            status=cmake_vars.get("Compiler_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_Compiler_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get("Compiler_VERSION"),
            version_required=cmake_vars.get("NVFUSER_REQUIREMENT_Compiler_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_Compiler_LOCATION_VAR", ""), ""),
        )

    elif name == "GitSubmodules":
        return GitSubmodulesRequirement(
            name=name,
            found=cmake_vars.get("GitSubmodules_FOUND", "FALSE"),
            status=cmake_vars.get("GitSubmodules_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_GitSubmodules_OPTIONAL", "FALSE"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_GitSubmodules_LOCATION_VAR", ""), ""),
        )

    elif name == "Ninja":
        return NinjaRequirement(
            name=name,
            found=cmake_vars.get("Ninja_FOUND", "FALSE"),
            status=cmake_vars.get("Ninja_STATUS", "UNKNOWN"),
            optional=cmake_vars.get("NVFUSER_REQUIREMENT_Ninja_OPTIONAL", "FALSE"),
            location=cmake_vars.get(cmake_vars.get("NVFUSER_REQUIREMENT_Ninja_LOCATION_VAR", ""), ""),
        )

    else:
        # Unknown dependency - fallback to generic VersionRequirement
        return VersionRequirement(
            name=name,
            found=cmake_vars.get(f"{name}_FOUND", "FALSE"),
            status=cmake_vars.get(f"{name}_STATUS", "UNKNOWN"),
            optional=cmake_vars.get(f"NVFUSER_REQUIREMENT_{name}_OPTIONAL", "FALSE"),
            version_found=cmake_vars.get(f"{name}_VERSION"),
            version_required=cmake_vars.get(f"NVFUSER_REQUIREMENT_{name}_VERSION_MIN"),
            location=cmake_vars.get(cmake_vars.get(f"NVFUSER_REQUIREMENT_{name}_LOCATION_VAR", ""), ""),
        )


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
