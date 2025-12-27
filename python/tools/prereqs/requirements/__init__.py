# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Requirement class registry and factory."""

from .base import Requirement, VersionRequirement, BooleanRequirement, RequirementStatus
from .python import PythonRequirement
from .torch import TorchRequirement
from .llvm import LLVMRequirement
from .cuda_toolkit import CUDAToolkitRequirement
from .pybind11 import Pybind11Requirement
from .compiler import CompilerRequirement
from .git_submodules import GitSubmodulesRequirement
from .ninja import NinjaRequirement

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
]
