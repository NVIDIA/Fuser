# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Dependency-specific installation help modules.

Each dependency has its own module that provides platform-specific
installation instructions.
"""

from .base import HelpProvider
from .python import PythonHelp
from .torch import TorchHelp
from .cuda import CUDAToolkitHelp, TorchCUDAConstraintHelp
from .pybind11 import Pybind11Help
from .llvm import LLVMHelp
from .compiler import CompilerHelp
from .git import GitSubmodulesHelp
from .ninja import NinjaHelp
from .cmake import CMakeHelp
from .generic import GenericHelp

__all__ = [
    "HelpProvider",
    "PythonHelp",
    "TorchHelp",
    "CUDAToolkitHelp",
    "TorchCUDAConstraintHelp",
    "Pybind11Help",
    "LLVMHelp",
    "CompilerHelp",
    "GitSubmodulesHelp",
    "NinjaHelp",
    "CMakeHelp",
    "GenericHelp",
]
