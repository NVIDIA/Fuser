#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive nvFuser Environment Configuration Tool

This tool provides an interactive interface (similar to ccmake) for configuring
nvFuser environment variables. It helps users set up debug flags, feature toggles,
and runtime options without needing to remember complex NVFUSER_* variable names.

Usage:
    python tools/configure_env.py                    # Interactive TUI mode
    python tools/configure_env.py --simple           # Simple prompt mode
    python tools/configure_env.py --generate-script  # Generate shell script
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal


# Category metadata - display names in desired order
# Dict maintains insertion order in Python 3.7+, so order here = display order
CATEGORY_NAMES = {
    "environment": "Environment & Compiler Settings (CC, CXX, CUDA_HOME, etc.)",
    "build": "Build Configuration (NVFUSER_BUILD_*)",
    "build_advanced": "Advanced Build Options (NVFUSER_BUILD_*)",
    "dump": "Debug/Diagnostic Options (NVFUSER_DUMP)",
    "enable": "Feature Enable Options (NVFUSER_ENABLE)",
    "disable": "Feature Disable Options (NVFUSER_DISABLE)",
    "profiler": "Profiler Options (NVFUSER_PROF)",
    "compilation": "Runtime Compilation Control",
    "misc": "Miscellaneous Options",
}


@dataclass
class EnvVarOption:
    """Represents a single environment variable option."""

    name: str  # The actual environment variable name (e.g., "CC", "NVFUSER_BUILD_NO_PYTHON", or value for lists)
    description: str
    var_type: Literal["bool", "string", "int", "multi"]
    category: str
    env_var: str | None = (
        None  # For list items (dump/enable/disable), this is the parent env var
    )
    default: str = ""
    choices: list[str] = field(default_factory=list)
    current_value: str | None = None

    def get_display_name(self) -> str:
        """Get the display name for this option."""
        return self.name

    def get_env_var_name(self) -> str:
        """Get the actual environment variable name (for list items, return the parent)."""
        if self.env_var is not None:
            return self.env_var
        return self.name


# Define all nvFuser environment variables organized by category
ENV_VAR_DEFINITIONS = [
    # ========================================================================
    # ENVIRONMENT AND COMPILER CONFIGURATION
    # ========================================================================
    EnvVarOption("CC", "C compiler to use", "string", "environment"),
    EnvVarOption("CXX", "C++ compiler to use", "string", "environment"),
    EnvVarOption("CUDA_HOME", "CUDA installation directory", "string", "environment"),
    EnvVarOption(
        "TORCH_CUDA_ARCH_LIST",
        "Target CUDA architectures (e.g., '8.0;9.0')",
        "string",
        "environment",
    ),
    EnvVarOption("CFLAGS", "Additional C compiler flags", "string", "environment"),
    EnvVarOption("CXXFLAGS", "Additional C++ compiler flags", "string", "environment"),
    EnvVarOption("LDFLAGS", "Additional linker flags", "string", "environment"),
    EnvVarOption(
        "NVFUSER_SOURCE_DIR", "nvFuser source directory", "string", "environment"
    ),
    # ========================================================================
    # BUILD-TIME OPTIONS (NVFUSER_BUILD_*)
    # ========================================================================
    # Build Configuration
    EnvVarOption(
        "NVFUSER_BUILD_BUILD_TYPE",
        "Build type:",
        "multi",
        "build",
        choices=["Release", "Debug", "RelWithDebInfo"],
    ),
    EnvVarOption(
        "NVFUSER_BUILD_CPP_STANDARD",
        "C++ standard version",
        "int",
        "build",
        default="20",
    ),
    EnvVarOption("NVFUSER_BUILD_NO_PYTHON", "Skip Python bindings", "bool", "build"),
    EnvVarOption(
        "NVFUSER_BUILD_NO_CUTLASS", "Skip building CUTLASS kernels", "bool", "build"
    ),
    EnvVarOption("NVFUSER_BUILD_NO_TEST", "Skip C++ tests", "bool", "build"),
    EnvVarOption("NVFUSER_BUILD_NO_BENCHMARK", "Skip benchmarks", "bool", "build"),
    EnvVarOption(
        "NVFUSER_BUILD_NO_NINJA", "Use make instead of ninja", "bool", "build"
    ),
    EnvVarOption(
        "NVFUSER_BUILD_ENABLE_PCH", "Enable precompiled headers", "bool", "build"
    ),
    # Advanced Build Options
    EnvVarOption(
        "NVFUSER_BUILD_WITH_UCC",
        "Build with UCC support for multi-device",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_WITH_ASAN",
        "Build with Address Sanitizer",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_WITHOUT_DISTRIBUTED",
        "Build without multidevice support",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK",
        "Enable explicit error checking",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_CUTLASS_MAX_JOBS",
        "Max parallel jobs for CUTLASS build",
        "int",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_DIR", "Custom build directory", "string", "build_advanced"
    ),
    EnvVarOption(
        "NVFUSER_BUILD_INSTALL_DIR",
        "Custom install directory",
        "string",
        "build_advanced",
    ),
    EnvVarOption("MAX_JOBS", "Max parallel compilation jobs", "int", "build_advanced"),
    EnvVarOption(
        "NVFUSER_BUILD_VERSION_TAG",
        "Tag for build nvfuser version",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_WHEEL_NAME",
        "Wheel name for pip package",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_INSTALL_REQUIRES",
        "Package dependencies for install (comma-separated)",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_EXTRAS_REQUIRE",
        "Extra requirements (Python dict string)",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_NVMMH_INCLUDE_DIR",
        "Location to find nvMatmulHeuristics.h",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_OVERWRITE_VERSION",
        "Overwrite version",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_CMAKE_ONLY",
        "Only generate build directory without building",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "NVFUSER_BUILD_SETUP",
        "Run build setup",
        "bool",
        "build_advanced",
    ),
    # ========================================================================
    # RUNTIME OPTIONS
    # ========================================================================
    # Debug and Diagnostic Options (NVFUSER_DUMP)
    EnvVarOption(
        "fusion_ir_original",
        "Dump the original fusion IR built by the Python API",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_ir",
        "Dump the Fusion IR before lowering (fed to KernelExecutor::compile)",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "kernel_ir",
        "Dump the compiler Kernel IR",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "cuda_kernel",
        "Dump the generated CUDA C++ kernel code",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "cuda_full",
        "Dump the complete CUDA C++ code",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption("ptx", "Dump compiled PTX", "bool", "dump", env_var="NVFUSER_DUMP"),
    EnvVarOption(
        "sass", "Dump disassembled SASS", "bool", "dump", env_var="NVFUSER_DUMP"
    ),
    EnvVarOption(
        "sass_to_file",
        "Dump disassembled SASS to file",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "launch_param",
        "Dump the launch parameters of kernel",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "kernel_args",
        "Print the runtime kernel arguments when launching kernels",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "scheduler_params",
        "Dump scheduler heuristic parameters",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "scheduler_verbose",
        "Dump detailed scheduler logging",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    # Additional NVFUSER_DUMP options
    EnvVarOption(
        "fusion_ir_concretized",
        "Dump the Fusion IR after concretization",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_ir_preseg",
        "Dump the Fusion IR after pre-segmenter optimization and before segmentation",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_ir_presched",
        "Dump the segmented Fusion IR before scheduling",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_ir_graph",
        "Dump a GraphViz graph of the Fusion IR",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_ir_math",
        "Dump just the compute (math) part of the Fusion IR for conciseness",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "cuda_to_file",
        "Dump CUDA strings to file",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "cubin",
        "Dump compiled CUBIN",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "fusion_args",
        "Print the runtime fusion arguments",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "global_zeroed_memory",
        "Print the log for zeroed global memory allocator",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "host_ir",
        "Dump the Host IR program",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "host_ir_lowering",
        "Dump the Host IR after each lowering pass",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "host_ir_jit",
        "Dump the LLVM IR lowered from Host IR",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "inlining",
        "Verbose information about tensor inlining",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "segmented_fusion",
        "Dump segmented fusion graph",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "draw_segmented_fusion",
        "Dump segmented fusion graph drawing",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "segmenter_logging",
        "Dump detailed segmenter logging",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "dynamic_shared_memory",
        "Dump the dynamic shared memory allocation",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "pre_segmenter_logging",
        "Pre-segmenter logging",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "python_definition",
        "Python Frontend Fusion Definition",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "python_definition_segments",
        "Python Frontend Fusion Definition of segments",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "python_frontend_debug",
        "Python Frontend debug information",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "transform_propagator",
        "Print propagation path and replay result",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "bank_conflict",
        "Dump bank conflict info",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "sync_map",
        "RAW dependency info",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "ca_map",
        "Dump the computeAt map",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "parallel_dimensions",
        "Dump known parallel dimensions",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "buffer_reuse_verbose",
        "Dump the analysis details of local/shared buffer re-use",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "lower_verbose",
        "Print all passes' transform in GpuLower::lower",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "expr_simplify",
        "Print all passes' transform in simplifyExpr",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "expr_sort",
        "Print merging decisions on expression sorting",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "expr_sort_verbose",
        "Print verbose debug info on expression sorting",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "occupancy",
        "Dump occupancy information",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "index_type",
        "Print the index type of the launched kernel",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "predicate_elimination",
        "Print the predicate elimination information",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "indexing_verbose",
        "Print verbose debug info on indexing",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "communication",
        "Print multi-GPU communications posted",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "compile_params",
        "Print NVRTC compile parameters",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "ptxas_verbose",
        "Print the ptxas verbose log including register usage",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "perf_debug_verbose",
        "Print verbose information when running kernels",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "ftrace",
        "Dump the function trace of selected internal functions",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    EnvVarOption(
        "cutlass_compile",
        "Dump compile commands and compile times for CutlassExecutor",
        "bool",
        "dump",
        env_var="NVFUSER_DUMP",
    ),
    # Feature Enable Options (NVFUSER_ENABLE)
    EnvVarOption(
        "cutlass_scheduler",
        "Enable the CUTLASS scheduler and executor",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "fuse_matmul",
        "Enable automatic fusion of matmul and linear ops",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "kernel_profile",
        "Enable intra-kernel performance profiling",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "fast_math",
        "Enable fast math optimizations (--use_fast_math)",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "tma_pointwise",
        "Enable TMA pointwise kernel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "tma_reduction",
        "Enable TMA reduction kernel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    # Additional NVFUSER_ENABLE options
    EnvVarOption(
        "fuse_multiple_matmuls",
        "Allow fusing more than one matmul in a single kernel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "id_model",
        "Enable IdModel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "id_model_extra_validation",
        "Enable extra error checking when building IdModel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "io_to_lower_precision",
        "Enable castInputOutputToLowerPrecision",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "kernel_db",
        "Enable Kernel Database",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "kernel_debug",
        "Enable debug mode in nvrtc",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "kernel_lineinfo",
        "Embed line info to compiled kernel and dump full CUDA C++ code",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "memory_promotion",
        "Enable promotion of memory types for non-pointwise ops",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "reuse_zeroed_memory",
        "Re-use zeroed memory used for grid synchronization",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "static_fusion_count",
        "Enable using single static count in kernel name",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "warn_register_spill",
        "Enable warnings of register spill",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "tma_inner_persistent",
        "Enable TMA inner persistent kernel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "ws_normalization",
        "Enable warp specialized persistent kernel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "host_ir_lowering",
        "Enable FusionKernelRuntime lowering to host IR",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "host_ir_jit",
        "Enable Host IR JIT compilation with LLVM",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "insert_resharding_after",
        "Insert resharding set after the expression",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "p2p_protocol",
        "Prescribe P2P protocol: put|get",
        "multi",
        "enable",
        env_var="NVFUSER_ENABLE",
        choices=["", "put", "get"],
    ),
    EnvVarOption(
        "multicast_protocol",
        "Prescribe multicast protocol: memcpy|multimem|batch_memcpy",
        "multi",
        "enable",
        env_var="NVFUSER_ENABLE",
        choices=["", "memcpy", "multimem", "batch_memcpy"],
    ),
    EnvVarOption(
        "parallel_serde",
        "Enable deserializing FusionExecutorCache in parallel",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "wait_debugger",
        "Wait for gdb attach at start for specified rank (value is rank number)",
        "int",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    EnvVarOption(
        "infer_contiguity",
        "Enable contiguity inference",
        "bool",
        "enable",
        env_var="NVFUSER_ENABLE",
    ),
    # Feature Disable Options (NVFUSER_DISABLE)
    EnvVarOption(
        "compile_to_sass",
        "Disable direct compilation to SASS (compile to PTX instead)",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "expr_simplify",
        "Disable expression simplifier",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "predicate_elimination",
        "Disable predicate elimination",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "fallback",
        "Disable fallback to eager mode on errors",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    # Additional NVFUSER_DISABLE options
    EnvVarOption(
        "contig_indexing",
        "Disable contiguous indexing",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "index_hoist",
        "Disable index hoisting",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "magic_zero",
        "Disable nvfuser_zero",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "fma",
        "Disable FMA instructions (warning: negatively affects performance)",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "greedy_scheduler",
        "Disable the greedy scheduler",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "resize_scheduler",
        "Disable the resize scheduler",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "grouped_grid_welford_outer_opt",
        "Disable use of outer-optimized grouped grid welford kernel",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "nvrtc_caching",
        "Disable compilation caching by nvrtc",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "parallel_compile",
        "Disable compiling Fusion segments in parallel",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "kernel_reuse",
        "Disable re-using cached FusionKernelRuntimes with different input shapes",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "matmul_expr_eval",
        "Disable ATen evaluation for entire fusion containing matmul",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "nvtx",
        "Disable NVTX instrumentation",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "python_inline_definitions",
        "Disable printing of inline definitions",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "var_name_remapping",
        "Disable variable name remapping",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "welford_vectorization",
        "Disable vectorization of Welford ops",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "reuse_mismatched_type_registers",
        "Disable explicitly re-using registers unless types match",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    EnvVarOption(
        "multidevice",
        "Disable creation of multidevice communicator (mainly for debugging)",
        "bool",
        "disable",
        env_var="NVFUSER_DISABLE",
    ),
    # Profiler Options (NVFUSER_PROF)
    EnvVarOption(
        "NVFUSER_PROF",
        "Profiler mode",
        "multi",
        "profiler",
        choices=[
            "",
            "enable",
            "enable.nocupti",
            "print",
            "print.nocupti",
            "print.verbose",
        ],
    ),
    # Compilation Control
    EnvVarOption(
        "NVFUSER_MAX_REG_COUNT",
        "Maximum number of registers per thread",
        "int",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_JIT_OPT_LEVEL",
        "PTX optimization level (0-4, default 4)",
        "int",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_USE_BLOCK_SYNC_ATOMIC",
        "Use atomic-based block synchronization instead of barriers",
        "bool",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_EXTERNAL_SRC",
        "Load kernel code from external CUDA files (comma-separated paths)",
        "string",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_TRACE",
        "Generate Chrome tracing format JSON file (specify file path)",
        "string",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_MATMUL_HEURISTIC_PLUGIN",
        "Shared library path for custom matmul scheduling heuristics",
        "string",
        "compilation",
    ),
    EnvVarOption(
        "NVFUSER_NUM_THREADS",
        "Number of threads for parallel compilation (default 8)",
        "int",
        "compilation",
    ),
    # ========================================================================
    # Misc OPTIONS
    # ========================================================================
    EnvVarOption(
        "DEBUG_SERDE",
        "Serde debugging flags",
        "multi",
        "misc",
        choices=["debug", "disable"],
    ),
    EnvVarOption(
        "NVFUSER_MASTER_ADDR",
        "Master node address for multi-node execution (required for multi-node)",
        "string",
        "misc",
    ),
    EnvVarOption(
        "NVFUSER_MASTER_PORT",
        "Master node port for multi-node execution (default: 29500)",
        "int",
        "misc",
    ),
]


class EnvVarConfig:
    """Manages the current configuration state."""

    def __init__(self):
        # Use (name, env_var) tuple as key to handle options with same name
        # but different env vars (e.g., expr_simplify in DUMP vs DISABLE)
        self.options: dict[tuple[str, str | None], EnvVarOption] = {
            (opt.name, opt.env_var): opt for opt in ENV_VAR_DEFINITIONS
        }
        # Also maintain a list of all options for iteration
        self.all_options: list[EnvVarOption] = ENV_VAR_DEFINITIONS.copy()
        self.categories: dict[str, list[EnvVarOption]] = self._organize_by_category()
        self._load_current_values()

    def _organize_by_category(self) -> dict[str, list[EnvVarOption]]:
        """Organize options by category."""
        categories: dict[str, list[EnvVarOption]] = {}
        for opt in self.all_options:
            if opt.category not in categories:
                categories[opt.category] = []
            categories[opt.category].append(opt)
        return categories

    def _load_current_values(self):
        """Load current values from environment."""
        for opt in self.all_options:
            env_var_name = opt.get_env_var_name()

            if opt.category in ["dump", "enable", "disable"]:
                # These are comma-separated list values
                list_val = os.environ.get(env_var_name, "")
                if list_val:
                    list_items = [item.strip() for item in list_val.split(",")]
                    if opt.name in list_items:
                        opt.current_value = "1"
            else:
                # Regular environment variables
                if env_var_name in os.environ:
                    val = os.environ[env_var_name]
                    if opt.var_type == "bool":
                        if val.upper() in ["ON", "1", "YES", "TRUE", "Y"]:
                            opt.current_value = "1"
                    else:
                        opt.current_value = val

    def get_env_exports(self) -> dict[str, str]:
        """Generate environment variable exports based on current configuration."""
        exports: dict[str, str] = {}

        # Collect values for list-based env vars (DUMP, ENABLE, DISABLE)
        list_vars: dict[str, list[str]] = {}  # Maps env_var_name -> list of values

        for opt in self.all_options:
            if opt.category in ["dump", "enable", "disable"]:
                # List items
                if opt.current_value == "1":
                    env_var = opt.get_env_var_name()
                    if env_var not in list_vars:
                        list_vars[env_var] = []
                    list_vars[env_var].append(opt.name)
            else:
                # Regular env vars
                if opt.current_value is not None:
                    env_var = opt.get_env_var_name()
                    if opt.var_type == "bool":
                        if opt.current_value == "1":
                            exports[env_var] = "1"
                    else:
                        exports[env_var] = opt.current_value

        # Add list-based env vars as comma-separated strings
        for env_var, values in list_vars.items():
            exports[env_var] = ",".join(values)

        return exports

    def get_unset_vars(self) -> list[str]:
        """Get list of ALL variable names that should be unset (empty or unconfigured)

        This unsets all known nvFuser variables that aren't configured, ensuring a
        clean slate when the script is sourced.
        """
        unset_vars: set[str] = set()  # Use set to avoid duplicates

        # Track which env vars have values
        env_vars_with_values: set[str] = set()

        for opt in self.all_options:
            if opt.category in ["dump", "enable", "disable"]:
                # List-based vars - only track if any are set
                if opt.current_value == "1":
                    env_vars_with_values.add(opt.get_env_var_name())
            else:
                # Regular env vars
                env_var = opt.get_env_var_name()
                if opt.current_value:
                    env_vars_with_values.add(env_var)
                else:
                    unset_vars.add(env_var)

        # Unset list vars that have no values set
        for list_var in ["NVFUSER_DUMP", "NVFUSER_ENABLE", "NVFUSER_DISABLE"]:
            if list_var not in env_vars_with_values:
                unset_vars.add(list_var)

        return sorted(list(unset_vars))


def simple_prompt_mode(config: EnvVarConfig):
    """Simple prompt-based configuration mode (no curses)."""
    # ANSI color codes
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("=" * 70)
    print("nvFuser Environment Configuration Tool - Simple Mode")
    print("=" * 70)
    print()

    for category in CATEGORY_NAMES.keys():
        if category not in config.categories:
            continue
        opts = config.categories[category]
        print(f"\n{CYAN}{BOLD}{CATEGORY_NAMES[category]}{RESET}")
        print("-" * 70)

        for opt in opts:
            # Show current value in green if set
            if opt.current_value:
                current = f"{GREEN}{opt.current_value}{RESET}"
            else:
                current = "(not set)"

            # Format the option name with type indicator
            opt_name = opt.name
            if opt.var_type == "multi":
                opt_name += " [multi]"

            print(f"\n{opt_name}:")
            print(f"  Description: {opt.description}")
            print(f"  Current: {current}")

            if opt.var_type == "bool":
                response = input("  Enable? [y/N]: ").strip().lower()
                opt.current_value = "1" if response in ["y", "yes"] else None
            elif opt.var_type == "multi":
                print(f"  Choices: {', '.join(repr(c) for c in opt.choices)}")
                default = opt.choices[0] if opt.choices else ""
                response = input(f"  Select [{default}]: ").strip()
                opt.current_value = response if response else default
            elif opt.var_type in ["int", "string"]:
                response = input("  Value: ").strip()
                opt.current_value = response if response else None

    print("\n" + "=" * 70)
    print(f"{BOLD}Configuration Summary{RESET}")
    print("=" * 70)

    exports = config.get_env_exports()
    if not exports:
        print("No environment variables configured.")
    else:
        for var, val in exports.items():
            print(f'export {var}="{GREEN}{val}{RESET}"')

    print("\nSave configuration? [Y/n]: ", end="")
    if input().strip().lower() not in ["n", "no"]:
        unsets = config.get_unset_vars()
        save_config(exports, unsets)
        print("\nConfiguration saved to: nvfuser_env.sh")
        print("To apply: source nvfuser_env.sh")


def save_config(
    exports: dict[str, str],
    unsets: list[str] | None = None,
    filename: str = "nvfuser_env.sh",
):
    """Save configuration to shell script with both exports and unsets"""
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# nvFuser Environment Configuration\n")
        f.write("# Generated by tools/configure_env.py\n\n")

        # Unset unconfigured variables first
        if unsets is not None:
            f.write("# Unset unconfigured variables\n")
            for var in sorted(unsets):
                f.write(f"unset {var}\n")
            f.write("\n")

        # Export configured variables
        if exports:
            f.write("# Export configured variables\n")
            for var, val in sorted(exports.items()):
                f.write(f'export {var}="{val}"\n')

    os.chmod(filename, 0o755)


def generate_script_mode(config: EnvVarConfig):
    """Generate a shell script with current environment configuration."""
    exports = config.get_env_exports()

    if not exports:
        print("No environment variables currently set.")
        print("Run interactive mode to configure variables first.")
        return

    unsets = config.get_unset_vars()
    save_config(exports, unsets)
    print("\nConfiguration saved to: nvfuser_env.sh")
    print("To apply: source nvfuser_env.sh")


def try_curses_mode(config: EnvVarConfig):
    """Try to run curses-based TUI mode."""
    try:
        import curses
        from curses_ui import run_curses_ui

        curses.wrapper(lambda stdscr: run_curses_ui(stdscr, config))
    except ImportError:
        print("Error: curses module not available.")
        print("Falling back to simple mode.")
        simple_prompt_mode(config)
    except Exception as e:
        print(f"Error running TUI mode: {e}")
        print("Falling back to simple mode.")
        simple_prompt_mode(config)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive nvFuser environment configuration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run interactive TUI
  %(prog)s --simple           # Run simple prompt mode
  %(prog)s --generate-script  # Generate shell script from current env
        """,
    )

    parser.add_argument(
        "--simple", action="store_true", help="Use simple prompt mode instead of TUI"
    )

    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="Generate shell script from current environment",
    )

    parser.add_argument(
        "--output",
        default="nvfuser_env.sh",
        help="Output filename for generated script (default: nvfuser_env.sh)",
    )

    args = parser.parse_args()

    config = EnvVarConfig()

    if args.generate_script:
        generate_script_mode(config)
    elif args.simple:
        simple_prompt_mode(config)
    else:
        try_curses_mode(config)


if __name__ == "__main__":
    main()
