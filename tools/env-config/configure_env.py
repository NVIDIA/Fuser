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
from typing import List, Dict, Optional, Literal


@dataclass
class EnvVarOption:
    """Represents a single environment variable option."""

    name: str  # The actual environment variable name (e.g., "CC", "NVFUSER_BUILD_NO_PYTHON", or value for lists)
    description: str
    var_type: Literal["bool", "string", "int", "multi"]
    category: str
    env_var: Optional[
        str
    ] = None  # For list items (dump/enable/disable), this is the parent env var
    default: str = ""
    choices: List[str] = field(default_factory=list)
    current_value: Optional[str] = None

    def get_display_name(self) -> str:
        """Get the display name for this option."""
        return self.name

    def get_env_var_name(self) -> str:
        """Get the actual environment variable name (for list items, return the parent)."""
        if self.env_var:
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
        "NVFUSER_SOURCE_DIR", "nvFuser source directory", "string", "environment"
    ),
    EnvVarOption(
        "TORCH_CUDA_ARCH_LIST",
        "Target CUDA architectures (e.g., '8.0;9.0')",
        "string",
        "environment",
    ),
    EnvVarOption("CFLAGS", "Additional C compiler flags", "string", "environment"),
    EnvVarOption("CXXFLAGS", "Additional C++ compiler flags", "string", "environment"),
    EnvVarOption("LDFLAGS", "Additional linker flags", "string", "environment"),
    # ========================================================================
    # BUILD-TIME OPTIONS (NVFUSER_BUILD_*)
    # ========================================================================
    # Build Configuration
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
        "NVFUSER_BUILD_BUILD_TYPE",
        "Build type: Release, Debug, RelWithDebInfo",
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
]


class EnvVarConfig:
    """Manages the current configuration state."""

    def __init__(self):
        self.options = {opt.name: opt for opt in ENV_VAR_DEFINITIONS}
        self.categories = self._organize_by_category()
        self._load_current_values()

    def _organize_by_category(self) -> Dict[str, List[EnvVarOption]]:
        """Organize options by category."""
        categories = {}
        for opt in self.options.values():
            if opt.category not in categories:
                categories[opt.category] = []
            categories[opt.category].append(opt)
        return categories

    def _load_current_values(self):
        """Load current values from environment."""
        for opt in self.options.values():
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

    def get_env_exports(self) -> Dict[str, str]:
        """Generate environment variable exports based on current configuration."""
        exports = {}

        # Collect values for list-based env vars (DUMP, ENABLE, DISABLE)
        list_vars = {}  # Maps env_var_name -> list of values

        for opt in self.options.values():
            if opt.category in ["dump", "enable", "disable"]:
                # List items
                if opt.current_value == "1":
                    env_var = opt.get_env_var_name()
                    if env_var not in list_vars:
                        list_vars[env_var] = []
                    list_vars[env_var].append(opt.name)
            else:
                # Regular env vars
                if opt.current_value:
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

    def get_unset_vars(self) -> List[str]:
        """Get list of ALL variable names that should be unset (empty or unconfigured)

        This unsets all known nvFuser variables that aren't configured, ensuring a
        clean slate when the script is sourced.
        """
        unset_vars = set()  # Use set to avoid duplicates

        # Track which env vars have values
        env_vars_with_values = set()

        for opt in self.options.values():
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

    category_names = {
        "build": "Build Configuration (NVFUSER_BUILD_*)",
        "build_advanced": "Advanced Build Options (NVFUSER_BUILD_*)",
        "environment": "Environment & Compiler Settings (CC, CXX, CUDA_HOME, etc.)",
        "dump": "Debug/Diagnostic Options (NVFUSER_DUMP)",
        "enable": "Feature Enable Options (NVFUSER_ENABLE)",
        "disable": "Feature Disable Options (NVFUSER_DISABLE)",
        "profiler": "Profiler Options (NVFUSER_PROF)",
        "compilation": "Runtime Compilation Control",
    }

    for category, opts in config.categories.items():
        print(f"\n{CYAN}{BOLD}{category_names.get(category, category.upper())}{RESET}")
        print("-" * 70)

        for opt in opts:
            # Show current value in green if set
            if opt.current_value:
                current = f"{GREEN}{opt.current_value}{RESET}"
            else:
                current = "(not set)"

            print(f"\n{opt.name}:")
            print(f"  Description: {opt.description}")
            print(f"  Current: {current}")

            if opt.var_type == "bool":
                response = input("  Enable? [y/N]: ").strip().lower()
                opt.current_value = "1" if response in ["y", "yes"] else None
            elif opt.var_type == "multi":
                print(f"  Choices: {', '.join(opt.choices)}")
                response = input(f"  Select [{opt.choices[0]}]: ").strip()
                opt.current_value = response if response else opt.choices[0]
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
    exports: Dict[str, str], unsets: List[str] = None, filename: str = "nvfuser_env.sh"
):
    """Save configuration to shell script with both exports and unsets"""
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# nvFuser Environment Configuration\n")
        f.write("# Generated by tools/configure_env.py\n\n")

        # Unset unconfigured variables first
        if unsets:
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
