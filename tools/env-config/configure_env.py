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

    name: str
    description: str
    var_type: Literal["bool", "string", "int", "multi"]
    category: str
    default: str = ""
    choices: List[str] = field(default_factory=list)
    current_value: Optional[str] = None

    def get_display_name(self) -> str:
        """Get the display name for this option (full env var name)."""
        # For dump, enable, disable - show as they appear in comma-separated list
        if self.category in ["dump", "enable", "disable"]:
            return self.name

        # For build options, show full NVFUSER_BUILD_* name
        if self.category in ["build", "build_advanced"]:
            if self.name == "max_jobs":
                return "MAX_JOBS"
            elif self.name in ["build_dir", "install_dir"]:
                return f"NVFUSER_BUILD_{self.name.upper()}"
            elif self.name == "cutlass_max_jobs":
                return "NVFUSER_CUTLASS_MAX_JOBS"
            elif self.name in [
                "build_type",
                "cpp_standard",
                "enable_pch",
                "explicit_error_check",
            ]:
                return f"NVFUSER_BUILD_{self.name.upper()}"
            else:
                # no_python -> NVFUSER_BUILD_NO_PYTHON
                return f"NVFUSER_BUILD_{self.name.upper()}"

        # For environment options, show actual env var name
        if self.category == "environment":
            env_map = {
                "cc": "CC",
                "cxx": "CXX",
                "cuda_home": "CUDA_HOME",
                "nvfuser_source_dir": "NVFUSER_SOURCE_DIR",
                "torch_cuda_arch_list": "TORCH_CUDA_ARCH_LIST",
                "cflags": "CFLAGS",
                "cxxflags": "CXXFLAGS",
                "ldflags": "LDFLAGS",
            }
            return env_map.get(self.name, self.name.upper())

        # For compilation options
        if self.category == "compilation":
            return f"NVFUSER_{self.name.upper()}"

        # For profiler
        if self.category == "profiler":
            return "NVFUSER_PROF"

        # Default: uppercase the name
        return self.name.upper()


# Define all nvFuser environment variables organized by category
ENV_VAR_DEFINITIONS = [
    # ========================================================================
    # ENVIRONMENT AND COMPILER CONFIGURATION
    # ========================================================================
    EnvVarOption("cc", "C compiler to use (CC)", "string", "environment"),
    EnvVarOption("cxx", "C++ compiler to use (CXX)", "string", "environment"),
    EnvVarOption(
        "cuda_home", "CUDA installation directory (CUDA_HOME)", "string", "environment"
    ),
    EnvVarOption(
        "nvfuser_source_dir",
        "nvFuser source directory (NVFUSER_SOURCE_DIR)",
        "string",
        "environment",
    ),
    EnvVarOption(
        "torch_cuda_arch_list",
        "Target CUDA architectures (TORCH_CUDA_ARCH_LIST, e.g., '8.0;9.0')",
        "string",
        "environment",
    ),
    EnvVarOption(
        "cflags", "Additional C compiler flags (CFLAGS)", "string", "environment"
    ),
    EnvVarOption(
        "cxxflags", "Additional C++ compiler flags (CXXFLAGS)", "string", "environment"
    ),
    EnvVarOption(
        "ldflags", "Additional linker flags (LDFLAGS)", "string", "environment"
    ),
    # ========================================================================
    # BUILD-TIME OPTIONS (NVFUSER_BUILD_*)
    # ========================================================================
    # Build Configuration
    EnvVarOption(
        "no_python", "Skip Python bindings (NVFUSER_BUILD_NO_PYTHON)", "bool", "build"
    ),
    EnvVarOption(
        "no_cutlass",
        "Skip building CUTLASS kernels (NVFUSER_BUILD_NO_CUTLASS)",
        "bool",
        "build",
    ),
    EnvVarOption("no_test", "Skip C++ tests (NVFUSER_BUILD_NO_TEST)", "bool", "build"),
    EnvVarOption(
        "no_benchmark", "Skip benchmarks (NVFUSER_BUILD_NO_BENCHMARK)", "bool", "build"
    ),
    EnvVarOption(
        "no_ninja",
        "Use make instead of ninja (NVFUSER_BUILD_NO_NINJA)",
        "bool",
        "build",
    ),
    EnvVarOption(
        "build_type",
        "Build type: Release, Debug, RelWithDebInfo (NVFUSER_BUILD_BUILD_TYPE)",
        "multi",
        "build",
        choices=["Release", "Debug", "RelWithDebInfo"],
    ),
    EnvVarOption(
        "cpp_standard",
        "C++ standard version (NVFUSER_BUILD_CPP_STANDARD)",
        "int",
        "build",
        default="20",
    ),
    EnvVarOption(
        "enable_pch",
        "Enable precompiled headers (NVFUSER_BUILD_ENABLE_PCH)",
        "bool",
        "build",
    ),
    # Advanced Build Options
    EnvVarOption(
        "build_with_ucc",
        "Build with UCC support for multi-device (NVFUSER_BUILD_WITH_UCC)",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "build_with_asan",
        "Build with Address Sanitizer (NVFUSER_BUILD_WITH_ASAN)",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "build_without_distributed",
        "Build without multidevice support (NVFUSER_BUILD_WITHOUT_DISTRIBUTED)",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "explicit_error_check",
        "Enable explicit error checking (NVFUSER_BUILD_EXPLICIT_ERROR_CHECK)",
        "bool",
        "build_advanced",
    ),
    EnvVarOption(
        "cutlass_max_jobs",
        "Max parallel jobs for CUTLASS build (NVFUSER_CUTLASS_MAX_JOBS)",
        "int",
        "build_advanced",
    ),
    EnvVarOption(
        "build_dir",
        "Custom build directory (NVFUSER_BUILD_DIR)",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "install_dir",
        "Custom install directory (NVFUSER_BUILD_INSTALL_DIR)",
        "string",
        "build_advanced",
    ),
    EnvVarOption(
        "max_jobs", "Max parallel compilation jobs (MAX_JOBS)", "int", "build_advanced"
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
    ),
    EnvVarOption(
        "fusion_ir",
        "Dump the Fusion IR before lowering (fed to KernelExecutor::compile)",
        "bool",
        "dump",
    ),
    EnvVarOption("kernel_ir", "Dump the compiler Kernel IR", "bool", "dump"),
    EnvVarOption(
        "cuda_kernel", "Dump the generated CUDA C++ kernel code", "bool", "dump"
    ),
    EnvVarOption("cuda_full", "Dump the complete CUDA C++ code", "bool", "dump"),
    EnvVarOption("ptx", "Dump compiled PTX", "bool", "dump"),
    EnvVarOption("sass", "Dump disassembled SASS", "bool", "dump"),
    EnvVarOption("sass_to_file", "Dump disassembled SASS to file", "bool", "dump"),
    EnvVarOption(
        "launch_param", "Dump the launch parameters of kernel", "bool", "dump"
    ),
    EnvVarOption(
        "kernel_args",
        "Print the runtime kernel arguments when launching kernels",
        "bool",
        "dump",
    ),
    EnvVarOption(
        "scheduler_params", "Dump scheduler heuristic parameters", "bool", "dump"
    ),
    EnvVarOption(
        "scheduler_verbose", "Dump detailed scheduler logging", "bool", "dump"
    ),
    # Feature Enable Options (NVFUSER_ENABLE)
    EnvVarOption(
        "cutlass_scheduler",
        "Enable the CUTLASS scheduler and executor",
        "bool",
        "enable",
    ),
    EnvVarOption(
        "fuse_matmul",
        "Enable automatic fusion of matmul and linear ops",
        "bool",
        "enable",
    ),
    EnvVarOption(
        "kernel_profile", "Enable intra-kernel performance profiling", "bool", "enable"
    ),
    EnvVarOption(
        "fast_math",
        "Enable fast math optimizations (--use_fast_math)",
        "bool",
        "enable",
    ),
    EnvVarOption("tma_pointwise", "Enable TMA pointwise kernel", "bool", "enable"),
    EnvVarOption("tma_reduction", "Enable TMA reduction kernel", "bool", "enable"),
    # Feature Disable Options (NVFUSER_DISABLE)
    EnvVarOption(
        "compile_to_sass",
        "Disable direct compilation to SASS (compile to PTX instead)",
        "bool",
        "disable",
    ),
    EnvVarOption("expr_simplify", "Disable expression simplifier", "bool", "disable"),
    EnvVarOption(
        "predicate_elimination", "Disable predicate elimination", "bool", "disable"
    ),
    EnvVarOption(
        "fallback", "Disable fallback to eager mode on errors", "bool", "disable"
    ),
    # Profiler Options (NVFUSER_PROF)
    EnvVarOption(
        "profiler",
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
        "max_reg_count", "Maximum number of registers per thread", "int", "compilation"
    ),
    EnvVarOption(
        "jit_opt_level", "PTX optimization level (0-4, default 4)", "int", "compilation"
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
        # Load BUILD options (NVFUSER_BUILD_*)
        build_env_map = {
            "no_python": "NVFUSER_BUILD_NO_PYTHON",
            "no_cutlass": "NVFUSER_BUILD_NO_CUTLASS",
            "no_test": "NVFUSER_BUILD_NO_TEST",
            "no_benchmark": "NVFUSER_BUILD_NO_BENCHMARK",
            "no_ninja": "NVFUSER_BUILD_NO_NINJA",
            "build_with_ucc": "NVFUSER_BUILD_WITH_UCC",
            "build_with_asan": "NVFUSER_BUILD_WITH_ASAN",
            "build_without_distributed": "NVFUSER_BUILD_WITHOUT_DISTRIBUTED",
            "explicit_error_check": "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK",
            "enable_pch": "NVFUSER_BUILD_ENABLE_PCH",
            "build_type": "NVFUSER_BUILD_BUILD_TYPE",
            "cpp_standard": "NVFUSER_BUILD_CPP_STANDARD",
            "cutlass_max_jobs": "NVFUSER_CUTLASS_MAX_JOBS",
            "build_dir": "NVFUSER_BUILD_DIR",
            "install_dir": "NVFUSER_BUILD_INSTALL_DIR",
            "max_jobs": "MAX_JOBS",
        }

        for opt_name, env_var in build_env_map.items():
            if env_var in os.environ and opt_name in self.options:
                val = os.environ[env_var]
                opt = self.options[opt_name]
                if opt.var_type == "bool":
                    if val.upper() in ["ON", "1", "YES", "TRUE", "Y"]:
                        opt.current_value = "1"
                else:
                    opt.current_value = val

        # Load ENVIRONMENT options
        env_map = {
            "cc": "CC",
            "cxx": "CXX",
            "cuda_home": "CUDA_HOME",
            "nvfuser_source_dir": "NVFUSER_SOURCE_DIR",
            "torch_cuda_arch_list": "TORCH_CUDA_ARCH_LIST",
            "cflags": "CFLAGS",
            "cxxflags": "CXXFLAGS",
            "ldflags": "LDFLAGS",
        }

        for opt_name, env_var in env_map.items():
            if env_var in os.environ and opt_name in self.options:
                self.options[opt_name].current_value = os.environ[env_var]

        # Load NVFUSER_DUMP values
        dump_val = os.environ.get("NVFUSER_DUMP", "")
        if dump_val:
            dump_opts = dump_val.split(",")
            for opt_name in dump_opts:
                if opt_name in self.options:
                    self.options[opt_name].current_value = "1"

        # Load NVFUSER_ENABLE values
        enable_val = os.environ.get("NVFUSER_ENABLE", "")
        if enable_val:
            enable_opts = enable_val.split(",")
            for opt_name in enable_opts:
                if opt_name in self.options:
                    self.options[opt_name].current_value = "1"

        # Load NVFUSER_DISABLE values
        disable_val = os.environ.get("NVFUSER_DISABLE", "")
        if disable_val:
            disable_opts = disable_val.split(",")
            for opt_name in disable_opts:
                if opt_name in self.options:
                    self.options[opt_name].current_value = "1"

        # Load NVFUSER_PROF
        prof_val = os.environ.get("NVFUSER_PROF", "")
        if prof_val and "profiler" in self.options:
            self.options["profiler"].current_value = prof_val

        # Load compilation options
        if "NVFUSER_MAX_REG_COUNT" in os.environ:
            self.options["max_reg_count"].current_value = os.environ[
                "NVFUSER_MAX_REG_COUNT"
            ]

        if "NVFUSER_JIT_OPT_LEVEL" in os.environ:
            self.options["jit_opt_level"].current_value = os.environ[
                "NVFUSER_JIT_OPT_LEVEL"
            ]

    def get_env_exports(self) -> Dict[str, str]:
        """Generate environment variable exports based on current configuration."""
        exports = {}

        # Handle BUILD options (NVFUSER_BUILD_*)
        build_env_map = {
            "no_python": "NVFUSER_BUILD_NO_PYTHON",
            "no_cutlass": "NVFUSER_BUILD_NO_CUTLASS",
            "no_test": "NVFUSER_BUILD_NO_TEST",
            "no_benchmark": "NVFUSER_BUILD_NO_BENCHMARK",
            "no_ninja": "NVFUSER_BUILD_NO_NINJA",
            "build_with_ucc": "NVFUSER_BUILD_WITH_UCC",
            "build_with_asan": "NVFUSER_BUILD_WITH_ASAN",
            "build_without_distributed": "NVFUSER_BUILD_WITHOUT_DISTRIBUTED",
            "explicit_error_check": "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK",
            "enable_pch": "NVFUSER_BUILD_ENABLE_PCH",
            "build_type": "NVFUSER_BUILD_BUILD_TYPE",
            "cpp_standard": "NVFUSER_BUILD_CPP_STANDARD",
            "cutlass_max_jobs": "NVFUSER_CUTLASS_MAX_JOBS",
            "build_dir": "NVFUSER_BUILD_DIR",
            "install_dir": "NVFUSER_BUILD_INSTALL_DIR",
            "max_jobs": "MAX_JOBS",
        }

        for opt_name, env_var in build_env_map.items():
            if opt_name in self.options:
                opt = self.options[opt_name]
                if opt.current_value:
                    if opt.var_type == "bool" and opt.current_value == "1":
                        exports[env_var] = "1"
                    elif opt.var_type in ["int", "string", "multi"]:
                        exports[env_var] = opt.current_value

        # Handle ENVIRONMENT options
        env_map = {
            "cc": "CC",
            "cxx": "CXX",
            "cuda_home": "CUDA_HOME",
            "nvfuser_source_dir": "NVFUSER_SOURCE_DIR",
            "torch_cuda_arch_list": "TORCH_CUDA_ARCH_LIST",
            "cflags": "CFLAGS",
            "cxxflags": "CXXFLAGS",
            "ldflags": "LDFLAGS",
        }

        for opt_name, env_var in env_map.items():
            if opt_name in self.options:
                opt = self.options[opt_name]
                if opt.current_value:
                    exports[env_var] = opt.current_value

        # Collect NVFUSER_DUMP options
        dump_opts = []
        for opt in self.categories.get("dump", []):
            if opt.current_value == "1":
                dump_opts.append(opt.name)
        if dump_opts:
            exports["NVFUSER_DUMP"] = ",".join(dump_opts)

        # Collect NVFUSER_ENABLE options
        enable_opts = []
        for opt in self.categories.get("enable", []):
            if opt.current_value == "1":
                enable_opts.append(opt.name)
        if enable_opts:
            exports["NVFUSER_ENABLE"] = ",".join(enable_opts)

        # Collect NVFUSER_DISABLE options
        disable_opts = []
        for opt in self.categories.get("disable", []):
            if opt.current_value == "1":
                disable_opts.append(opt.name)
        if disable_opts:
            exports["NVFUSER_DISABLE"] = ",".join(disable_opts)

        # Handle profiler
        for opt in self.categories.get("profiler", []):
            if opt.current_value and opt.current_value != "":
                exports["NVFUSER_PROF"] = opt.current_value

        # Handle compilation options
        for opt in self.categories.get("compilation", []):
            if opt.current_value:
                var_name = f"NVFUSER_{opt.name.upper()}"
                exports[var_name] = opt.current_value

        return exports

    def get_unset_vars(self) -> List[str]:
        """Get list of ALL variable names that should be unset (empty or unconfigured)

        This unsets all known nvFuser variables that aren't configured, ensuring a
        clean slate when the script is sourced.
        """
        unset_vars = []

        # Build environment variable mapping
        build_env_map = {
            "no_python": "NVFUSER_BUILD_NO_PYTHON",
            "no_cutlass": "NVFUSER_BUILD_NO_CUTLASS",
            "no_test": "NVFUSER_BUILD_NO_TEST",
            "no_benchmark": "NVFUSER_BUILD_NO_BENCHMARK",
            "no_ninja": "NVFUSER_BUILD_NO_NINJA",
            "build_with_ucc": "NVFUSER_BUILD_WITH_UCC",
            "build_with_asan": "NVFUSER_BUILD_WITH_ASAN",
            "build_without_distributed": "NVFUSER_BUILD_WITHOUT_DISTRIBUTED",
            "explicit_error_check": "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK",
            "enable_pch": "NVFUSER_BUILD_ENABLE_PCH",
            "build_type": "NVFUSER_BUILD_BUILD_TYPE",
            "cpp_standard": "NVFUSER_BUILD_CPP_STANDARD",
            "cutlass_max_jobs": "NVFUSER_CUTLASS_MAX_JOBS",
            "build_dir": "NVFUSER_BUILD_DIR",
            "install_dir": "NVFUSER_BUILD_INSTALL_DIR",
            "max_jobs": "MAX_JOBS",
        }

        # Check build options
        for opt_name, env_var in build_env_map.items():
            if opt_name in self.options:
                opt = self.options[opt_name]
                if not opt.current_value:
                    unset_vars.append(env_var)

        # Environment options
        env_map = {
            "cc": "CC",
            "cxx": "CXX",
            "cuda_home": "CUDA_HOME",
            "nvfuser_source_dir": "NVFUSER_SOURCE_DIR",
            "torch_cuda_arch_list": "TORCH_CUDA_ARCH_LIST",
            "cflags": "CFLAGS",
            "cxxflags": "CXXFLAGS",
            "ldflags": "LDFLAGS",
        }

        for opt_name, env_var in env_map.items():
            if opt_name in self.options:
                opt = self.options[opt_name]
                if not opt.current_value:
                    unset_vars.append(env_var)

        # Check if any dump options are set
        dump_opts_set = any(
            opt.current_value == "1" for opt in self.categories.get("dump", [])
        )
        if not dump_opts_set:
            unset_vars.append("NVFUSER_DUMP")

        # Check if any enable options are set
        enable_opts_set = any(
            opt.current_value == "1" for opt in self.categories.get("enable", [])
        )
        if not enable_opts_set:
            unset_vars.append("NVFUSER_ENABLE")

        # Check if any disable options are set
        disable_opts_set = any(
            opt.current_value == "1" for opt in self.categories.get("disable", [])
        )
        if not disable_opts_set:
            unset_vars.append("NVFUSER_DISABLE")

        # Check profiler
        profiler_set = any(
            opt.current_value and opt.current_value != ""
            for opt in self.categories.get("profiler", [])
        )
        if not profiler_set:
            unset_vars.append("NVFUSER_PROF")

        # Check compilation options
        for opt in self.categories.get("compilation", []):
            if not opt.current_value:
                var_name = f"NVFUSER_{opt.name.upper()}"
                unset_vars.append(var_name)

        return unset_vars


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
