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

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Load environment options from YAML
from options_loader import load_options_from_yaml, EnvVarOption

# Load configuration from YAML file
_yaml_path = Path(__file__).parent / "env_options.yaml"
CATEGORY_NAMES, ENV_VAR_DEFINITIONS = load_options_from_yaml(_yaml_path)


class EnvVarConfig:
    """Manages the current configuration state."""

    def __init__(self) -> None:
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

    def _load_current_values(self) -> None:
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


def simple_prompt_mode(config: EnvVarConfig) -> None:
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

            match opt.var_type:
                case "bool":
                    response = input("  Enable? [y/N]: ").strip().lower()
                    opt.current_value = "1" if response in ["y", "yes"] else None
                case "multi":
                    print(f"  Choices: {', '.join(repr(c) for c in opt.choices)}")
                    default = opt.choices[0] if opt.choices else ""
                    response = input(f"  Select [{default}]: ").strip()
                    opt.current_value = response if response else default
                case "int" | "string":
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
) -> None:
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

    # Set secure permissions (600 for apply scripts, 755 for user-generated scripts)
    # If filename starts with a dot, it's likely a temporary apply script
    if os.path.basename(filename).startswith("."):
        os.chmod(filename, 0o600)  # Apply scripts: owner read/write only
    else:
        os.chmod(filename, 0o755)  # User scripts: executable by all


def generate_script_mode(config: EnvVarConfig) -> None:
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


def try_curses_mode(config: EnvVarConfig) -> None:
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


def main() -> None:
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
