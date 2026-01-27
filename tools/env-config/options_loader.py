#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Load environment options from YAML configuration.

This module provides a loader for env_options.yaml that replaces the
hardcoded Python definitions in configure_env.py.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvVarOption:
    """Represents a single environment variable option."""

    name: str
    description: str
    var_type: Literal["bool", "string", "int", "multi"]
    category: str
    env_var: str | None = None
    default: str = ""
    choices: list[str] = field(default_factory=list)
    current_value: str | None = None
    source: str = ""  # Optional: tracks where in options.h this came from

    def get_display_name(self) -> str:
        """Get the display name for this option."""
        return self.name

    def get_env_var_name(self) -> str:
        """Get the actual environment variable name."""
        if self.env_var is not None:
            return self.env_var
        return self.name


def load_options_from_yaml(
    yaml_path: str | Path,
) -> tuple[dict[str, str], list[EnvVarOption]]:
    """Load environment options from YAML file.

    Returns:
        Tuple of (CATEGORY_NAMES dict, ENV_VAR_DEFINITIONS list)
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Options YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract category names
    category_names = {}
    for cat_key, cat_info in config.get("categories", {}).items():
        category_names[cat_key] = cat_info["display_name"]

    # Extract options
    options = []
    for opt_data in config.get("options", []):
        option = EnvVarOption(
            name=opt_data["name"],
            description=opt_data["description"],
            var_type=opt_data["type"],
            category=opt_data["category"],
            env_var=opt_data.get("env_var"),
            default=opt_data.get("default", ""),
            choices=opt_data.get("choices", []),
            source=opt_data.get("source", ""),
        )
        options.append(option)

    return category_names, options


# For testing
if __name__ == "__main__":
    import sys

    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "env_options.yaml"

    try:
        category_names, options = load_options_from_yaml(yaml_file)

        print(f"Loaded from {yaml_file}:")
        print(f"  Categories: {len(category_names)}")
        print(f"  Options: {len(options)}")
        print()

        # Count by category
        cat_counts = {}
        for opt in options:
            cat = opt.category
            if cat not in cat_counts:
                cat_counts[cat] = 0
            cat_counts[cat] += 1

        print("Options by category:")
        for cat, count in sorted(cat_counts.items()):
            cat_name = category_names.get(cat, cat)
            print(f"  {cat_name[:50]:50s}: {count:3d}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
