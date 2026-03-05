#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# nvFuser Configuration Tool - Shell Integration
#
# Setup (ONE LINE in ~/.bashrc or ~/.zshrc):
#   eval "$(/absolute/path/to/Fuser/tools/env-config/nvfuser-config-shell.sh)"
#
# Usage:
#   nvfuser-configure       # Configure and auto-apply

# Detect if being sourced vs executed
_is_sourced() {
    if [ -n "$ZSH_VERSION" ]; then
        case $ZSH_EVAL_CONTEXT in
            *:file:*) return 0;;
            *) return 1;;
        esac
    elif [ -n "$BASH_VERSION" ]; then
        [ "${BASH_SOURCE[0]}" != "${0}" ]
    else
        case "$(basename "$0" 2>/dev/null)" in
            sh|dash|ksh) return 0;;
            *) return 1;;
        esac
    fi
}

# Main implementation (shared by both eval and sourced versions)
_nvfuser_configure_impl() {
    local SCRIPT_DIR="$1"
    shift

    local TOOL_PATH="$SCRIPT_DIR/configure_env.py"

    # If file not found, try PATH
    if [ ! -f "$TOOL_PATH" ]; then
        TOOL_PATH="$(command -v configure_env.py 2>/dev/null || echo "")"
        if [ -z "$TOOL_PATH" ]; then
            echo "Error: configure_env.py not found" >&2
            echo "Expected location: $SCRIPT_DIR/configure_env.py" >&2
            return 1
        fi
    fi

    python "$TOOL_PATH" "$@"
    local exit_code=$?

    # Find the apply script (created with unpredictable name for security)
    local APPLY_SCRIPT
    for script in "$(pwd)"/.nvfuser-apply.*.sh; do
        if [ -f "$script" ]; then
            APPLY_SCRIPT="$script"
            break
        fi
    done

    # Debug output
    if [ "$NVFUSER_CONFIG_DEBUG" = "1" ]; then
        echo "[DEBUG] Python exit code: $exit_code" >&2
        echo "[DEBUG] Apply script: ${APPLY_SCRIPT:-(not found)}" >&2
    fi

    # Verify and source apply script if it exists
    # Security: Check ownership and permissions to prevent TOCTOU attacks
    if [ $exit_code -eq 0 ] && [ -n "$APPLY_SCRIPT" ] && [ -f "$APPLY_SCRIPT" ]; then
        if [ -O "$APPLY_SCRIPT" ] && [ "$(stat -c '%a' "$APPLY_SCRIPT" 2>/dev/null)" = "600" ]; then
            . "$APPLY_SCRIPT" && rm -f "$APPLY_SCRIPT" && echo "✓ Configuration applied to current shell"
        else
            echo "Warning: Apply script has unexpected ownership/permissions, skipping for security" >&2
            rm -f "$APPLY_SCRIPT"
        fi
    fi

    return $exit_code
}

# If executed (not sourced), output function definition for eval
if ! _is_sourced; then
    # Get script directory and this script's path
    if [ -n "$BASH_VERSION" ]; then
        THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    else
        THIS_SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    fi

    SCRIPT_DIR="$(dirname "$THIS_SCRIPT")"

    # Output function that sources this script and calls the implementation
    cat << EOF
nvfuser-configure() {
    # Source the script to get the implementation function
    . "$THIS_SCRIPT"
    _nvfuser_configure_impl "$SCRIPT_DIR" "\$@"
}
EOF
    exit 0
fi

# If sourced, define function directly
nvfuser-configure() {
    # Detect script directory
    local SCRIPT_DIR
    if [ -n "$ZSH_VERSION" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
    elif [ -n "$BASH_VERSION" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    else
        SCRIPT_DIR="$(pwd)"
    fi

    _nvfuser_configure_impl "$SCRIPT_DIR" "$@"
}
