#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# nvFuser Configuration Tool - Shell Integration
#
# Works with bash, zsh, and other POSIX shells
#
# Setup (ONE LINE in ~/.bashrc or ~/.zshrc):
#   eval "$(/absolute/path/to/Fuser/tools/env-config/nvfuser-config-shell.sh)"
#
# Usage:
#   nvfuser-configure       # Configure and auto-apply

# Detect if being sourced vs executed - works in both bash and zsh
# If sourced, return 0; if executed, return 1
_is_sourced() {
    if [ -n "$ZSH_VERSION" ]; then
        # zsh: use ZSH_EVAL_CONTEXT
        case $ZSH_EVAL_CONTEXT in
            *:file:*) return 0;;  # Sourced
            *:file) return 1;;     # Executed
            *) return 1;;          # Executed
        esac
    elif [ -n "$BASH_VERSION" ]; then
        # bash: compare BASH_SOURCE with $0
        if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
            return 0  # Sourced
        else
            return 1  # Executed
        fi
    else
        # Other shells: assume sourced if $0 looks like a shell name
        case "$(basename "$0" 2>/dev/null)" in
            sh|dash|ksh) return 0;;  # Likely sourced
            *) return 1;;            # Likely executed
        esac
    fi
}

# If being executed (not sourced), print the function definition for eval
if ! _is_sourced; then
    # Get the absolute path to this script's directory
    # This will be embedded in the function definition
    if [ -n "$BASH_VERSION" ]; then
        EMBED_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    else
        # For sh/dash, use $0
        EMBED_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    fi

    # Print function with embedded path
    cat << EOF
nvfuser-configure() {
    local SCRIPT_DIR="$EMBED_SCRIPT_DIR"
    local APPLY_SCRIPT="\$(pwd)/.nvfuser_apply_now.sh"
    local TOOL_PATH="\$SCRIPT_DIR/configure_env.py"

    # If file not found, try alternative locations
    if [ ! -f "\$TOOL_PATH" ]; then
        # Try to find configure_env.py in PATH
        TOOL_PATH="\$(command -v configure_env.py 2>/dev/null || echo "")"
        if [ -z "\$TOOL_PATH" ]; then
            echo "Error: configure_env.py not found"
            echo "Expected location: $EMBED_SCRIPT_DIR/configure_env.py"
            echo "Make sure you use the full path in your setup command"
            return 1
        fi
    fi

    python "\$TOOL_PATH" "\$@"
    local exit_code=\$?

    # Find the apply script (Python creates it with unpredictable name for security)
    local APPLY_SCRIPT
    for script in "\$(pwd)"/.nvfuser-apply.*.sh; do
        if [ -f "\$script" ]; then
            APPLY_SCRIPT="\$script"
            break
        fi
    done

    # Debug: Show what happened
    if [ "\$NVFUSER_CONFIG_DEBUG" = "1" ]; then
        echo "[DEBUG] Python exit code: \$exit_code"
        echo "[DEBUG] Apply script path: \${APPLY_SCRIPT:-(not found)}"
        if [ -n "\$APPLY_SCRIPT" ] && [ -f "\$APPLY_SCRIPT" ]; then
            echo "[DEBUG] Apply script exists"
        else
            echo "[DEBUG] Apply script NOT found"
        fi
    fi

    # If apply script was generated, verify and source it
    # Security: Verify ownership and permissions to prevent TOCTOU attacks
    if [ \$exit_code -eq 0 ] && [ -n "\$APPLY_SCRIPT" ] && [ -f "\$APPLY_SCRIPT" ]; then
        # Verify file is owned by current user and has mode 600
        if [ -O "\$APPLY_SCRIPT" ] && [ "\$(stat -c '%a' "\$APPLY_SCRIPT" 2>/dev/null)" = "600" ]; then
            . "\$APPLY_SCRIPT" && rm -f "\$APPLY_SCRIPT" && echo "✓ Configuration applied to current shell"
        else
            echo "Warning: Apply script has unexpected ownership/permissions, skipping for security" >&2
            rm -f "\$APPLY_SCRIPT"
        fi
    fi

    return \$exit_code
}
EOF
    exit 0
fi

# If being sourced, define the function directly
nvfuser-configure() {
    # Portable way to get script directory (works in bash and zsh)
    if [ -n "$ZSH_VERSION" ]; then
        # zsh: use parameter expansion modifiers
        local SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
    elif [ -n "$BASH_VERSION" ]; then
        # bash: use BASH_SOURCE
        local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    else
        # Fallback: use current directory
        local SCRIPT_DIR="$(pwd)"
    fi

    local APPLY_SCRIPT="$(pwd)/.nvfuser-apply."*.sh
    local TOOL_PATH="$SCRIPT_DIR/configure_env.py"

    # If SCRIPT_DIR detection failed, try to find configure_env.py in PATH
    if [ ! -f "$TOOL_PATH" ]; then
        TOOL_PATH="$(command -v configure_env.py 2>/dev/null || echo "")"
        if [ -z "$TOOL_PATH" ]; then
            echo "Error: configure_env.py not found"
            echo "Make sure nvfuser/tools is in your PATH or use absolute path in setup"
            return 1
        fi
    fi

    python "$TOOL_PATH" "$@"
    local exit_code=$?

    # Find the apply script (Python creates it with unpredictable name for security)
    local APPLY_SCRIPT
    for script in "$(pwd)"/.nvfuser-apply.*.sh; do
        if [ -f "$script" ]; then
            APPLY_SCRIPT="$script"
            break
        fi
    done

    # Debug: Show what happened
    if [ "$NVFUSER_CONFIG_DEBUG" = "1" ]; then
        echo "[DEBUG] Python exit code: $exit_code"
        echo "[DEBUG] Apply script path: ${APPLY_SCRIPT:-(not found)}"
        if [ -n "$APPLY_SCRIPT" ] && [ -f "$APPLY_SCRIPT" ]; then
            echo "[DEBUG] Apply script exists"
        else
            echo "[DEBUG] Apply script NOT found"
        fi
    fi

    # If apply script was generated, verify and source it
    # Security: Verify ownership and permissions to prevent TOCTOU attacks
    if [ $exit_code -eq 0 ] && [ -n "$APPLY_SCRIPT" ] && [ -f "$APPLY_SCRIPT" ]; then
        # Verify file is owned by current user and has mode 600
        if [ -O "$APPLY_SCRIPT" ] && [ "$(stat -c '%a' "$APPLY_SCRIPT" 2>/dev/null)" = "600" ]; then
            . "$APPLY_SCRIPT" && rm -f "$APPLY_SCRIPT" && echo "✓ Configuration applied to current shell"
        else
            echo "Warning: Apply script has unexpected ownership/permissions, skipping for security" >&2
            rm -f "$APPLY_SCRIPT"
        fi
    fi

    return $exit_code
}
