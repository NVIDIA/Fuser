<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->
# nvFuser Environment Configuration Tool

Interactive tool for managing nvFuser's build and runtime environment variables.

## Quick Setup (Portable, Project-Scoped - Recommended)

Add these lines to your `~/.zshrc` or `~/.bashrc` for automatic loading when you enter an nvFuser directory:

**For bash:**
```bash
cat >> ~/.bashrc << 'EOF'

# nvFuser: Auto-load configuration tool
_load_nvfuser_shell() { [[ -f .nvfuser-shell ]] && source .nvfuser-shell; }
PROMPT_COMMAND="_load_nvfuser_shell${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
_load_nvfuser_shell
EOF
source ~/.bashrc
```

**For zsh (Option 1 - Simple):**
```bash
cat >> ~/.zshrc << 'EOF'

# nvFuser: Auto-load configuration tool
chpwd() {
  [[ -f .nvfuser-shell ]] && source .nvfuser-shell
}
[[ -f .nvfuser-shell ]] && source .nvfuser-shell
EOF
source ~/.zshrc
```

**For zsh (Option 2 - If you have existing chpwd hooks):**
```bash
cat >> ~/.zshrc << 'EOF'

# nvFuser: Auto-load configuration tool
_load_nvfuser_shell() { [[ -f .nvfuser-shell ]] && source .nvfuser-shell; }
chpwd_functions+=(_load_nvfuser_shell)
_load_nvfuser_shell
EOF
source ~/.zshrc
```

**Verify:** `cd` into the Fuser directory, then run `type nvfuser-configure` - you should see the function definition.

**Benefits:**
- ✅ Portable (no hard-coded paths)
- ✅ Project-scoped (only loaded in nvFuser directories)
- ✅ "Apply" feature works (modifies current shell)
- ✅ Works across all your nvFuser checkouts
- ✅ Fast (guard in .nvfuser-shell prevents re-loading)
- ✅ Clean (doesn't pollute global namespace when outside project)


**Verify:** `type nvfuser-configure` should show the function definition.

**Note:** Running this multiple times will add duplicate lines. If you need to remove duplicates later, edit your RC file manually.

## Manual Setup

If you prefer not to use the quick setup, add these lines to your `~/.bashrc` (bash) or `~/.zshrc` (zsh):

**For bash:**
```bash
_load_nvfuser_shell() { [[ -f .nvfuser-shell ]] && source .nvfuser-shell; }
PROMPT_COMMAND="_load_nvfuser_shell${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
_load_nvfuser_shell
```

**For zsh (simple):**
```bash
chpwd() { [[ -f .nvfuser-shell ]] && source .nvfuser-shell; }
[[ -f .nvfuser-shell ]] && source .nvfuser-shell
```

**For zsh (with existing hooks):**
```bash
_load_nvfuser_shell() { [[ -f .nvfuser-shell ]] && source .nvfuser-shell; }
chpwd_functions+=(_load_nvfuser_shell)
_load_nvfuser_shell
```

Then reload your shell:
```bash
source ~/.bashrc  # for bash
source ~/.zshrc   # for zsh
```

This is portable and will work for all nvFuser checkouts on all machines.

## How It Works

The setup line `[[ -f .nvfuser-shell ]] && source .nvfuser-shell` runs automatically in every directory you `cd` into. When you're in an nvFuser directory, it finds the `.nvfuser-shell` file and loads the `nvfuser-configure` function. When you leave the directory, the function remains available in that shell session but won't be loaded in new shells unless you're in an nvFuser directory.

This approach is:
- **Portable:** No hard-coded paths, works on any machine
- **Project-scoped:** Only activates in nvFuser directories
- **Fast:** Minimal overhead
- **Standard:** Similar to how `.env`, `.nvmrc`, and other tools work


## Usage

```bash
nvfuser-configure
```

**Navigation:**
- ↑↓ or j/k - Move up/down
- {} - Jump to prev/next section
- PgUp/PgDn - Jump to top/bottom
- **/** - Search for options (like vim)
- **n** - Jump to next search match
- **N** - Jump to previous search match

**Apply Workflow (Immediate):**
1. Navigate (↑↓), Toggle (Enter), Edit (e) to configure variables
2. Press **a** to apply
3. Confirm with **y**
4. TUI exits, configuration is immediately active in your shell

**Generate Workflow (Reusable Script):**
1. Navigate (↑↓), Toggle (Enter), Edit (e) to configure variables
2. Press **g** to generate
3. Enter filename (default: nvfuser_env.sh)
4. TUI exits with script saved
5. Apply anytime: `source nvfuser_env.sh`

## What It Manages

**Build-time** (NVFUSER_BUILD_*, MAX_JOBS):
- Build type (Release/Debug), enable/disable components (python, cutlass, tests, benchmarks)
- Compiler flags, directories, parallelism

**Environment** (CC, CXX, CUDA_HOME, etc.):
- C/C++ compiler selection, CUDA paths, target architectures, compiler/linker flags

**Runtime** (NVFUSER_DUMP, NVFUSER_ENABLE/DISABLE):
- Debug output (fusion_ir, cuda_kernel, ptx, sass, scheduler info)
- Features (fast_math, cutlass_scheduler, tma kernels)
- Profiler modes

## Key Features

- ✅ Green highlighting for configured values
- ✅ Smart quit prompt (never lose changes)
- ✅ Auto-detection of current environment
- ✅ **Apply** immediately updates environment and exits
- ✅ **Generate** creates reusable scripts with proper unset logic
- ✅ Generated scripts clean up unconfigured variables for reproducibility

## Quick Examples

```bash
# Example 1: Debug build (Apply immediately)
nvfuser-configure
# Set: build_type=Debug, fusion_ir=on, cuda_kernel=on
# Press a→y
# Ready to build with debug configuration!

# Example 2: Custom compiler (Generate reusable script)
nvfuser-configure
# Set: cc=gcc-13, cxx=g++-13, cuda_home=/usr/local/cuda-12.6
# Press g
# Enter: my_compiler_config.sh
source my_compiler_config.sh
# Use this script anytime you want this configuration

# Example 3: Search for options
nvfuser-configure
# Press /
# Type: matmul
# Press Enter to jump to first match
# Press n to cycle through matches (n for next, N for previous)

# Example 4: Inspect current configuration
env | grep -E '^(NVFUSER_|MAX_JOBS|CC|CXX|CUDA_HOME)' | sort

# Example 5: Clear specific variables manually
unset NVFUSER_DUMP NVFUSER_ENABLE
```

## Alternative (without shell setup)

```bash
python tools/env-config/configure_env.py              # TUI mode
python tools/env-config/configure_env.py --simple     # Prompt mode
# Note: Without shell function setup, you'll need to manually source generated scripts
# After pressing 'a': source .nvfuser_apply.*.sh  (generated with random suffix)
# After pressing 'g': source nvfuser_env.sh
```

## Understanding Generated Scripts

Generated scripts include both:
- **unset** commands for unconfigured variables (ensures clean slate)
- **export** commands for configured variables

Example generated script:
```bash
#!/bin/bash
# nvFuser Environment Configuration
# Generated by tools/configure_env.py

# Unset unconfigured variables
unset NVFUSER_BUILD_BUILD_TYPE
unset NVFUSER_DISABLE
unset NVFUSER_DUMP

# Export configured variables
export CUDA_HOME="/usr/local/cuda-12.6"
export MAX_JOBS="16"
export NVFUSER_ENABLE="fast_math,kernel_profile"
```

This ensures reproducibility - sourcing the script multiple times gives identical results.

## See Also

- Variable reference: `../../CLAUDE.md` (Runtime Configuration section)
- Quick reference: `ENV_QUICK_REFERENCE.txt`
- Main tools README: `../README.md`
