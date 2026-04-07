<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# nvFuser Tools

This directory contains utilities and tools for working with nvFuser.

## Environment Configuration Tools

Located in `tools/env-config/` directory.

### Quick Setup

Add to your shell configuration file (ONE LINE):

```bash
# For bash: add to ~/.bashrc
# For zsh: add to ~/.zshrc
eval "$(/absolute/path/to/Fuser/tools/env-config/nvfuser-config-shell.sh)"
```

Then run: `nvfuser-configure`

**Note:** Works with bash, zsh, and other POSIX-compatible shells.

See [env-config/README.md](env-config/README.md) for detailed setup instructions.

### Interactive Configuration Tool (`env-config/configure_env.py`)

Interactive interface for configuring nvFuser's 49 build and runtime environment variables.

**Features:**
- TUI mode (similar to ccmake) with instant preview
- Apply configuration immediately to current shell
- Generate reusable scripts with proper variable cleanup
- Auto-detection of current environment state
- Simple prompt mode for non-curses systems

**Usage:**
```bash
# Interactive TUI (recommended)
nvfuser-configure

# Or without shell setup
python tools/env-config/configure_env.py

# Simple prompt mode
python tools/env-config/configure_env.py --simple

# Generate script from current environment
python tools/env-config/configure_env.py --generate-script
```

**Documentation:** [env-config/README.md](env-config/README.md)

### Quick Reference (`ENV_QUICK_REFERENCE.txt`)

Printable reference card for common environment variables and workflows.

```bash
cat tools/ENV_QUICK_REFERENCE.txt
```

## Code Generation Tools

### cpp-repro-gen.py

### cpp-repro-gen.py

Converts Python fusion definitions to C++ test code.

#### Usage

Prepare a python file containing only the fusion definition, for example `examples/repro.py`. Then run

```
python cpp-repro-gen.py < examples/repro.py > examples/repro.cpp
```

to get the generated C++ test.

Note that `cpp-repro-gen.py` has no knowledge about the actual sizes of the input tensors if they are symbolic sizes, so you will get code like:

```C++
auto t0 = at::randn({-1}, options);
```

You can either manually modify the test, or use `--symbolic_sizes` to specify the symbolic sizes.

Example:

```
python cpp-repro-gen.py --symbolic_sizes 768 768 1024 768 < examples/repro.py > examples/repro.cpp
```

### Codegen Diff Tools

See the `codediff` [subdirectory](codediff/README.md).

## Build and Development Tools

### Memory Utilities (`memory.py`)

Helper functions for checking available system memory during builds.

### Build Prerequisites (`prereqs/`)

Tools for checking and managing build dependencies.

### Version Generation (`gen_nvfuser_version.py`)

Generates version information for Python packages.

### Dependency Checking (`check_dependencies.py`)

Validates that all required dependencies are available.

## Examples

The `examples/` directory contains example scripts demonstrating tool usage:

- `configure_env_example.py` - Demonstrates environment configuration CLI and programmatic usage

## See Also

- [../CLAUDE.md](../CLAUDE.md) - Complete nvFuser documentation
- Build system: [../python/utils.py](../python/utils.py)
- CMake configuration: [../CMakeLists.txt](../CMakeLists.txt)
