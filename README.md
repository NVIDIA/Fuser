<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Fuser

A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser")

## Installation

We publish nightly wheel packages on https://pypi.nvidia.com, while build against stable torch version on https://pypi.org and https://pypi.nvidia.com.
**Wheels are published for Python version: _3.10_, _3.12_**.

Note that nvfuser built against stable torch version isn't compatible with nightly pytorch wheel, so ensure you pick the right version suiting your environment.


### Nightly nvfuser pip wheel

You can install a nightly nvfuser pip package built against torch nightly code base with
`pip install --pre nvfuser-cu128 --extra-index-url https://pypi.nvidia.com`

As we build against nightly torch wheel and there's no compatibility promised on nightly wheels,
we have explicitly marked the nightly torch wheel as an optional dependency.
You can choose to install the torch wheel along with nvfuser package,
e.g.  `pip install --pre "nvfuser-cu128[torch]" --extra-index-url https://pypi.nvidia.com`.
Note that this may uninstall your local pytorch installation and install the compatible nightly pytorch.

### Nvfuser pip wheel against pytorch stable release

Nvfuser pip wheel built against stable torch releases is published on both https://pypi.org and https://pypi.nvidia.com.
Pick the right cuda toolkit version to match your torch installation. e.g. `pip install nvfuser-cu128-torch27`.

For old nvfuser builds against old version pytorch, e.g. `nvfuser-cuXXY-torchZW`,
there are packages available at [PyPI](https://pypi.org/search/?q=nvfuser).
We build and publish builds against the latest stable pytorch on https://pypi.org on 1st and 15th of every month regularly and
when major changes are added.

We always recommend use of the latest nvfuser build with latest cuda and pytorch versions.

PyPI: [https://pypi.org/project/nvfuser/](https://pypi.org/search/?q=nvfuser)

## Developer

Docs: https://github.com/NVIDIA/Fuser/wiki

### Building From Source

#### Prerequisites

**Required:**
- **C++ Compiler** with C++20 support:
  - GCC >= 13.1, or
  - Clang >= 19
- **Python** >= 3.10
- **CMake** >= 3.18
  * **Ninja** build system (recommended for faster builds)
- **CUDA Toolkit** >= 12.6 (recommend 12.8+)
- **PyTorch** >= 2.9 (recommend latest stable/nightly)
  - PyTorch **must** be built w/ CUDA
  - The PyTorch CUDA version must match your CUDA Toolkit version.
- **pybind11** >= 3.0
- **LLVM** >= 18.1

**Optional:**
- **nvidia-matmul-heuristics** (enhanced matmul scheduling)

#### Build Steps

1. Clone the repository and initialize submodules:
```bash
git clone --recursive https://github.com/NVIDIA/Fuser.git
cd Fuser
```

If you already cloned without `--recursive`, initialize submodules:
```bash
git submodule update --init --recursive
```

2. Install Python dependencies:
```bash
pip install -r python/requirements.txt
```

3. Build and install nvFuser:
```bash
pip install --no-build-isolation -e python -v
```

The build system will automatically validate all dependencies and provide helpful error messages if anything is missing.

#### Build Options

You can customize the build using environment variables:

**Build Configuration:**
- `MAX_JOBS=<n>` - Control compilation parallelism (e.g., `MAX_JOBS=8`)
- `NVFUSER_BUILD_BUILD_TYPE` - Build in (Debug/RelWithDebInfo/Release) mode
- `NVFUSER_BUILD_DIR=<path>` - Custom build directory
- `NVFUSER_BUILD_INSTALL_DIR=<path>` - Custom install directory

**Build Targets:**
- `NVFUSER_BUILD_NO_PYTHON=1` - Skip Python bindings
- `NVFUSER_BUILD_NO_TEST=1` - Skip C++ tests
- `NVFUSER_BUILD_NO_BENCHMARK=1` - Skip benchmarks

**Advanced Options:**
- `NVFUSER_BUILD_WITH_UCC=1` - Enable UCC support for multi-device operations
- `NVFUSER_BUILD_WITHOUT_DISTRIBUTED=1` - Build without multi-device support
- `NVFUSER_BUILD_NO_NINJA=1` - Use make instead of ninja
- `NVFUSER_BUILD_CPP_STANDARD=<n>` - Specify C++ standard (default: 20)

Example with custom options:
```bash
MAX_JOBS=8 NVFUSER_BUILD_BUILD_TYPE=Debug pip install --no-build-isolation -e python -v
```

#### Verifying the Installation

Test your installation:
```python
python -c "import nvfuser; print(nvfuser.__version__)"
```

Run the Python test suite:
```bash
pytest tests/python/
```

Run C++ tests (if built):
```bash
./build/bin/test_nvfuser
```
