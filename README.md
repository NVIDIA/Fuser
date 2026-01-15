<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Fuser

A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser")

## PyPI Installation
**PyPI:** [https://pypi.org/project/nvfuser](https://pypi.org/project/nvfuser)

nvFuser provides pre-built wheels for Python 3.10 and 3.12, available through
multiple channels depending on your PyTorch version requirements.

### Nightly Builds

Nightly `nvFuser` wheels are built against `PyTorch:nightly` and published to
`https://pypi.nvidia.com`:

```bash
pip install --pre nvfuser-cuXXY --extra-index-url https://pypi.nvidia.com
```
> [!note]
> nvFuser supports CUDA 12.6+. `cuXXY` denotes the CUDA major `XX` and minor
> `Y` version.  If you have CUDA 12.8 use `nvfuser-cu128`.

To install nvFuser with a compatible PyTorch nightly build:

```bash
pip install --pre "nvfuser-cu128[torch]" --extra-index-url https://pypi.nvidia.com
```

> [!warning]
> Installing with the `[torch]` extra will **replace** your existing PyTorch
> installation with a compatible nightly build.

### Stable Releases

Stable wheels are built against PyTorch stable releases and published to both
`https://pypi.org` and `https://pypi.nvidia.com`. Select the package matching your
CUDA Toolkit version:

```bash
pip install nvfuser-cu128-torch29
```

Releases are published on the 1st and 15th of each month, and when significant
changes are introduced. For legacy versions, see [PyPI](https://pypi.org/search/?q=nvfuser).

**Recommendation:** Use the latest nvFuser build with the most recent CUDA
Toolkit and PyTorch versions for optimal performance and features.

> [!important]
> Stable nvFuser release wheels are not guaranteed to be compatible with
> PyTorch nightly builds. Select the appropriate package for your environment.

## Building From Source

### Required:

- C++20 compliant compiler:
  - `GCC` >= `13.1` or `Clang` >= `19`
- `Python` >= `3.10`
- `CMake` >= `3.18`
- `Ninja`
- `CUDA Toolkit` >= `12.6` (recommend `12.8+`)
- `PyTorch` >= `2.9` (recommend latest `stable`/`nightly` release)
- `pybind11` >= `3.0`
- `LLVM` >= `18.1`

> [!note]
>
> - `PyTorch` **MUST** be built w/ `CUDA` support.
> - The `PyTorch CUDA version` **MUST** match the `CUDAToolkit version`.

### Optional:

- `nvidia-matmul-heuristics` (enhanced matmul scheduling)

### Build Steps

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
pip install -r requirements.txt
```

3. Build and install nvFuser:

```bash
pip install --no-build-isolation -e python -v
```

The build system will automatically validate all dependencies and provide
helpful error messages if anything is missing.

### Build Options

You can customize the build using environment variables:

**Build Configuration:**

- `MAX_JOBS=<n>` - Control compilation parallelism (e.g., `MAX_JOBS=8`)
- `NVFUSER_BUILD_BUILD_TYPE` - Build in (`Debug`/`RelWithDebInfo`/`Release`)
  mode.
- `NVFUSER_BUILD_DIR=<path>` - Custom build directory (default:
  `./python/build`)
- `NVFUSER_BUILD_INSTALL_DIR=<path>` - Custom install directory (default:
  `./nvfuser`)

**Build Targets:**

- `NVFUSER_BUILD_NO_PYTHON=1` - Skip Python bindings.
- `NVFUSER_BUILD_NO_TEST=1` - Skip C++ tests.
- `NVFUSER_BUILD_NO_BENCHMARK=1` - Skip benchmarks.

**Advanced Options:**

- `NVFUSER_BUILD_WITH_UCC=1` - Enable UCC support for multi-device operations.
- `NVFUSER_BUILD_WITHOUT_DISTRIBUTED=1` - Build without multi-device support.
- `NVFUSER_BUILD_CPP_STANDARD=<n>` - Specify C++ standard (default: 20).

Example with custom options:

```bash
MAX_JOBS=8 NVFUSER_BUILD_BUILD_TYPE=Debug pip install --no-build-isolation -e python -v
```

### Verifying the Installation

Test your installation with a simple fusion:

```python
python -c "import nvfuser; print('nvFuser successfully imported from:', nvfuser.__file__)"
```

Run the Python test suite:

```bash
pytest tests/python/
```

Run C++ tests (if built):

```bash
./build/bin/test_nvfuser
```
