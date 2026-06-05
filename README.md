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

2. Install system dependencies:

The build process requires a few utilities to be installed. The following is a
probably-incomplete list, expressed in terms of what must be added to Docker
image ubuntu:24.04. The commands should be run as `root`.

```bash
apt update # may get errors re: needing ca-certificates if running in a fresh Docker container
apt-get -y install ca-certificates
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt-get -y install \
  build-essential \
  cmake \
  cuda-toolkit-13-2 \
  git \
  libcurl4-openssl-dev \
  libedit-dev \
  libnccl-dev \
  libzstd-dev \
  llvm-dev \
  python3-pip \
  virtualenv \
  #
```

3. Set up CUDA

Some environment variables need to be set up to build against CUDA. An example
of how to do this is:

```bash
cat >> ~/.bashrc <<'ENDOFHERE'
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin":"${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
ENDOFHERE
```

Then log out and in, or otherwise restart your shell.

4. Create a Python virtual environment

By default, `pip` installs Python packages system-wide. This should never be
done on a system that uses a package-manager, and indeed recent versions of
`pip` refuse to do so, to prevent conflicts with system packages. The correct
approach is to create a Python virtual environment. `pip` supports this, and
it avoids conflicts with system-provided packages:

```bash
virtualenv venv
. ./venv/bin/activate
```

5. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Some build-time dependencies are missing from `requirements.txt`. To install
them, execute:

```bash
pip install setuptools wheel numpy
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

6. Optionally select a GPU architecture to build for

*IF* you are building within Docker, you will need to either:
- Import your GPU into the container (not described here), or
- Set environment variables to tell the build process which GPU architecture
  to compile for (see below).

To find your GPU architecture, run the following on the host:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

To tell the build process which GPU to build for:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

7. Build and install nvFuser:

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
python -c "import nvfuser_direct as nvfuser; print('nvFuser successfully imported from:', nvfuser.__file__)"
```

Run the Python test suite:

```bash
pytest tests/python/
```

Run C++ tests (if built):

```bash
./build/bin/test_nvfuser
```
