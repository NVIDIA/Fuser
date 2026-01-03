<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Fuser

A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser")

## pip install

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

## Install from source

```bash
git clone https://github.com/NVIDIA/Fuser.git
cd Fuser
pip install -r python/requirements.txt

pip install --no-build-isolation -e python -v
# If you are in the internal PJNL container, use `_bn` instead for sccache.
```

Supported compilers:

**GCC:**

We support all "supported releases" of gcc as specified in [the official site](https://gcc.gnu.org/).
As of 3/2/2025, they are:

- gcc 12.4
- gcc 13.3
- gcc 14.2

**Clang:**

- clang 16+

Supported C++ standard:

- C++20

### Test

The easiest way to test a PR is to comment `!test` on the PR to trigger CI.

You can also run `tools/run_nvfuser_tests.py` to test locally.

## nvFuser Internals

https://github.com/NVIDIA/Fuser/blob/main/doc/README.md
