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

built-env | cuda 11.8 | cuda 12.4 | cuda12.6 | cuda 12.8
:---: | :---: | :---: | :---: | :---: |
torch 2.6 | nvfuser-cu118-torch26 | nvfuser-cu124-torch26 | nvfuser-cu126-torch26 | N/A |
torch nightly (pypi.nvidia.com) | nvfuser-cu118 | N/A | nvfuser-cu126 | nvfuser-cu128 |

Note that nvfuser built against stable torch version isn't compatible with nightly pytorch wheel, so ensure you pick the right version suiting your environment.

### nightly nvfuser pip wheel

You can instll a nightly nvfuser pip package built against torch nightly code base with `pip install --pre nvfuser-cu121 --extra-index-url https://pypi.nvidia.com`

As we build against nightly torch wheel and there's no compatibility promised on nightly wheels, we have explicitly marked the nightly torch wheel as an optinoal dependency. You can choose to install the torch wheel along with nvfuser package. e.g.
`pip install --pre "nvfuser-cu121[torch]" --extra-index-url https://pypi.nvidia.com`.
Note that this may uninstall your local pytorch installation and install the compatible nightly pytorch.

### nvfuser pip wheel against pytorch stable release

nvfuser pip wheel built against stable torch releases is published on both pypi.org and pypi.nvidia.com. Pick the right cuda toolkit version to match your torch installation. e.g. `pip install nvfuser-cu121-torch26`

PyPI: [https://pypi.org/project/nvfuser/](https://pypi.org/search/?q=nvfuser)

## Developer

Docs: https://github.com/NVIDIA/Fuser/wiki

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
