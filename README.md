<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Fuser

A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser")

## Installation

We publish nightly wheel packages on https://pypi.nvidia.com

built-env | cuda 11.8 | cuda 12.1
:---: | :---: | :---:
torch 2.2 | nvfuser-cu118-torch22 | nvfuser-cu121-torch22
torch nightly wheel | nvfuser-cu118 | nvfuser-cu121

Note that nvfuser built against torch-2.2 isn't compatible with nightly pytorch wheel, so ensure you pick the right version suiting your environment.

### nightly nvfuser pip wheel

You can instll a nightly nvfuser pip package built against torch nightly code base with `pip install --pre nvfuser-cu121 --extra-index-url https://pypi.nvidia.com`

As we build against nightly torch wheel and there's no compatibility promised on nightly wheels, we have explicitly marked the nightly torch wheel as an optinoal dependency. You can choose to install the torch wheel along with nvfuser package. e.g.
`pip install --pre "nvfuser-cu121[torch]" --extra-index-url https://pypi.nvidia.com`.
Note that this may uninstall your local pytorch installation and install the compatible nightly pytorch.

### nvfuser pip wheel against pytorch stable release

nvfuser pip wheel built against stable torch releases is published on pypi.org. Pick the right cuda toolkit version to match your torch installation. e.g. `pip install nvfuser-cu121-torch22`

PyPI: [https://pypi.org/project/nvfuser/](https://pypi.org/search/?q=nvfuser)

## Developer

Docs: https://github.com/NVIDIA/Fuser/wiki

Supported compilers:

**GCC:**

We support all "supported releases" of gcc as specified in [the official site](https://gcc.gnu.org/).
As of 5/3/2024, they are:

- gcc 11.4
- gcc 12.3
- gcc 13.2
- gcc 14.1

**Clang:**

- clang 14+

Supported C++ standard:

- C++17
- C++20

We are actively considering dropping C++17 support
