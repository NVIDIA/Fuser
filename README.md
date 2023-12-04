# Fuser

A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser")

## Installation

We publish nightly wheel packages on https://pypi.nvidia.com

built-env | cuda 11.8 | cuda 12.1
:---: | :---: | :---:
torch 2.1 | nvfuser-cu118-torch21 | nvfuser-cu121-torch21
torch nightly wheel | nvfuser-cu118 | nvfuser-cu121

Note that nvfuser built against torch-2.1 isn't compatible with nightly pytorch wheel, so ensure you pick the right version suiting your environment.

You can instll a given nvfuser version with `pip install --pre nvfuser-cu121 --extra-index-url https://pypi.nvidia.com`

As we build against nightly torch wheel and there's no compatibility promised on nightly wheels, we have explicitly marked the nightly torch wheel as an optinoal dependency. You can choose to install the torch wheel along with nvfuser package. e.g.
`pip install --pre "nvfuser-cu121[torch]" --extra-index-url https://pypi.nvidia.com`.
Note that this may uninstall your local pytorch installation and install the compatible nightly pytorch.

Versioned nvfuser will be published on pypi.org [WIP]

PyPI: [https://pypi.org/project/nvfuser/](https://pypi.org/search/?q=nvfuser)


## Developer

Getting started: https://github.com/NVIDIA/Fuser/wiki/Getting-started
Build: https://github.com/NVIDIA/Fuser/wiki/Building-fuser-project

Supported compilers:
- gcc 11.4+
- clang14+

Supported C++ standard:
- C++17
- C++20

We are actively considering dropping C++17 support
