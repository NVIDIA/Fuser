# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage:
# [MAX_JOBS] python setup.py develop [args]
#
# Environment variables used during build:
#
#  MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
# NvFuser build arguments:
#
#   --cmake-only
#     Only generate ./build directory with cmake setup
#
#   --no-python
#     Skips python API target `libnvfuser.so`, i.e. `_C.cpython-xxx.so`
#
#   --no-test
#     Skips cpp tests `test_nvfuser`
#
#   --no-benchmark
#     Skips benchmark target `nvfuser_bench`
#
#   --no-ninja
#     In case you want to use make instead of ninja for build
#
#   --build-with-ucc
#     Build nvfuser with UCC support. You may need to specify environment variables of UCC_HOME, UCC_DIR, UCX_HOME, UCX_DIR.
#
#   --build-without-distributed
#     Build nvfuser without multidevice support
#
#   --debug
#     Building nvfuser in debug mode
#
#   --debinfo
#     Building nvfuser in release mode with debug info, a.k.a. RelwithDebInfo
#
#   --build-dir=<ABSOLUTE PATH>
#     Specify in which directory to build nvfuser. If not specified, the default build directory is "./build".
#
#   --install-dir=<ABSOLUTE PATH>
#     Specify in which directory to install nvfuser. If not specified, the default install directory is "./python/nvfuser".
#
#   -version-tag=TAG
#     Specify the tag for build nvfuser version, this is used for pip wheel
#     package nightly where we might want to add a date tag
#     nvfuser-VERSION+TAG+gitSHA1-....-whl
#
#   -install_requires=pkg0[,pkg1...]
#     this is used for pip wheel build to specify package required for install
#     e.g. -install_requires=nvidia-cuda-nvrtc-cu12
#
#   -wheel-name=NAME
#     Specify the wheel name this is used for pip wheel package where we want
#     to identify the cuda toolkit version
#
#   --cpp=STANDARD
#     Specify the C++ standard to use for building nvfuser. The default is C++20.
#

# TODO Remove nvfuser symbolic link to python/nvfuser
# TODO Remove tools/gen_nvfuser_version.py symbolic link to python/tools/gen_nvfuser_version.py
# TODO Remove tools/memory.py symbolic link to python/tools/memory.py

import sys


from python.utils import (
    run,
    create_build_config,
)


def version_tag(config):
    from python.tools.gen_nvfuser_version import get_version

    version = get_version()
    if config.overwrite_version:
        version = version.split("+")[0]
        if len(config.version_tag) != 0:
            # use "." to be pypi friendly
            version = ".".join([version, config.version_tag])
    return version


def main():
    # Parse arguments using argparse
    config, forward_args = create_build_config()

    if "clean" in sys.argv:
        # only disables BUILD_SETUP, but keep the argument for setuptools
        config.build_setup = False

    if config.cpp_standard < 20:
        raise ValueError("nvfuser requires C++20 standard or higher")

    sys.argv = [sys.argv[0]] + forward_args

    run(config, version_tag(config), relative_path=".")


if __name__ == "__main__":
    main()
