# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage:
# pip install --no-build-isolation -e python -v
#   This build command is equivalent to: python setup.py develop
#   Options:
#     -v: verbose output
#     --no-build-isolation: don't build in a temporary directory
#     -e: install in development mode
#
# Environment variables used during build:
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
#   NVFUSER_BUILD_CMAKE_ONLY
#     Only generate ./build directory with cmake setup
#
#   NVFUSER_BUILD_NO_PYTHON
#     Skips python API target `libnvfuser.so`, i.e. `_C.cpython-xxx.so`
#
#   NVFUSER_BUILD_NO_TEST
#     Skips cpp tests `test_nvfuser`
#
#   NVFUSER_BUILD_NO_BENCHMARK
#     Skips benchmark target `nvfuser_bench`
#
#   NVFUSER_BUILD_NO_NINJA
#     In case you want to use make instead of ninja for build
#
#   NVFUSER_BUILD_WITH_UCC
#     Build nvfuser with UCC support. You may need to specify environment variables of UCC_HOME, UCC_DIR, UCX_HOME, UCX_DIR.
#
#   NVFUSER_BUILD_WITHOUT_DISTRIBUTED
#     Build nvfuser without multidevice support
#
#   NVFUSER_BUILD_BUILD_TYPE=Debug
#     Building nvfuser in debug mode
#
#   NVFUSER_BUILD_BUILD_TYPE=RelwithDebInfo
#     Building nvfuser in release mode with debug info, a.k.a. RelwithDebInfo
#
#   NVFUSER_BUILD_DIR=<ABSOLUTE PATH>
#     Specify in which directory to build nvfuser. If not specified, the default build directory is "./build".
#
#   NVFUSER_BUILD_INSTALL_DIR=<ABSOLUTE PATH>
#     Specify in which directory to install nvfuser. If not specified, the default install directory is "./python/nvfuser".
#
#   NVFUSER_BUILD_VERSION_TAG=TAG
#     Specify the tag for build nvfuser version, this is used for pip wheel
#     package nightly where we might want to add a date tag
#     nvfuser-VERSION+TAG+gitSHA1-....-whl
#
#   NVFUSER_BUILD_INSTALL_REQUIRES=pkg0[,pkg1...]
#     this is used for pip wheel build to specify package required for install
#     e.g. NVFUSER_BUILD_INSTALL_REQUIRES=nvidia-cuda-nvrtc-cu12
#
#   NVFUSER_BUILD_WHEEL_NAME=NAME
#     Specify the wheel name this is used for pip wheel package where we want
#     to identify the cuda toolkit version
#
#   NVFUSER_BUILD_CPP_STANDARD=STANDARD
#     Specify the C++ standard to use for building nvfuser. The default is C++20.
#
#   NVFUSER_BUILD_WITH_LLVM
#     Build nvfuser with LLVM support

import sys

from utils import (
    run,
    create_build_config,
    override_build_config_from_env,
)


def version_tag(config):
    from tools.gen_nvfuser_version import get_version

    version = get_version()
    if config.overwrite_version:
        version = version.split("+")[0]
        if len(config.version_tag) != 0:
            # use "." to be pypi friendly
            version = ".".join([version, config.version_tag])
    return version


def main():
    # Parse arguments using argparse
    # Use argparse to create description of arguments from command line
    config, forward_args = create_build_config()

    # Override build config from environment variables
    override_build_config_from_env(config)

    if "clean" in sys.argv:
        # only disables BUILD_SETUP, but keep the argument for setuptools
        config.build_setup = False

    if config.cpp_standard < 20:
        raise ValueError("nvfuser requires C++20 standard or higher")

    sys.argv = [sys.argv[0]] + forward_args

    run(config, version_tag(config), relative_path="..")


if __name__ == "__main__":
    main()
