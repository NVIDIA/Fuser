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
#   NVFUSER_BUILD_TYPE=Debug
#     Building nvfuser in debug mode
#
#   NVFUSER_BUILD_TYPE=RelwithDebInfo
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

import sys
from setuptools import Extension, setup, find_packages

from utils import (
    create_build_config,
    override_build_config_from_env,
    cmake,
    version_tag,
    build_whl,
    build_ext,
    create_clean,
)

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


def main():
    # NOTE(crcrpar): Deliberately build basically two dynamic libraries here so that they can
    # be treated as "nvfuser_package_data". This function call will put the two of "nvfuser" and
    # "nvfuser_codegen" into "./nvfuser/lib", and the former will be "nvfuser._C".
    if config.build_setup:
        cmake(config, relative_path="..")
    if not config.cmake_only:
        # NOTE: package include files for cmake
        # TODO(crcrpar): Better avoid hardcoding `libnvfuser_codegen.so`
        # might can be treated by using `exclude_package_data`.
        nvfuser_package_data = [
            "lib/libnvfuser_codegen.so",
            "include/nvfuser/*.h",
            "include/nvfuser/struct.inl",
            "include/nvfuser/C++20/type_traits",
            "include/nvfuser/device_lower/*.h",
            "include/nvfuser/device_lower/analysis/*.h",
            "include/nvfuser/device_lower/pass/*.h",
            "include/nvfuser/dynamic_type/*",
            "include/nvfuser/dynamic_type/C++20/*",
            "include/nvfuser/kernel_db/*.h",
            "include/nvfuser/multidevice/*.h",
            "include/nvfuser/ops/*.h",
            "include/nvfuser/ir/*.h",
            "include/nvfuser/python_frontend/*.h",
            "include/nvfuser/scheduler/*.h",
            "include/nvfuser/serde/*.h",
            "include/nvfuser/flatbuffers/*.h",
            "include/nvfuser/host_ir/*.h",
            "include/nvfuser/id_model/*.h",
            "share/cmake/nvfuser/NvfuserConfig*",
            # TODO(crcrpar): it'd be better to ship the following two binaries.
            # Would need some change in CMakeLists.txt.
            # "bin/test_nvfuser",
            # "bin/nvfuser_bench"
        ]

        setup(
            name=config.wheel_name,
            version=version_tag(config),
            url="https://github.com/NVIDIA/Fuser",
            description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
            packages=find_packages(),
            ext_modules=[Extension(name="nvfuser._C", sources=[])],
            license_files=("LICENSE",),
            cmdclass={
                "bdist_wheel": build_whl,
                "build_ext": build_ext,
                "clean": create_clean(relative_path=".."),
            },
            package_data={
                "nvfuser": nvfuser_package_data,
            },
            install_requires=config.install_requires,
            extras_require={
                "test": ["numpy", "expecttest", "pytest"],
                **config.extras_require,
            },
            license="BSD-3-Clause",
        )


if __name__ == "__main__":
    main()
