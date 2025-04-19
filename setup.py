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

import os
import shutil
import sys

import setuptools
import setuptools.command.build_ext
from setuptools import Extension, setup, find_packages

from python.utils import (
    create_build_config,
    cmake,
    build_whl,
    build_ext,
)

# Parse arguments using argparse
config, forward_args = create_build_config()

if "clean" in sys.argv:
    # only disables BUILD_SETUP, but keep the argument for setuptools
    config.build_setup = False

if config.cpp_standard < 20:
    raise ValueError("nvfuser requires C++20 standard or higher")


class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob

        with open(".gitignore", "r") as f:
            ignores = f.read()
            for entry in ignores.split("\n"):
                # ignore comment in .gitignore
                if len(entry) >= 1 and entry[0] != "#":
                    for filename in glob.glob(entry):
                        print("removing: ", filename)
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


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
    # NOTE(crcrpar): Deliberately build basically two dynamic libraries here so that they can
    # be treated as "nvfuser_package_data". This function call will put the two of "nvfuser" and
    # "nvfuser_codegen" into "./nvfuser/lib", and the former will be "nvfuser._C".
    if config.build_setup:
        cmake(config, relative_path=".")
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
                "clean": clean,
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
