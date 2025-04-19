# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Usage:
# pip install --no-build-isolation --e python -v
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

import os
import shutil

import setuptools
from setuptools import Extension, setup, find_packages

from utils import (
    check_env_flag_bool,
    BuildConfig,
    cmake,
    version_tag,
    build_whl,
    build_ext,
)

# Command line arguments don't work on PEP517 builds and will be silently ignored,
# so we need to pass those options as environment variables instead.
# Create a BuildConfig from environment variables
config = BuildConfig(
    cmake_only=check_env_flag_bool("NVFUSER_BUILD_CMAKE_ONLY", False),
    build_setup=check_env_flag_bool("NVFUSER_BUILD_SETUP", True),
    no_python=check_env_flag_bool("NVFUSER_BUILD_NO_PYTHON", False),
    no_test=check_env_flag_bool("NVFUSER_BUILD_NO_TEST", False),
    no_benchmark=check_env_flag_bool("NVFUSER_BUILD_NO_BENCHMARK", False),
    no_ninja=check_env_flag_bool("NVFUSER_BUILD_NO_NINJA", False),
    build_with_ucc=check_env_flag_bool("NVFUSER_BUILD_WITH_UCC", False),
    build_with_asan=check_env_flag_bool("NVFUSER_BUILD_WITH_ASAN", False),
    build_without_distributed=check_env_flag_bool(
        "NVFUSER_BUILD_WITHOUT_DISTRIBUTED", False
    ),
    build_with_system_nvtx=check_env_flag_bool("NVFUSER_BUILD_WITH_SYSTEM_NVTX", True),
    explicit_error_check=check_env_flag_bool(
        "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK", False
    ),
    build_type=os.getenv("NVFUSER_BUILD_TYPE", "Release"),
    wheel_name=os.getenv("NVFUSER_BUILD_WHEEL_NAME", "nvfuser"),
    build_dir=os.getenv("NVFUSER_BUILD_DIR", ""),
    install_dir=os.getenv("NVFUSER_BUILD_INSTALL_DIR", ""),
    install_requires=os.getenv("NVFUSER_BUILD_INSTALL_REQUIRES", "").split(","),
    extras_require=eval(os.getenv("NVFUSER_BUILD_EXTRA_REQUIRES", "{}")),
    cpp_standard=int(os.getenv("NVFUSER_BUILD_CPP_STANDARD", 20)),
)

# Apply remaining options
if "NVFUSER_BUILD_VERSION_TAG" in os.environ:
    config.overwrite_version = True
    config.version_tag = os.getenv("NVFUSER_BUILD_VERSION_TAG")
else:
    config.overwrite_version = False
    config.version_tag = None

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

        with open("../.gitignore", "r") as f:
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
