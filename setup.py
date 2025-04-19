# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Environment variables used during build:
#
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
# build argument:
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

import multiprocessing
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_ext
from setuptools import Extension, setup, find_packages

from python.utils import parse_args, BuildConfig

# Parse arguments and set global variables accordingly
args, forward_args = parse_args()


# Create a BuildConfig from args
config = BuildConfig(
    cmake_only=args.cmake_only,
    no_python=args.no_python,
    no_test=args.no_test,
    no_benchmark=args.no_benchmark,
    no_ninja=args.no_ninja,
    build_with_ucc=args.build_with_ucc,
    build_with_asan=args.build_with_asan,
    build_without_distributed=args.build_without_distributed,
    build_with_system_nvtx=not args.no_system_nvtx,
    explicit_error_check=args.explicit_error_check,
    wheel_name=args.wheel_name,
    build_dir=args.build_dir,
    install_dir=args.install_dir,
    cpp_standard=args.cpp_standard,
)

# Apply remaining options
if args.debug_mode:
    config.build_type = "Debug"
if args.debinfo_mode:
    config.build_type = "RelwithDebInfo"
if args.install_requires:
    config.install_requires = args.install_requires.split(",")
if args.extras_require:
    config.extras_require = eval(args.extras_require)
if args.version_tag:
    config.version_tag = args.version_tag
    config.overwrite_version = True

# append forward_args to sys.argv
# forward_args is a list of arguments that are not handled by argparse
sys.argv.extend(forward_args)


if config.cpp_standard < 20:
    raise ValueError("nvfuser requires C++20 standard or higher")


def get_cmake_bin():
    # TODO: double check cmake version here and retrieve later version if necessary
    return "cmake"


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


class build_ext(setuptools.command.build_ext.build_ext):
    def build_extension(self, ext):
        if ext.name == "nvfuser._C":
            # Copy files on necessity.
            filename = self.get_ext_filename(self.get_ext_fullname(ext.name))
            fileext = os.path.splitext(filename)[1]

            libnvfuser_path = os.path.join("./nvfuser/lib", f"libnvfuser{fileext}")
            assert os.path.exists(libnvfuser_path)
            install_dst = os.path.join(self.build_lib, filename)
            if not os.path.exists(os.path.dirname(install_dst)):
                os.makedirs(os.path.dirname(install_dst))
            self.copy_file(libnvfuser_path, install_dst)
        else:
            super().build_extension(ext)


class concat_third_party_license:
    def __init__(self, directory="third_party"):
        self.license_file = "LICENSE"
        self.directory = directory

    def __enter__(self):
        # read original license file
        with open(self.license_file, "r") as f:
            self.nvfuser_license_txt = f.read()

        licenses = {"LICENSE", "LICENSE.txt", "LICENSE.rst", "COPYING.BSD"}

        # aggregated license, we key on project name
        aggregated_license = {}
        for root, dirs, files in os.walk(self.directory):
            license = list(licenses & set(files))
            if license:
                project_name = root.split("/")[-1]
                # let's worry about multiple license when we see it.
                assert len(license) == 1
                license_entry = os.path.join(root, license[0])
                if project_name in aggregated_license:
                    # Only add it if the license is different
                    aggregated_license[project_name].append(license_entry)
                else:
                    aggregated_license[project_name] = [license_entry]
        return aggregated_license

    def __exit__(self, exception_type, exception_value, traceback):
        # restore original license file
        with open(self.license_file, "w") as f:
            f.write(self.nvfuser_license_txt)


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    build_whl = None
else:

    class build_whl(bdist_wheel):
        def run(self):
            with concat_third_party_license() as tp_licenses:
                if len(tp_licenses) != 0:
                    with open("LICENSE", "a") as f:
                        f.write("\n\n")
                        f.write(
                            "NVIDIA/fuser depends on libraries with license listed below:"
                        )

                for project_name, license_files in tp_licenses.items():
                    # check all license files are identical
                    with open(license_files[0], "r") as f:
                        license_ref = f.read()

                    def check_file(file_name):
                        with open(file_name, "r") as f:
                            return f.read() == license_ref

                    identical_flag = all(map(check_file, license_files[1:]))
                    if not identical_flag:
                        raise RuntimeError(
                            "inconsistent license found for project: ",
                            project_name,
                            " check its license files under: ",
                            license_files,
                        )

                    with open("LICENSE", "a") as f:
                        f.write("\n\nProject Name: " + project_name)
                        f.write("\nLicense Files:\n")
                        for file_name in license_files:
                            f.write("\t" + file_name)
                        f.write("\n" + license_ref)

                # generate whl before we restore LICENSE
                super().run()


def version_tag():
    from tools.gen_nvfuser_version import get_version

    version = get_version()
    if config.overwrite_version:
        version = version.split("+")[0]
        if len(config.version_tag) != 0:
            # use "." to be pypi friendly
            version = ".".join([version, config.version_tag])
    return version


def cmake():
    from tools.memory import get_available_memory_gb

    # make build directories
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmake_build_dir = (
        os.path.join(cwd, "python", "build")
        if not config.build_dir
        else config.build_dir
    )
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)

    install_prefix = (
        os.path.join(cwd, "python", "nvfuser")
        if not config.install_dir
        else config.install_dir
    )

    from tools.gen_nvfuser_version import (
        get_pytorch_cmake_prefix,
        get_pytorch_use_distributed,
    )

    # this is used to suppress import error.
    # so we can get the right pytorch prefix for cmake
    import logging

    logger = logging.getLogger("nvfuser")
    logger_level = logger.getEffectiveLevel()
    logger.setLevel(logging.CRITICAL)

    pytorch_cmake_config = "-DCMAKE_PREFIX_PATH=" + get_pytorch_cmake_prefix()

    logger.setLevel(logger_level)

    pytorch_use_distributed = get_pytorch_use_distributed()

    # generate cmake directory
    cmd_str = [
        get_cmake_bin(),
        pytorch_cmake_config,
        "-DCMAKE_BUILD_TYPE=" + config.build_type,
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        f"-DNVFUSER_CPP_STANDARD={config.cpp_standard}",
        f"-DUSE_DISTRIBUTED={pytorch_use_distributed}",
        "-B",
        cmake_build_dir,
    ]
    if config.build_with_ucc:
        cmd_str.append("-DNVFUSER_STANDALONE_BUILD_WITH_UCC=ON")
    if config.explicit_error_check:
        cmd_str.append("-DNVFUSER_EXPLICIT_ERROR_CHECK=ON")
    if not config.no_ninja:
        cmd_str.append("-G")
        cmd_str.append("Ninja")
    if not config.no_test:
        cmd_str.append("-DBUILD_TEST=ON")
    if not config.no_python:
        cmd_str.append("-DBUILD_PYTHON=ON")
        cmd_str.append(f"-DPython_EXECUTABLE={sys.executable}")
    if not config.no_benchmark:
        cmd_str.append("-DBUILD_NVFUSER_BENCHMARK=ON")
    if config.build_with_asan:
        cmd_str.append("-DNVFUSER_BUILD_WITH_ASAN=ON")
    if config.build_without_distributed:
        cmd_str.append("-DNVFUSER_DISTRIBUTED=OFF")
    if config.build_with_system_nvtx:
        cmd_str.append("-DUSE_SYSTEM_NVTX=ON")
    cmd_str.append(".")

    print(f"Configuring CMake with {' '.join(cmd_str)}")
    subprocess.check_call(cmd_str)

    max_jobs = multiprocessing.cpu_count()
    mem_gb_per_task = 3  # Currently compilation of nvFuser souce code takes ~3GB of memory per task, we should adjust this value if it changes in the future.
    available_mem = get_available_memory_gb()
    if available_mem > 0:
        max_jobs_mem = int(available_mem / mem_gb_per_task)
        max_jobs = min(max_jobs, max_jobs_mem)

    if not config.cmake_only:
        # build binary
        max_jobs = os.getenv("MAX_JOBS", str(max_jobs))
        print(f"Using {max_jobs} jobs for compilation")
        cmd_str = [
            get_cmake_bin(),
            "--build",
            cmake_build_dir,
            "--target",
            "install",
            "--",
            "-j",
            max_jobs,
        ]
        subprocess.check_call(cmd_str)


def main():
    # NOTE(crcrpar): Deliberately build basically two dynamic libraries here so that they can
    # be treated as "nvfuser_package_data". This function call will put the two of "nvfuser" and
    # "nvfuser_codegen" into "./nvfuser/lib", and the former will be "nvfuser._C".
    if config.build_setup:
        cmake()
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
            version=version_tag(),
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
