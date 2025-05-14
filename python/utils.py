# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import multiprocessing
import subprocess
import sys
import shutil
from dataclasses import dataclass
import setuptools.command.build_ext


@dataclass
class BuildConfig:
    cmake_only: bool = False
    build_setup: bool = True
    no_python: bool = False
    no_test: bool = False
    no_benchmark: bool = False
    no_ninja: bool = False
    build_with_ucc: bool = False
    build_with_asan: bool = False
    build_without_distributed: bool = False
    build_with_system_nvtx: bool = True
    explicit_error_check: bool = False
    overwrite_version: bool = False
    version_tag: str = None
    build_type: str = "Release"
    wheel_name: str = "nvfuser"
    build_dir: str = ""
    install_dir: str = ""
    install_requires: list = None
    extras_require: dict = None
    cpp_standard: int = 20

    def __post_init__(self):
        # dataclass cannot have mutable default values in the class definition
        if self.install_requires is None:
            self.install_requires = []
        if self.extras_require is None:
            self.extras_require = {}


def check_env_flag_bool_default(name: str, default: str = "") -> bool:
    if name not in os.environ:
        return default
    return os.getenv(name).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_env_flag_bool(name: str) -> bool:
    assert name in os.environ
    return os.getenv(name).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVFUSER build options", add_help=False
    )

    # Add arguments that don't go to setuptools
    parser.add_argument(
        "--cmake-only",
        dest="cmake_only",
        action="store_true",
        help="Only generate ./build directory with cmake setup",
    )
    parser.add_argument(
        "--no-python",
        dest="no_python",
        action="store_true",
        help="Skips python API target libnvfuser.so",
    )
    parser.add_argument(
        "--no-test",
        dest="no_test",
        action="store_true",
        help="Skips cpp tests test_nvfuser",
    )
    parser.add_argument(
        "--no-benchmark",
        dest="no_benchmark",
        action="store_true",
        help="Skips benchmark target nvfuser_bench",
    )
    parser.add_argument(
        "--no-ninja",
        dest="no_ninja",
        action="store_true",
        help="Use make instead of ninja for build",
    )
    parser.add_argument(
        "--build-with-ucc",
        dest="build_with_ucc",
        action="store_true",
        help="Build nvfuser with UCC support",
    )
    parser.add_argument(
        "--explicit-error-check",
        dest="explicit_error_check",
        action="store_true",
        help="Enable explicit error checking",
    )
    parser.add_argument(
        "--build-with-asan",
        dest="build_with_asan",
        action="store_true",
        help="Build with Address Sanitizer",
    )
    parser.add_argument(
        "--build-without-distributed",
        dest="build_without_distributed",
        action="store_true",
        help="Build nvfuser without multidevice support",
    )
    parser.add_argument(
        "--no-system-nvtx",
        dest="no_system_nvtx",
        action="store_true",
        help="Disable system NVTX",
    )
    parser.add_argument(
        "--debug",
        dest="debug_mode",
        action="store_true",
        help="Building nvfuser in debug mode",
    )
    parser.add_argument(
        "--debinfo",
        dest="debinfo_mode",
        action="store_true",
        help="Building nvfuser in release mode with debug info",
    )
    parser.add_argument(
        "--build-dir",
        dest="build_dir",
        type=str,
        default="",
        help="Specify in which directory to build nvfuser",
    )
    parser.add_argument(
        "--install-dir",
        dest="install_dir",
        type=str,
        default="",
        help="Specify in which directory to install nvfuser",
    )
    parser.add_argument(
        "-install_requires",
        dest="install_requires",
        type=str,
        help="Specify package required for installation",
    )
    parser.add_argument(
        "--extras_require",
        dest="extras_require",
        type=str,
        help="Specify extra requirements",
    )
    parser.add_argument(
        "-version-tag",
        dest="version_tag",
        type=str,
        help="Specify the tag for build nvfuser version",
    )
    parser.add_argument(
        "-wheel-name",
        dest="wheel_name",
        type=str,
        default="nvfuser",
        help="Specify the wheel name",
    )
    parser.add_argument(
        "--cpp",
        dest="cpp_standard",
        type=int,
        help="Specify the C++ standard to use",
        default=20,
    )

    # Use parse_known_args to separate our arguments from setuptools arguments
    args, forward_args = parser.parse_known_args()
    return args, forward_args


# Create BuildConfig using argparse
def create_build_config():
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
    return config, forward_args


# Override BuildConfig with environment variables. Only change if variable
# exists. Do not use default to override argparse.
def override_build_config_from_env(config):
    # Command line arguments don't work on PEP517 builds and will be silently ignored,
    # so we need to pass those options as environment variables instead.
    if "NVFUSER_BUILD_CMAKE_ONLY" in os.environ:
        config.cmake_only = get_env_flag_bool("NVFUSER_BUILD_CMAKE_ONLY")
    if "NVFUSER_BUILD_SETUP" in os.environ:
        config.build_setup = get_env_flag_bool("NVFUSER_BUILD_SETUP")
    if "NVFUSER_BUILD_NO_PYTHON" in os.environ:
        config.no_python = get_env_flag_bool("NVFUSER_BUILD_NO_PYTHON")
    if "NVFUSER_BUILD_NO_TEST" in os.environ:
        config.no_test = get_env_flag_bool("NVFUSER_BUILD_NO_TEST")
    if "NVFUSER_BUILD_NO_BENCHMARK" in os.environ:
        config.no_benchmark = get_env_flag_bool("NVFUSER_BUILD_NO_BENCHMARK")
    if "NVFUSER_BUILD_NO_NINJA" in os.environ:
        config.no_ninja = get_env_flag_bool("NVFUSER_BUILD_NO_NINJA")
    if "NVFUSER_BUILD_WITH_UCC" in os.environ:
        config.build_with_ucc = get_env_flag_bool("NVFUSER_BUILD_WITH_UCC")
    if "NVFUSER_BUILD_WITH_ASAN" in os.environ:
        config.build_with_asan = get_env_flag_bool("NVFUSER_BUILD_WITH_ASAN")
    if "NVFUSER_BUILD_WITHOUT_DISTRIBUTED" in os.environ:
        config.build_without_distributed = get_env_flag_bool(
            "NVFUSER_BUILD_WITHOUT_DISTRIBUTED"
        )
    if "NVFUSER_BUILD_WITH_SYSTEM_NVTX" in os.environ:
        config.build_with_system_nvtx = get_env_flag_bool(
            "NVFUSER_BUILD_WITH_SYSTEM_NVTX"
        )
    if "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK" in os.environ:
        config.explicit_error_check = get_env_flag_bool(
            "NVFUSER_BUILD_EXPLICIT_ERROR_CHECK"
        )
    if "NVFUSER_BUILD_OVERWRITE_VERSION" in os.environ:
        config.overwrite_version = get_env_flag_bool("NVFUSER_BUILD_OVERWRITE_VERSION")
    if "NVFUSER_BUILD_VERSION_TAG" in os.environ:
        config.version_tag = os.getenv("NVFUSER_BUILD_VERSION_TAG")
    if "NVFUSER_BUILD_BUILD_TYPE" in os.environ:
        config.build_type = os.getenv("NVFUSER_BUILD_BUILD_TYPE")
    if "NVFUSER_BUILD_WHEEL_NAME" in os.environ:
        config.wheel_name = os.getenv("NVFUSER_BUILD_WHEEL_NAME")
    if "NVFUSER_BUILD_DIR" in os.environ:
        config.build_dir = os.getenv("NVFUSER_BUILD_DIR")
    if "NVFUSER_BUILD_INSTALL_DIR" in os.environ:
        config.install_dir = os.getenv("NVFUSER_BUILD_INSTALL_DIR")
    if "NVFUSER_BUILD_INSTALL_REQUIRES" in os.environ:
        config.install_requires = os.getenv("NVFUSER_BUILD_INSTALL_REQUIRES").split(",")
    if "NVFUSER_BUILD_EXTRAS_REQUIRE" in os.environ:
        config.extras_require = eval(os.getenv("NVFUSER_BUILD_EXTRAS_REQUIRE"))
    if "NVFUSER_BUILD_CPP_STANDARD" in os.environ:
        config.cpp_standard = int(os.getenv("NVFUSER_BUILD_CPP_STANDARD"))
    if "NVFUSER_BUILD_VERSION_TAG" in os.environ:
        config.overwrite_version = True
        config.version_tag = os.getenv("NVFUSER_BUILD_VERSION_TAG")


def get_default_install_prefix():
    cwd = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cwd, "nvfuser_common")


class build_ext(setuptools.command.build_ext.build_ext):
    def __init__(self, *args, **kwargs):
        install_dir = kwargs.pop("install_dir", "")
        self.install_dir = install_dir if install_dir else get_default_install_prefix()
        super().__init__(*args, **kwargs)

    def copy_library(self, ext, library_name):
        # Copy files on necessity.
        filename = self.get_ext_filename(self.get_ext_fullname(ext.name))
        fileext = os.path.splitext(filename)[1]

        libnvfuser_path = os.path.join(
            os.path.join(self.install_dir, "lib"), f"{library_name}{fileext}"
        )
        assert os.path.exists(libnvfuser_path)
        install_dst = os.path.join(self.build_lib, filename)
        if not os.path.exists(os.path.dirname(install_dst)):
            os.makedirs(os.path.dirname(install_dst))
        self.copy_file(libnvfuser_path, install_dst)

    def build_extension(self, ext):
        if ext.name == "nvfuser._C":
            self.copy_library(ext, "libnvfuser")
        elif ext.name == "nvfuser_next._C_NEXT":
            self.copy_library(ext, "libnvfuser_next")
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


def get_cmake_bin():
    # TODO: double check cmake version here and retrieve later version if necessary
    return "cmake"


def cmake(config, relative_path):
    from tools.memory import get_available_memory_gb

    # make build directories
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmake_build_dir = (
        os.path.join(cwd, "build") if not config.build_dir else config.build_dir
    )
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)

    install_prefix = (
        config.install_dir if config.install_dir else get_default_install_prefix()
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

    def on_or_off(flag: bool) -> str:
        return "ON" if flag else "OFF"

    # generate cmake directory
    #
    # cmake options are sticky: when -DFOO=... isn't specified, cmake's option
    # FOO prefers the cached value over the default value. This behavior
    # confused me several times (e.g.
    # https://github.com/NVIDIA/Fuser/pull/4319) when I ran `pip install -e`,
    # so I chose to always pass these options even for default values. Doing so
    # explicitly overrides cached values and ensures consistent behavior across
    # clean and dirty builds.
    cmd_str = [
        get_cmake_bin(),
        pytorch_cmake_config,
        f"-DCMAKE_BUILD_TYPE={config.build_type}",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        f"-DNVFUSER_CPP_STANDARD={config.cpp_standard}",
        f"-DUSE_DISTRIBUTED={pytorch_use_distributed}",
        f"-DNVFUSER_BUILD_WITH_ASAN={on_or_off(config.build_with_asan)}",
        f"-DNVFUSER_STANDALONE_BUILD_WITH_UCC={on_or_off(config.build_with_ucc)}",
        f"-DNVFUSER_EXPLICIT_ERROR_CHECK={on_or_off(config.explicit_error_check)}",
        f"-DBUILD_TEST={on_or_off(not config.no_test)}",
        f"-DBUILD_PYTHON={on_or_off(not config.no_python)}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DBUILD_NVFUSER_BENCHMARK={on_or_off(not config.no_benchmark)}",
        f"-DNVFUSER_DISTRIBUTED={on_or_off(not config.build_without_distributed)}",
        f"-DUSE_SYSTEM_NVTX={on_or_off(config.build_with_system_nvtx)}",
        "-B",
        cmake_build_dir,
    ]
    if not config.no_ninja:
        cmd_str.append("-G")
        cmd_str.append("Ninja")
    cmd_str.append(relative_path)

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


def create_clean(relative_path):
    class clean(setuptools.Command):
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            import glob

            gitignore_path = os.path.join(relative_path, ".gitignore")
            assert os.path.exists(gitignore_path)
            with open(gitignore_path, "r") as f:
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

    return clean


def run(config, version_tag, relative_path):
    from setuptools import Extension, setup, find_packages

    # NOTE(crcrpar): Deliberately build basically two dynamic libraries here so that they can
    # be treated as "nvfuser_package_data". This function call will put the two of "nvfuser" and
    # "nvfuser_codegen" into "./nvfuser/lib", and the former will be "nvfuser._C".
    if config.build_setup:
        cmake(config, relative_path)
    if not config.cmake_only:
        # NOTE: package include files for cmake
        # TODO(crcrpar): Better avoid hardcoding `libnvfuser_codegen.so`
        # might can be treated by using `exclude_package_data`.
        nvfuser_common_package_data = [
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
            version=version_tag,
            url="https://github.com/NVIDIA/Fuser",
            description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
            packages=find_packages(),
            ext_modules=[
                Extension(name="nvfuser._C", sources=[]),
                Extension(name="nvfuser_next._C_NEXT", sources=[]),
            ],
            license_files=("LICENSE",),
            cmdclass={
                "bdist_wheel": build_whl,
                "build_ext": lambda *args, **kwargs: build_ext(
                    *args, install_dir=config.install_dir, **kwargs
                ),
                "clean": create_clean(relative_path),
            },
            package_data={
                "nvfuser_common": nvfuser_common_package_data,
            },
            install_requires=config.install_requires,
            extras_require={
                "test": ["numpy", "expecttest", "pytest"],
                **config.extras_require,
            },
            license="BSD-3-Clause",
        )
