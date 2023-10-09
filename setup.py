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
#     Skips cpp tests `nvfuser_tests`
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
#   --debug
#     Building nvfuser in debug mode
#
#   --debinfo
#     Building nvfuser in release mode with debug info, a.k.a. RelwithDebInfo
#
#   --build-dir=<ABSOLUTE PATH>
#     Specify in which directory to build nvfuser. If not specified, the default build directory is "./build".
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

import multiprocessing
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_ext
from setuptools import Extension, setup

# pick args used by this script
CMAKE_ONLY = False
BUILD_SETUP = True
NO_PYTHON = False
NO_TEST = False
NO_BENCHMARK = False
NO_NINJA = False
BUILD_WITH_UCC = False
BUILD_WITH_ASAN = False
PATCH_NVFUSER = True
OVERWRITE_VERSION = False
VERSION_TAG = None
BUILD_TYPE = "Release"
WHEEL_NAME = "nvfuser"
BUILD_DIR = ""
INSTALL_REQUIRES = []
forward_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--cmake-only":
        CMAKE_ONLY = True
        continue
    if arg == "--no-python":
        NO_PYTHON = True
        continue
    if arg == "--no-test":
        NO_TEST = True
        continue
    if arg == "--no-benchmark":
        NO_BENCHMARK = True
        continue
    if arg == "--no-ninja":
        NO_NINJA = True
        continue
    if arg == "--build-with-ucc":
        BUILD_WITH_UCC = True
        continue
    if arg == "--build-with-asan":
        BUILD_WITH_ASAN = True
        continue
    if arg == "--debug":
        BUILD_TYPE = "Debug"
        continue
    if arg == "--debinfo":
        BUILD_TYPE = "RelwithDebInfo"
        continue
    if arg.startswith("--build-dir"):
        BUILD_DIR = arg.split("=")[1]
        continue
    if arg.startswith("-install_requires="):
        INSTALL_REQUIRES = arg.split("=")[1].split(",")
        continue
    if arg.startswith("-version-tag="):
        OVERWRITE_VERSION = True
        VERSION_TAG = arg.split("=")[1]
        continue
    if arg.startswith("-wheel-name="):
        WHEEL_NAME = arg.split("=")[1]
        continue
    if arg in ["clean"]:
        # only disables BUILD_SETUP, but keep the argument for setuptools
        BUILD_SETUP = False
    if arg in ["bdist_wheel"]:
        # bdist_wheel doesn't install entry-points, so we can't really patch it yet
        PATCH_NVFUSER = False
    forward_args.append(arg)
sys.argv = forward_args


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
    if OVERWRITE_VERSION:
        version = version.split("+")[0]
        if len(VERSION_TAG) != 0:
            # use "." to be pypi friendly
            version = ".".join([version, VERSION_TAG])
    return version


from tools.memory import get_available_memory_gb


def cmake(install_prefix: str = "./nvfuser"):
    # make build directories
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmake_build_dir = os.path.join(cwd, "build") if not BUILD_DIR else BUILD_DIR
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)

    from tools.gen_nvfuser_version import get_pytorch_cmake_prefix

    # this is used to suppress import error.
    # so we can get the right pytorch prefix for cmake
    import logging

    logger = logging.getLogger("nvfuser")
    logger_level = logger.getEffectiveLevel()
    logger.setLevel(logging.CRITICAL)

    pytorch_cmake_config = "-DCMAKE_PREFIX_PATH=" + get_pytorch_cmake_prefix()

    logger.setLevel(logger_level)

    # generate cmake directory
    cmd_str = [
        get_cmake_bin(),
        pytorch_cmake_config,
        "-DCMAKE_BUILD_TYPE=" + BUILD_TYPE,
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        "-B",
        cmake_build_dir,
    ]
    if BUILD_WITH_UCC:
        cmd_str.append("-DNVFUSER_STANDALONE_BUILD_WITH_UCC=ON")
    if not NO_NINJA:
        cmd_str.append("-G")
        cmd_str.append("Ninja")
    if not NO_TEST:
        cmd_str.append("-DBUILD_TEST=ON")
    if not NO_PYTHON:
        cmd_str.append("-DBUILD_PYTHON=ON")
        cmd_str.append(f"-DPython_EXECUTABLE={sys.executable}")
    if not NO_BENCHMARK:
        cmd_str.append("-DBUILD_NVFUSER_BENCHMARK=ON")
    if BUILD_WITH_ASAN:
        cmd_str.append("-DNVFUSER_BUILD_WITH_ASAN=ON")
    cmd_str.append(".")

    print(f"Configuring CMake with {' '.join(cmd_str)}")
    subprocess.check_call(cmd_str)

    max_jobs = multiprocessing.cpu_count()
    mem_gb_per_task = 3  # Currently compilation of nvFuser souce code takes ~3GB of memory per task, we should adjust this value if it changes in the future.
    available_mem = get_available_memory_gb()
    if available_mem > 0:
        max_jobs_mem = int(available_mem / mem_gb_per_task)
        max_jobs = min(max_jobs, max_jobs_mem)

    if not CMAKE_ONLY:
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
    if BUILD_SETUP:
        cmake()
    if not CMAKE_ONLY:
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
            "share/cmake/nvfuser/NvfuserConfig*",
            "contrib/*",
            "contrib/nn/*",
            # TODO(crcrpar): it'd be better to ship the following two binaries.
            # Would need some change in CMakeLists.txt.
            # "bin/nvfuser_tests",
            # "bin/nvfuser_bench"
        ]

        setup(
            name=WHEEL_NAME,
            version=version_tag(),
            url="https://github.com/NVIDIA/Fuser",
            description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
            packages=["nvfuser", "nvfuser_python_utils"],
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
            install_requires=INSTALL_REQUIRES,
            extra_requires={
                "test": ["numpy", "expecttest", "pytest"],
            },
            license="BSD-3-Clause",
        )


if __name__ == "__main__":
    main()
