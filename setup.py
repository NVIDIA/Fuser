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
#   --debug
#     Building nvfuser in debug mode
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
PATCH_NVFUSER = True
OVERWRITE_VERSION = False
VERSION_TAG = None
BUILD_TYPE = "Release"
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
    if arg == "--debug":
        BUILD_TYPE = "Debug"
        continue
    if arg.startswith("-install_requires="):
        INSTALL_REQUIRES = arg.split("=")[1].split(",")
        continue
    if arg.startswith("-version-tag="):
        OVERWRITE_VERSION = True
        VERSION_TAG = arg.split("=")[1]
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
    def build_extensions(self):
        for i, ext in enumerate(self.extensions):
            if ext.name == "nvfuser._C":
                # NOTE: nvfuser pybind target is built with cmake, we remove the entry for ext_modules
                del self.extensions[i]

                # Copy nvfuser extension to proper file name
                fullname = self.get_ext_fullname("nvfuser._C")
                filename = self.get_ext_filename(fullname)
                fileext = os.path.splitext(filename)[1]
                cwd = os.path.dirname(os.path.abspath(__file__))
                src = os.path.join(cwd, "nvfuser", "lib", "libnvfuser" + fileext)
                dst = os.path.join(cwd, filename)
                if os.path.exists(src):
                    print(
                        "handling nvfuser pybind API, copying from {} to {}".format(
                            src, dst
                        )
                    )
                    self.copy_file(src, dst)

        setuptools.command.build_ext.build_ext.build_extensions(self)


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


def cmake():
    # make build directories
    build_dir_name = "build"
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmake_build_dir = os.path.join(cwd, "build")
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)

    # this is used to suppress import error.
    # so we can get the right pytorch prefix for cmake
    import logging

    logger = logging.getLogger("nvfuser")
    logger_level = logger.getEffectiveLevel()
    logger.setLevel(logging.CRITICAL)

    from tools.gen_nvfuser_version import get_pytorch_cmake_prefix
    pytorch_cmake_config = "-DCMAKE_PREFIX_PATH=" + get_pytorch_cmake_prefix()

    logger.setLevel(logger_level)

    # generate cmake directory
    cmd_str = [
        get_cmake_bin(),
        pytorch_cmake_config,
        "-DCMAKE_BUILD_TYPE=" + BUILD_TYPE,
        "-B",
        build_dir_name,
    ]
    if not NO_NINJA:
        cmd_str.append("-G")
        cmd_str.append("Ninja")
    cmd_str.append(".")
    if not NO_TEST:
        cmd_str.append("-DBUILD_TEST=ON")
    if not NO_PYTHON:
        cmd_str.append("-DBUILD_PYTHON=ON")
    if not NO_BENCHMARK:
        cmd_str.append("-DBUILD_NVFUSER_BENCHMARK=ON")

    subprocess.check_call(cmd_str)

    if not CMAKE_ONLY:
        # build binary
        max_jobs = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))
        cmd_str = [
            get_cmake_bin(),
            "--build",
            build_dir_name,
            "--target",
            "install",
            "--",
            "-j",
            max_jobs,
        ]
        subprocess.check_call(cmd_str)


def version_tag():
    from tools.gen_nvfuser_version import get_version

    version = get_version()
    if OVERWRITE_VERSION:
        version = version.split("+")[0]
        if len(VERSION_TAG) != 0:
            version = "+".join([version, VERSION_TAG])
    return version


def main():
    if BUILD_SETUP:
        cmake()

    if not CMAKE_ONLY:
        # NOTE: package include files for cmake
        nvfuser_package_data = [
            "*.so",
            "lib/*.so",
            "include/nvfuser/*.h",
            "include/nvfuser/kernel_db/*.h",
            "include/nvfuser/multidevice/*.h",
            "include/nvfuser/ops/*.h",
            "include/nvfuser/python_frontend/*.h",
            "include/nvfuser/scheduler/*.h",
            "include/nvfuser/serde*.h",
            "share/cmake/nvfuser/NvfuserConfig*",
        ]

        setup(
            name="nvfuser",
            version=version_tag(),
            url="https://github.com/NVIDIA/Fuser",
            description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
            packages=["nvfuser", "nvfuser_python_utils"],
            ext_modules=[Extension(name=str("nvfuser._C"), sources=[])],
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
            entry_points={
                "console_scripts": [
                    "patch-nvfuser = nvfuser_python_utils:patch_installation",
                ],
            },
            license="BSD-3-Clause",
        )

        if BUILD_SETUP and PATCH_NVFUSER:
            subprocess.check_call(["patch-nvfuser"])


if __name__ == "__main__":
    main()
