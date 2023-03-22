import sys
import subprocess
import os
import multiprocessing
import shutil

import setuptools
from setuptools import setup
from distutils.version import LooseVersion
from distutils.file_util import copy_file

# pick args used by this script
CMAKE_ONLY = False
BUILD_SETUP = True
forward_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--cmake-only":
        CMAKE_ONLY = True
        continue
    if arg in ["clean"]:
        # only disables BUILD_SETUP, but keep the argument for setuptools
        BUILD_SETUP = False
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
        import re

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

                if not tp_licenses.empty():
                    with open("LICENSE", "a") as f:
                        f.write("\n\n")
                        f.write(
                            "NVIDIA/fuser depends on libraries with license listed below:"
                        )

                for project_name, license_files in tp_licenses:
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


def cmake():
    # make build directories
    build_dir_name = "build"
    cwd = os.path.dirname(os.path.abspath(__file__))
    cmake_build_dir = os.path.join(cwd, "build")
    if not os.path.exists(cmake_build_dir):
        os.makedirs(cmake_build_dir)

    import torch.utils

    pytorch_cmake_config = "-DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path

    # generate cmake directory
    cmd_str = [get_cmake_bin(), pytorch_cmake_config, "-B", build_dir_name, "."]
    subprocess.check_call(cmd_str)

    if not CMAKE_ONLY:
        # build binary
        max_jobs = str(multiprocessing.cpu_count())
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

        # copy nvfuser pybind extension
        src = os.path.join(cmake_build_dir, "libnvfuser.so")
        dst = os.path.join(cwd, "nvfuser", "_C.cpython-310-x86_64-linux-gnu.so")
        copy_file(src, dst)


def main():
    # NOTE: package include files for cmake
    nvfuser_package_data = [
        "*.so",
        "lib/*.so",
    ]

    if BUILD_SETUP:
        cmake()

    setup(
        name="nvfuser",
        # query nvfuser version
        version="0.0.1",
        description="A Fusion Code Generator for NVIDIA GPUs (commonly known as 'nvFuser')",
        packages=["nvfuser"],
        license_files=("LICENSE",),
        cmdclass={
            "bdist_wheel": build_whl,
            "clean": clean,
        },
        package_data={
            "nvfuser": nvfuser_package_data,
        },
    )


if __name__ == "__main__":
    main()
