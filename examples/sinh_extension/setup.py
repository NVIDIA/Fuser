# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pathlib
import importlib.util
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise RuntimeError("PyTorch cpp_extension is required for building this package.")

nvfuser_csrc_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "csrc"
)
dynamic_type_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "dynamic_type", "src"
)
flatbuffers_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "third_party",
    "flatbuffers",
    "include",
)

# Ensure nvfuser is installed before trying to find its path
try:
    nvfuser_spec = importlib.util.find_spec("nvfuser_common")
    if nvfuser_spec is None or nvfuser_spec.origin is None:
        raise ImportError("Could not find nvfuser. Is it installed?")
    nvfuser_lib_dir = str(pathlib.Path(nvfuser_spec.origin).parent / "lib")
except ImportError as e:
    print(f"Error finding nvfuser path: {e}")
    print("Ensure 'nvfuser' is installed in the build environment.")
    raise e


# Inherit from PyTorch BuildExtension
# Modify build_extension from setuptools.command.build_ext to move shared
# library to nvfuser_extension package
class custom_build_ext(BuildExtension):
    def build_extension(self, ext):
        # Handle different os and cpu arch
        def _find_library_path():
            for item in os.listdir("build"):
                if item.startswith("lib"):
                    return item
            raise RuntimeException("Cannot find lib in build directory")

        # Call PyTorch BuildExtension first that overloads
        # setuptools.command.build_ext
        super().build_extension(ext)
        if ext.name == "nvfuser_extension":
            # Copy files on necessity.
            filename = self.get_ext_filename(self.get_ext_fullname(ext.name))
            fileext = os.path.splitext(filename)[1]

            extension_path = os.path.join("./build/", _find_library_path(), filename)
            assert os.path.exists(extension_path)

            install_dst = os.path.join("nvfuser_extension", filename)
            if not os.path.exists(os.path.dirname(install_dst)):
                os.makedirs(os.path.dirname(install_dst))
            self.copy_file(extension_path, install_dst)


setup(
    # Name is now in pyproject.toml
    # Version is now in pyproject.toml
    ext_modules=[
        CUDAExtension(
            name="nvfuser_extension",  # The name of the *compiled* module
            # pkg tells setuptools where the compiled module should go.
            # Assumes you have a Python package directory named 'nvfuser_extension'
            pkg="nvfuser_extension",
            include_dirs=[nvfuser_csrc_dir, dynamic_type_dir, flatbuffers_dir],
            libraries=["nvfuser_codegen"],
            library_dirs=[nvfuser_lib_dir],
            extra_link_args=[f"-Wl,-rpath,{nvfuser_lib_dir}"],
            sources=["main.cpp"],
            extra_compile_args={"cxx": ["-std=c++20"]},
        )
    ],
    # Keep cmdclass to use custom build extension logic
    cmdclass={"build_ext": custom_build_ext},
)
