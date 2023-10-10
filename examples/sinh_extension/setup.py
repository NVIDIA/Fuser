# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
import pathlib
import importlib.util

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

nvfuser_lib_dir = str(pathlib.Path(importlib.util.find_spec('nvfuser').origin).parent/'lib')

setup(
    name="nvfuser_extension",
    ext_modules=[
        CUDAExtension(
            name="nvfuser_extension",
            pkg="nvfuser_extension",
            include_dirs=[nvfuser_csrc_dir, dynamic_type_dir, flatbuffers_dir],
            libraries=["nvfuser_codegenx"],
            library_dirs=[nvfuser_lib_dir],
            extra_link_args=[f"-Wl,-rpath,{nvfuser_lib_dir}"],
            sources=["main.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
