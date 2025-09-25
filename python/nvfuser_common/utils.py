# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os
import importlib.util


__all__ = [
    "cmake_prefix_path",
]


cmake_prefix_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "nvfuser_common",
    "share",
    "cmake",
    "nvfuser",
)


def find_include_directory():
    spec = importlib.util.find_spec("nvfuser_common")
    assert spec is not None, "Unable to find nvfuser_common. Use pip install nvfuser."
    import nvfuser_common

    module_directory = os.path.dirname(nvfuser_common.__file__)
    include_directory = os.path.join(module_directory, "include")
    assert os.path.exists(include_directory)
    return include_directory
