# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import os


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
