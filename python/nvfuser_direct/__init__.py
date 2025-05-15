# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

assert (
    "nvfuser" not in sys.modules
), "Cannot import nvfuser_direct if nvfuser module is already imported."

import os
import torch

# This is needed when libnvfuser_direct.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

from . import _C_DIRECT  # noqa: F401,F403
from ._C_DIRECT import *  # noqa: F401,F403
