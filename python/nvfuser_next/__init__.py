# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import torch

# This is needed when libnvfuser_next.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

from . import _C_NEXT  # noqa: F401,F403
from ._C_NEXT import *  # noqa: F401,F403
