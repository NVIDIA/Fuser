# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Tuple
from . import _C_DIRECT  # noqa: F401,F403
from ._C_DIRECT import nvf_cutlass  # noqa: F401,F403

assert torch.cuda.get_device_capability() >= (
    10,
    0,
), "Nvfp4 Requires compute capability of 10 or above."


def cutlass_nvfp4_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    nvf_cutlass.nvfp4_scaled_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out
