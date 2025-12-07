# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import cuequivariance_ops_torch

from . import triangle_attention_flex

triangle_attention_flex = triangle_attention_flex.triangle_attention
triangle_attention_cuequivariance = cuequivariance_ops_torch.triangle_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triangle_attention requires CUDA"
)


def test_triangle_attention_matches_cuequivariance():
    torch.manual_seed(0)
    batch, n_tokens, n_heads = 2, 3, 2
    q_len, k_len, head_dim = 4, 5, 32
    device = torch.device("cuda")
    dtype = torch.float16

    q_base = torch.randn(
        batch, n_tokens, n_heads, q_len, head_dim, device=device, dtype=dtype
    )
    k_base = torch.randn(
        batch, n_tokens, n_heads, k_len, head_dim, device=device, dtype=dtype
    )
    v_base = torch.randn(
        batch, n_tokens, n_heads, k_len, head_dim, device=device, dtype=dtype
    )
    bias_base = torch.randn(
        batch, 1, n_heads, q_len, k_len, device=device, dtype=torch.float32
    )
    mask = torch.rand(batch, n_tokens, 1, 1, k_len, device=device) > 0.3

    out_flex, lse_flex, max_flex = triangle_attention_flex(
        q_base, k_base, v_base, bias_base, mask=mask, return_aux=True
    )
    out_cue, lse_cue, max_cue = triangle_attention_cuequivariance(
        q_base, k_base, v_base, bias_base, mask=mask, return_aux=True
    )

    torch.testing.assert_close(out_flex, out_cue, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(lse_flex, lse_cue, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(max_flex, max_cue, rtol=1e-3, atol=1e-4)
