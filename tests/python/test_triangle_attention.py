# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

triangle_attention_cuequivariance = None
try:  # cuequivariance_ops_torch is optional; fall back to other impls when missing
    import cuequivariance_ops_torch
except Exception:  # pragma: no cover - best effort import
    cuequivariance_ops_torch = None
else:
    triangle_attention_cuequivariance = cuequivariance_ops_torch.triangle_attention

from . import triangle_attention_cudnn, triangle_attention_flex

triangle_attention_cudnn = triangle_attention_cudnn.triangle_attention
triangle_attention_flex = triangle_attention_flex.triangle_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triangle_attention requires CUDA"
)


def test_triangle_attention_matches_cuequivariance():
    torch.manual_seed(0)
    batch, n_tokens, n_heads = 2, 3, 2
    q_len, k_len, head_dim = 4, 5, 32
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn(
        batch, n_tokens, n_heads, q_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch, n_tokens, n_heads, k_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch, n_tokens, n_heads, k_len, head_dim, device=device, dtype=dtype
    )
    bias = torch.randn(
        batch, 1, n_heads, q_len, k_len, device=device, dtype=torch.float32
    )
    mask = torch.rand(batch, n_tokens, 1, 1, k_len, device=device) > 0.3

    impls = {
        "cudnn": triangle_attention_cudnn,
        "flex": triangle_attention_flex,
    }
    if triangle_attention_cuequivariance is not None:
        impls["cuequivariance"] = triangle_attention_cuequivariance

    outputs = {
        name: impl(q, k, v, bias, mask, return_aux=True) for name, impl in impls.items()
    }

    ref_name = next(iter(outputs))
    ref_out, ref_lse, ref_max = outputs[ref_name]

    for name, (out, lse, max_scores) in outputs.items():

        def msg(m):
            return f"[{name} vs {ref_name}]\n\n{m}"

        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3, msg=msg)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=1e-3, msg=msg)
        torch.testing.assert_close(max_scores, ref_max, rtol=1e-3, atol=1e-3, msg=msg)
