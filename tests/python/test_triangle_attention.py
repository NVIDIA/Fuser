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


def _clone_for_grad(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


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
    scale = 0.7

    q_flex, k_flex, v_flex, bias_flex = _clone_for_grad(
        q_base, k_base, v_base, bias_base
    )
    out_flex, lse_flex, max_flex = triangle_attention_flex(
        q_flex, k_flex, v_flex, bias_flex, mask=mask, scale=scale, return_aux=True
    )
    out_flex.sum().backward()

    q_cue, k_cue, v_cue, bias_cue = _clone_for_grad(q_base, k_base, v_base, bias_base)
    out_cue, lse_cue, max_cue = triangle_attention_cuequivariance(
        q_cue, k_cue, v_cue, bias_cue, mask=mask, scale=scale, return_aux=True
    )
    out_cue.sum().backward()

    torch.testing.assert_close(out_flex, out_cue, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(lse_flex, lse_cue, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(max_flex, max_cue, rtol=1e-3, atol=1e-4)

    for grad_flex, grad_cue in zip(
        (q_flex.grad, k_flex.grad, v_flex.grad, bias_flex.grad),
        (q_cue.grad, k_cue.grad, v_cue.grad, bias_cue.grad),
    ):
        torch.testing.assert_close(grad_flex, grad_cue, rtol=1e-3, atol=1e-4)


def test_triangle_attention_supports_batchless_inputs():
    torch.manual_seed(1)
    n_tokens, n_heads, q_len, k_len, head_dim = 4, 3, 5, 6, 16
    device = torch.device("cuda")

    q = torch.randn(
        n_tokens, n_heads, q_len, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        n_tokens, n_heads, k_len, head_dim, device=device, dtype=torch.float16
    )
    v = torch.randn(
        n_tokens, n_heads, k_len, head_dim, device=device, dtype=torch.float16
    )
    bias = torch.randn(1, n_heads, q_len, k_len, device=device, dtype=torch.float32)
    mask = torch.rand(n_tokens, 1, 1, k_len, device=device) > 0.5

    out_flex = triangle_attention_flex(q, k, v, bias, mask=mask)
    out_cue = triangle_attention_cuequivariance(q, k, v, bias, mask=mask)

    assert out_flex.shape == (1, n_tokens, n_heads, q_len, head_dim)
    assert out_cue.shape == out_flex.shape
    assert out_flex.dtype == q.dtype
    torch.testing.assert_close(out_flex, out_cue, rtol=1e-3, atol=1e-4)
