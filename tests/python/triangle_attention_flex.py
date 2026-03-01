# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Triangle attention implementation backed by torch.nn.attention.flex_attention.
This mirrors cuequivariance_ops_torch.triangle_attention so it can be validated
against other implementations in the test suite.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from torch.nn.attention.flex_attention import (
    AuxRequest,
    flex_attention as _flex_attention_impl,
)

__all__ = ["triangle_attention"]

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _compile_flex_attention():
    try:
        return torch.compile(_flex_attention_impl, fullgraph=True)
    except Exception:
        return _flex_attention_impl


_flex_attention = _compile_flex_attention()


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, int, int, int, int, int]:
    if q.device.type != "cuda":
        raise ValueError(
            f"triangle_attention requires CUDA tensors, but got device {q.device}"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"triangle_attention only supports float16, bfloat16, or float32 inputs, but got {q.dtype}"
        )
    for name, tensor in (("k", k), ("v", v), ("bias", bias), ("mask", mask)):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, but got {tensor.device}")

    batch, n_tokens, n_heads, q_len, head_dim = q.shape
    k_batch, k_tokens, k_heads, k_len, k_dim = k.shape
    v_batch, v_tokens, v_heads, v_len, v_dim = v.shape

    if (batch, n_tokens, n_heads) != (k_batch, k_tokens, k_heads):
        raise ValueError(
            "q, k, and v must have matching batch, token, and head dimensions"
        )
    if (batch, n_tokens, n_heads) != (v_batch, v_tokens, v_heads):
        raise ValueError(
            "q, k, and v must have matching batch, token, and head dimensions"
        )
    if k_len != v_len:
        raise ValueError(
            f"k and v must have matching key sequence length, got {k_len} and {v_len}"
        )
    if k_dim != head_dim or v_dim != head_dim:
        raise ValueError(
            "q, k, and v must have the same head dimension on the last axis"
        )

    if bias.shape != (batch, 1, n_heads, q_len, k_len):
        raise ValueError(
            f"bias must have shape (B, 1, H, Q, K), but got {tuple(bias.shape)}"
        )
    if mask.shape != (batch, n_tokens, 1, 1, k_len):
        raise ValueError(
            f"mask must have shape (B, N, 1, 1, K), but got {tuple(mask.shape)}"
        )

    return batch, n_tokens, n_heads, q_len, k_len, head_dim


def triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    return_aux: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch, n_tokens, n_heads, q_len, k_len, head_dim = _validate_inputs(
        q, k, v, bias, mask
    )

    batch_tokens = batch * n_tokens
    scale = 1.0 / math.sqrt(head_dim)

    q_flat = q.reshape(batch_tokens, n_heads, q_len, head_dim).contiguous()
    k_flat = k.reshape(batch_tokens, n_heads, k_len, head_dim).contiguous()
    v_flat = v.reshape(batch_tokens, n_heads, k_len, head_dim).contiguous()

    bias_flat = bias.reshape(-1)
    mask_flat = mask.reshape(-1)
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=q.dtype)

    def _score_mod(
        scores: torch.Tensor,
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = batch_idx.to(dtype=torch.long)
        head_idx = head_idx.to(dtype=torch.long)
        q_idx = q_idx.to(dtype=torch.long)
        k_idx = k_idx.to(dtype=torch.long)
        b = torch.div(batch_idx, n_tokens, rounding_mode="floor")
        token = torch.remainder(batch_idx, n_tokens)

        bias_lin_idx = (((b * n_heads) + head_idx) * q_len + q_idx) * k_len + k_idx
        mask_lin_idx = ((b * n_tokens) + token) * k_len + k_idx
        # Gather through flattened views so we only track one linear index per tensor.
        # Dynamo cannot specialize the multi-axis indexing form bias[b, 0, h, qi, ki]
        # when indices depend on tensors, but index_select on a flat view is supported
        # and avoids materializing the expanded [B, N, H, Q, K] layout.
        gathered_bias = torch.index_select(
            bias_flat, 0, bias_lin_idx.reshape(-1)
        ).reshape_as(bias_lin_idx)
        gathered_mask = torch.index_select(
            mask_flat, 0, mask_lin_idx.reshape(-1)
        ).reshape_as(mask_lin_idx)
        modified = scores + gathered_bias
        return torch.where(gathered_mask, modified, neg_inf)

    aux_request = AuxRequest(lse=True, max_scores=True) if return_aux else None

    flex_out = _flex_attention(
        q_flat,
        k_flat,
        v_flat,
        score_mod=_score_mod,
        scale=scale,
        return_aux=aux_request,
    )

    if aux_request is None:
        out_flat = flex_out
        aux_output = None
    else:
        out_flat, aux_output = flex_out

    out = out_flat.reshape(batch, n_tokens, n_heads, q_len, head_dim)

    if return_aux:
        assert aux_output is not None
        lse = aux_output.lse.reshape(batch, n_tokens, n_heads, q_len)
        max_scores = aux_output.max_scores.reshape(batch, n_tokens, n_heads, q_len)
        return out, lse, max_scores
    return out
