# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Reference triangle attention implementation built on top of the cuDNN frontend
SDPA API.  The interface mirrors cuequivariance_ops_torch.triangle_attention so
that we can validate this path against the vendor kernel in unit tests.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import cudnn
import torch
from torch import Tensor

__all__ = ["triangle_attention"]

_NEG_INF = -1e9
_FORWARD_GRAPH_CACHE: dict[
    Tuple[
        int, torch.dtype, int, int, int, int, int, int, float, int, int, int, int, int
    ],
    "_CudnnSdpaGraph",
] = {}

_TORCH_TO_CUDNN_DTYPE = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}


def _ensure_rank(tensor: Tensor, name: str, expected_dims: Sequence[int]) -> Tensor:
    if tensor.dim() in expected_dims:
        return tensor
    expected = ", ".join(str(dim) for dim in expected_dims)
    raise ValueError(
        f"{name} must have rank in ({expected}), but got tensor with shape {tuple(tensor.shape)}"
    )


def _ensure_5d(tensor: Tensor, name: str) -> Tensor:
    tensor = _ensure_rank(tensor, name, (4, 5))
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def _torch_dtype_to_cudnn(dtype: torch.dtype) -> cudnn.data_type:
    try:
        return _TORCH_TO_CUDNN_DTYPE[dtype]
    except KeyError as exc:
        raise TypeError(
            f"triangle_attention only supports float16, bfloat16, or float32 inputs, but got {dtype}"
        ) from exc


def _graph_key(
    device_index: int,
    dtype: torch.dtype,
    batch: int,
    n_tokens: int,
    n_heads: int,
    q_len: int,
    k_len: int,
    head_dim: int,
    scale: float,
    bias_batch_broadcast: bool,
    bias_token_broadcast: bool,
    mask_present: bool,
    mask_batch_broadcast: bool,
    mask_token_broadcast: bool,
) -> Tuple[
    int, torch.dtype, int, int, int, int, int, int, float, int, int, int, int, int
]:
    return (
        device_index,
        dtype,
        batch,
        n_tokens,
        n_heads,
        q_len,
        k_len,
        head_dim,
        scale,
        int(bias_batch_broadcast),
        int(bias_token_broadcast),
        int(mask_present),
        int(mask_batch_broadcast),
        int(mask_token_broadcast),
    )


def _expand_bias_for_scores(
    bias: Tensor,
    batch: int,
    n_tokens: int,
    n_heads: int,
    q_len: int,
    k_len: int,
) -> Tuple[Tensor, bool, bool]:
    bias = bias.to(dtype=torch.float32)
    batch_broadcast = bias.shape[0] == 1
    token_broadcast = bias.shape[1] == 1
    expand_sizes = (
        batch if batch_broadcast else bias.shape[0],
        n_tokens if token_broadcast else bias.shape[1],
        n_heads,
        q_len,
        k_len,
    )
    bias_view = bias.expand(expand_sizes).reshape(
        batch * n_tokens, n_heads, q_len, k_len
    )
    return bias_view, batch_broadcast, token_broadcast


def _prepare_mask_bias(
    mask: Tensor,
    batch: int,
    n_tokens: int,
    n_heads: int,
    q_len: int,
    k_len: int,
) -> Tuple[Tensor, bool, bool]:
    mask_float = (~mask).to(dtype=torch.float32) * _NEG_INF
    batch_broadcast = mask_float.shape[0] == 1
    token_broadcast = mask_float.shape[1] == 1
    expand_sizes = (
        batch if batch_broadcast else mask_float.shape[0],
        n_tokens if token_broadcast else mask_float.shape[1],
        1,
        1,
        k_len,
    )
    expanded = mask_float.expand(expand_sizes)
    mask_flat = expanded.reshape(batch * n_tokens, 1, 1, k_len).expand(
        batch * n_tokens, n_heads, q_len, k_len
    )
    return mask_flat, batch_broadcast, token_broadcast


class _CudnnSdpaGraph:
    """Owns a compiled cuDNN graph for forward SDPA with the requested shape."""

    def __init__(
        self,
        key: Tuple[
            int,
            torch.dtype,
            int,
            int,
            int,
            int,
            int,
            int,
            float,
            int,
            int,
            int,
            int,
            int,
        ],
    ) -> None:
        (
            device_index,
            dtype,
            batch,
            n_tokens,
            n_heads,
            q_len,
            k_len,
            head_dim,
            scale,
            bias_batch_broadcast,
            bias_token_broadcast,
            mask_present,
            mask_batch_broadcast,
            mask_token_broadcast,
        ) = key
        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.scale = scale
        io_dtype = _torch_dtype_to_cudnn(dtype)
        self.has_mask = bool(mask_present)
        self.batch = batch
        self.n_tokens = n_tokens
        batch_tokens = batch * n_tokens

        with torch.cuda.device(device_index):
            self.handle = cudnn.create_handle()
            self.graph = cudnn.pygraph(
                handle=self.handle,
                name="triangle_attention_fwd",
                io_data_type=io_dtype,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )

            sample_q = torch.empty(
                (batch_tokens, n_heads, q_len, head_dim),
                device=self.device,
                dtype=dtype,
            )
            sample_k = torch.empty_like(sample_q)
            sample_v = torch.empty_like(sample_q)
            sample_bias = torch.empty(
                (batch_tokens, n_heads, q_len, k_len),
                device=self.device,
                dtype=torch.float32,
            )
            sample_stats = torch.empty(
                (batch_tokens, n_heads, q_len, 1),
                device=self.device,
                dtype=torch.float32,
            )

            self.q = self.graph.tensor_like(sample_q)
            self.k = self.graph.tensor_like(sample_k)
            self.v = self.graph.tensor_like(sample_v)
            self.bias = self.graph.tensor_like(sample_bias)
            self.score_max = self.graph.tensor_like(sample_stats)
            self.score_sum_exp = self.graph.tensor_like(sample_stats)

            def _make_score_mod(mask_tensor):
                def score_mod(
                    graph_obj: cudnn.pygraph, scores: cudnn.tensor
                ) -> cudnn.tensor:
                    return graph_obj.bias(scores, mask_tensor, name="mask_bias")

                return score_mod

            score_modifier = None
            if self.has_mask:
                sample_mask = torch.empty(
                    (batch_tokens, n_heads, q_len, k_len),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.mask_tensor = self.graph.tensor_like(sample_mask)
                score_modifier = _make_score_mod(self.mask_tensor)

            out, stats = self.graph.sdpa(
                q=self.q,
                k=self.k,
                v=self.v,
                bias=self.bias,
                attn_scale=self.scale,
                generate_stats=True,
                score_max=self.score_max,
                score_sum_exp=self.score_sum_exp,
                implementation=cudnn.attention_implementation.AUTO,
                score_mod=score_modifier,
            )

            out.set_output(True).set_dim(sample_q.shape).set_stride(sample_q.stride())
            stats.set_output(True).set_dim(sample_stats.shape).set_stride(
                sample_stats.stride()
            ).set_data_type(cudnn.data_type.FLOAT)
            self.score_max.set_output(True).set_dim(sample_stats.shape).set_stride(
                sample_stats.stride()
            )
            self.score_sum_exp.set_output(True).set_dim(sample_stats.shape).set_stride(
                sample_stats.stride()
            )

            self.graph.validate()
            self.graph.build_operation_graph()
            self.graph.create_execution_plans(
                [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
            )
            self.graph.check_support()
            self.graph.build_plans()

            workspace_bytes = max(1, self.graph.get_workspace_size())
            self.workspace = torch.empty(
                workspace_bytes, device=self.device, dtype=torch.uint8
            )

            self.out_tensor = out
            self.stats_tensor = stats

    def run(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
        out: Tensor,
        stats: Tensor,
        score_max: Tensor,
        score_sum_exp: Tensor,
        mask_bias: Optional[Tensor],
    ) -> None:
        variant_pack = {
            self.q: q,
            self.k: k,
            self.v: v,
            self.bias: bias,
            self.out_tensor: out,
            self.stats_tensor: stats,
            self.score_max: score_max,
            self.score_sum_exp: score_sum_exp,
        }
        if self.has_mask:
            assert self.mask_tensor is not None and mask_bias is not None
            variant_pack[self.mask_tensor] = mask_bias
        self.graph.execute(variant_pack, self.workspace, handle=self.handle)


class _TriangleAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
        mask: Optional[Tensor],
        scale: Optional[float],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        q_original_shape = tuple(q.shape)
        k_original_shape = tuple(k.shape)
        v_original_shape = tuple(v.shape)
        q = _ensure_5d(q, "q")
        k = _ensure_5d(k, "k")
        v = _ensure_5d(v, "v")

        if q.device.type != "cuda":
            raise ValueError(
                f"triangle_attention requires CUDA tensors, but got device {q.device}"
            )
        for name, tensor in (("k", k), ("v", v)):
            if tensor.device != q.device:
                raise ValueError(
                    f"{name} must be on {q.device}, but got {tensor.device}"
                )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError(
                f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
            )

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

        if scale is None:
            scale_val = 1.0 / math.sqrt(head_dim)
        else:
            scale_val = float(scale)

        bias_input = _ensure_rank(bias, "bias", (4, 5))
        bias_original_ndim = bias_input.dim()
        if bias_input.device != q.device:
            raise ValueError(f"bias must be on {q.device}, but got {bias_input.device}")
        if bias_input.dim() == 4:
            bias_input = bias_input.unsqueeze(0)
        if bias_input.shape[0] not in (1, batch):
            raise ValueError(
                f"bias batch dimension must be 1 or {batch}, but got {bias_input.shape[0]}"
            )
        if bias_input.shape[1] not in (1, n_tokens):
            raise ValueError(
                f"bias token dimension must be 1 or {n_tokens}, but got {bias_input.shape[1]}"
            )
        if bias_input.shape[2] != n_heads:
            raise ValueError(
                f"bias head dimension must match q ({n_heads}), but got {bias_input.shape[2]}"
            )
        if bias_input.shape[3] != q_len or bias_input.shape[4] != k_len:
            raise ValueError(
                "bias must have shape (B, N, H, Q, K) with matching Q/K dimensions"
            )
        bias_base_shape = tuple(bias_input.shape)
        (
            bias_view,
            bias_batch_broadcast,
            bias_token_broadcast,
        ) = _expand_bias_for_scores(bias_input, batch, n_tokens, n_heads, q_len, k_len)

        if mask is not None:
            mask = _ensure_rank(mask, "mask", (4, 5)).to(dtype=torch.bool)
            if mask.device != q.device:
                raise ValueError(f"mask must be on {q.device}, but got {mask.device}")
            if mask.dim() == 4:
                mask = mask.unsqueeze(0)
            if mask.shape[0] not in (1, batch):
                raise ValueError(
                    f"mask batch dimension must be 1 or {batch}, but got {mask.shape[0]}"
                )
            if mask.shape[1] not in (1, n_tokens):
                raise ValueError(
                    f"mask token dimension must be 1 or {n_tokens}, but got {mask.shape[1]}"
                )
            if mask.shape[2:] != (1, 1, k_len):
                raise ValueError(
                    "mask must have trailing shape (1, 1, K) to broadcast over heads and queries"
                )
            (
                mask_bias_view,
                mask_batch_broadcast,
                mask_token_broadcast,
            ) = _prepare_mask_bias(mask, batch, n_tokens, n_heads, q_len, k_len)
        else:
            mask_bias_view = None
            mask_batch_broadcast = False
            mask_token_broadcast = False

        batch_tokens = batch * n_tokens
        q_flat = q.reshape(batch_tokens, n_heads, q_len, head_dim).contiguous()
        k_flat = k.reshape(batch_tokens, n_heads, k_len, head_dim).contiguous()
        v_flat = v.reshape(batch_tokens, n_heads, k_len, head_dim).contiguous()

        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = _graph_key(
            device_index,
            q.dtype,
            batch,
            n_tokens,
            n_heads,
            q_len,
            k_len,
            head_dim,
            scale_val,
            bias_batch_broadcast,
            bias_token_broadcast,
            mask_bias_view is not None,
            mask_batch_broadcast,
            mask_token_broadcast,
        )

        runner = _FORWARD_GRAPH_CACHE.get(key)
        if runner is None:
            runner = _CudnnSdpaGraph(key)
            _FORWARD_GRAPH_CACHE[key] = runner

        out_flat = torch.empty_like(q_flat)
        stats_flat = torch.empty(
            batch_tokens, n_heads, q_len, 1, device=q.device, dtype=torch.float32
        )
        max_scores_flat = torch.empty_like(stats_flat)
        sum_exp_flat = torch.empty_like(stats_flat)

        runner.run(
            q_flat,
            k_flat,
            v_flat,
            bias_view,
            out_flat,
            stats_flat,
            max_scores_flat,
            sum_exp_flat,
            mask_bias_view if mask_bias_view is not None else None,
        )

        max_scores_values = max_scores_flat.squeeze(-1)
        sum_exp_values = sum_exp_flat.squeeze(-1)
        tiny = torch.finfo(sum_exp_values.dtype).tiny
        lse_flat = max_scores_values + torch.log(sum_exp_values.clamp_min(tiny))

        out = out_flat.reshape(batch, n_tokens, n_heads, q_len, head_dim)
        lse = lse_flat.reshape(batch, n_tokens, n_heads, q_len)
        max_scores = max_scores_values.reshape(batch, n_tokens, n_heads, q_len)

        return out, lse, max_scores

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        *grad_outputs,
    ):
        raise RuntimeError("triangle_attention_flex only supports forward mode")


def triangle_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
    return_aux: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """
    Triangle attention via cuDNN SDPA frontend.
    """

    out, lse, max_scores = _TriangleAttentionFunction.apply(q, k, v, bias, mask, scale)
    if return_aux:
        return out, lse, max_scores
    return out
