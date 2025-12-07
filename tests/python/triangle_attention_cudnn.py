# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Reference triangle attention implementation built on top of the cuDNN frontend
SDPA API.  The interface mirrors cuequivariance_ops_torch.triangle_attention so
that we can validate this path against the vendor kernel in unit tests.
"""

import math
import cudnn
import torch
from typing import Tuple, Union

__all__ = ["triangle_attention"]

_TORCH_TO_CUDNN_DTYPE = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}


def _torch_dtype_to_cudnn(dtype: torch.dtype) -> cudnn.data_type:
    try:
        return _TORCH_TO_CUDNN_DTYPE[dtype]
    except KeyError as exc:
        raise TypeError(
            f"triangle_attention only supports float16, bfloat16, or float32 inputs, but got {dtype}"
        ) from exc


class _CudnnSdpaGraph:
    """Owns a compiled cuDNN graph for forward SDPA with the requested shape."""

    def __init__(
        self,
        device_index: int,
        dtype: torch.dtype,
        batch: int,
        n_tokens: int,
        n_heads: int,
        q_len: int,
        k_len: int,
        head_dim: int,
    ) -> None:
        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.scale = 1.0 / math.sqrt(head_dim)
        io_dtype = _torch_dtype_to_cudnn(dtype)
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
            sample_k = torch.empty(
                (batch_tokens, n_heads, k_len, head_dim),
                device=self.device,
                dtype=dtype,
            )
            sample_v = torch.empty_like(sample_k)
            sample_stats = torch.empty(
                (batch_tokens, n_heads, q_len, 1),
                device=self.device,
                dtype=torch.float32,
            )

            self.q = self.graph.tensor_like(sample_q)
            self.k = self.graph.tensor_like(sample_k)
            self.v = self.graph.tensor_like(sample_v)
            self.score_max = self.graph.tensor_like(sample_stats)
            self.score_sum_exp = self.graph.tensor_like(sample_stats)

            def _make_score_mod(bias, mask):
                def score_mod(
                    graph: cudnn.pygraph, scores: cudnn.tensor
                ) -> cudnn.tensor:
                    # scores: [B * N, H, Q, K]
                    # bias: [B, 1, H, Q, K]
                    # mask: [B, N, 1, 1, K]
                    scores = graph.reshape(scores)
                    scores.set_dim([batch, n_tokens, n_heads, q_len, k_len])

                    # TODO: bias and mask
                    scores = graph.bias(scores, bias)

                    scores = graph.reshape(scores)
                    scores.set_dim([batch * n_tokens, n_heads, q_len, k_len])
                    return scores

                return score_mod

            sample_bias = torch.empty(
                (batch, 1, n_heads, q_len, k_len),
                device=self.device,
                dtype=torch.float32,
            )
            self.bias = self.graph.tensor_like(sample_bias)
            sample_mask = torch.empty(
                (batch, n_tokens, 1, 1, k_len),
                device=self.device,
                dtype=torch.float32,
            )
            self.mask = self.graph.tensor_like(sample_mask)
            score_modifier = _make_score_mod(self.mask, self.bias)

            out, stats = self.graph.sdpa(
                q=self.q,
                k=self.k,
                v=self.v,
                attn_scale=self.scale,
                generate_stats=True,
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

            self.workspace = torch.empty(
                self.graph.get_workspace_size(), device=self.device, dtype=torch.uint8
            )

            self.out_tensor = out
            self.stats_tensor = stats

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        out: torch.Tensor,
        stats: torch.Tensor,
        score_max: torch.Tensor,
        score_sum_exp: torch.Tensor,
    ) -> None:
        variant_pack = {
            self.q: q,
            self.k: k,
            self.v: v,
            self.bias: bias,
            self.mask: mask,
            self.out_tensor: out,
            self.stats_tensor: stats,
            self.score_max: score_max,
            self.score_sum_exp: score_sum_exp,
        }
        self.graph.execute(variant_pack, self.workspace, handle=self.handle)


class _TriangleAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if bias.device != q.device:
            raise ValueError(f"bias must be on {q.device}, but got {bias.device}")
        if bias.shape != (batch, 1, n_heads, q_len, k_len):
            raise ValueError(
                f"bias must have shape (B, 1, H, Q, K), but got {tuple(bias.shape)}"
            )

        if mask.device != q.device:
            raise ValueError(f"mask must be on {q.device}, but got {mask.device}")
        if mask.shape != (batch, n_tokens, 1, 1, k_len):
            raise ValueError(
                f"mask must have shape (B, N, 1, 1, K), but got {tuple(mask.shape)}"
            )

        batch_tokens = batch * n_tokens
        q_flat = q.reshape(-1, n_heads, q_len, head_dim).contiguous()
        k_flat = k.reshape(-1, n_heads, k_len, head_dim).contiguous()
        v_flat = v.reshape(-1, n_heads, k_len, head_dim).contiguous()

        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        runner = _CudnnSdpaGraph(
            device_index,
            q.dtype,
            batch,
            n_tokens,
            n_heads,
            q_len,
            k_len,
            head_dim,
        )

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
            bias,
            mask,
            out_flat,
            stats_flat,
            max_scores_flat,
            sum_exp_flat,
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    return_aux: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Triangle attention via cuDNN SDPA frontend.
    """

    out, lse, max_scores = _TriangleAttentionFunction.apply(q, k, v, bias, mask)
    if return_aux:
        return out, lse, max_scores
    return out
