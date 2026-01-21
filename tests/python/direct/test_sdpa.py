# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import itertools
import math
import pytest
import torch
import torch.nn.functional as F
from enum import Enum, auto
from functools import partial

from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from python.direct_utils import (
    is_pre_ampere,
    define_sdpa_rng_state,
    verify_stride_order,
)


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
def test_softmax_logsumexp(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition) -> None:
        q = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            dtype=DataType.BFloat16,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            dtype=DataType.BFloat16,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            dtype=DataType.BFloat16,
        )
        (
            _,
            lse,
            *_,
        ) = fd.ops.sdpfa_fwd(q, k, v, dropout_p=None, is_causal=None, scale=None)
        fd.add_output(lse)

    n, h, l, s, e = 1, 1, 4, 4, 2
    inputs = [
        torch.ones((n, h, l, e), dtype=torch.bfloat16, device="cuda"),
        torch.ones((n, h, s, e), dtype=torch.bfloat16, device="cuda"),
        torch.ones((n, h, s, e), dtype=torch.bfloat16, device="cuda"),
    ]

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
            fusion_func,
            inputs,
        )
    # Ignoring size-1 dimensions, `q @ k^T / sqrt(e)` generates a `l`x`s`
    # matrix full of `sqrt(e)`s.  Therefore, the logsumexp of each row is
    # expected to be log(exp(sqrt(e)) * s) = log(s) + sqrt(e).
    torch.testing.assert_close(
        nvf_out[0].cpu(), torch.full((n, h, l), math.log(s) + e**0.5)
    )


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
def test_sdpa_fwd(nvfuser_direct_test):
    def fusion_func(
        fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
    ):
        q = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        dropout_p, is_causal, scale = None, None, None
        if has_dropout:
            dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
        if has_causal:
            is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
        if has_scale:
            scale = fd.define_scalar(value=None, dtype=DataType.Double)
        attn, *_ = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        fd.add_output(attn)

    N, H, L, S, E = 4, 8, 16, 16, 8
    qkv = [
        torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda"),
        torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda"),
        torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda"),
    ]

    dropout_vals = [None, 0.0, 0.2]
    is_causal_vals = [None, True, False]
    scale_vals = [None, 1 / E**0.5, 1e-3]
    # TODO: Try to move this to pytest_ops.py. Currently, it does not work since the API between nvFuser and torch differs.
    for dropout_p, is_causal, scale in itertools.product(
        dropout_vals, is_causal_vals, scale_vals
    ):
        with nvfuser_direct_test.subTest(
            dropout_p=dropout_p, is_causal=is_causal, scale=scale
        ):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            has_dropout = True if dropout_p is not None else False
            has_causal = True if is_causal is not None else False
            has_scale = True if scale is not None else False
            inputs = [*qkv]
            for param in [dropout_p, is_causal, scale]:
                if param is not None:
                    inputs.append(param)
            nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
                partial(
                    fusion_func,
                    has_dropout=has_dropout,
                    has_causal=has_causal,
                    has_scale=has_scale,
                ),
                inputs,
                new_fusion_expected=None,
            )

            # Torch does not accept NoneType dropout_p, is_causal.
            dropout_p = 0.0 if dropout_p is None else dropout_p
            is_causal = False if is_causal is None else is_causal

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                torch.manual_seed(0)
                ref_out = F.scaled_dot_product_attention(
                    *qkv, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )
            torch.testing.assert_close(nvf_out[0], ref_out)


# Memory layout of query, key, value and output tensors.
class Layout(Enum):
    NHSE = auto()
    NSHE = auto()


@pytest.mark.skipif(
    is_pre_ampere(),
    reason="Flash Attention is only supported on Ampere and newer devices.",
)
@pytest.mark.parametrize("layout", [Layout.NHSE, Layout.NSHE])
def test_sdpa_fwd_bias_mask(nvfuser_direct_test, layout: Layout):
    match layout:
        case Layout.NHSE:
            stride_order = [3, 2, 1, 0]
        case Layout.NSHE:
            stride_order = [3, 1, 2, 0]

    with FusionDefinition() as fd:
        q = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=stride_order,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=stride_order,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            stride_order=stride_order,
        )
        bias = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
        )
        mask = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.Bool,
        )
        attn, *_ = fd.ops.sdpfa_fwd(q, k, v, bias=bias, mask=mask)
        fd.add_output(attn)

    N, H, L, S, E = 11, 7, 5, 3, 2
    match layout:
        case Layout.NHSE:
            q = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda")
            k = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda")
            v = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda")
        case Layout.NSHE:
            q = torch.randn(
                (N, L, H, E), dtype=torch.bfloat16, device="cuda"
            ).transpose(1, 2)
            k = torch.randn(
                (N, S, H, E), dtype=torch.bfloat16, device="cuda"
            ).transpose(1, 2)
            v = torch.randn(
                (N, S, H, E), dtype=torch.bfloat16, device="cuda"
            ).transpose(1, 2)
    bias = torch.randn((N, H, L, S), dtype=torch.bfloat16, device="cuda")
    mask = torch.rand((N, H, L, S), device="cuda") > 0.3

    (out,) = fd.execute([q, k, v, bias, mask])

    attn_mask = (bias + torch.where(mask, 0.0, float("-inf"))).to(dtype=bias.dtype)
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask
    )
    torch.testing.assert_close(out, ref_out)
    verify_stride_order(out.stride(), stride_order)


def test_sdpa_bwd(nvfuser_direct_test):
    N, H, L, S, E = 4, 8, 16, 16, 8

    grad_output = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda")
    q = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda")

    dropout_vals = [None, 0.0, 0.2]
    is_causal_vals = [None, True, False]
    scale_vals = [None, 1 / E**0.5, 1e-3]

    def fusion_func(
        fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
    ):
        grad_output = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        q = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        output = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        logsumexp = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=True,
            dtype=DataType.Float,
            is_cpu=False,
        )
        philox_seed, philox_offset = define_sdpa_rng_state(fd)

        dropout_p, is_causal, scale = None, None, None
        if has_dropout:
            dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
        if has_causal:
            is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
        if has_scale:
            scale = fd.define_scalar(value=None, dtype=DataType.Double)

        grad_query, grad_key, grad_value = fd.ops.sdpfa_bwd(
            grad_output,
            q,
            k,
            v,
            output,
            logsumexp,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale,
        )
        fd.add_output(grad_query)
        fd.add_output(grad_key)
        fd.add_output(grad_value)

    for dropout_p, is_causal, scale in itertools.product(
        dropout_vals, is_causal_vals, scale_vals
    ):
        with nvfuser_direct_test.subTest(
            dropout_p=dropout_p, is_causal=is_causal, scale=scale
        ):
            # Torch does not accept NoneType dropout_p, is_causal.
            at_dropout_p = 0.0 if dropout_p is None else dropout_p
            at_is_causal = False if is_causal is None else is_causal

            (
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                query_seq_len,
                key_seq_len,
                philox_seed,
                philox_offset,
                _,
            ) = torch.ops.aten._scaled_dot_product_flash_attention(
                q,
                k,
                v,
                at_dropout_p,
                at_is_causal,
                return_debug_mask=False,
                scale=scale,
            )
            ref_grad = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                grad_output,
                q,
                k,
                v,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                query_seq_len,
                key_seq_len,
                at_dropout_p,
                at_is_causal,
                philox_seed,
                philox_offset,
                scale=scale,
            )

            has_dropout = True if dropout_p is not None else False
            has_causal = True if is_causal is not None else False
            has_scale = True if scale is not None else False

            inputs = [
                grad_output,
                q,
                k,
                v,
                output,
                logsumexp,
                philox_seed,
                philox_offset,
            ]
            for param in [dropout_p, is_causal, scale]:
                if param is not None:
                    inputs.append(param)

            nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
                partial(
                    fusion_func,
                    has_dropout=has_dropout,
                    has_causal=has_causal,
                    has_scale=has_scale,
                ),
                inputs,
                new_fusion_expected=None,
            )
            torch.testing.assert_close(nvf_out[0], ref_grad[0])
            torch.testing.assert_close(nvf_out[1], ref_grad[1])
            torch.testing.assert_close(nvf_out[2], ref_grad[2])


def test_sdpa_fwd_bwd(nvfuser_direct_test):
    N, H, L, S, E = 4, 8, 16, 16, 8

    dropout_vals = [None, 0.0, 0.2]
    is_causal_vals = [None, True, False]
    scale_vals = [None, 1 / E**0.5, 1e-3]

    def fusion_func(
        fd: FusionDefinition, has_dropout: bool, has_causal: bool, has_scale: bool
    ):
        q = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        grad_out = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=True,
            dtype=DataType.BFloat16,
            is_cpu=False,
        )

        dropout_p, is_causal, scale = None, None, None
        if has_dropout:
            dropout_p = fd.define_scalar(value=None, dtype=DataType.Double)
        if has_causal:
            is_causal = fd.define_scalar(value=None, dtype=DataType.Bool)
        if has_scale:
            scale = fd.define_scalar(value=None, dtype=DataType.Double)

        output, logsumexp, philox_seed, philox_offset = fd.ops.sdpfa_fwd(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
        grad_query, grad_key, grad_value = fd.ops.sdpfa_bwd(
            grad_out,
            q,
            k,
            v,
            output,
            logsumexp,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale,
        )

        fd.add_output(output)
        fd.add_output(grad_query)
        fd.add_output(grad_key)
        fd.add_output(grad_value)

    for dropout_p, is_causal, scale in itertools.product(
        dropout_vals, is_causal_vals, scale_vals
    ):
        with nvfuser_direct_test.subTest(
            dropout_p=dropout_p, is_causal=is_causal, scale=scale
        ):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            q = torch.randn(
                (N, H, L, E),
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            k = torch.randn(
                (N, H, S, E),
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            v = torch.randn(
                (N, H, S, E),
                dtype=torch.bfloat16,
                device="cuda",
                requires_grad=True,
            )
            grad_output = torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda")

            has_dropout = True if dropout_p is not None else False
            has_causal = True if is_causal is not None else False
            has_scale = True if scale is not None else False

            inputs = [q, k, v, grad_output]
            for param in [dropout_p, is_causal, scale]:
                if param is not None:
                    inputs.append(param)

            # Torch does not accept NoneType dropout_p, is_causal.
            dropout_p = 0.0 if dropout_p is None else dropout_p
            is_causal = False if is_causal is None else is_causal

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                torch.manual_seed(0)
                ref_out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )
                ref_out.backward(grad_output)

            nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
                partial(
                    fusion_func,
                    has_dropout=has_dropout,
                    has_causal=has_causal,
                    has_scale=has_scale,
                ),
                inputs,
                new_fusion_expected=None,
            )
            torch.testing.assert_close(nvf_out[0], ref_out)
            torch.testing.assert_close(nvf_out[1], q.grad)
            torch.testing.assert_close(nvf_out[2], k.grad)
            torch.testing.assert_close(nvf_out[3], v.grad)
