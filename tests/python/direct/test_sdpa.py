# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import itertools
import math
import pytest
import torch
import torch.nn.functional as F
from functools import partial
from nvfuser_direct import (
    FusionDefinition,
    DataType,
)
from python.direct_utils import is_pre_ampere


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
            contiguity=[True, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        k = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        v = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
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
        attn, *_ = fd.ops.sdpfa_fwd(q, k, v, dropout_p, is_causal, scale)
        fd.add_output(attn)

    N, H, L, S, E = 4, 8, 16, 16, 8
    qkv = [
        torch.randn((N, H, L, E), dtype=torch.bfloat16, device="cuda:0"),
        torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
        torch.randn((N, H, S, E), dtype=torch.bfloat16, device="cuda:0"),
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
