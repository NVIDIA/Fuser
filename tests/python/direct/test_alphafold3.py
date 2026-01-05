# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


# This file contains certain building blocks of the AlphaFold3 model.

import pytest
import torch
from dataclasses import dataclass
from enum import Enum, auto

from nvfuser_direct import FusionDefinition, DataType


@dataclass
class ModelConfig:
    c_z: int = 128
    c_hidden: int = 32
    n_heads: int = 4


_DEFAULT_CONFIG = ModelConfig()


class Direction(Enum):
    INCOMING = auto()  # aka ending node
    OUTGOING = auto()  # aka starting node


@pytest.mark.parametrize(
    "direction", [Direction.OUTGOING, Direction.INCOMING], ids=lambda d: d.name.lower()
)
def test_triangle_updates(direction):
    pass


# https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/#triangle-attention
@pytest.mark.parametrize(
    "direction", [Direction.OUTGOING, Direction.INCOMING], ids=lambda d: d.name.lower()
)
def test_triangle_attention(direction):
    c_z, c_hidden, h = (
        _DEFAULT_CONFIG.c_z,
        _DEFAULT_CONFIG.c_hidden,
        _DEFAULT_CONFIG.n_heads,
    )

    with FusionDefinition() as fd:
        z_in = fd.define_tensor(
            shape=[-1, -1, -1, c_z],
            dtype=DataType.BFloat16,
            contiguity=True,
        )  # [b, i, j, c_z]
        if direction == Direction.INCOMING:
            z_in = fd.ops.permute(z_in, [0, 2, 1, 3])
        w_q = fd.define_tensor(
            shape=[h * c_hidden, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_k = fd.define_tensor(
            shape=[h * c_hidden, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_b = fd.define_tensor(shape=[h, c_z], dtype=DataType.BFloat16, contiguity=True)
        mask = fd.define_tensor(
            shape=[-1, -1, -1], dtype=DataType.Bool, contiguity=True
        )  # [b, i, j]
        if direction == Direction.INCOMING:
            mask = fd.ops.permute(mask, [0, 2, 1])
        w_v = fd.define_tensor(
            shape=[h * c_hidden, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_g = fd.define_tensor(
            shape=[h * c_hidden, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_o = fd.define_tensor(
            shape=[c_z, h * c_hidden], dtype=DataType.BFloat16, contiguity=True
        )

        batch_size = fd.ops.size(z_in, 0)
        n_tokens = fd.ops.size(z_in, 1)

        q = fd.ops.linear(z_in, w_q)
        q_h = fd.ops.reshape(
            q, [batch_size, n_tokens, n_tokens, h, -1]
        )  # [b, i, j, h, c_hidden]
        q_h = fd.ops.permute(q_h, [0, 1, 3, 2, 4])  # [b, i, h, j, c_hidden]

        k = fd.ops.linear(z_in, w_k)
        k_h = fd.ops.reshape(
            k, [batch_size, n_tokens, n_tokens, h, -1]
        )  # [b, i, k, h, c_hidden]
        k_h = fd.ops.permute(k_h, [0, 1, 3, 2, 4])  # [b, i, h, k, c_hidden]

        b_h = fd.ops.linear(z_in, w_b)  # [b, j, k, h]
        b_h = fd.ops.permute(b_h, [0, 3, 1, 2])  # [b, h, j, k]
        b_h = fd.ops.broadcast_in_dim(
            b_h,
            shape=[batch_size, 1, h, n_tokens, n_tokens],
            broadcast_dims=[0, 2, 3, 4],
        )  # [b, 1, h, j, k]

        mask = fd.ops.broadcast_in_dim(
            mask,
            shape=[batch_size, n_tokens, 1, 1, n_tokens],
            broadcast_dims=[0, 1, 4],
        )  # [b, i, 1, 1, k]

        v = fd.ops.linear(z_in, w_v)
        v_h = fd.ops.reshape(
            v, [batch_size, n_tokens, n_tokens, h, -1]
        )  # [b, i, k, h, c_hidden]
        v_h = fd.ops.permute(v_h, [0, 1, 3, 2, 4])  # [b, i, h, k, c_hidden]

        # k_h.T: [b, i, h, c_hidden, k]
        # attention_matrix: [b, i, h, j, k]
        o_h, _, _, _ = fd.ops.sdpfa_fwd(
            q_h, k_h, v_h, bias=b_h, mask=mask, is_causal=False
        )  # [b, i, h, j, c_hidden]

        g = fd.ops.linear(z_in, w_g)
        g = fd.ops.sigmoid(g)
        g_h = fd.ops.reshape(
            g, [batch_size, n_tokens, n_tokens, h, -1]
        )  # [b, i, j, h, c_hidden]
        g_h = fd.ops.permute(g_h, [0, 1, 3, 2, 4])  # [b, i, h, j, c_hidden]

        o_h = fd.ops.mul(o_h, g_h)  # [b, i, h, j, c_hidden]
        o_h = fd.ops.cast(o_h, dtype=DataType.BFloat16)

        o = fd.ops.permute(o_h, [0, 1, 3, 2, 4])  # [b, i, j, h, c_hidden]
        o = fd.ops.reshape(
            o, [batch_size, n_tokens, n_tokens, -1]
        )  # [b, i, j, h * c_hidden]

        z_out = fd.ops.linear(o, w_o)  # [b, i, j, c_z]
        if direction == Direction.INCOMING:
            z_out = fd.ops.permute(z_out, [0, 2, 1, 3])
        fd.add_output(z_out)

    batch_size = 3
    n_tokens = 5
    z_in = torch.testing.make_tensor(
        batch_size, n_tokens, n_tokens, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_q = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_k = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_b = torch.testing.make_tensor(h, c_z, dtype=torch.bfloat16, device="cuda")
    mask = torch.testing.make_tensor(
        batch_size, n_tokens, n_tokens, dtype=torch.bool, device="cuda"
    )
    w_v = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_g = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_o = torch.testing.make_tensor(
        c_z, h * c_hidden, dtype=torch.bfloat16, device="cuda"
    )
    (z_out,) = fd.execute([z_in, w_q, w_k, w_b, mask, w_v, w_g, w_o])
    assert z_out.shape == (batch_size, n_tokens, n_tokens, c_z)
