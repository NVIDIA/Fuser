# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


# This file contains certain building blocks of the AlphaFold3 model.

from dataclasses import dataclass

import torch

from nvfuser_direct import FusionDefinition, DataType


@dataclass
class ModelConfig:
    c_z: int = 128
    c_hidden: int = 32
    n_heads: int = 2


_DEFAULT_CONFIG = ModelConfig()


def test_triangle_updates_outgoing():
    pass


def test_triangle_updates_incoming():
    pass


# https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/#triangle-attention
def test_triangle_attention_starting_node():
    c_z, c_hidden, h = (
        _DEFAULT_CONFIG.c_z,
        _DEFAULT_CONFIG.c_hidden,
        _DEFAULT_CONFIG.n_heads,
    )

    with FusionDefinition() as fd:
        z_in = fd.define_tensor(
            shape=[-1, -1, -1, c_z], dtype=DataType.BFloat16
        )  # [b, i, j, c_z]
        w_q = fd.define_tensor(shape=[h * c_hidden, c_z], dtype=DataType.BFloat16)
        w_k = fd.define_tensor(shape=[h * c_hidden, c_z], dtype=DataType.BFloat16)
        w_b = fd.define_tensor(shape=[h, c_z], dtype=DataType.BFloat16)
        w_v = fd.define_tensor(shape=[h * c_hidden, c_z], dtype=DataType.BFloat16)
        w_g = fd.define_tensor(shape=[h * c_hidden, c_z], dtype=DataType.BFloat16)
        w_o = fd.define_tensor(shape=[c_z, h * c_hidden], dtype=DataType.BFloat16)

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
        )  # [b, i, j, h, c_hidden]
        k_h = fd.ops.permute(k_h, [0, 1, 3, 2, 4])  # [b, i, h, j, c_hidden]

        b_h = fd.ops.linear(z_in, w_b)

        v = fd.ops.linear(z_in, w_v)
        v_h = fd.ops.reshape(
            v, [batch_size, n_tokens, n_tokens, h, -1]
        )  # [b, i, j, h, c_hidden]
        v_h = fd.ops.permute(v_h, [0, 1, 3, 2, 4])  # [b, i, h, j, c_hidden]

        # TODO: b_h should be added here. fd.ops.sdpfa_fwd hasn't yet supported custom masks.
        o_h, _, _, _ = fd.ops.sdpfa_fwd(
            q_h, k_h, v_h, is_causal=False
        )  # [b, i, h, j, c_hidden]

        g = fd.ops.linear(z_in, w_g)
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
        fd.add_output(z_out)

    batch_size = 2
    n_tokens = 3
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
    w_v = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_g = torch.testing.make_tensor(
        h * c_hidden, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_o = torch.testing.make_tensor(
        c_z, h * c_hidden, dtype=torch.bfloat16, device="cuda"
    )
    (z_out,) = fd.execute([z_in, w_q, w_k, w_b, w_v, w_g, w_o])
    assert z_out.shape == (batch_size, n_tokens, n_tokens, c_z)


def test_triangle_attention_ending_node():
    pass
