# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


# This file contains certain building blocks of the AlphaFold3 model.

import pytest
import torch
from dataclasses import dataclass
from enum import Enum, auto

import nvfuser_direct as nvfuser
from nvfuser_direct import FusionDefinition, DataType, TensorView


@dataclass
class ModelConfig:
    c_z: int = 128
    c_hidden: int = 32
    n_heads: int = 4


_DEFAULT_CONFIG = ModelConfig()


class Direction(Enum):
    INCOMING = auto()  # aka ending node
    OUTGOING = auto()  # aka starting node


def layer_norm(
    fd: FusionDefinition, x: TensorView, w: TensorView, b: TensorView
) -> TensorView:
    io_dtype = x.dtype()
    x = fd.ops.cast(x, dtype=DataType.Float)
    var, mean = fd.ops.var_mean(x, dims=[-1], correction=0, keepdim=True)
    y = fd.ops.sub(x, mean)
    var = fd.ops.add(var, fd.define_scalar(1e-5))
    y = fd.ops.mul(y, fd.ops.rsqrt(var))
    shape = fd.ops.shape(x)
    w = fd.ops.broadcast_in_dim(w, shape=shape, broadcast_dims=[-1])
    y = fd.ops.mul(y, w)
    b = fd.ops.broadcast_in_dim(b, shape=shape, broadcast_dims=[-1])
    y = fd.ops.add(y, b)
    y = fd.ops.cast(y, dtype=io_dtype)
    return y


def gating(
    fd: FusionDefinition,
    z: TensorView,
    w_p: TensorView,
    z_in: TensorView,
    w_g: TensorView,
) -> TensorView:
    io_dtype = z.dtype()
    p = fd.ops.linear(z, w_p)
    g = fd.ops.linear(z_in, w_g)
    g = fd.ops.sigmoid(g)
    z = fd.ops.mul(p, g)
    return fd.ops.cast(z, dtype=io_dtype)


# https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/#triangle-updates
#
# Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure
# prediction with AlphaFold. Nature 596, 583–589 (2021).
# https://doi.org/10.1038/s41586-021-03819-2
# (see Supplementary Methods 1.6.5 for details)
@pytest.mark.mpi
@pytest.mark.parametrize(
    "direction", [Direction.OUTGOING, Direction.INCOMING], ids=lambda d: d.name.lower()
)
def test_triangle_updates(direction, multidevice_test):
    d = multidevice_test.size
    cp_size = 1
    if d % (cp_size * cp_size) != 0:
        pytest.skip(
            f"We only support even split, so {d} has to be divisible by {cp_size * cp_size} for {cp_size=}."
        )
    dp_size = d // (cp_size * cp_size)

    c_z = _DEFAULT_CONFIG.c_z

    with FusionDefinition() as fd:
        z_in_tv = fd.define_tensor(
            shape=[-1, -1, -1, c_z],
            dtype=DataType.BFloat16,
            contiguity=True,
        )  # [b, i, j, c_z]
        w_norm_in = fd.define_tensor(
            shape=[c_z], dtype=DataType.BFloat16, contiguity=True
        )
        b_norm_in = fd.define_tensor(
            shape=[c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_p_in = fd.define_tensor(
            shape=[c_z * 2, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_g_in = fd.define_tensor(
            shape=[c_z * 2, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_norm_out = fd.define_tensor(
            shape=[c_z], dtype=DataType.BFloat16, contiguity=True
        )
        b_norm_out = fd.define_tensor(
            shape=[c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_p_out = fd.define_tensor(
            shape=[c_z, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        w_g_out = fd.define_tensor(
            shape=[c_z, c_z], dtype=DataType.BFloat16, contiguity=True
        )
        # Masking is used in an internal implementation: http://nv/e-4
        mask_tv = fd.define_tensor(
            shape=[-1, -1, -1], dtype=DataType.Bool, contiguity=True
        )  # [b, i, j]

        batch_size = fd.ops.size(z_in_tv, 0)
        n_tokens = fd.ops.size(z_in_tv, 1)

        z_in = layer_norm(fd, z_in_tv, w_norm_in, b_norm_in)
        z = gating(fd, z_in_tv, w_p_in, z_in, w_g_in)
        mask = fd.ops.broadcast_in_dim(
            mask_tv,
            shape=[batch_size, n_tokens, n_tokens, c_z],
            broadcast_dims=[0, 1, 2],
        )
        z = fd.ops.where(mask, z, 0.0)
        a = fd.ops.slice(z, [0, 0, 0, 0], [batch_size, n_tokens, n_tokens, c_z])
        b = fd.ops.slice(z, [0, 0, 0, c_z], [batch_size, n_tokens, n_tokens, c_z * 2])

        match direction:
            case Direction.OUTGOING:
                # z_out = einsum("bikc,bjkc->bijc", a, b)
                a = fd.ops.permute(a, [0, 3, 1, 2])  # [b, c, i, k]
                b = fd.ops.permute(b, [0, 3, 2, 1])  # [b, c, k, j]
            case Direction.INCOMING:
                # z_out = einsum("bkic,bkjc->bijc", a, b)
                a = fd.ops.permute(a, [0, 3, 2, 1])  # [b, c, i, k]
                b = fd.ops.permute(b, [0, 3, 1, 2])  # [b, c, k, j]
        z = fd.ops.matmul(a, b)  # [b, c, i, j]
        z = fd.ops.permute(z, [0, 2, 3, 1])  # [b, i, j, c]

        z = layer_norm(fd, z, w_norm_out, b_norm_out)
        z = gating(fd, z, w_p_out, z_in, w_g_out)
        fd.add_output(z)

        mesh = nvfuser.multidevice.DeviceMesh(
            torch.arange(d).reshape(dp_size, cp_size, cp_size)
        )
        for tv in [
            z_in_tv,
            w_norm_in,
            b_norm_in,
            w_p_in,
            w_g_in,
            w_norm_out,
            b_norm_out,
            w_p_out,
            w_g_out,
            mask_tv,
        ]:
            tv.set_device_mesh(mesh)

        for tv in [z_in, mask]:
            tv.outer_split(2, cp_size)
            tv.axis(2).parallelize(nvfuser.ParallelType.mesh_x)
            tv.outer_split(1, cp_size)
            tv.axis(1).parallelize(nvfuser.ParallelType.mesh_y)
            tv.outer_split(0, dp_size)
            tv.axis(0).parallelize(nvfuser.ParallelType.mesh_z)

    batch_per_rank = 3
    n_tokens_per_rank = 5
    z_in_ref = torch.testing.make_tensor(
        batch_per_rank * dp_size,
        n_tokens_per_rank * cp_size,
        n_tokens_per_rank * cp_size,
        c_z,
        dtype=torch.bfloat16,
        device="cpu",
    )
    mask_ref = torch.testing.make_tensor(
        batch_per_rank * dp_size,
        n_tokens_per_rank * cp_size,
        n_tokens_per_rank * cp_size,
        dtype=torch.bool,
        device="cpu",
    )

    z_in = multidevice_test.shard_tensor(z_in_ref, z_in_tv)
    w_norm_in = torch.testing.make_tensor(c_z, dtype=torch.bfloat16, device="cuda")
    b_norm_in = torch.testing.make_tensor(c_z, dtype=torch.bfloat16, device="cuda")
    w_p_in = torch.testing.make_tensor(
        c_z * 2, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_g_in = torch.testing.make_tensor(
        c_z * 2, c_z, dtype=torch.bfloat16, device="cuda"
    )
    w_norm_out = torch.testing.make_tensor(c_z, dtype=torch.bfloat16, device="cuda")
    b_norm_out = torch.testing.make_tensor(c_z, dtype=torch.bfloat16, device="cuda")
    w_p_out = torch.testing.make_tensor(c_z, c_z, dtype=torch.bfloat16, device="cuda")
    w_g_out = torch.testing.make_tensor(c_z, c_z, dtype=torch.bfloat16, device="cuda")
    mask = multidevice_test.shard_tensor(mask_ref, mask_tv)
    (z_out,) = fd.execute(
        [
            z_in,
            w_norm_in,
            b_norm_in,
            w_p_in,
            w_g_in,
            w_norm_out,
            b_norm_out,
            w_p_out,
            w_g_out,
            mask,
        ]
    )
    assert z_out.shape == (batch_per_rank, n_tokens_per_rank, n_tokens_per_rank, c_z)
