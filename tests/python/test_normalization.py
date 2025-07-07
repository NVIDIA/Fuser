# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import torch
import torch.nn as nn

from nvfuser import FusionDefinition, DataType
from nvfuser.contrib.nn.normalization import InstanceNorm3dNVFuser


def assert_close(a: torch.Tensor, b: torch.Tensor):
    """Given two Tensors, compare with a reasonable precision.

    If the dtypes mismatch, use a custom rule to cast one or the other
    """
    # increasing order of precision
    precedence = [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    # demote inputs so we use the more permissive test
    if precedence.index(a.dtype) < precedence.index(b.dtype):
        b = b.type(a.dtype)
    else:
        a = a.type(b.dtype)

    if a.dtype in [torch.float16, torch.bfloat16]:
        # torch.nn.InstanceNorm3d fails rtol=6, atols=4e-2 for half precision
        torch.testing.assert_close(a, b, rtol=10, atol=5e-2)
    else:  # use default tolerance
        torch.testing.assert_close(a, b)


dtypes = {
    "float32": torch.float,
    "float64": torch.double,
    "float16": torch.half,
}
if torch.cuda.get_device_capability() >= (8, 0):
    dtypes["bfloat16"] = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,channel_size,spatial_size,compare",
    [
        (5, 7, 3, True),
        # check size=1 dimensions
        (1, 7, 3, True),  # NOTE: FAILS!
        (5, 1, 3, True),
        # (5, 7, 1, True), # eager instance norm needs more than one spatial element
        (1, 1, 3, True),
        # Don't check output for larger inputs, but check that they run
        # (16, 1, 64, False),
        # (16, 2, 64, False),
        # (1, 16, 64, False),
        # (2, 16, 64, False),
        # (16, 16, 64, False),
    ],
)
@pytest.mark.parametrize("memory_format", ["contiguous", "channels_last", "strided"])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("track_running_stats", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", dtypes.keys())
def test_instance_norm(
    dtype,
    training,
    track_running_stats,
    memory_format,
    affine,
    batch_size,
    channel_size,
    spatial_size,
    compare,
):
    dtype = dtypes[dtype]
    m = InstanceNorm3dNVFuser(
        channel_size,
        affine=affine,
        track_running_stats=track_running_stats,
        device="cuda",
        dtype=dtype,
    )
    reference_m = torch.nn.InstanceNorm3d(
        channel_size,
        affine=affine,
        track_running_stats=track_running_stats,
        device="cuda",
        dtype=torch.float64,
    )
    torch.manual_seed(42)
    for i in range(2):  # exercise JIT + caching
        inp = torch.rand(
            (
                batch_size,
                channel_size,
                spatial_size,
                spatial_size,
                2 * spatial_size if memory_format == "strided" else spatial_size,
            ),
            device="cuda",
            requires_grad=True,
            dtype=dtype,
        )
        if memory_format == "channels_last":
            inp = inp.to(memory_format=torch.channels_last_3d)
        elif memory_format == "strided":
            inp = inp[..., ::2]

        inp = inp.detach()
        inp.requires_grad = True

        inp2 = inp.clone().type(torch.float64).detach()
        inp2.requires_grad = True

        if training:
            m.train()
            reference_m.train()
        else:
            m.eval()
            reference_m.eval()

        out = m(inp)
        out2 = reference_m(inp2)
        if compare:
            assert_close(out, out2)

        if m.running_mean is None:
            assert reference_m.running_mean is None
            assert m.running_var is None
            assert reference_m.running_var is None
        else:
            if compare:
                assert_close(m.running_mean, reference_m.running_mean)

        if not training:
            return

        grad_out = torch.randn_like(inp)
        out.backward(grad_out)
        out2.backward(grad_out)
        if compare:
            assert_close(inp.grad, inp2.grad)

            # compare weight gradients
            if m.weight is not None:
                assert_close(m.weight.grad, reference_m.weight.grad)
            if m.bias is not None:
                assert_close(m.bias.grad, reference_m.bias.grad)


@pytest.mark.skip(
    reason="disable failing test, see https://github.com/NVIDIA/Fuser/issues/1728"
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="more than 1 GPU required")
def test_instance_norm_multigpu():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.norm = InstanceNorm3dNVFuser(4)

        def forward(self, x):
            x = self.norm(x)
            x = torch.sum(x, dim=(1, 2, 3, 4))
            return x

    device = torch.device("cuda:1")
    model = Model().to(device)

    x = torch.randn(2, 4, 128, 128, 128, device=device, requires_grad=True)
    y = torch.randn(2, device=device)
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y.float())
    loss.backward()


# Test that split extents are properly replaced with constants
# See https://github.com/NVIDIA/Fuser/issues/2702
def test_issue2702():
    def create_fusion(fd: FusionDefinition) -> None:
        T4 = fd.define_tensor(
            shape=[1, -1, -1, -1],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T75 = fd.ops.reshape(T4, new_shape=[1, 8, 4, 8192, 128])
        T90 = fd.ops.cast(T75, dtype=DataType.Float)
        T91 = fd.ops.sum(T90, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T92 = fd.ops.cast(T91, dtype=DataType.BFloat16)
        fd.add_output(T92)

    with FusionDefinition() as fd:
        create_fusion(fd)

    ins = [torch.randn((1, 32, 8192, 128), dtype=torch.bfloat16, device="cuda:0")]
    outs = fd.execute(ins)

    torch.testing.assert_close(outs[0], ins[0].view(8, 4, 8192, 128).sum(1))


# https://nvbugspro.nvidia.com/bug/5374765
# Avoid setting 2 vectorization loop domains for a single tensor.
def test_ws_tma_vectorization_loop_domain():
    def create_fusion(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[4096, 3072],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T1 = fd.define_tensor(
            shape=[4096, 3072],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        T2 = fd.define_tensor(
            shape=[3072],
            contiguity=[True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 1],
            contiguity=[None, True, None],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T5 = fd.define_tensor(
            shape=[1, 4096, 3072],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.ops.reshape(T0, new_shape=[1, 4096, 3072])
        T15 = fd.ops.reshape(T1, new_shape=[1, 4096, 3072])
        T16 = fd.ops.cast(T2, dtype=DataType.Float)
        T17 = fd.ops.cast(T10, dtype=DataType.Float)
        T18 = fd.ops.cast(T15, dtype=DataType.Float)
        S19 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T20 = fd.ops.add(S19, T16)
        T21 = fd.ops.add(T18, T17)
        T26 = fd.ops.broadcast_in_dim(T20, shape=[1, 4096, 3072], broadcast_dims=[2])
        T27 = fd.ops.mul(T26, T21)
        T28 = fd.ops.cast(T3, dtype=DataType.Float)
        T29 = fd.ops.mul(T28, T27)
        T30 = fd.ops.sum(T29, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T35 = fd.ops.broadcast_in_dim(T30, shape=[1, 4096, 1], broadcast_dims=[1])
        T36 = fd.ops.mul(T4, T4)
        T37 = fd.ops.mul(T4, T36)
        S38 = fd.define_scalar(-0.500000, dtype=DataType.Double)
        T39 = fd.ops.mul(S38, T35)
        T40 = fd.ops.mul(T39, T37)
        S41 = fd.define_scalar(3072.00, dtype=DataType.Double)
        S42 = fd.ops.reciprocal(S41)
        T43 = fd.ops.mul(T40, S42)
        T44 = fd.ops.sum(T43, dims=[0, 2], keepdim=False, dtype=DataType.Null)
        T48 = fd.ops.broadcast_in_dim(T44, shape=[1, 4096], broadcast_dims=[1])
        T53 = fd.ops.broadcast_in_dim(T48, shape=[1, 4096, 1], broadcast_dims=[0, 1])
        T58 = fd.ops.broadcast_in_dim(
            T53, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T63 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 4096, 3072], broadcast_dims=[0, 1, 2]
        )
        T64 = fd.ops.mul(T28, T58)
        T65 = fd.ops.mul(T63, T27)
        T66 = fd.ops.add(T65, T64)
        T67 = fd.ops.add(T66, T64)
        T68 = fd.ops.cast(T5, dtype=DataType.Float)
        T69 = fd.ops.add(T68, T67)
        T70 = fd.ops.mul(T28, T63)
        T71 = fd.ops.mul(T70, T21)
        T72 = fd.ops.sum(T71, dims=[0, 1], keepdim=False, dtype=DataType.Null)
        T73 = fd.ops.cast(T69, dtype=DataType.BFloat16)
        T74 = fd.ops.cast(T72, dtype=DataType.BFloat16)
        T78 = fd.ops.reshape(T73, new_shape=[4096, 3072])
        T79 = fd.ops.permute(T78, dims=[1, 0])
        fd.add_output(T79)
        fd.add_output(T78)
        fd.add_output(T73)
        fd.add_output(T74)

    with FusionDefinition() as fd:
        create_fusion(fd)

    inputs = [
        torch.testing.make_tensor((4096, 3072), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((4096, 3072), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor((3072,), dtype=torch.bfloat16, device="cuda:0"),
        torch.testing.make_tensor(
            (1, 4096, 3072), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor((1, 4096, 1), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor(
            (1, 4096, 3072), dtype=torch.bfloat16, device="cuda:0"
        ),
    ]
    fd.validate(inputs)
