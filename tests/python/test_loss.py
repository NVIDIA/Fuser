# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
from torch.testing import assert_close

from nvfuser import (
    FusionDefinition,
    FusionCache,
    DataType,
    Tensor,
    version,
    compute_contiguity,
    compute_tensor_descriptor,
)
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

from nvfuser.testing.utils import (
    is_pre_volta,
    is_pre_ampere,
    is_pre_hopper,
    debug_serde,
    NVFuserTest,
    verify_stride_order,
)


def nvfuser_fusion_id0(fd: FusionDefinition, inputs) -> None:
    T0 = fd.from_pytorch(inputs[0])
    T1 = fd.from_pytorch(inputs[1])
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    S3 = fd.define_scalar(-100, dtype=DataType.Int)
    T9 = fd.ops.pad(T1, [0, 1, 0, 0], S3)
    T19 = fd.ops.slice(
        T9,
        start_indices=[0, 1],
        end_indices=[1, 4097],
        strides=[1, 1],
        manual_normalization=0,
    )
    T20 = fd.ops.stride_order(T19, stride_order=[1, 0])
    T24 = fd.ops.reshape(T2, new_shape=[4096, 152064])
    T27 = fd.ops.reshape(T20, new_shape=[4096])
    S28 = fd.define_scalar(-100, dtype=DataType.Int)
    S29 = fd.define_scalar(0.00000, dtype=DataType.Float)
    T30 = fd.ops.ne(T27, S28)
    T31 = fd.ops.where(T30, T27, S29)
    V32 = fd.ops.shape(T27)
    S33 = fd.ops.at(V32, index=-1)
    T36 = fd.ops.broadcast_in_dim(T31, shape=[S33, 1], broadcast_dims=[0])
    T37 = fd.ops.take_along_axis(T24, T36, dim=1)
    V38 = fd.ops.shape(T27)
    S39 = fd.ops.at(V38, index=-1)
    T41 = fd.ops.reshape(T37, new_shape=[S39])
    T42 = fd.ops.max(T24, dims=[1], keepdim=False, dtype=DataType.Null)
    V43 = fd.ops.shape(T27)
    S44 = fd.ops.at(V43, index=-1)
    T47 = fd.ops.broadcast_in_dim(T42, shape=[S44, 1], broadcast_dims=[0])
    T48 = fd.ops.sub(T24, T47)
    T49 = fd.ops.exp(T48)
    T50 = fd.ops.sum(T49, dims=[1], keepdim=False, dtype=DataType.Null)
    T51 = fd.ops.log(T50)
    T52 = fd.ops.sub(T41, T42)
    T53 = fd.ops.sub(T52, T51)
    T54 = fd.ops.neg(T53)
    T55 = fd.ops.where(T30, T54, S29)
    T56 = fd.ops.sum(T30, dims=[0], keepdim=False, dtype=DataType.Null)
    T57 = fd.ops.cast(T56, dtype=DataType.Float)
    T58 = fd.ops.sum(T55, dims=[0], keepdim=False, dtype=DataType.Null)
    T59 = fd.ops.div(T58, T57)
    fd.add_output(T59)


inputs = [
    torch.randn(
        1, 4096, 152064, requires_grad=False, device="cuda", dtype=torch.bfloat16
    ),
    torch.randint(
        0,
        128,
        (
            1,
            4096,
        ),
        requires_grad=False,
        device="cuda",
    ),
]

fun = nvfuser_fusion_id0

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd, inputs)

outputs = fd.execute(inputs)
print(outputs[0])
