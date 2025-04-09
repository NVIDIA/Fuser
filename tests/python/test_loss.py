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
from nvfuser import FusionDefinition, DataType, FusionCache
from utils import NVFuserTest, is_pre_ampere, define_sdpa_rng_state




class TestLoss(NVFuserTest):
    def test_cross_entropy_loss(self):

        def fusion_func_old(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(shape=[1, 8192, 32064], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
            T1 = fd.define_tensor(shape=[1, 8192], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
            T2 = fd.ops.cast(T0, dtype=DataType.Float)
            S3 = fd.define_scalar(-100, dtype=DataType.Int)
            T9 = fd.ops.pad(T1, [0, 1, 0, 0], S3)
            T19 = fd.ops.slice(T9, start_indices=[0, 1], end_indices=[1, 8193], strides=[1, 1], manual_normalization=0)
            T20 = fd.ops.stride_order(T19, stride_order=[1, 0])
            T24 = fd.ops.reshape(T2, new_shape=[8192, 32064])
            T27 = fd.ops.reshape(T20, new_shape=[8192])
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




        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(shape=[1, 8192, 32064], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
            T1 = fd.define_tensor(shape=[1, 8192], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
            S3 = fd.define_scalar(-100, dtype=DataType.Int)
            S29 = fd.define_scalar(0.00000, dtype=DataType.Int)
            S29_float = fd.define_scalar(0.00000, dtype=DataType.Float)

            # padding and slicing the labels. Not sure why we need the stride_order. 
            T9 = fd.ops.pad(T1, [0, 1, 0, 0], S3)
            T19 = fd.ops.slice(T9, start_indices=[0, 1], end_indices=[1, 8193], strides=[1, 1], manual_normalization=0)
            T20 = fd.ops.stride_order(T19, stride_order=[1, 0])

            S28 = fd.define_scalar(-100, dtype=DataType.Int)
            S29 = fd.define_scalar(0.00000, dtype=DataType.Int)
            T30 = fd.ops.ne(T20, S28)
            T31 = fd.ops.where(T30, T20, S29)

            # take along axis, cast then reshape
            V32 = fd.ops.shape(T31)
            S33 = fd.ops.at(V32, index=-1)
            T336 = fd.ops.broadcast_in_dim(T31, shape=[1, S33, 1], broadcast_dims=[0, 1])
            T37 = fd.ops.take_along_axis(T0, T336, dim=2)
            T38 = fd.ops.cast(T37, dtype=DataType.Float)
            T41 = fd.ops.reshape(T38, new_shape=[S33])


            T2 = fd.ops.cast(T0, dtype=DataType.Float)
            T24 = fd.ops.reshape(T2, new_shape=[8192, 32064])
            T27 = fd.ops.reshape(T20, new_shape=[8192])


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
            T30_reshape = fd.ops.reshape(T30, new_shape=[S33])
            
            T55 = fd.ops.where(T30_reshape, T54, S29_float)
            T56 = fd.ops.sum(T30_reshape, dims=[0], keepdim=False, dtype=DataType.Null)
            T57 = fd.ops.cast(T56, dtype=DataType.Float)
            T58 = fd.ops.sum(T55, dims=[0], keepdim=False, dtype=DataType.Null)
            T59 = fd.ops.div(T58, T57)
            fd.add_output(T59)

        inputs = [
            torch.randn(
                1, 8192, 32064, requires_grad=False, device="cuda", dtype=torch.bfloat16
            ),
            torch.randint(
                0,
                128,
                (
                    1,
                    8192,
                ),
                requires_grad=False,
                device="cuda",
            ),
        ]

        nvf_out, _ = self.exec_nvfuser(fusion_func, inputs)
        print(nvf_out[0])
        print(nvf_out[0].to(torch.bfloat16))

        b = torch.squeeze(inputs[0], dim=0)
        labels = torch.squeeze(inputs[1], dim=0)
        x = torch.nn.functional.cross_entropy(b, labels)
        print(x)