# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from nvfuser import FusionDefinition, DataType
from nvfuser.testing.utils import NVFuserTest


class TestRepro(NVFuserTest):
    def test_issue4444(self):
        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[1, 64, 16384, 128],
                contiguity=[None, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T1 = fd.define_tensor(
                shape=[16384, 128],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T2 = fd.define_tensor(
                shape=[16384, 128],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T3 = fd.define_tensor(
                shape=[1, 64, 16384, 128],
                contiguity=[None, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T4 = fd.define_tensor(
                shape=[1, 64, 16384, 128],
                contiguity=[None, True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T20 = fd.ops.slice(
                T0,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 64, 16384, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T21 = fd.ops.cast(T20, dtype=DataType.Float)
            T27 = fd.ops.broadcast_in_dim(
                T1, shape=[1, 64, 16384, 128], broadcast_dims=[2, 3]
            )
            T28 = fd.ops.mul(T27, T21)
            T29 = fd.ops.cast(T28, dtype=DataType.BFloat16)
            T35 = fd.ops.broadcast_in_dim(
                T2, shape=[1, 64, 16384, 128], broadcast_dims=[2, 3]
            )
            T51 = fd.ops.slice(
                T29,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 64, 16384, 64],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T52 = fd.ops.mul(T35, T21)
            S53 = fd.define_scalar(0, dtype=DataType.Int)
            T59 = fd.ops.full(
                shape=[1, 64, 16384, 0], fill_value=S53, dtype=DataType.BFloat16
            )
            T75 = fd.ops.slice(
                T3,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 64, 16384, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T76 = fd.ops.cast(T51, dtype=DataType.Float)
            S77 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T87 = fd.ops.pad(T59, [0, 128, 0, 0, 0, 0, 0, 0], S77)
            T88 = fd.ops.cast(T75, dtype=DataType.Float)
            T89 = fd.ops.neg(T76)
            T90 = fd.ops.cast(T87, dtype=DataType.Float)
            T91 = fd.ops.mul(T27, T88)
            T92 = fd.ops.cast(T89, dtype=DataType.BFloat16)
            T93 = fd.ops.add(T90, T52)
            T94 = fd.ops.cast(T91, dtype=DataType.BFloat16)
            S95 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T105 = fd.ops.pad(T92, [64, 0, 0, 0, 0, 0, 0, 0], S95)
            T121 = fd.ops.slice(
                T94,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 64, 16384, 64],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T122 = fd.ops.mul(T35, T88)
            T123 = fd.ops.cast(T105, dtype=DataType.Float)
            T124 = fd.ops.cast(T121, dtype=DataType.Float)
            T140 = fd.ops.slice(
                T29,
                start_indices=[0, 0, 0, 64],
                end_indices=[1, 64, 16384, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T141 = fd.ops.add(T93, T123)
            T142 = fd.ops.neg(T124)
            S143 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T153 = fd.ops.pad(T140, [0, 64, 0, 0, 0, 0, 0, 0], S143)
            T154 = fd.ops.cast(T142, dtype=DataType.BFloat16)
            T155 = fd.ops.add(T90, T122)
            T156 = fd.ops.cast(T153, dtype=DataType.Float)
            S157 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T167 = fd.ops.pad(T154, [64, 0, 0, 0, 0, 0, 0, 0], S157)
            T168 = fd.ops.add(T141, T156)
            T169 = fd.ops.cast(T167, dtype=DataType.Float)
            T170 = fd.ops.cast(T168, dtype=DataType.BFloat16)
            T186 = fd.ops.slice(
                T94,
                start_indices=[0, 0, 0, 64],
                end_indices=[1, 64, 16384, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T187 = fd.ops.add(T155, T169)
            T194 = fd.ops.reshape(T4, new_shape=[1, 8, 8, 16384, 128])
            T201 = fd.ops.reshape(T170, new_shape=[1, 8, 8, 16384, 128])
            S202 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T212 = fd.ops.pad(T186, [0, 64, 0, 0, 0, 0, 0, 0], S202)
            T213 = fd.ops.cast(T194, dtype=DataType.Float)
            T214 = fd.ops.cast(T201, dtype=DataType.Float)
            T215 = fd.ops.cast(T212, dtype=DataType.Float)
            T216 = fd.ops.sum(T213, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T217 = fd.ops.sum(T214, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T218 = fd.ops.add(T187, T215)
            T219 = fd.ops.cast(T216, dtype=DataType.BFloat16)
            T220 = fd.ops.cast(T217, dtype=DataType.BFloat16)
            T221 = fd.ops.cast(T218, dtype=DataType.BFloat16)
            T228 = fd.ops.broadcast_in_dim(
                T219, shape=[1, 8, 1, 16384, 128], broadcast_dims=[1, 3, 4]
            )
            T235 = fd.ops.broadcast_in_dim(
                T220, shape=[1, 8, 1, 16384, 128], broadcast_dims=[1, 3, 4]
            )
            T242 = fd.ops.reshape(T221, new_shape=[1, 8, 8, 16384, 128])
            T243 = fd.ops.cat([T242, T235, T228], dim=2, manual_padding=0)
            T244 = fd.ops.permute(T243, dims=[0, 3, 1, 2, 4])
            T249 = fd.ops.reshape(T244, new_shape=[1, 16384, 10240])
            T253 = fd.ops.reshape(T249, new_shape=[16384, 10240])
            T254 = fd.ops.permute(T253, dims=[1, 0])
            fd.add_output(T253)
            fd.add_output(T254)

        inputs = [
            torch.testing.make_tensor(
                (1, 64, 16384, 128), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (16384, 128), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (16384, 128), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 64, 16384, 128), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 64, 16384, 128), dtype=torch.bfloat16, device="cuda:0"
            ),
        ]

        self.exec_nvfuser(fusion_func, inputs, validate=True)
