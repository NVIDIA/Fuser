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

    def test_issue4459(self):
        def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[4, 32],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[0, 1],
            )
            T1 = fd.define_tensor(
                shape=[4, 32, 1, 1, 1],
                contiguity=[True, True, None, None, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[4, 3, 2, 1, 0],
            )
            T2 = fd.define_tensor(
                shape=[4, 32, 10, 64, 64],
                contiguity=[True, True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[4, 3, 2, 1, 0],
            )
            T3 = fd.define_tensor(
                shape=[320],
                contiguity=[True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[0],
            )
            T4 = fd.define_tensor(
                shape=[320],
                contiguity=[True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[0],
            )
            T5 = fd.define_tensor(
                shape=[4, 320, 66, 66],
                contiguity=[True, True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[3, 2, 1, 0],
            )
            T12 = fd.ops.broadcast_in_dim(
                T0, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            T19 = fd.ops.broadcast_in_dim(
                T12, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
            )
            T26 = fd.ops.broadcast_in_dim(
                T1, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
            )
            T27 = fd.ops.sub(T2, T19)
            T33 = fd.ops.reshape(T3, new_shape=[1, 320, 1, 1])
            T34 = fd.ops.mul(T27, T26)
            T40 = fd.ops.reshape(T4, new_shape=[1, 320, 1, 1])
            T46 = fd.ops.broadcast_in_dim(
                T33, shape=[4, 320, 64, 64], broadcast_dims=[0, 1, 2, 3]
            )
            T52 = fd.ops.reshape(T34, new_shape=[4, 320, 64, 64])
            T58 = fd.ops.broadcast_in_dim(
                T40, shape=[4, 320, 64, 64], broadcast_dims=[0, 1, 2, 3]
            )
            T59 = fd.ops.mul(T52, T46)
            T60 = fd.ops.add(T59, T58)
            T61 = fd.ops.neg(T60)
            T62 = fd.ops.exp(T61)
            S63 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T73 = fd.ops.pad(T5, [-1, -1, -1, -1, 0, 0, 0, 0], S63)
            S74 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T75 = fd.ops.add(S74, T62)
            T76 = fd.ops.mul(T60, T73)
            T77 = fd.ops.reciprocal(T75)
            T78 = fd.ops.neg(T76)
            T79 = fd.ops.mul(T78, T77)
            T80 = fd.ops.mul(T79, T77)
            T81 = fd.ops.mul(T80, T62)
            T82 = fd.ops.neg(T81)
            T83 = fd.ops.mul(T77, T73)
            T84 = fd.ops.add(T83, T82)
            T85 = fd.ops.mul(T46, T84)
            T92 = fd.ops.reshape(T85, new_shape=[4, 32, 10, 64, 64])
            T93 = fd.ops.mul(T27, T92)
            T94 = fd.ops.sum(T93, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
            T101 = fd.ops.broadcast_in_dim(
                T94, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            S102 = fd.define_scalar(3.00000, dtype=DataType.Double)
            T103 = fd.ops.pow(T1, S102)
            S104 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T105 = fd.ops.mul(S104, T101)
            T106 = fd.ops.mul(T26, T92)
            T107 = fd.ops.mul(T105, T103)
            T108 = fd.ops.neg(T106)
            T109 = fd.ops.sum(T107, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
            T110 = fd.ops.sum(T108, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
            T117 = fd.ops.broadcast_in_dim(
                T0, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            T124 = fd.ops.broadcast_in_dim(
                T109, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            T131 = fd.ops.broadcast_in_dim(
                T110, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            T138 = fd.ops.broadcast_in_dim(
                T117, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
            )
            T145 = fd.ops.broadcast_in_dim(
                T124, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
            )
            T146 = fd.ops.sum(T131, dims=[2, 3, 4], keepdim=False, dtype=DataType.Null)
            T147 = fd.ops.sub(T2, T138)
            S148 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T149 = fd.ops.mul(S148, T145)
            T156 = fd.ops.broadcast_in_dim(
                T146, shape=[4, 32, 1, 1, 1], broadcast_dims=[0, 1]
            )
            T157 = fd.ops.mul(T149, T147)
            T164 = fd.ops.broadcast_in_dim(
                T156, shape=[4, 32, 10, 64, 64], broadcast_dims=[0, 1, 2, 3, 4]
            )
            S165 = fd.define_scalar(40960.0, dtype=DataType.Double)
            S166 = fd.ops.reciprocal(S165)
            T167 = fd.ops.mul(T157, S166)
            S168 = fd.define_scalar(2.44141e-05, dtype=DataType.Double)
            T169 = fd.ops.mul(S168, T164)
            T170 = fd.ops.add(T169, T167)
            T171 = fd.ops.add(T106, T170)
            T177 = fd.ops.reshape(T171, new_shape=[4, 320, 64, 64])
            T184 = fd.ops.reshape(T177, new_shape=[1, 4, 320, 64, 64])
            T185 = fd.ops.permute(T184, dims=[0, 3, 4, 1, 2])
            T192 = fd.ops.reshape(T185, new_shape=[1, 1, 4096, 4, 320])
            T193 = fd.ops.mul(T52, T84)
            T194 = fd.ops.sum(T192, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T195 = fd.ops.sum(T193, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
            T196 = fd.ops.sum(T84, dims=[0, 2, 3], keepdim=False, dtype=DataType.Null)
            T200 = fd.ops.reshape(T194, new_shape=[16384, 320])
            T206 = fd.ops.broadcast_in_dim(
                T195, shape=[1, 320, 1, 1], broadcast_dims=[1]
            )
            T212 = fd.ops.broadcast_in_dim(
                T196, shape=[1, 320, 1, 1], broadcast_dims=[1]
            )
            T213 = fd.ops.permute(T200, dims=[1, 0])
            T214 = fd.ops.sum(T194, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T217 = fd.ops.reshape(T206, new_shape=[320])
            T220 = fd.ops.reshape(T212, new_shape=[320])
            fd.add_output(T177)
            fd.add_output(T200)
            fd.add_output(T213)
            fd.add_output(T214)
            fd.add_output(T217)
            fd.add_output(T220)

        inputs = [
            torch.randn(128, dtype=torch.float32, device="cuda:0").as_strided(
                (4, 32), (1, 4)
            ),
            torch.testing.make_tensor(
                (4, 32, 1, 1, 1), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (4, 32, 10, 64, 64), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor((320,), dtype=torch.float32, device="cuda:0"),
            torch.testing.make_tensor((320,), dtype=torch.float32, device="cuda:0"),
            torch.testing.make_tensor(
                (4, 320, 66, 66), dtype=torch.float32, device="cuda:0"
            ),
        ]

        self.exec_nvfuser(nvfuser_fusion_id0, inputs, validate=True)
