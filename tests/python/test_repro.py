# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from nvfuser import FusionDefinition, DataType
from python.utils import NVFuserTest


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

        with FusionDefinition() as fd:
            fusion_func(fd)

        inputs = [
            torch.testing.make_tensor(
                (1, 64, 16384, 128),
                dtype=torch.bfloat16,
                device="cuda:0",
                low=-1,
                high=1,
            ),
            torch.testing.make_tensor(
                (16384, 128), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (16384, 128), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 64, 16384, 128),
                dtype=torch.bfloat16,
                device="cuda:0",
                low=-1,
                high=1,
            ),
            torch.testing.make_tensor(
                (1, 64, 16384, 128),
                dtype=torch.bfloat16,
                device="cuda:0",
                low=-1,
                high=1,
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

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

        with FusionDefinition() as fd:
            nvfuser_fusion_id0(fd)

        inputs = [
            torch.randn(128, dtype=torch.float32, device="cuda:0").as_strided(
                (4, 32), (1, 4)
            ),
            torch.testing.make_tensor(
                (4, 32, 1, 1, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (4, 32, 10, 64, 64),
                dtype=torch.float32,
                device="cuda:0",
                low=-1,
                high=1,
            ),
            torch.testing.make_tensor(
                (320,), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (320,), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (4, 320, 66, 66), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
        ]
        fd.validate(inputs)

    def test_issue4670(self):
        def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
            S0 = fd.define_scalar(129, dtype=DataType.Int)
            S1 = fd.define_scalar(0, dtype=DataType.Int)
            S2 = fd.define_scalar(1, dtype=DataType.Int)
            T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int)
            T4 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
            S5 = fd.define_scalar(128, dtype=DataType.Int)
            S6 = fd.ops.size(T3, dim=0)
            V7 = fd.define_vector([S5, S6], dtype=DataType.Int)
            T8 = fd.ops.expand(T4, shape=V7)
            S9 = fd.define_scalar(128, dtype=DataType.Int)
            S10 = fd.define_scalar(0, dtype=DataType.Int)
            S11 = fd.define_scalar(1, dtype=DataType.Int)
            T12 = fd.ops.iota(S9, S10, S11, dtype=DataType.Int)
            T13 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
            S14 = fd.ops.size(T12, dim=0)
            S15 = fd.define_scalar(129, dtype=DataType.Int)
            V16 = fd.define_vector([S14, S15], dtype=DataType.Int)
            T17 = fd.ops.expand(T13, shape=V16)
            T18 = fd.ops.gt(T8, T17)
            T19 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
            S20 = fd.define_scalar(128, dtype=DataType.Int)
            V21 = fd.define_vector([S20, S6], dtype=DataType.Int)
            T22 = fd.ops.expand(T19, shape=V21)
            T23 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
            S24 = fd.define_scalar(129, dtype=DataType.Int)
            V25 = fd.define_vector([S14, S24], dtype=DataType.Int)
            T26 = fd.ops.expand(T23, shape=V25)
            T27 = fd.ops.sub(T22, T26)
            S28 = fd.define_scalar(1, dtype=DataType.Int)
            T29 = fd.ops.ge(T27, S28)
            S30 = fd.define_scalar(-3.38953e38, dtype=DataType.BFloat16)
            S31 = fd.define_scalar(128, dtype=DataType.Int)
            S32 = fd.define_scalar(129, dtype=DataType.Int)
            V33 = fd.define_vector([S31, S32], dtype=DataType.Int)
            T34 = fd.ops.full(shape=V33, fill_value=S30, dtype=DataType.BFloat16)
            S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T36 = fd.ops.where(T29, T34, S35)
            fd.add_output(T18)
            fd.add_output(T36)

        with FusionDefinition() as fd:
            nvfuser_fusion_id0(fd)

        inputs = []
        fd.validate(inputs)

    def test_ws_tma_normalization1(self):
        # Bug 5374765: Gemma-7b model fail with "Found two vectorized domains ... only one is allowed"
        def nvfuser_fusion_id8(fd: FusionDefinition) -> None:
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
            T26 = fd.ops.broadcast_in_dim(
                T20, shape=[1, 4096, 3072], broadcast_dims=[2]
            )
            T27 = fd.ops.mul(T26, T21)
            T28 = fd.ops.cast(T3, dtype=DataType.Float)
            T29 = fd.ops.mul(T28, T27)
            T30 = fd.ops.sum(T29, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T35 = fd.ops.broadcast_in_dim(T30, shape=[1, 4096, 1], broadcast_dims=[1])
            S36 = fd.define_scalar(3.00000, dtype=DataType.Float)
            T37 = fd.ops.pow(T4, S36)
            S38 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T39 = fd.ops.mul(S38, T35)
            T40 = fd.ops.mul(T39, T37)
            S41 = fd.define_scalar(3072.00, dtype=DataType.Double)
            S42 = fd.ops.reciprocal(S41)
            T43 = fd.ops.mul(T40, S42)
            T44 = fd.ops.sum(T43, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T48 = fd.ops.broadcast_in_dim(T44, shape=[1, 4096], broadcast_dims=[1])
            T53 = fd.ops.broadcast_in_dim(
                T48, shape=[1, 4096, 1], broadcast_dims=[0, 1]
            )
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
            nvfuser_fusion_id8(fd)

        inputs = [
            torch.testing.make_tensor(
                (4096, 3072), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (4096, 3072), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (3072,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 3072), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 3072), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

    def test_ws_tma_normalization2(self):
        # Bug 5374766: Multiple model fail with "Invalid tensor to circular-buffer"
        def nvfuser_fusion_id4(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[147456, 128],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[128],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.define_tensor(
                shape=[288, 512],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T3 = fd.define_tensor(
                shape=[288, 512, 128],
                contiguity=[True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T4 = fd.define_tensor(
                shape=[288, 512, 1],
                contiguity=[True, True, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T9 = fd.ops.reshape(T0, new_shape=[288, 512, 128])
            T14 = fd.ops.broadcast_in_dim(T1, shape=[288, 512, 128], broadcast_dims=[2])
            T19 = fd.ops.broadcast_in_dim(
                T2, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T20 = fd.ops.cast(T9, dtype=DataType.Float)
            T21 = fd.ops.cast(T14, dtype=DataType.Float)
            T26 = fd.ops.broadcast_in_dim(
                T19, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
            )
            T27 = fd.ops.cast(T3, dtype=DataType.Float)
            T28 = fd.ops.mul(T21, T20)
            T29 = fd.ops.sub(T27, T26)
            T30 = fd.ops.mul(T29, T28)
            T31 = fd.ops.sum(T30, dims=[2], keepdim=False, dtype=DataType.Null)
            T36 = fd.ops.broadcast_in_dim(
                T31, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T41 = fd.ops.broadcast_in_dim(
                T4, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
            )
            S42 = fd.define_scalar(3.00000, dtype=DataType.Double)
            T43 = fd.ops.pow(T4, S42)
            S44 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T45 = fd.ops.mul(S44, T36)
            T46 = fd.ops.mul(T41, T28)
            T47 = fd.ops.mul(T45, T43)
            T48 = fd.ops.neg(T46)
            T49 = fd.ops.sum(T47, dims=[2], keepdim=False, dtype=DataType.Null)
            T50 = fd.ops.sum(T48, dims=[2], keepdim=False, dtype=DataType.Null)
            T55 = fd.ops.broadcast_in_dim(
                T2, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T60 = fd.ops.broadcast_in_dim(
                T49, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T65 = fd.ops.broadcast_in_dim(
                T50, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T70 = fd.ops.broadcast_in_dim(
                T55, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
            )
            T75 = fd.ops.broadcast_in_dim(
                T60, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
            )
            T76 = fd.ops.sum(T65, dims=[2], keepdim=False, dtype=DataType.Null)
            T77 = fd.ops.sub(T27, T70)
            S78 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T79 = fd.ops.mul(S78, T75)
            T84 = fd.ops.broadcast_in_dim(
                T76, shape=[288, 512, 1], broadcast_dims=[0, 1]
            )
            T85 = fd.ops.mul(T79, T77)
            T90 = fd.ops.broadcast_in_dim(
                T84, shape=[288, 512, 128], broadcast_dims=[0, 1, 2]
            )
            S91 = fd.define_scalar(128.000, dtype=DataType.Double)
            S92 = fd.ops.reciprocal(S91)
            T93 = fd.ops.mul(T85, S92)
            S94 = fd.define_scalar(0.00781250, dtype=DataType.Double)
            T95 = fd.ops.mul(S94, T90)
            T96 = fd.ops.add(T95, T93)
            T97 = fd.ops.add(T46, T96)
            T98 = fd.ops.cast(T97, dtype=DataType.BFloat16)
            T99 = fd.ops.mul(T29, T41)
            T100 = fd.ops.mul(T99, T20)
            T101 = fd.ops.cast(T9, dtype=DataType.Float)
            T105 = fd.ops.reshape(T98, new_shape=[147456, 128])
            T106 = fd.ops.sum(T97, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T107 = fd.ops.sum(T100, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T108 = fd.ops.sum(T101, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T109 = fd.ops.permute(T105, dims=[1, 0])
            T110 = fd.ops.cast(T106, dtype=DataType.BFloat16)
            T111 = fd.ops.cast(T107, dtype=DataType.BFloat16)
            T112 = fd.ops.cast(T108, dtype=DataType.BFloat16)
            fd.add_output(T109)
            fd.add_output(T110)
            fd.add_output(T105)
            fd.add_output(T98)
            fd.add_output(T111)
            fd.add_output(T112)

        with FusionDefinition() as fd:
            nvfuser_fusion_id4(fd)

        inputs = [
            torch.testing.make_tensor(
                (147456, 128), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (128,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (288, 512), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (288, 512, 128), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (288, 512, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

    def test_ws_tma_normalization3(self):
        # Bug 5374767: Mistral-7B-v0.1 fails with "The non-allocating compute-at IDs are not found in the allocation domain"
        def nvfuser_fusion_id14(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[4096, 4096],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[4096],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.define_tensor(
                shape=[1, 4096, 4096],
                contiguity=[True, None, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 2, 0],
            )
            T3 = fd.define_tensor(
                shape=[1, 4096, 1],
                contiguity=[None, True, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T4 = fd.define_tensor(
                shape=[1, 4096, 4096],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T9 = fd.ops.reshape(T0, new_shape=[1, 4096, 4096])
            T10 = fd.ops.cast(T1, dtype=DataType.Float)
            T11 = fd.ops.cast(T9, dtype=DataType.Float)
            T16 = fd.ops.broadcast_in_dim(
                T10, shape=[1, 4096, 4096], broadcast_dims=[2]
            )
            T17 = fd.ops.mul(T16, T11)
            T18 = fd.ops.cast(T2, dtype=DataType.Float)
            T19 = fd.ops.mul(T18, T17)
            T20 = fd.ops.sum(T19, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T25 = fd.ops.broadcast_in_dim(T20, shape=[1, 4096, 1], broadcast_dims=[1])
            S26 = fd.define_scalar(3.00000, dtype=DataType.Double)
            T27 = fd.ops.pow(T3, S26)
            S28 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T29 = fd.ops.mul(S28, T25)
            T30 = fd.ops.mul(T29, T27)
            S31 = fd.define_scalar(4096.00, dtype=DataType.Double)
            S32 = fd.ops.reciprocal(S31)
            T33 = fd.ops.mul(T30, S32)
            T34 = fd.ops.sum(T33, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T38 = fd.ops.broadcast_in_dim(T34, shape=[1, 4096], broadcast_dims=[1])
            T43 = fd.ops.broadcast_in_dim(
                T38, shape=[1, 4096, 1], broadcast_dims=[0, 1]
            )
            T48 = fd.ops.broadcast_in_dim(
                T43, shape=[1, 4096, 4096], broadcast_dims=[0, 1, 2]
            )
            T53 = fd.ops.broadcast_in_dim(
                T3, shape=[1, 4096, 4096], broadcast_dims=[0, 1, 2]
            )
            T54 = fd.ops.mul(T18, T48)
            T55 = fd.ops.mul(T53, T17)
            T56 = fd.ops.add(T55, T54)
            T57 = fd.ops.add(T56, T54)
            T58 = fd.ops.mul(T18, T53)
            T59 = fd.ops.cast(T4, dtype=DataType.Float)
            T60 = fd.ops.mul(T58, T11)
            T61 = fd.ops.add(T59, T57)
            T62 = fd.ops.sum(T60, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T63 = fd.ops.cast(T61, dtype=DataType.BFloat16)
            T64 = fd.ops.cast(T62, dtype=DataType.BFloat16)
            fd.add_output(T64)
            fd.add_output(T63)

        with FusionDefinition() as fd:
            nvfuser_fusion_id14(fd)

        inputs = [
            torch.testing.make_tensor(
                (4096, 4096), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (4096,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 4096), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 4096, 4096), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

    def test_ws_tma_normalization4(self):
        # Bug 5374768: Multiple model fail with "Tried to access out of boundary index"
        def nvfuser_fusion_id4(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[28672, 2048],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[2048],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.define_tensor(
                shape=[14, 2048, 2048],
                contiguity=[True, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T3 = fd.define_tensor(
                shape=[14, 2048, 1],
                contiguity=[True, True, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T8 = fd.ops.reshape(T0, new_shape=[14, 2048, 2048])
            T9 = fd.ops.cast(T1, dtype=DataType.Float)
            T10 = fd.ops.cast(T8, dtype=DataType.Float)
            T15 = fd.ops.broadcast_in_dim(
                T9, shape=[14, 2048, 2048], broadcast_dims=[2]
            )
            T16 = fd.ops.mul(T15, T10)
            T17 = fd.ops.cast(T2, dtype=DataType.Float)
            T18 = fd.ops.mul(T17, T16)
            T19 = fd.ops.sum(T18, dims=[2], keepdim=False, dtype=DataType.Null)
            T24 = fd.ops.broadcast_in_dim(
                T19, shape=[14, 2048, 1], broadcast_dims=[0, 1]
            )
            S25 = fd.define_scalar(3.00000, dtype=DataType.Double)
            T26 = fd.ops.pow(T3, S25)
            S27 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T28 = fd.ops.mul(S27, T24)
            T29 = fd.ops.mul(T28, T26)
            S30 = fd.define_scalar(2048.00, dtype=DataType.Double)
            S31 = fd.ops.reciprocal(S30)
            T32 = fd.ops.mul(T29, S31)
            T33 = fd.ops.sum(T32, dims=[2], keepdim=False, dtype=DataType.Null)
            T38 = fd.ops.broadcast_in_dim(
                T33, shape=[14, 2048, 1], broadcast_dims=[0, 1]
            )
            T43 = fd.ops.broadcast_in_dim(
                T38, shape=[14, 2048, 2048], broadcast_dims=[0, 1, 2]
            )
            T48 = fd.ops.broadcast_in_dim(
                T3, shape=[14, 2048, 2048], broadcast_dims=[0, 1, 2]
            )
            T49 = fd.ops.mul(T17, T43)
            T50 = fd.ops.mul(T48, T16)
            T51 = fd.ops.add(T50, T49)
            T52 = fd.ops.add(T51, T49)
            T53 = fd.ops.cast(T52, dtype=DataType.BFloat16)
            T54 = fd.ops.mul(T17, T48)
            T55 = fd.ops.mul(T54, T10)
            T59 = fd.ops.reshape(T53, new_shape=[28672, 2048])
            T60 = fd.ops.sum(T55, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T61 = fd.ops.permute(T59, dims=[1, 0])
            T62 = fd.ops.cast(T60, dtype=DataType.BFloat16)
            fd.add_output(T61)
            fd.add_output(T59)
            fd.add_output(T53)
            fd.add_output(T62)

        with FusionDefinition() as fd:
            nvfuser_fusion_id4(fd)

        inputs = [
            torch.testing.make_tensor(
                (28672, 2048), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (2048,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (14, 2048, 2048), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (14, 2048, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

    def test_ws_tma_normalization5(self):
        # Bug 5374769: stablecode-completion-alpha-3b fails with "The non-allocating compute-at IDs are not found in the allocation domain"
        def nvfuser_fusion_id9(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[16384, 2560],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[2560],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T2 = fd.define_tensor(
                shape=[1, 16384],
                contiguity=[None, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T3 = fd.define_tensor(
                shape=[1, 16384, 2560],
                contiguity=[True, None, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 2, 0],
            )
            T4 = fd.define_tensor(
                shape=[1, 16384, 1],
                contiguity=[None, True, None],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T5 = fd.define_tensor(
                shape=[16384, 2560],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T6 = fd.define_tensor(
                shape=[2560],
                contiguity=[True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[0],
            )
            T7 = fd.define_tensor(
                shape=[1, 16384, 2560],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T12 = fd.ops.reshape(T0, new_shape=[1, 16384, 2560])
            T17 = fd.ops.broadcast_in_dim(
                T1, shape=[1, 16384, 2560], broadcast_dims=[2]
            )
            T22 = fd.ops.broadcast_in_dim(
                T2, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T23 = fd.ops.cast(T12, dtype=DataType.Float)
            T24 = fd.ops.cast(T17, dtype=DataType.Float)
            T29 = fd.ops.broadcast_in_dim(
                T22, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T30 = fd.ops.cast(T3, dtype=DataType.Float)
            T31 = fd.ops.mul(T24, T23)
            T32 = fd.ops.sub(T30, T29)
            T33 = fd.ops.mul(T32, T31)
            T34 = fd.ops.sum(T33, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T39 = fd.ops.broadcast_in_dim(T34, shape=[1, 16384, 1], broadcast_dims=[1])
            T44 = fd.ops.broadcast_in_dim(
                T4, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T49 = fd.ops.reshape(T5, new_shape=[1, 16384, 2560])
            T54 = fd.ops.broadcast_in_dim(
                T6, shape=[1, 16384, 2560], broadcast_dims=[2]
            )
            S55 = fd.define_scalar(3.00000, dtype=DataType.Double)
            T56 = fd.ops.pow(T4, S55)
            S57 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T58 = fd.ops.mul(S57, T39)
            T59 = fd.ops.mul(T44, T31)
            T60 = fd.ops.cast(T49, dtype=DataType.Float)
            T61 = fd.ops.cast(T54, dtype=DataType.Float)
            T62 = fd.ops.mul(T58, T56)
            T63 = fd.ops.neg(T59)
            T64 = fd.ops.mul(T61, T60)
            T65 = fd.ops.sum(T62, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T66 = fd.ops.sum(T63, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T67 = fd.ops.mul(T32, T64)
            T71 = fd.ops.broadcast_in_dim(T65, shape=[1, 16384], broadcast_dims=[1])
            T76 = fd.ops.broadcast_in_dim(T66, shape=[1, 16384, 1], broadcast_dims=[1])
            T77 = fd.ops.sum(T67, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T82 = fd.ops.broadcast_in_dim(
                T2, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T87 = fd.ops.broadcast_in_dim(
                T71, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T88 = fd.ops.sum(T76, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T93 = fd.ops.broadcast_in_dim(T77, shape=[1, 16384, 1], broadcast_dims=[1])
            T98 = fd.ops.broadcast_in_dim(
                T82, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T103 = fd.ops.broadcast_in_dim(
                T87, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T107 = fd.ops.broadcast_in_dim(T88, shape=[1, 16384], broadcast_dims=[1])
            S108 = fd.define_scalar(-0.500000, dtype=DataType.Double)
            T109 = fd.ops.mul(S108, T93)
            T110 = fd.ops.mul(T44, T64)
            T111 = fd.ops.sub(T30, T98)
            S112 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T113 = fd.ops.mul(S112, T103)
            T118 = fd.ops.broadcast_in_dim(
                T107, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T119 = fd.ops.mul(T109, T56)
            T120 = fd.ops.neg(T110)
            T121 = fd.ops.mul(T113, T111)
            T126 = fd.ops.broadcast_in_dim(
                T118, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T127 = fd.ops.sum(T119, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T128 = fd.ops.sum(T120, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            S129 = fd.define_scalar(2560.00, dtype=DataType.Double)
            S130 = fd.ops.reciprocal(S129)
            T131 = fd.ops.mul(T121, S130)
            S132 = fd.define_scalar(0.000390625, dtype=DataType.Double)
            T133 = fd.ops.mul(S132, T126)
            T134 = fd.ops.cast(T7, dtype=DataType.Float)
            T138 = fd.ops.broadcast_in_dim(T127, shape=[1, 16384], broadcast_dims=[1])
            T143 = fd.ops.broadcast_in_dim(
                T128, shape=[1, 16384, 1], broadcast_dims=[1]
            )
            T144 = fd.ops.add(T133, T131)
            T145 = fd.ops.add(T134, T59)
            T150 = fd.ops.broadcast_in_dim(
                T138, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T151 = fd.ops.sum(T143, dims=[0, 2], keepdim=False, dtype=DataType.Null)
            T156 = fd.ops.broadcast_in_dim(
                T150, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            T160 = fd.ops.broadcast_in_dim(T151, shape=[1, 16384], broadcast_dims=[1])
            S161 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T162 = fd.ops.mul(S161, T156)
            T167 = fd.ops.broadcast_in_dim(
                T160, shape=[1, 16384, 1], broadcast_dims=[0, 1]
            )
            T168 = fd.ops.add(T145, T144)
            T169 = fd.ops.mul(T162, T111)
            T174 = fd.ops.broadcast_in_dim(
                T167, shape=[1, 16384, 2560], broadcast_dims=[0, 1, 2]
            )
            S175 = fd.define_scalar(2560.00, dtype=DataType.Double)
            S176 = fd.ops.reciprocal(S175)
            T177 = fd.ops.mul(T169, S176)
            S178 = fd.define_scalar(0.000390625, dtype=DataType.Double)
            T179 = fd.ops.mul(S178, T174)
            T180 = fd.ops.mul(T32, T44)
            T181 = fd.ops.add(T179, T177)
            T182 = fd.ops.add(T168, T110)
            T183 = fd.ops.mul(T180, T60)
            T184 = fd.ops.mul(T180, T23)
            T185 = fd.ops.cast(T49, dtype=DataType.Float)
            T186 = fd.ops.cast(T12, dtype=DataType.Float)
            T187 = fd.ops.add(T182, T181)
            T188 = fd.ops.sum(T183, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T189 = fd.ops.sum(T185, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T190 = fd.ops.sum(T184, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T191 = fd.ops.sum(T186, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T192 = fd.ops.cast(T187, dtype=DataType.BFloat16)
            T193 = fd.ops.cast(T188, dtype=DataType.BFloat16)
            T194 = fd.ops.cast(T189, dtype=DataType.BFloat16)
            T195 = fd.ops.cast(T190, dtype=DataType.BFloat16)
            T196 = fd.ops.cast(T191, dtype=DataType.BFloat16)
            fd.add_output(T196)
            fd.add_output(T195)
            fd.add_output(T194)
            fd.add_output(T193)
            fd.add_output(T192)

        with FusionDefinition() as fd:
            nvfuser_fusion_id9(fd)

        inputs = [
            torch.testing.make_tensor(
                (16384, 2560), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (2560,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 16384), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 16384, 2560), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 16384, 1), dtype=torch.float32, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (16384, 2560), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (2560,), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
            torch.testing.make_tensor(
                (1, 16384, 2560), dtype=torch.bfloat16, device="cuda:0", low=-1, high=1
            ),
        ]
        outputs = fd.execute(inputs)
        fd.validate_with_auto_inferred_outputs(outputs, inputs)

    def test_loop_promotion_cyclic_war(self):
        def nvfuser_fusion_id1(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[4096, 128],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[4096, 128],
                contiguity=[True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T2 = fd.define_tensor(
                shape=[1, 4096, 5120],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T3 = fd.define_tensor(
                shape=[1, 4096, 640],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T4 = fd.define_tensor(
                shape=[1, 4096, 640],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T5 = fd.define_tensor(
                shape=[1, 4096, 16640],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T6 = fd.define_tensor(
                shape=[1, 4096, 16640],
                contiguity=[None, True, True],
                dtype=DataType.BFloat16,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T7 = fd.define_tensor(
                shape=[1, 4096, 5120],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T8 = fd.define_tensor(
                shape=[1, 4096, 640],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T9 = fd.define_tensor(
                shape=[1, 4096, 640],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T10 = fd.define_tensor(
                shape=[1, 4096, 16640],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T11 = fd.define_tensor(
                shape=[1, 4096, 16640],
                contiguity=[None, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T16 = fd.ops.reshape(T0, new_shape=[1, 4096, 128])
            T22 = fd.ops.broadcast_in_dim(
                T16, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
            )
            T27 = fd.ops.reshape(T1, new_shape=[1, 4096, 128])
            T33 = fd.ops.broadcast_in_dim(
                T27, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
            )
            T39 = fd.ops.broadcast_in_dim(
                T22, shape=[1, 40, 4096, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T40 = fd.ops.cast(T39, dtype=DataType.Float)
            T46 = fd.ops.broadcast_in_dim(
                T33, shape=[1, 40, 4096, 128], broadcast_dims=[0, 1, 2, 3]
            )
            T47 = fd.ops.cast(T46, dtype=DataType.Float)
            T48 = fd.ops.cast(T2, dtype=DataType.Float)
            T49 = fd.ops.cast(T3, dtype=DataType.Float)
            T50 = fd.ops.cast(T4, dtype=DataType.Float)
            T51 = fd.ops.cast(T5, dtype=DataType.Float)
            T52 = fd.ops.cast(T6, dtype=DataType.Float)
            S53 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T54 = fd.ops.mul(T7, S53)
            S55 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T56 = fd.ops.mul(T8, S55)
            S57 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T58 = fd.ops.mul(T9, S57)
            S59 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T60 = fd.ops.mul(T10, S59)
            S61 = fd.define_scalar(2.00000, dtype=DataType.Double)
            T62 = fd.ops.mul(T11, S61)
            T63 = fd.ops.add(T48, T54)
            T64 = fd.ops.add(T49, T56)
            T65 = fd.ops.add(T50, T58)
            T66 = fd.ops.add(T51, T60)
            T67 = fd.ops.add(T52, T62)
            T68 = fd.ops.cast(T63, dtype=DataType.BFloat16)
            T74 = fd.ops.reshape(T68, new_shape=[1, 4096, 40, 128])
            T75 = fd.ops.cast(T64, dtype=DataType.BFloat16)
            T81 = fd.ops.reshape(T75, new_shape=[1, 4096, 5, 128])
            T82 = fd.ops.cast(T65, dtype=DataType.BFloat16)
            T88 = fd.ops.reshape(T82, new_shape=[1, 4096, 5, 128])
            T89 = fd.ops.cast(T66, dtype=DataType.BFloat16)
            T90 = fd.ops.neg(T66)
            T91 = fd.ops.cast(T67, dtype=DataType.BFloat16)
            T92 = fd.ops.permute(T74, dims=[0, 2, 1, 3])
            T93 = fd.ops.permute(T81, dims=[0, 2, 1, 3])
            T94 = fd.ops.permute(T88, dims=[0, 2, 1, 3])
            T95 = fd.ops.exp(T90)
            T105 = fd.ops.broadcast_in_dim(
                T93, shape=[1, 1, 8, 5, 1, 4096, 1, 128], broadcast_dims=[1, 3, 5, 7]
            )
            T111 = fd.ops.reshape(T105, new_shape=[1, 40, 4096, 128])
            T121 = fd.ops.broadcast_in_dim(
                T94, shape=[1, 1, 8, 5, 1, 4096, 1, 128], broadcast_dims=[1, 3, 5, 7]
            )
            T127 = fd.ops.reshape(T121, new_shape=[1, 40, 4096, 128])
            T128 = fd.ops.cast(T92, dtype=DataType.Float)
            T144 = fd.ops.slice(
                T92,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 40, 4096, 64],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T160 = fd.ops.slice(
                T92,
                start_indices=[0, 0, 0, 64],
                end_indices=[1, 40, 4096, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T161 = fd.ops.cast(T160, dtype=DataType.Float)
            T162 = fd.ops.neg(T161)
            T163 = fd.ops.cast(T162, dtype=DataType.BFloat16)
            T164 = fd.ops.cast(T111, dtype=DataType.Float)
            T180 = fd.ops.slice(
                T111,
                start_indices=[0, 0, 0, 0],
                end_indices=[1, 40, 4096, 64],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T196 = fd.ops.slice(
                T111,
                start_indices=[0, 0, 0, 64],
                end_indices=[1, 40, 4096, 128],
                strides=[1, 1, 1, 1],
                manual_normalization=0,
            )
            T197 = fd.ops.cast(T196, dtype=DataType.Float)
            T198 = fd.ops.neg(T197)
            T199 = fd.ops.cast(T198, dtype=DataType.BFloat16)
            S200 = fd.define_scalar(1.00000, dtype=DataType.Double)
            T201 = fd.ops.add(S200, T95)
            T202 = fd.ops.mul(T128, T40)
            T203 = fd.ops.cat([T163, T144], dim=-1, manual_padding=0)
            T204 = fd.ops.mul(T164, T40)
            T205 = fd.ops.cat([T199, T180], dim=-1, manual_padding=0)
            T206 = fd.ops.reciprocal(T201)
            T207 = fd.ops.cast(T203, dtype=DataType.Float)
            T208 = fd.ops.cast(T205, dtype=DataType.Float)
            T209 = fd.ops.mul(T207, T47)
            T210 = fd.ops.mul(T208, T47)
            T211 = fd.ops.mul(T66, T206)
            T212 = fd.ops.add(T202, T209)
            T213 = fd.ops.add(T204, T210)
            T214 = fd.ops.mul(T211, T67)
            T215 = fd.ops.cast(T212, dtype=DataType.BFloat16)
            T216 = fd.ops.cast(T213, dtype=DataType.BFloat16)
            T217 = fd.ops.cast(T214, dtype=DataType.BFloat16)
            fd.add_output(T89)
            fd.add_output(T91)
            fd.add_output(T127)
            fd.add_output(T215)
            fd.add_output(T216)
            fd.add_output(T217)
            fd.add_output(T214)

        with FusionDefinition() as fd:
            nvfuser_fusion_id1(fd)

        inputs = [
            torch.testing.make_tensor(
                (4096, 128), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (4096, 128), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 5120), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 640), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 640), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 16640), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 16640), dtype=torch.bfloat16, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 5120), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 640), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 640), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 16640), dtype=torch.float32, device="cuda:0"
            ),
            torch.testing.make_tensor(
                (1, 4096, 16640), dtype=torch.float32, device="cuda:0"
            ),
        ]
        fd.execute(inputs)

    def test_issue4670(self):
        def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
            S0 = fd.define_scalar(129, dtype=DataType.Int)
            S1 = fd.define_scalar(0, dtype=DataType.Int)
            S2 = fd.define_scalar(1, dtype=DataType.Int)
            T3 = fd.ops.iota(S0, S1, S2, dtype=DataType.Int)
            T4 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
            S5 = fd.define_scalar(128, dtype=DataType.Int)
            S6 = fd.ops.size(T3, dim=0)
            V7 = fd.define_vector([S5, S6], dtype=DataType.Int)
            T8 = fd.ops.expand(T4, shape=V7)
            S9 = fd.define_scalar(128, dtype=DataType.Int)
            S10 = fd.define_scalar(0, dtype=DataType.Int)
            S11 = fd.define_scalar(1, dtype=DataType.Int)
            T12 = fd.ops.iota(S9, S10, S11, dtype=DataType.Int)
            T13 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
            S14 = fd.ops.size(T12, dim=0)
            S15 = fd.define_scalar(129, dtype=DataType.Int)
            V16 = fd.define_vector([S14, S15], dtype=DataType.Int)
            T17 = fd.ops.expand(T13, shape=V16)
            T18 = fd.ops.gt(T8, T17)
            T19 = fd.ops.broadcast(T3, is_broadcast_dim=[True, False])
            S20 = fd.define_scalar(128, dtype=DataType.Int)
            V21 = fd.define_vector([S20, S6], dtype=DataType.Int)
            T22 = fd.ops.expand(T19, shape=V21)
            T23 = fd.ops.broadcast(T12, is_broadcast_dim=[False, True])
            S24 = fd.define_scalar(129, dtype=DataType.Int)
            V25 = fd.define_vector([S14, S24], dtype=DataType.Int)
            T26 = fd.ops.expand(T23, shape=V25)
            T27 = fd.ops.sub(T22, T26)
            S28 = fd.define_scalar(1, dtype=DataType.Int)
            T29 = fd.ops.ge(T27, S28)
            S30 = fd.define_scalar(-3.38953e38, dtype=DataType.BFloat16)
            S31 = fd.define_scalar(128, dtype=DataType.Int)
            S32 = fd.define_scalar(129, dtype=DataType.Int)
            V33 = fd.define_vector([S31, S32], dtype=DataType.Int)
            T34 = fd.ops.full(shape=V33, fill_value=S30, dtype=DataType.BFloat16)
            S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
            T36 = fd.ops.where(T29, T34, S35)
            fd.add_output(T18)
            fd.add_output(T36)

        with FusionDefinition() as fd:
            nvfuser_fusion_id0(fd)

        inputs = []
        fd.execute(inputs)
