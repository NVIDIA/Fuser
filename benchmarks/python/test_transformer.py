# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
This file micro-benchmarks the forward pass and the backprop of a Transformer
block used in GPT-3. The nvFusions are dumped from Thunder. To regenerate
the nvFusions and the inputs, run the following:

1. `git clone https://github.com/Lightning-AI/lightning-thunder.git`

2. `git fetch origin wjy/sharded`

3. `git checkout wjy/sharded`
This branch adds the GPT-3 block benchmark and turns on certain knobs so the
entire Transformer block fits into one nvFusion.

4. Apply the following patch
```
diff --git a/nvfuser/__init__.py b/nvfuser/__init__.py
index 8be5df3d..69bcf450 100644
--- a/nvfuser/__init__.py
+++ b/nvfuser/__init__.py
@@ -214,8 +214,8 @@ class FusionDefinition(_C._FusionDefinition):
                 capture_debug_output=capture_debug_output,
                 profile=profile,
             )
+            print(self.getReproErrorString("executing", inputs))
         except Exception as err:
-            logger.exception(self.getReproErrorString("executing", inputs))
             raise

         return result
```

5. `pytest thunder/benchmarks/targets.py -k 'test_nanogpt_block[backward-thunder]' -s`
In stdout, you'll find the forward nvFusion executed once followed by the
backward nvFusion executed many times.
"""

from nvfuser import FusionDefinition, DataType
from .core import run_benchmark, clear_cuda_cache
import torch


def create_transformer_forward(fd: FusionDefinition) -> None:
    S0 = fd.define_scalar(None, dtype=DataType.Int)
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    S4 = fd.define_scalar(None, dtype=DataType.Int)
    S5 = fd.define_scalar(None, dtype=DataType.Int)
    T6 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T7 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T8 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T9 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T10 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T11 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T12 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T13 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T14 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T15 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T16 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T17 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T18 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T19 = fd.ops.cast(T18, dtype=DataType.Float)
    T20, T21 = fd.ops.var_mean(T19, dims=[2], correction=0, keepdim=False)
    S22 = fd.define_scalar(1, dtype=DataType.Int)
    S23 = fd.define_scalar(2048, dtype=DataType.Int)
    S24 = fd.define_scalar(1, dtype=DataType.Int)
    V25 = fd.define_vector([S22, S23, S24], dtype=DataType.Int)
    T26 = fd.ops.broadcast_in_dim(T20, shape=V25, broadcast_dims=[0, 1])
    S27 = fd.define_scalar(1, dtype=DataType.Int)
    S28 = fd.define_scalar(2048, dtype=DataType.Int)
    S29 = fd.define_scalar(1, dtype=DataType.Int)
    V30 = fd.define_vector([S27, S28, S29], dtype=DataType.Int)
    T31 = fd.ops.broadcast_in_dim(T21, shape=V30, broadcast_dims=[0, 1])
    S32 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T33 = fd.ops.add(T26, S32)
    T34 = fd.ops.rsqrt(T33)
    S35 = fd.define_scalar(1, dtype=DataType.Int)
    S36 = fd.define_scalar(2048, dtype=DataType.Int)
    S37 = fd.define_scalar(12288, dtype=DataType.Int)
    V38 = fd.define_vector([S35, S36, S37], dtype=DataType.Int)
    T39 = fd.ops.broadcast_in_dim(T31, shape=V38, broadcast_dims=[0, 1, 2])
    T40 = fd.ops.sub(T19, T39)
    S41 = fd.define_scalar(1, dtype=DataType.Int)
    S42 = fd.define_scalar(2048, dtype=DataType.Int)
    S43 = fd.define_scalar(12288, dtype=DataType.Int)
    V44 = fd.define_vector([S41, S42, S43], dtype=DataType.Int)
    T45 = fd.ops.broadcast_in_dim(T34, shape=V44, broadcast_dims=[0, 1, 2])
    T46 = fd.ops.mul(T40, T45)
    S47 = fd.define_scalar(1, dtype=DataType.Int)
    S48 = fd.define_scalar(2048, dtype=DataType.Int)
    S49 = fd.define_scalar(12288, dtype=DataType.Int)
    V50 = fd.define_vector([S47, S48, S49], dtype=DataType.Int)
    T51 = fd.ops.broadcast_in_dim(T11, shape=V50, broadcast_dims=[2])
    T52 = fd.ops.cast(T51, dtype=DataType.Float)
    T53 = fd.ops.mul(T46, T52)
    S54 = fd.define_scalar(1, dtype=DataType.Int)
    S55 = fd.define_scalar(2048, dtype=DataType.Int)
    S56 = fd.define_scalar(12288, dtype=DataType.Int)
    V57 = fd.define_vector([S54, S55, S56], dtype=DataType.Int)
    T58 = fd.ops.broadcast_in_dim(T10, shape=V57, broadcast_dims=[2])
    T59 = fd.ops.cast(T58, dtype=DataType.Float)
    T60 = fd.ops.add(T53, T59)
    T61 = fd.ops.cast(T60, dtype=DataType.BFloat16)
    T62 = fd.ops.linear(T61, T7, T6)
    T63 = fd.ops.slice(
        T62, start_indices=[0, 0, 0], end_indices=[1, 2048, 12288], strides=[1, 1, 1]
    )
    T64 = fd.ops.slice(
        T62,
        start_indices=[0, 0, 12288],
        end_indices=[1, 2048, 24576],
        strides=[1, 1, 1],
    )
    T65 = fd.ops.slice(
        T62,
        start_indices=[0, 0, 24576],
        end_indices=[1, 2048, 36864],
        strides=[1, 1, 1],
    )
    S66 = fd.define_scalar(1, dtype=DataType.Int)
    S67 = fd.define_scalar(2048, dtype=DataType.Int)
    S68 = fd.define_scalar(96, dtype=DataType.Int)
    S69 = fd.define_scalar(128, dtype=DataType.Int)
    V70 = fd.define_vector([S66, S67, S68, S69], dtype=DataType.Int)
    T71 = fd.ops.reshape(T64, new_shape=V70)
    T72 = fd.ops.permute(T71, dims=[0, 2, 1, 3])
    S73 = fd.define_scalar(1, dtype=DataType.Int)
    S74 = fd.define_scalar(2048, dtype=DataType.Int)
    S75 = fd.define_scalar(96, dtype=DataType.Int)
    S76 = fd.define_scalar(128, dtype=DataType.Int)
    V77 = fd.define_vector([S73, S74, S75, S76], dtype=DataType.Int)
    T78 = fd.ops.reshape(T63, new_shape=V77)
    T79 = fd.ops.permute(T78, dims=[0, 2, 1, 3])
    S80 = fd.define_scalar(1, dtype=DataType.Int)
    S81 = fd.define_scalar(2048, dtype=DataType.Int)
    S82 = fd.define_scalar(96, dtype=DataType.Int)
    S83 = fd.define_scalar(128, dtype=DataType.Int)
    V84 = fd.define_vector([S80, S81, S82, S83], dtype=DataType.Int)
    T85 = fd.ops.reshape(T65, new_shape=V84)
    T86 = fd.ops.permute(T85, dims=[0, 2, 1, 3])
    T87 = fd.ops.cast(T79, dtype=DataType.Float)
    S88 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T89 = fd.ops.mul(T87, S88)
    T90 = fd.ops.cast(T89, dtype=DataType.BFloat16)
    T91 = fd.ops.permute(T72, dims=[0, 1, 3, 2])
    T92 = fd.ops.cast(T91, dtype=DataType.Float)
    S93 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T94 = fd.ops.mul(T92, S93)
    T95 = fd.ops.cast(T94, dtype=DataType.BFloat16)
    T96 = fd.ops.matmul(T90, T95)
    S97 = fd.define_scalar(2048, dtype=DataType.Int)
    S98 = fd.define_scalar(0, dtype=DataType.Int)
    S99 = fd.define_scalar(1, dtype=DataType.Int)
    T100 = fd.ops.iota(S97, S98, S99, dtype=DataType.Int)
    S101 = fd.define_scalar(2048, dtype=DataType.Int)
    S102 = fd.define_scalar(1, dtype=DataType.Int)
    V103 = fd.define_vector([S101, S102], dtype=DataType.Int)
    T104 = fd.ops.broadcast_in_dim(T100, shape=V103, broadcast_dims=[0])
    S105 = fd.define_scalar(1, dtype=DataType.Int)
    S106 = fd.define_scalar(2048, dtype=DataType.Int)
    V107 = fd.define_vector([S105, S106], dtype=DataType.Int)
    T108 = fd.ops.broadcast_in_dim(T100, shape=V107, broadcast_dims=[1])
    S109 = fd.define_scalar(0, dtype=DataType.Int)
    T110 = fd.ops.add(T104, S109)
    S111 = fd.define_scalar(2048, dtype=DataType.Int)
    S112 = fd.define_scalar(2048, dtype=DataType.Int)
    V113 = fd.define_vector([S111, S112], dtype=DataType.Int)
    T114 = fd.ops.broadcast_in_dim(T110, shape=V113, broadcast_dims=[0, 1])
    S115 = fd.define_scalar(2048, dtype=DataType.Int)
    S116 = fd.define_scalar(2048, dtype=DataType.Int)
    V117 = fd.define_vector([S115, S116], dtype=DataType.Int)
    T118 = fd.ops.broadcast_in_dim(T108, shape=V117, broadcast_dims=[0, 1])
    T119 = fd.ops.ge(T114, T118)
    S120 = fd.define_scalar(1, dtype=DataType.Int)
    S121 = fd.define_scalar(96, dtype=DataType.Int)
    S122 = fd.define_scalar(2048, dtype=DataType.Int)
    S123 = fd.define_scalar(2048, dtype=DataType.Int)
    V124 = fd.define_vector([S120, S121, S122, S123], dtype=DataType.Int)
    T125 = fd.ops.broadcast_in_dim(T119, shape=V124, broadcast_dims=[2, 3])
    S126 = fd.define_scalar(float("-inf"), dtype=DataType.Double)
    T127 = fd.ops.where(T125, T96, S126)
    T128 = fd.ops.cast(T127, dtype=DataType.Float)
    T129 = fd.ops.max(T128, dims=[3], keepdim=False, dtype=DataType.Null)
    S130 = fd.define_scalar(1, dtype=DataType.Int)
    S131 = fd.define_scalar(96, dtype=DataType.Int)
    S132 = fd.define_scalar(2048, dtype=DataType.Int)
    S133 = fd.define_scalar(1, dtype=DataType.Int)
    V134 = fd.define_vector([S130, S131, S132, S133], dtype=DataType.Int)
    T135 = fd.ops.broadcast_in_dim(T129, shape=V134, broadcast_dims=[0, 1, 2])
    S136 = fd.define_scalar(1, dtype=DataType.Int)
    S137 = fd.define_scalar(96, dtype=DataType.Int)
    S138 = fd.define_scalar(2048, dtype=DataType.Int)
    S139 = fd.define_scalar(2048, dtype=DataType.Int)
    V140 = fd.define_vector([S136, S137, S138, S139], dtype=DataType.Int)
    T141 = fd.ops.broadcast_in_dim(T135, shape=V140, broadcast_dims=[0, 1, 2, 3])
    T142 = fd.ops.sub(T128, T141)
    T143 = fd.ops.exp(T142)
    T144 = fd.ops.sum(T143, dims=[3], keepdim=False, dtype=DataType.Null)
    S145 = fd.define_scalar(1, dtype=DataType.Int)
    S146 = fd.define_scalar(96, dtype=DataType.Int)
    S147 = fd.define_scalar(2048, dtype=DataType.Int)
    S148 = fd.define_scalar(1, dtype=DataType.Int)
    V149 = fd.define_vector([S145, S146, S147, S148], dtype=DataType.Int)
    T150 = fd.ops.broadcast_in_dim(T144, shape=V149, broadcast_dims=[0, 1, 2])
    S151 = fd.define_scalar(1, dtype=DataType.Int)
    S152 = fd.define_scalar(96, dtype=DataType.Int)
    S153 = fd.define_scalar(2048, dtype=DataType.Int)
    S154 = fd.define_scalar(2048, dtype=DataType.Int)
    V155 = fd.define_vector([S151, S152, S153, S154], dtype=DataType.Int)
    T156 = fd.ops.broadcast_in_dim(T150, shape=V155, broadcast_dims=[0, 1, 2, 3])
    T157 = fd.ops.reciprocal(T156)
    T158 = fd.ops.mul(T143, T157)
    T159 = fd.ops.cast(T158, dtype=DataType.BFloat16)
    S160 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S161 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S162 = fd.define_scalar(1, dtype=DataType.Int)
    S163 = fd.define_scalar(96, dtype=DataType.Int)
    S164 = fd.define_scalar(2048, dtype=DataType.Int)
    S165 = fd.define_scalar(2048, dtype=DataType.Int)
    V166 = fd.define_vector([S162, S163, S164, S165], dtype=DataType.Int)
    T167 = fd.ops.uniform(
        S160, S161, shape=V166, rng_seed=S0, rng_offset=S1, dtype=DataType.BFloat16
    )
    S168 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T169 = fd.ops.lt(T167, S168)
    T170 = fd.ops.cast(T169, dtype=DataType.Float)
    T171 = fd.ops.mul(T158, T170)
    S172 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T173 = fd.ops.mul(T171, S172)
    T174 = fd.ops.cast(T173, dtype=DataType.BFloat16)
    T175 = fd.ops.matmul(T174, T86)
    T176 = fd.ops.permute(T175, dims=[0, 2, 1, 3])
    T177 = fd.ops.stride_order(T176, stride_order=[3, 2, 1, 0])
    S178 = fd.define_scalar(1, dtype=DataType.Int)
    S179 = fd.define_scalar(2048, dtype=DataType.Int)
    S180 = fd.define_scalar(12288, dtype=DataType.Int)
    V181 = fd.define_vector([S178, S179, S180], dtype=DataType.Int)
    T182 = fd.ops.reshape(T177, new_shape=V181)
    T183 = fd.ops.linear(T182, T9, T8)
    S184 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S185 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S186 = fd.define_scalar(1, dtype=DataType.Int)
    S187 = fd.define_scalar(2048, dtype=DataType.Int)
    S188 = fd.define_scalar(12288, dtype=DataType.Int)
    V189 = fd.define_vector([S186, S187, S188], dtype=DataType.Int)
    T190 = fd.ops.uniform(
        S184, S185, shape=V189, rng_seed=S2, rng_offset=S3, dtype=DataType.BFloat16
    )
    S191 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T192 = fd.ops.lt(T190, S191)
    T193 = fd.ops.cast(T183, dtype=DataType.Float)
    T194 = fd.ops.cast(T192, dtype=DataType.Float)
    T195 = fd.ops.mul(T193, T194)
    S196 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T197 = fd.ops.mul(T195, S196)
    T198 = fd.ops.add(T19, T197)
    T199, T200 = fd.ops.var_mean(T198, dims=[2], correction=0, keepdim=False)
    S201 = fd.define_scalar(1, dtype=DataType.Int)
    S202 = fd.define_scalar(2048, dtype=DataType.Int)
    S203 = fd.define_scalar(1, dtype=DataType.Int)
    V204 = fd.define_vector([S201, S202, S203], dtype=DataType.Int)
    T205 = fd.ops.broadcast_in_dim(T199, shape=V204, broadcast_dims=[0, 1])
    S206 = fd.define_scalar(1, dtype=DataType.Int)
    S207 = fd.define_scalar(2048, dtype=DataType.Int)
    S208 = fd.define_scalar(1, dtype=DataType.Int)
    V209 = fd.define_vector([S206, S207, S208], dtype=DataType.Int)
    T210 = fd.ops.broadcast_in_dim(T200, shape=V209, broadcast_dims=[0, 1])
    S211 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T212 = fd.ops.add(T205, S211)
    T213 = fd.ops.rsqrt(T212)
    S214 = fd.define_scalar(1, dtype=DataType.Int)
    S215 = fd.define_scalar(2048, dtype=DataType.Int)
    S216 = fd.define_scalar(12288, dtype=DataType.Int)
    V217 = fd.define_vector([S214, S215, S216], dtype=DataType.Int)
    T218 = fd.ops.broadcast_in_dim(T210, shape=V217, broadcast_dims=[0, 1, 2])
    T219 = fd.ops.sub(T198, T218)
    S220 = fd.define_scalar(1, dtype=DataType.Int)
    S221 = fd.define_scalar(2048, dtype=DataType.Int)
    S222 = fd.define_scalar(12288, dtype=DataType.Int)
    V223 = fd.define_vector([S220, S221, S222], dtype=DataType.Int)
    T224 = fd.ops.broadcast_in_dim(T213, shape=V223, broadcast_dims=[0, 1, 2])
    T225 = fd.ops.mul(T219, T224)
    S226 = fd.define_scalar(1, dtype=DataType.Int)
    S227 = fd.define_scalar(2048, dtype=DataType.Int)
    S228 = fd.define_scalar(12288, dtype=DataType.Int)
    V229 = fd.define_vector([S226, S227, S228], dtype=DataType.Int)
    T230 = fd.ops.broadcast_in_dim(T13, shape=V229, broadcast_dims=[2])
    T231 = fd.ops.cast(T230, dtype=DataType.Float)
    T232 = fd.ops.mul(T225, T231)
    S233 = fd.define_scalar(1, dtype=DataType.Int)
    S234 = fd.define_scalar(2048, dtype=DataType.Int)
    S235 = fd.define_scalar(12288, dtype=DataType.Int)
    V236 = fd.define_vector([S233, S234, S235], dtype=DataType.Int)
    T237 = fd.ops.broadcast_in_dim(T12, shape=V236, broadcast_dims=[2])
    T238 = fd.ops.cast(T237, dtype=DataType.Float)
    T239 = fd.ops.add(T232, T238)
    T240 = fd.ops.cast(T239, dtype=DataType.BFloat16)
    T241 = fd.ops.linear(T240, T15, T14)
    T242 = fd.ops.cast(T241, dtype=DataType.Float)
    T243 = fd.ops.mul(T242, T242)
    T244 = fd.ops.mul(T243, T242)
    S245 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T246 = fd.ops.mul(S245, T242)
    S247 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T248 = fd.ops.mul(S247, T244)
    T249 = fd.ops.add(T242, T248)
    S250 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T251 = fd.ops.mul(S250, T249)
    T252 = fd.ops.tanh(T251)
    S253 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T254 = fd.ops.add(S253, T252)
    T255 = fd.ops.mul(T246, T254)
    T256 = fd.ops.cast(T255, dtype=DataType.BFloat16)
    T257 = fd.ops.linear(T256, T17, T16)
    S258 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S259 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S260 = fd.define_scalar(1, dtype=DataType.Int)
    S261 = fd.define_scalar(2048, dtype=DataType.Int)
    S262 = fd.define_scalar(12288, dtype=DataType.Int)
    V263 = fd.define_vector([S260, S261, S262], dtype=DataType.Int)
    T264 = fd.ops.uniform(
        S258, S259, shape=V263, rng_seed=S4, rng_offset=S5, dtype=DataType.BFloat16
    )
    S265 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T266 = fd.ops.lt(T264, S265)
    T267 = fd.ops.cast(T257, dtype=DataType.Float)
    T268 = fd.ops.cast(T266, dtype=DataType.Float)
    T269 = fd.ops.mul(T267, T268)
    S270 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T271 = fd.ops.mul(T269, S270)
    T272 = fd.ops.add(T198, T271)
    T273 = fd.ops.cast(T272, dtype=DataType.BFloat16)
    fd.add_output(T200)
    fd.add_output(T213)
    fd.add_output(T273)
    fd.add_output(T21)
    fd.add_output(T159)
    fd.add_output(T34)
    fd.add_output(T174)


def test_transformer_forward(
    benchmark, disable_validation: bool, disable_benchmarking: bool
):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        create_transformer_forward(fd)

    inputs = [
        2757501781750758,
        29,
        2757501781750758,
        30,
        2757501781750758,
        31,
        torch.randn((36864,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864,), (1,)
        ),
        torch.randn((452984832,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864, 12288), (12288, 1)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((150994944,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 12288), (12288, 1)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((49152,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152,), (1,)
        ),
        torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152, 12288), (12288, 1)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 49152), (49152, 1)
        ),
        torch.randn((25165824,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 12288), (25165824, 12288, 1)
        ),
    ]

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


def create_transformer_backward(fd: FusionDefinition) -> None:
    S0 = fd.define_scalar(None, dtype=DataType.Int)
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    S4 = fd.define_scalar(None, dtype=DataType.Int)
    S5 = fd.define_scalar(None, dtype=DataType.Int)
    T6 = fd.define_tensor(
        shape=[1, -1],
        contiguity=[None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T7 = fd.define_tensor(
        shape=[1, -1, 1],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T8 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T9 = fd.define_tensor(
        shape=[1, -1],
        contiguity=[None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T10 = fd.define_tensor(
        shape=[1, -1, -1, -1],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T11 = fd.define_tensor(
        shape=[1, -1, 1],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T12 = fd.define_tensor(
        shape=[1, -1, -1, -1],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T13 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T14 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T15 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T16 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T17 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T18 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T19 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T20 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T21 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T22 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T23 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T24 = fd.define_tensor(
        shape=[1, -1, -1],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T25 = fd.ops.cast(T24, dtype=DataType.Float)
    S26 = fd.define_scalar(1, dtype=DataType.Int)
    S27 = fd.define_scalar(2048, dtype=DataType.Int)
    S28 = fd.define_scalar(1, dtype=DataType.Int)
    V29 = fd.define_vector([S26, S27, S28], dtype=DataType.Int)
    T30 = fd.ops.broadcast_in_dim(T9, shape=V29, broadcast_dims=[0, 1])
    S31 = fd.define_scalar(1, dtype=DataType.Int)
    S32 = fd.define_scalar(2048, dtype=DataType.Int)
    S33 = fd.define_scalar(12288, dtype=DataType.Int)
    V34 = fd.define_vector([S31, S32, S33], dtype=DataType.Int)
    T35 = fd.ops.broadcast_in_dim(T30, shape=V34, broadcast_dims=[0, 1, 2])
    T36 = fd.ops.sub(T25, T35)
    S37 = fd.define_scalar(1, dtype=DataType.Int)
    S38 = fd.define_scalar(2048, dtype=DataType.Int)
    S39 = fd.define_scalar(12288, dtype=DataType.Int)
    V40 = fd.define_vector([S37, S38, S39], dtype=DataType.Int)
    T41 = fd.ops.broadcast_in_dim(T11, shape=V40, broadcast_dims=[0, 1, 2])
    T42 = fd.ops.mul(T36, T41)
    S43 = fd.define_scalar(1, dtype=DataType.Int)
    S44 = fd.define_scalar(2048, dtype=DataType.Int)
    S45 = fd.define_scalar(12288, dtype=DataType.Int)
    V46 = fd.define_vector([S43, S44, S45], dtype=DataType.Int)
    T47 = fd.ops.broadcast_in_dim(T18, shape=V46, broadcast_dims=[2])
    T48 = fd.ops.cast(T47, dtype=DataType.Float)
    T49 = fd.ops.mul(T42, T48)
    S50 = fd.define_scalar(1, dtype=DataType.Int)
    S51 = fd.define_scalar(2048, dtype=DataType.Int)
    S52 = fd.define_scalar(12288, dtype=DataType.Int)
    V53 = fd.define_vector([S50, S51, S52], dtype=DataType.Int)
    T54 = fd.ops.broadcast_in_dim(T17, shape=V53, broadcast_dims=[2])
    T55 = fd.ops.cast(T54, dtype=DataType.Float)
    T56 = fd.ops.add(T49, T55)
    T57 = fd.ops.cast(T56, dtype=DataType.BFloat16)
    T58 = fd.ops.linear(T57, T14, T13)
    T59 = fd.ops.slice(
        T58, start_indices=[0, 0, 0], end_indices=[1, 2048, 12288], strides=[1, 1, 1]
    )
    T60 = fd.ops.slice(
        T58,
        start_indices=[0, 0, 12288],
        end_indices=[1, 2048, 24576],
        strides=[1, 1, 1],
    )
    T61 = fd.ops.slice(
        T58,
        start_indices=[0, 0, 24576],
        end_indices=[1, 2048, 36864],
        strides=[1, 1, 1],
    )
    S62 = fd.define_scalar(1, dtype=DataType.Int)
    S63 = fd.define_scalar(2048, dtype=DataType.Int)
    S64 = fd.define_scalar(96, dtype=DataType.Int)
    S65 = fd.define_scalar(128, dtype=DataType.Int)
    V66 = fd.define_vector([S62, S63, S64, S65], dtype=DataType.Int)
    T67 = fd.ops.reshape(T60, new_shape=V66)
    T68 = fd.ops.permute(T67, dims=[0, 2, 1, 3])
    S69 = fd.define_scalar(1, dtype=DataType.Int)
    S70 = fd.define_scalar(2048, dtype=DataType.Int)
    S71 = fd.define_scalar(96, dtype=DataType.Int)
    S72 = fd.define_scalar(128, dtype=DataType.Int)
    V73 = fd.define_vector([S69, S70, S71, S72], dtype=DataType.Int)
    T74 = fd.ops.reshape(T59, new_shape=V73)
    T75 = fd.ops.permute(T74, dims=[0, 2, 1, 3])
    S76 = fd.define_scalar(1, dtype=DataType.Int)
    S77 = fd.define_scalar(2048, dtype=DataType.Int)
    S78 = fd.define_scalar(96, dtype=DataType.Int)
    S79 = fd.define_scalar(128, dtype=DataType.Int)
    V80 = fd.define_vector([S76, S77, S78, S79], dtype=DataType.Int)
    T81 = fd.ops.reshape(T61, new_shape=V80)
    T82 = fd.ops.permute(T81, dims=[0, 2, 1, 3])
    T83 = fd.ops.cast(T75, dtype=DataType.Float)
    S84 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T85 = fd.ops.mul(T83, S84)
    T86 = fd.ops.cast(T85, dtype=DataType.BFloat16)
    T87 = fd.ops.permute(T68, dims=[0, 1, 3, 2])
    T88 = fd.ops.cast(T87, dtype=DataType.Float)
    S89 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T90 = fd.ops.mul(T88, S89)
    T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
    S92 = fd.define_scalar(2048, dtype=DataType.Int)
    S93 = fd.define_scalar(0, dtype=DataType.Int)
    S94 = fd.define_scalar(1, dtype=DataType.Int)
    T95 = fd.ops.iota(S92, S93, S94, dtype=DataType.Int)
    S96 = fd.define_scalar(2048, dtype=DataType.Int)
    S97 = fd.define_scalar(1, dtype=DataType.Int)
    V98 = fd.define_vector([S96, S97], dtype=DataType.Int)
    T99 = fd.ops.broadcast_in_dim(T95, shape=V98, broadcast_dims=[0])
    S100 = fd.define_scalar(1, dtype=DataType.Int)
    S101 = fd.define_scalar(2048, dtype=DataType.Int)
    V102 = fd.define_vector([S100, S101], dtype=DataType.Int)
    T103 = fd.ops.broadcast_in_dim(T95, shape=V102, broadcast_dims=[1])
    S104 = fd.define_scalar(0, dtype=DataType.Int)
    T105 = fd.ops.add(T99, S104)
    S106 = fd.define_scalar(2048, dtype=DataType.Int)
    S107 = fd.define_scalar(2048, dtype=DataType.Int)
    V108 = fd.define_vector([S106, S107], dtype=DataType.Int)
    T109 = fd.ops.broadcast_in_dim(T105, shape=V108, broadcast_dims=[0, 1])
    S110 = fd.define_scalar(2048, dtype=DataType.Int)
    S111 = fd.define_scalar(2048, dtype=DataType.Int)
    V112 = fd.define_vector([S110, S111], dtype=DataType.Int)
    T113 = fd.ops.broadcast_in_dim(T103, shape=V112, broadcast_dims=[0, 1])
    T114 = fd.ops.ge(T109, T113)
    S115 = fd.define_scalar(1, dtype=DataType.Int)
    S116 = fd.define_scalar(96, dtype=DataType.Int)
    S117 = fd.define_scalar(2048, dtype=DataType.Int)
    S118 = fd.define_scalar(2048, dtype=DataType.Int)
    V119 = fd.define_vector([S115, S116, S117, S118], dtype=DataType.Int)
    T120 = fd.ops.broadcast_in_dim(T114, shape=V119, broadcast_dims=[2, 3])
    S121 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S122 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S123 = fd.define_scalar(1, dtype=DataType.Int)
    S124 = fd.define_scalar(96, dtype=DataType.Int)
    S125 = fd.define_scalar(2048, dtype=DataType.Int)
    S126 = fd.define_scalar(2048, dtype=DataType.Int)
    V127 = fd.define_vector([S123, S124, S125, S126], dtype=DataType.Int)
    T128 = fd.ops.uniform(
        S121, S122, shape=V127, rng_seed=S0, rng_offset=S1, dtype=DataType.BFloat16
    )
    S129 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T130 = fd.ops.lt(T128, S129)
    T131 = fd.ops.cast(T130, dtype=DataType.Float)
    T132 = fd.ops.matmul(T12, T82)
    T133 = fd.ops.permute(T132, dims=[0, 2, 1, 3])
    T134 = fd.ops.stride_order(T133, stride_order=[3, 2, 1, 0])
    S135 = fd.define_scalar(1, dtype=DataType.Int)
    S136 = fd.define_scalar(2048, dtype=DataType.Int)
    S137 = fd.define_scalar(12288, dtype=DataType.Int)
    V138 = fd.define_vector([S135, S136, S137], dtype=DataType.Int)
    T139 = fd.ops.reshape(T134, new_shape=V138)
    T140 = fd.ops.linear(T139, T16, T15)
    S141 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S142 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S143 = fd.define_scalar(1, dtype=DataType.Int)
    S144 = fd.define_scalar(2048, dtype=DataType.Int)
    S145 = fd.define_scalar(12288, dtype=DataType.Int)
    V146 = fd.define_vector([S143, S144, S145], dtype=DataType.Int)
    T147 = fd.ops.uniform(
        S141, S142, shape=V146, rng_seed=S2, rng_offset=S3, dtype=DataType.BFloat16
    )
    S148 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T149 = fd.ops.lt(T147, S148)
    T150 = fd.ops.cast(T140, dtype=DataType.Float)
    T151 = fd.ops.cast(T149, dtype=DataType.Float)
    T152 = fd.ops.mul(T150, T151)
    S153 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T154 = fd.ops.mul(T152, S153)
    T155 = fd.ops.add(T25, T154)
    S156 = fd.define_scalar(1, dtype=DataType.Int)
    S157 = fd.define_scalar(2048, dtype=DataType.Int)
    S158 = fd.define_scalar(1, dtype=DataType.Int)
    V159 = fd.define_vector([S156, S157, S158], dtype=DataType.Int)
    T160 = fd.ops.broadcast_in_dim(T6, shape=V159, broadcast_dims=[0, 1])
    S161 = fd.define_scalar(1, dtype=DataType.Int)
    S162 = fd.define_scalar(2048, dtype=DataType.Int)
    S163 = fd.define_scalar(12288, dtype=DataType.Int)
    V164 = fd.define_vector([S161, S162, S163], dtype=DataType.Int)
    T165 = fd.ops.broadcast_in_dim(T160, shape=V164, broadcast_dims=[0, 1, 2])
    T166 = fd.ops.sub(T155, T165)
    S167 = fd.define_scalar(1, dtype=DataType.Int)
    S168 = fd.define_scalar(2048, dtype=DataType.Int)
    S169 = fd.define_scalar(12288, dtype=DataType.Int)
    V170 = fd.define_vector([S167, S168, S169], dtype=DataType.Int)
    T171 = fd.ops.broadcast_in_dim(T7, shape=V170, broadcast_dims=[0, 1, 2])
    T172 = fd.ops.mul(T166, T171)
    S173 = fd.define_scalar(1, dtype=DataType.Int)
    S174 = fd.define_scalar(2048, dtype=DataType.Int)
    S175 = fd.define_scalar(12288, dtype=DataType.Int)
    V176 = fd.define_vector([S173, S174, S175], dtype=DataType.Int)
    T177 = fd.ops.broadcast_in_dim(T20, shape=V176, broadcast_dims=[2])
    T178 = fd.ops.cast(T177, dtype=DataType.Float)
    T179 = fd.ops.mul(T172, T178)
    S180 = fd.define_scalar(1, dtype=DataType.Int)
    S181 = fd.define_scalar(2048, dtype=DataType.Int)
    S182 = fd.define_scalar(12288, dtype=DataType.Int)
    V183 = fd.define_vector([S180, S181, S182], dtype=DataType.Int)
    T184 = fd.ops.broadcast_in_dim(T19, shape=V183, broadcast_dims=[2])
    T185 = fd.ops.cast(T184, dtype=DataType.Float)
    T186 = fd.ops.add(T179, T185)
    T187 = fd.ops.cast(T186, dtype=DataType.BFloat16)
    T188 = fd.ops.linear(T187, T22, T21)
    T189 = fd.ops.cast(T188, dtype=DataType.Float)
    T190 = fd.ops.mul(T189, T189)
    T191 = fd.ops.mul(T190, T189)
    S192 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T193 = fd.ops.mul(S192, T189)
    S194 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T195 = fd.ops.mul(S194, T191)
    T196 = fd.ops.add(T189, T195)
    S197 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T198 = fd.ops.mul(S197, T196)
    T199 = fd.ops.tanh(T198)
    S200 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T201 = fd.ops.add(S200, T199)
    T202 = fd.ops.mul(T193, T201)
    T203 = fd.ops.cast(T202, dtype=DataType.BFloat16)
    S204 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S205 = fd.define_scalar(1.00000, dtype=DataType.Double)
    S206 = fd.define_scalar(1, dtype=DataType.Int)
    S207 = fd.define_scalar(2048, dtype=DataType.Int)
    S208 = fd.define_scalar(12288, dtype=DataType.Int)
    V209 = fd.define_vector([S206, S207, S208], dtype=DataType.Int)
    T210 = fd.ops.uniform(
        S204, S205, shape=V209, rng_seed=S4, rng_offset=S5, dtype=DataType.BFloat16
    )
    S211 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T212 = fd.ops.lt(T210, S211)
    T213 = fd.ops.cast(T212, dtype=DataType.Float)
    T214 = fd.ops.cast(T8, dtype=DataType.Float)
    S215 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T216 = fd.ops.mul(S215, T214)
    T217 = fd.ops.mul(T213, T216)
    T218 = fd.ops.cast(T217, dtype=DataType.BFloat16)
    S219 = fd.define_scalar(2048, dtype=DataType.Int)
    S220 = fd.define_scalar(12288, dtype=DataType.Int)
    V221 = fd.define_vector([S219, S220], dtype=DataType.Int)
    T222 = fd.ops.reshape(T218, new_shape=V221)
    T223 = fd.ops.matmul(T222, T23)
    S224 = fd.define_scalar(1, dtype=DataType.Int)
    S225 = fd.define_scalar(2048, dtype=DataType.Int)
    S226 = fd.define_scalar(49152, dtype=DataType.Int)
    V227 = fd.define_vector([S224, S225, S226], dtype=DataType.Int)
    T228 = fd.ops.reshape(T223, new_shape=V227)
    T229 = fd.ops.permute(T222, dims=[1, 0])
    S230 = fd.define_scalar(2048, dtype=DataType.Int)
    S231 = fd.define_scalar(49152, dtype=DataType.Int)
    V232 = fd.define_vector([S230, S231], dtype=DataType.Int)
    T233 = fd.ops.reshape(T203, new_shape=V232)
    T234 = fd.ops.matmul(T229, T233)
    T235 = fd.ops.sum(T217, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T236 = fd.ops.cast(T235, dtype=DataType.BFloat16)
    T237 = fd.ops.cast(T228, dtype=DataType.Float)
    T238 = fd.ops.mul(T201, T237)
    T239 = fd.ops.mul(T193, T237)
    T240 = fd.ops.mul(T199, T199)
    S241 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T242 = fd.ops.sub(S241, T240)
    T243 = fd.ops.mul(T239, T242)
    S244 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T245 = fd.ops.mul(S244, T243)
    S246 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T247 = fd.ops.mul(S246, T245)
    S248 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T249 = fd.ops.mul(S248, T238)
    T250 = fd.ops.add(T245, T249)
    T251 = fd.ops.mul(T189, T247)
    T252 = fd.ops.mul(T190, T247)
    T253 = fd.ops.add(T250, T252)
    T254 = fd.ops.mul(T189, T251)
    T255 = fd.ops.add(T253, T254)
    T256 = fd.ops.add(T255, T254)
    T257 = fd.ops.cast(T256, dtype=DataType.BFloat16)
    S258 = fd.define_scalar(2048, dtype=DataType.Int)
    S259 = fd.define_scalar(49152, dtype=DataType.Int)
    V260 = fd.define_vector([S258, S259], dtype=DataType.Int)
    T261 = fd.ops.reshape(T257, new_shape=V260)
    T262 = fd.ops.matmul(T261, T22)
    S263 = fd.define_scalar(1, dtype=DataType.Int)
    S264 = fd.define_scalar(2048, dtype=DataType.Int)
    S265 = fd.define_scalar(12288, dtype=DataType.Int)
    V266 = fd.define_vector([S263, S264, S265], dtype=DataType.Int)
    T267 = fd.ops.reshape(T262, new_shape=V266)
    T268 = fd.ops.permute(T261, dims=[1, 0])
    S269 = fd.define_scalar(2048, dtype=DataType.Int)
    S270 = fd.define_scalar(12288, dtype=DataType.Int)
    V271 = fd.define_vector([S269, S270], dtype=DataType.Int)
    T272 = fd.ops.reshape(T187, new_shape=V271)
    T273 = fd.ops.matmul(T268, T272)
    T274 = fd.ops.sum(T256, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T275 = fd.ops.cast(T274, dtype=DataType.BFloat16)
    T276 = fd.ops.cast(T267, dtype=DataType.Float)
    T277 = fd.ops.sum(T276, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T278 = fd.ops.cast(T277, dtype=DataType.BFloat16)
    T279 = fd.ops.mul(T178, T276)
    T280 = fd.ops.mul(T172, T276)
    T281 = fd.ops.sum(T280, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T282 = fd.ops.cast(T281, dtype=DataType.BFloat16)
    T283 = fd.ops.mul(T171, T279)
    T284 = fd.ops.mul(T166, T279)
    T285 = fd.ops.sum(T284, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S286 = fd.define_scalar(1, dtype=DataType.Int)
    S287 = fd.define_scalar(2048, dtype=DataType.Int)
    S288 = fd.define_scalar(1, dtype=DataType.Int)
    V289 = fd.define_vector([S286, S287, S288], dtype=DataType.Int)
    T290 = fd.ops.broadcast_in_dim(T285, shape=V289, broadcast_dims=[1])
    T291 = fd.ops.neg(T283)
    T292 = fd.ops.sum(T291, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S293 = fd.define_scalar(1, dtype=DataType.Int)
    S294 = fd.define_scalar(2048, dtype=DataType.Int)
    S295 = fd.define_scalar(1, dtype=DataType.Int)
    V296 = fd.define_vector([S293, S294, S295], dtype=DataType.Int)
    T297 = fd.ops.broadcast_in_dim(T292, shape=V296, broadcast_dims=[1])
    S298 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T299 = fd.ops.mul(S298, T290)
    S300 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T301 = fd.ops.pow(T7, S300)
    T302 = fd.ops.mul(T299, T301)
    T303 = fd.ops.sum(T297, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S304 = fd.define_scalar(1, dtype=DataType.Int)
    S305 = fd.define_scalar(2048, dtype=DataType.Int)
    V306 = fd.define_vector([S304, S305], dtype=DataType.Int)
    T307 = fd.ops.broadcast_in_dim(T303, shape=V306, broadcast_dims=[1])
    T308 = fd.ops.sum(T302, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S309 = fd.define_scalar(1, dtype=DataType.Int)
    S310 = fd.define_scalar(2048, dtype=DataType.Int)
    V311 = fd.define_vector([S309, S310], dtype=DataType.Int)
    T312 = fd.ops.broadcast_in_dim(T308, shape=V311, broadcast_dims=[1])
    S313 = fd.define_scalar(1, dtype=DataType.Int)
    S314 = fd.define_scalar(2048, dtype=DataType.Int)
    S315 = fd.define_scalar(1, dtype=DataType.Int)
    V316 = fd.define_vector([S313, S314, S315], dtype=DataType.Int)
    T317 = fd.ops.broadcast_in_dim(T307, shape=V316, broadcast_dims=[0, 1])
    S318 = fd.define_scalar(1, dtype=DataType.Int)
    S319 = fd.define_scalar(2048, dtype=DataType.Int)
    S320 = fd.define_scalar(12288, dtype=DataType.Int)
    V321 = fd.define_vector([S318, S319, S320], dtype=DataType.Int)
    T322 = fd.ops.broadcast_in_dim(T317, shape=V321, broadcast_dims=[0, 1, 2])
    S323 = fd.define_scalar(8.13802e-05, dtype=DataType.Double)
    T324 = fd.ops.mul(S323, T322)
    S325 = fd.define_scalar(1, dtype=DataType.Int)
    S326 = fd.define_scalar(2048, dtype=DataType.Int)
    S327 = fd.define_scalar(1, dtype=DataType.Int)
    V328 = fd.define_vector([S325, S326, S327], dtype=DataType.Int)
    T329 = fd.ops.broadcast_in_dim(T312, shape=V328, broadcast_dims=[0, 1])
    S330 = fd.define_scalar(1, dtype=DataType.Int)
    S331 = fd.define_scalar(2048, dtype=DataType.Int)
    S332 = fd.define_scalar(12288, dtype=DataType.Int)
    V333 = fd.define_vector([S330, S331, S332], dtype=DataType.Int)
    T334 = fd.ops.broadcast_in_dim(T329, shape=V333, broadcast_dims=[0, 1, 2])
    S335 = fd.define_scalar(1, dtype=DataType.Int)
    S336 = fd.define_scalar(2048, dtype=DataType.Int)
    S337 = fd.define_scalar(1, dtype=DataType.Int)
    V338 = fd.define_vector([S335, S336, S337], dtype=DataType.Int)
    T339 = fd.ops.broadcast_in_dim(T6, shape=V338, broadcast_dims=[0, 1])
    S340 = fd.define_scalar(1, dtype=DataType.Int)
    S341 = fd.define_scalar(2048, dtype=DataType.Int)
    S342 = fd.define_scalar(12288, dtype=DataType.Int)
    V343 = fd.define_vector([S340, S341, S342], dtype=DataType.Int)
    T344 = fd.ops.broadcast_in_dim(T339, shape=V343, broadcast_dims=[0, 1, 2])
    S345 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T346 = fd.ops.mul(S345, T334)
    T347 = fd.ops.sub(T155, T344)
    T348 = fd.ops.mul(T346, T347)
    S349 = fd.define_scalar(12288.0, dtype=DataType.Double)
    S350 = fd.ops.reciprocal(S349)
    T351 = fd.ops.mul(T348, S350)
    T352 = fd.ops.add(T324, T351)
    T353 = fd.ops.add(T283, T352)
    T354 = fd.ops.add(T214, T353)
    S355 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T356 = fd.ops.mul(S355, T354)
    T357 = fd.ops.mul(T151, T356)
    T358 = fd.ops.cast(T357, dtype=DataType.BFloat16)
    S359 = fd.define_scalar(2048, dtype=DataType.Int)
    S360 = fd.define_scalar(12288, dtype=DataType.Int)
    V361 = fd.define_vector([S359, S360], dtype=DataType.Int)
    T362 = fd.ops.reshape(T358, new_shape=V361)
    T363 = fd.ops.matmul(T362, T16)
    S364 = fd.define_scalar(1, dtype=DataType.Int)
    S365 = fd.define_scalar(2048, dtype=DataType.Int)
    S366 = fd.define_scalar(12288, dtype=DataType.Int)
    V367 = fd.define_vector([S364, S365, S366], dtype=DataType.Int)
    T368 = fd.ops.reshape(T363, new_shape=V367)
    T369 = fd.ops.permute(T362, dims=[1, 0])
    S370 = fd.define_scalar(2048, dtype=DataType.Int)
    S371 = fd.define_scalar(12288, dtype=DataType.Int)
    V372 = fd.define_vector([S370, S371], dtype=DataType.Int)
    T373 = fd.ops.reshape(T139, new_shape=V372)
    T374 = fd.ops.matmul(T369, T373)
    T375 = fd.ops.sum(T357, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T376 = fd.ops.cast(T375, dtype=DataType.BFloat16)
    S377 = fd.define_scalar(1, dtype=DataType.Int)
    S378 = fd.define_scalar(2048, dtype=DataType.Int)
    S379 = fd.define_scalar(96, dtype=DataType.Int)
    S380 = fd.define_scalar(128, dtype=DataType.Int)
    V381 = fd.define_vector([S377, S378, S379, S380], dtype=DataType.Int)
    T382 = fd.ops.reshape(T368, new_shape=V381)
    T383 = fd.ops.permute(T382, dims=[0, 2, 1, 3])
    T384 = fd.ops.permute(T82, dims=[0, 1, 3, 2])
    T385 = fd.ops.matmul(T383, T384)
    T386 = fd.ops.permute(T12, dims=[0, 1, 3, 2])
    T387 = fd.ops.matmul(T386, T383)
    T388 = fd.ops.cast(T385, dtype=DataType.Float)
    S389 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T390 = fd.ops.mul(S389, T388)
    T391 = fd.ops.mul(T131, T390)
    T392 = fd.ops.cast(T10, dtype=DataType.Float)
    T393 = fd.ops.mul(T392, T391)
    T394 = fd.ops.sum(T393, dims=[3], keepdim=False, dtype=DataType.Null)
    S395 = fd.define_scalar(1, dtype=DataType.Int)
    S396 = fd.define_scalar(96, dtype=DataType.Int)
    S397 = fd.define_scalar(2048, dtype=DataType.Int)
    S398 = fd.define_scalar(1, dtype=DataType.Int)
    V399 = fd.define_vector([S395, S396, S397, S398], dtype=DataType.Int)
    T400 = fd.ops.broadcast_in_dim(T394, shape=V399, broadcast_dims=[0, 1, 2])
    T401 = fd.ops.cast(T400, dtype=DataType.BFloat16)
    S402 = fd.define_scalar(1, dtype=DataType.Int)
    S403 = fd.define_scalar(96, dtype=DataType.Int)
    S404 = fd.define_scalar(2048, dtype=DataType.Int)
    S405 = fd.define_scalar(2048, dtype=DataType.Int)
    V406 = fd.define_vector([S402, S403, S404, S405], dtype=DataType.Int)
    T407 = fd.ops.broadcast_in_dim(T401, shape=V406, broadcast_dims=[0, 1, 2, 3])
    T408 = fd.ops.cast(T407, dtype=DataType.Float)
    T409 = fd.ops.sub(T391, T408)
    T410 = fd.ops.mul(T392, T409)
    T411 = fd.ops.cast(T410, dtype=DataType.BFloat16)
    S412 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T413 = fd.ops.where(T120, T411, S412)
    T414 = fd.ops.permute(T91, dims=[0, 1, 3, 2])
    T415 = fd.ops.matmul(T413, T414)
    T416 = fd.ops.permute(T86, dims=[0, 1, 3, 2])
    T417 = fd.ops.matmul(T416, T413)
    T418 = fd.ops.cast(T417, dtype=DataType.Float)
    S419 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T420 = fd.ops.mul(S419, T418)
    T421 = fd.ops.cast(T420, dtype=DataType.BFloat16)
    T422 = fd.ops.permute(T421, dims=[0, 1, 3, 2])
    T423 = fd.ops.cast(T415, dtype=DataType.Float)
    S424 = fd.define_scalar(0.297302, dtype=DataType.Double)
    T425 = fd.ops.mul(S424, T423)
    T426 = fd.ops.cast(T425, dtype=DataType.BFloat16)
    T427 = fd.ops.permute(T387, dims=[0, 2, 1, 3])
    S428 = fd.define_scalar(1, dtype=DataType.Int)
    S429 = fd.define_scalar(2048, dtype=DataType.Int)
    S430 = fd.define_scalar(12288, dtype=DataType.Int)
    V431 = fd.define_vector([S428, S429, S430], dtype=DataType.Int)
    T432 = fd.ops.reshape(T427, new_shape=V431)
    T433 = fd.ops.permute(T426, dims=[0, 2, 1, 3])
    S434 = fd.define_scalar(1, dtype=DataType.Int)
    S435 = fd.define_scalar(2048, dtype=DataType.Int)
    S436 = fd.define_scalar(12288, dtype=DataType.Int)
    V437 = fd.define_vector([S434, S435, S436], dtype=DataType.Int)
    T438 = fd.ops.reshape(T433, new_shape=V437)
    T439 = fd.ops.permute(T422, dims=[0, 2, 1, 3])
    S440 = fd.define_scalar(1, dtype=DataType.Int)
    S441 = fd.define_scalar(2048, dtype=DataType.Int)
    S442 = fd.define_scalar(12288, dtype=DataType.Int)
    V443 = fd.define_vector([S440, S441, S442], dtype=DataType.Int)
    T444 = fd.ops.reshape(T439, new_shape=V443)
    T445 = fd.ops.cat([T438, T444, T432], dim=2)
    S446 = fd.define_scalar(2048, dtype=DataType.Int)
    S447 = fd.define_scalar(36864, dtype=DataType.Int)
    V448 = fd.define_vector([S446, S447], dtype=DataType.Int)
    T449 = fd.ops.reshape(T445, new_shape=V448)
    T450 = fd.ops.matmul(T449, T14)
    S451 = fd.define_scalar(1, dtype=DataType.Int)
    S452 = fd.define_scalar(2048, dtype=DataType.Int)
    S453 = fd.define_scalar(12288, dtype=DataType.Int)
    V454 = fd.define_vector([S451, S452, S453], dtype=DataType.Int)
    T455 = fd.ops.reshape(T450, new_shape=V454)
    T456 = fd.ops.permute(T449, dims=[1, 0])
    S457 = fd.define_scalar(2048, dtype=DataType.Int)
    S458 = fd.define_scalar(12288, dtype=DataType.Int)
    V459 = fd.define_vector([S457, S458], dtype=DataType.Int)
    T460 = fd.ops.reshape(T57, new_shape=V459)
    T461 = fd.ops.matmul(T456, T460)
    T462 = fd.ops.cast(T445, dtype=DataType.Float)
    T463 = fd.ops.sum(T462, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T464 = fd.ops.cast(T463, dtype=DataType.BFloat16)
    T465 = fd.ops.cast(T455, dtype=DataType.Float)
    T466 = fd.ops.sum(T465, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T467 = fd.ops.cast(T466, dtype=DataType.BFloat16)
    T468 = fd.ops.mul(T48, T465)
    T469 = fd.ops.mul(T42, T465)
    T470 = fd.ops.sum(T469, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T471 = fd.ops.cast(T470, dtype=DataType.BFloat16)
    T472 = fd.ops.mul(T41, T468)
    T473 = fd.ops.mul(T36, T468)
    T474 = fd.ops.sum(T473, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S475 = fd.define_scalar(1, dtype=DataType.Int)
    S476 = fd.define_scalar(2048, dtype=DataType.Int)
    S477 = fd.define_scalar(1, dtype=DataType.Int)
    V478 = fd.define_vector([S475, S476, S477], dtype=DataType.Int)
    T479 = fd.ops.broadcast_in_dim(T474, shape=V478, broadcast_dims=[1])
    T480 = fd.ops.neg(T472)
    T481 = fd.ops.sum(T480, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S482 = fd.define_scalar(1, dtype=DataType.Int)
    S483 = fd.define_scalar(2048, dtype=DataType.Int)
    S484 = fd.define_scalar(1, dtype=DataType.Int)
    V485 = fd.define_vector([S482, S483, S484], dtype=DataType.Int)
    T486 = fd.ops.broadcast_in_dim(T481, shape=V485, broadcast_dims=[1])
    S487 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T488 = fd.ops.mul(S487, T479)
    S489 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T490 = fd.ops.pow(T11, S489)
    T491 = fd.ops.mul(T488, T490)
    T492 = fd.ops.sum(T486, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S493 = fd.define_scalar(1, dtype=DataType.Int)
    S494 = fd.define_scalar(2048, dtype=DataType.Int)
    V495 = fd.define_vector([S493, S494], dtype=DataType.Int)
    T496 = fd.ops.broadcast_in_dim(T492, shape=V495, broadcast_dims=[1])
    T497 = fd.ops.sum(T491, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S498 = fd.define_scalar(1, dtype=DataType.Int)
    S499 = fd.define_scalar(2048, dtype=DataType.Int)
    V500 = fd.define_vector([S498, S499], dtype=DataType.Int)
    T501 = fd.ops.broadcast_in_dim(T497, shape=V500, broadcast_dims=[1])
    S502 = fd.define_scalar(1, dtype=DataType.Int)
    S503 = fd.define_scalar(2048, dtype=DataType.Int)
    S504 = fd.define_scalar(1, dtype=DataType.Int)
    V505 = fd.define_vector([S502, S503, S504], dtype=DataType.Int)
    T506 = fd.ops.broadcast_in_dim(T496, shape=V505, broadcast_dims=[0, 1])
    S507 = fd.define_scalar(1, dtype=DataType.Int)
    S508 = fd.define_scalar(2048, dtype=DataType.Int)
    S509 = fd.define_scalar(12288, dtype=DataType.Int)
    V510 = fd.define_vector([S507, S508, S509], dtype=DataType.Int)
    T511 = fd.ops.broadcast_in_dim(T506, shape=V510, broadcast_dims=[0, 1, 2])
    S512 = fd.define_scalar(8.13802e-05, dtype=DataType.Double)
    T513 = fd.ops.mul(S512, T511)
    S514 = fd.define_scalar(1, dtype=DataType.Int)
    S515 = fd.define_scalar(2048, dtype=DataType.Int)
    S516 = fd.define_scalar(1, dtype=DataType.Int)
    V517 = fd.define_vector([S514, S515, S516], dtype=DataType.Int)
    T518 = fd.ops.broadcast_in_dim(T501, shape=V517, broadcast_dims=[0, 1])
    S519 = fd.define_scalar(1, dtype=DataType.Int)
    S520 = fd.define_scalar(2048, dtype=DataType.Int)
    S521 = fd.define_scalar(12288, dtype=DataType.Int)
    V522 = fd.define_vector([S519, S520, S521], dtype=DataType.Int)
    T523 = fd.ops.broadcast_in_dim(T518, shape=V522, broadcast_dims=[0, 1, 2])
    S524 = fd.define_scalar(1, dtype=DataType.Int)
    S525 = fd.define_scalar(2048, dtype=DataType.Int)
    S526 = fd.define_scalar(1, dtype=DataType.Int)
    V527 = fd.define_vector([S524, S525, S526], dtype=DataType.Int)
    T528 = fd.ops.broadcast_in_dim(T9, shape=V527, broadcast_dims=[0, 1])
    S529 = fd.define_scalar(1, dtype=DataType.Int)
    S530 = fd.define_scalar(2048, dtype=DataType.Int)
    S531 = fd.define_scalar(12288, dtype=DataType.Int)
    V532 = fd.define_vector([S529, S530, S531], dtype=DataType.Int)
    T533 = fd.ops.broadcast_in_dim(T528, shape=V532, broadcast_dims=[0, 1, 2])
    S534 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T535 = fd.ops.mul(S534, T523)
    T536 = fd.ops.sub(T25, T533)
    T537 = fd.ops.mul(T535, T536)
    S538 = fd.define_scalar(12288.0, dtype=DataType.Double)
    S539 = fd.ops.reciprocal(S538)
    T540 = fd.ops.mul(T537, S539)
    T541 = fd.ops.add(T513, T540)
    T542 = fd.ops.add(T472, T541)
    T543 = fd.ops.add(T354, T542)
    T544 = fd.ops.cast(T543, dtype=DataType.BFloat16)
    fd.add_output(T234)
    fd.add_output(T236)
    fd.add_output(T273)
    fd.add_output(T275)
    fd.add_output(T278)
    fd.add_output(T282)
    fd.add_output(T374)
    fd.add_output(T376)
    fd.add_output(T461)
    fd.add_output(T464)
    fd.add_output(T467)
    fd.add_output(T471)
    fd.add_output(T544)


def test_transformer_backward(
    benchmark, disable_validation: bool, disable_benchmarking: bool
):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        create_transformer_backward(fd)

    inputs = [
        2757501781750758,
        29,
        2757501781750758,
        30,
        2757501781750758,
        31,
        torch.randn((2048,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048), (2048, 1)
        ),
        torch.randn((2048,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 1), (2048, 1, 1)
        ),
        torch.randn((25165824,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 12288), (25165824, 12288, 1)
        ),
        torch.randn((2048,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048), (2048, 1)
        ),
        torch.randn((402653184,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 2048), (402653184, 4194304, 2048, 1)
        ),
        torch.randn((2048,), dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 1), (2048, 1, 1)
        ),
        torch.randn((402653184,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 2048), (402653184, 4194304, 2048, 1)
        ),
        torch.randn((36864,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864,), (1,)
        ),
        torch.randn((452984832,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864, 12288), (12288, 1)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((150994944,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 12288), (12288, 1)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((12288,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn((49152,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152,), (1,)
        ),
        torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152, 12288), (12288, 1)
        ),
        torch.randn((603979776,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 49152), (49152, 1)
        ),
        torch.randn((25165824,), dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 12288), (25165824, 12288, 1)
        ),
    ]

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
