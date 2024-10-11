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
from .core import run_benchmark
from nvfuser.pytorch_utils import clear_cuda_cache
import torch


def transformer_forward_fusion(fd: FusionDefinition) -> None:
    # MHA dropout.rng_offset
    S0 = fd.define_scalar(None, dtype=DataType.Int)
    # MHA dropout.rng_seed
    S1 = fd.define_scalar(None, dtype=DataType.Int)
    # MLP dropout.rng_offset
    S2 = fd.define_scalar(None, dtype=DataType.Int)
    # MLP dropout.rng_seed
    S3 = fd.define_scalar(None, dtype=DataType.Int)
    # x: input
    T4 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    # layer_norm0.weight
    T5 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # layer_norm0.bias
    T6 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # MHA linear0.weight
    T7 = fd.define_tensor(
        shape=[36864, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    # MHA linear0.bias
    T8 = fd.define_tensor(
        shape=[36864],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # MHA linear1.weight
    T9 = fd.define_tensor(
        shape=[12288, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    # MHA linear1.bias
    T10 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # layer_norm1.weight
    T11 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # layer_norm1.bias
    T12 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # MLP linear0.weight
    T13 = fd.define_tensor(
        shape=[49152, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    # MLP linear0.bias
    T14 = fd.define_tensor(
        shape=[49152],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    # MLP linear1.weight
    T15 = fd.define_tensor(
        shape=[12288, 49152],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    # MLP linear1.bias
    T16 = fd.define_tensor(
        shape=[12288],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    S17 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S18 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T23 = fd.ops.uniform(
        S17,
        S18,
        shape=[1, 2048, 12288],
        rng_seed=S1,
        rng_offset=S0,
        dtype=DataType.BFloat16,
    )
    S24 = fd.define_scalar(0.00000, dtype=DataType.Double)
    S25 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T30 = fd.ops.uniform(
        S24,
        S25,
        shape=[1, 2048, 12288],
        rng_seed=S3,
        rng_offset=S2,
        dtype=DataType.BFloat16,
    )
    T31 = fd.ops.cast(T4, dtype=DataType.Float)
    S32 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T33 = fd.ops.lt(T23, S32)
    S34 = fd.define_scalar(0.900000, dtype=DataType.Double)
    T35 = fd.ops.lt(T30, S34)
    T36, T37 = fd.ops.var_mean(T31, dims=[2], correction=0, keepdim=False)
    T42 = fd.ops.broadcast_in_dim(T36, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    T47 = fd.ops.broadcast_in_dim(T37, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    S48 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T49 = fd.ops.add(T42, S48)
    T54 = fd.ops.broadcast_in_dim(T47, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2])
    T55 = fd.ops.rsqrt(T49)
    T56 = fd.ops.sub(T31, T54)
    T61 = fd.ops.broadcast_in_dim(T55, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2])
    T62 = fd.ops.mul(T56, T61)
    T67 = fd.ops.broadcast_in_dim(T5, shape=[1, 2048, 12288], broadcast_dims=[2])
    T68 = fd.ops.cast(T67, dtype=DataType.Float)
    T69 = fd.ops.mul(T62, T68)
    T74 = fd.ops.broadcast_in_dim(T6, shape=[1, 2048, 12288], broadcast_dims=[2])
    T75 = fd.ops.cast(T74, dtype=DataType.Float)
    T76 = fd.ops.add(T69, T75)
    T77 = fd.ops.cast(T76, dtype=DataType.BFloat16)
    T78 = fd.ops.linear(T77, T7, T8)
    T91 = fd.ops.slice(
        T78, start_indices=[0, 0, 0], end_indices=[1, 2048, 12288], strides=[1, 1, 1]
    )
    T104 = fd.ops.slice(
        T78,
        start_indices=[0, 0, 12288],
        end_indices=[1, 2048, 24576],
        strides=[1, 1, 1],
    )
    T117 = fd.ops.slice(
        T78,
        start_indices=[0, 0, 24576],
        end_indices=[1, 2048, 36864],
        strides=[1, 1, 1],
    )
    T123 = fd.ops.reshape(T104, new_shape=[1, 2048, 96, 128])
    T124 = fd.ops.permute(T123, dims=[0, 2, 1, 3])
    T130 = fd.ops.reshape(T91, new_shape=[1, 2048, 96, 128])
    T131 = fd.ops.permute(T130, dims=[0, 2, 1, 3])
    T137 = fd.ops.reshape(T117, new_shape=[1, 2048, 96, 128])
    T138 = fd.ops.permute(T137, dims=[0, 2, 1, 3])
    S139 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S140 = fd.define_scalar(True, dtype=DataType.Bool)
    T141, T142, T143, T144 = fd.ops.sdpfa_fwd(T131, T124, T138, S139, S140, None)
    T145 = fd.ops.permute(T141, dims=[0, 2, 1, 3])
    T146 = fd.ops.stride_order(T145, stride_order=[3, 2, 1, 0])
    T151 = fd.ops.reshape(T146, new_shape=[1, 2048, 12288])
    T152 = fd.ops.linear(T151, T9, T10)
    T153 = fd.ops.cast(T152, dtype=DataType.Float)
    T154 = fd.ops.cast(T33, dtype=DataType.Float)
    T155 = fd.ops.mul(T153, T154)
    S156 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T157 = fd.ops.mul(T155, S156)
    T158 = fd.ops.add(T31, T157)
    T159, T160 = fd.ops.var_mean(T158, dims=[2], correction=0, keepdim=False)
    T165 = fd.ops.broadcast_in_dim(T159, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    T170 = fd.ops.broadcast_in_dim(T160, shape=[1, 2048, 1], broadcast_dims=[0, 1])
    S171 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T172 = fd.ops.add(T165, S171)
    T177 = fd.ops.broadcast_in_dim(
        T170, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2]
    )
    T178 = fd.ops.rsqrt(T172)
    T179 = fd.ops.sub(T158, T177)
    T184 = fd.ops.broadcast_in_dim(
        T178, shape=[1, 2048, 12288], broadcast_dims=[0, 1, 2]
    )
    T185 = fd.ops.mul(T179, T184)
    T190 = fd.ops.broadcast_in_dim(T11, shape=[1, 2048, 12288], broadcast_dims=[2])
    T191 = fd.ops.cast(T190, dtype=DataType.Float)
    T192 = fd.ops.mul(T185, T191)
    T197 = fd.ops.broadcast_in_dim(T12, shape=[1, 2048, 12288], broadcast_dims=[2])
    T198 = fd.ops.cast(T197, dtype=DataType.Float)
    T199 = fd.ops.add(T192, T198)
    T200 = fd.ops.cast(T199, dtype=DataType.BFloat16)
    T201 = fd.ops.linear(T200, T13, T14)
    T202 = fd.ops.cast(T201, dtype=DataType.Float)
    T203 = fd.ops.mul(T202, T202)
    T204 = fd.ops.mul(T203, T202)
    S205 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T206 = fd.ops.mul(S205, T204)
    T207 = fd.ops.add(T202, T206)
    S208 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T209 = fd.ops.mul(S208, T207)
    T210 = fd.ops.tanh(T209)
    S211 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T212 = fd.ops.mul(S211, T202)
    S213 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T214 = fd.ops.add(S213, T210)
    T215 = fd.ops.mul(T212, T214)
    T216 = fd.ops.cast(T215, dtype=DataType.BFloat16)
    T217 = fd.ops.linear(T216, T15, T16)
    T218 = fd.ops.cast(T217, dtype=DataType.Float)
    T219 = fd.ops.cast(T35, dtype=DataType.Float)
    T220 = fd.ops.mul(T218, T219)
    S221 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T222 = fd.ops.mul(T220, S221)
    T223 = fd.ops.add(T158, T222)
    T224 = fd.ops.cast(T223, dtype=DataType.BFloat16)
    # layer_norm0.welford_out.avg
    fd.add_output(T37)
    # layer_norm0.invstd
    fd.add_output(T55)
    # MHA linear0 output
    fd.add_output(T78)
    # MHA sdpa output
    fd.add_output(T141)
    # MHA sdpa logsum_exp
    fd.add_output(T142)
    # MHA sdpa philox_seed
    fd.add_output(T143)
    # MHA sdpa philox_offset
    fd.add_output(T144)
    # MHA dropout output
    fd.add_output(T158)
    # layer_norm1.welford_out.avg
    fd.add_output(T160)
    # layer_norm1.invstd
    fd.add_output(T178)
    # output
    fd.add_output(T224)


def test_transformer_forward(
    benchmark, disable_validation: bool, disable_benchmarking: bool
):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        transformer_forward_fusion(fd)

    inputs = [
        29,
        2142642406458297,
        30,
        2142642406458297,
        torch.randn(25165824, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 12288), (25165824, 12288, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn(452984832, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864, 12288), (12288, 1)
        ),
        torch.randn(36864, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (36864,), (1,)
        ),
        torch.randn(150994944, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 12288), (12288, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
        torch.randn(603979776, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152, 12288), (12288, 1)
        ),
        torch.randn(49152, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (49152,), (1,)
        ),
        torch.randn(603979776, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288, 49152), (49152, 1)
        ),
        torch.randn(12288, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (12288,), (1,)
        ),
    ]

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


def transformer_backward_fusion(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[12288, 49152],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T3 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T5 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T6 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T7 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T8 = fd.define_tensor(
        shape=[1, 2048, 49152],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T9 = fd.define_tensor(
        shape=[49152, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T10 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T11 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T12 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T13 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T14 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T15 = fd.define_tensor(
        shape=[1, 2048, 1],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T16 = fd.define_tensor(
        shape=[1, 2048],
        contiguity=[None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T17 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T18 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T19 = fd.define_tensor(
        shape=[12288, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T20 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T21 = fd.define_tensor(
        shape=[1, 96, 2048, 128],
        contiguity=[None, False, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T22 = fd.define_tensor(
        shape=[1, 96, 2048, 128],
        contiguity=[None, False, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T23 = fd.define_tensor(
        shape=[1, 96, 2048, 128],
        contiguity=[None, False, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T24 = fd.define_tensor(
        shape=[1, 96, 2048, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T25 = fd.define_tensor(
        shape=[1, 96, 2048],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T26 = fd.define_tensor(shape=[], contiguity=[], dtype=DataType.Int, is_cpu=True)
    T27 = fd.define_tensor(shape=[], contiguity=[], dtype=DataType.Int, is_cpu=True)
    T28 = fd.define_tensor(
        shape=[36864, 12288],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T29 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T30 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T31 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T32 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T33 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T34 = fd.define_tensor(
        shape=[1, 2048, 1],
        contiguity=[None, True, None],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T35 = fd.define_tensor(
        shape=[1, 2048],
        contiguity=[None, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T36 = fd.define_tensor(
        shape=[1, 2048, 12288],
        contiguity=[None, True, True],
        dtype=DataType.Float,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T37 = fd.ops.cast(T0, dtype=DataType.Float)
    S38 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T39 = fd.ops.mul(S38, T37)
    T40 = fd.ops.mul(T1, T39)
    T41 = fd.ops.cast(T40, dtype=DataType.BFloat16)
    T45 = fd.ops.reshape(T41, new_shape=[2048, 12288])
    T46 = fd.ops.matmul(T45, T2)
    T51 = fd.ops.reshape(T46, new_shape=[1, 2048, 49152])
    T52 = fd.ops.permute(T45, dims=[1, 0])
    T56 = fd.ops.reshape(T3, new_shape=[2048, 49152])
    T57 = fd.ops.matmul(T52, T56)
    T58 = fd.ops.sum(T40, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T59 = fd.ops.cast(T58, dtype=DataType.BFloat16)
    T60 = fd.ops.cast(T51, dtype=DataType.Float)
    T61 = fd.ops.mul(T4, T60)
    T62 = fd.ops.mul(T5, T60)
    T63 = fd.ops.mul(T6, T6)
    S64 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T65 = fd.ops.sub(S64, T63)
    T66 = fd.ops.mul(T62, T65)
    S67 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T68 = fd.ops.mul(S67, T66)
    S69 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T70 = fd.ops.mul(S69, T68)
    S71 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T72 = fd.ops.mul(S71, T61)
    T73 = fd.ops.add(T68, T72)
    T74 = fd.ops.mul(T7, T70)
    T75 = fd.ops.mul(T8, T70)
    T76 = fd.ops.add(T73, T75)
    T77 = fd.ops.mul(T7, T74)
    T78 = fd.ops.add(T76, T77)
    T79 = fd.ops.add(T78, T77)
    T80 = fd.ops.cast(T79, dtype=DataType.BFloat16)
    T84 = fd.ops.reshape(T80, new_shape=[2048, 49152])
    T85 = fd.ops.matmul(T84, T9)
    T90 = fd.ops.reshape(T85, new_shape=[1, 2048, 12288])
    T91 = fd.ops.permute(T84, dims=[1, 0])
    T95 = fd.ops.reshape(T10, new_shape=[2048, 12288])
    T96 = fd.ops.matmul(T91, T95)
    T97 = fd.ops.sum(T79, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T98 = fd.ops.cast(T97, dtype=DataType.BFloat16)
    T99 = fd.ops.cast(T90, dtype=DataType.Float)
    T100 = fd.ops.sum(T99, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T101 = fd.ops.cast(T100, dtype=DataType.BFloat16)
    T102 = fd.ops.mul(T11, T99)
    T103 = fd.ops.mul(T12, T99)
    T104 = fd.ops.sum(T103, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T105 = fd.ops.cast(T104, dtype=DataType.BFloat16)
    T106 = fd.ops.mul(T13, T102)
    T107 = fd.ops.mul(T14, T102)
    T108 = fd.ops.sum(T107, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S109 = fd.define_scalar(1, dtype=DataType.Int)
    S110 = fd.define_scalar(2048, dtype=DataType.Int)
    S111 = fd.define_scalar(1, dtype=DataType.Int)
    T113 = fd.ops.broadcast_in_dim(T108, shape=[S109, S110, S111], broadcast_dims=[1])
    T114 = fd.ops.neg(T106)
    T115 = fd.ops.sum(T114, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S116 = fd.define_scalar(1, dtype=DataType.Int)
    S117 = fd.define_scalar(2048, dtype=DataType.Int)
    S118 = fd.define_scalar(1, dtype=DataType.Int)
    T120 = fd.ops.broadcast_in_dim(T115, shape=[S116, S117, S118], broadcast_dims=[1])
    S121 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T122 = fd.ops.mul(S121, T113)
    S123 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T124 = fd.ops.pow(T15, S123)
    T125 = fd.ops.mul(T122, T124)
    T126 = fd.ops.sum(T120, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S127 = fd.define_scalar(1, dtype=DataType.Int)
    S128 = fd.define_scalar(2048, dtype=DataType.Int)
    T130 = fd.ops.broadcast_in_dim(T126, shape=[S127, S128], broadcast_dims=[1])
    T131 = fd.ops.sum(T125, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S132 = fd.define_scalar(1, dtype=DataType.Int)
    S133 = fd.define_scalar(2048, dtype=DataType.Int)
    T135 = fd.ops.broadcast_in_dim(T131, shape=[S132, S133], broadcast_dims=[1])
    S136 = fd.define_scalar(1, dtype=DataType.Int)
    S137 = fd.define_scalar(2048, dtype=DataType.Int)
    S138 = fd.define_scalar(1, dtype=DataType.Int)
    T140 = fd.ops.broadcast_in_dim(
        T130, shape=[S136, S137, S138], broadcast_dims=[0, 1]
    )
    S141 = fd.define_scalar(1, dtype=DataType.Int)
    S142 = fd.define_scalar(2048, dtype=DataType.Int)
    S143 = fd.define_scalar(12288, dtype=DataType.Int)
    T145 = fd.ops.broadcast_in_dim(
        T140, shape=[S141, S142, S143], broadcast_dims=[0, 1, 2]
    )
    S146 = fd.define_scalar(8.13802e-05, dtype=DataType.Double)
    T147 = fd.ops.mul(S146, T145)
    S148 = fd.define_scalar(1, dtype=DataType.Int)
    S149 = fd.define_scalar(2048, dtype=DataType.Int)
    S150 = fd.define_scalar(1, dtype=DataType.Int)
    T152 = fd.ops.broadcast_in_dim(
        T135, shape=[S148, S149, S150], broadcast_dims=[0, 1]
    )
    S153 = fd.define_scalar(1, dtype=DataType.Int)
    S154 = fd.define_scalar(2048, dtype=DataType.Int)
    S155 = fd.define_scalar(12288, dtype=DataType.Int)
    T157 = fd.ops.broadcast_in_dim(
        T152, shape=[S153, S154, S155], broadcast_dims=[0, 1, 2]
    )
    S158 = fd.define_scalar(1, dtype=DataType.Int)
    S159 = fd.define_scalar(2048, dtype=DataType.Int)
    S160 = fd.define_scalar(1, dtype=DataType.Int)
    T162 = fd.ops.broadcast_in_dim(T16, shape=[S158, S159, S160], broadcast_dims=[0, 1])
    S163 = fd.define_scalar(1, dtype=DataType.Int)
    S164 = fd.define_scalar(2048, dtype=DataType.Int)
    S165 = fd.define_scalar(12288, dtype=DataType.Int)
    T167 = fd.ops.broadcast_in_dim(
        T162, shape=[S163, S164, S165], broadcast_dims=[0, 1, 2]
    )
    S168 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T169 = fd.ops.mul(S168, T157)
    T170 = fd.ops.sub(T17, T167)
    T171 = fd.ops.mul(T169, T170)
    S172 = fd.define_scalar(12288.0, dtype=DataType.Double)
    S173 = fd.ops.reciprocal(S172)
    T174 = fd.ops.mul(T171, S173)
    T175 = fd.ops.add(T147, T174)
    T176 = fd.ops.add(T106, T175)
    T177 = fd.ops.add(T37, T176)
    S178 = fd.define_scalar(1.11111, dtype=DataType.Double)
    T179 = fd.ops.mul(S178, T177)
    T180 = fd.ops.mul(T18, T179)
    T181 = fd.ops.cast(T180, dtype=DataType.BFloat16)
    T185 = fd.ops.reshape(T181, new_shape=[2048, 12288])
    T186 = fd.ops.matmul(T185, T19)
    T191 = fd.ops.reshape(T186, new_shape=[1, 2048, 12288])
    T192 = fd.ops.permute(T185, dims=[1, 0])
    T196 = fd.ops.reshape(T20, new_shape=[2048, 12288])
    T197 = fd.ops.matmul(T192, T196)
    T198 = fd.ops.sum(T180, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T199 = fd.ops.cast(T198, dtype=DataType.BFloat16)
    T205 = fd.ops.reshape(T191, new_shape=[1, 2048, 96, 128])
    T206 = fd.ops.permute(T205, dims=[0, 2, 1, 3])
    S207 = fd.define_scalar(0.100000, dtype=DataType.Double)
    S208 = fd.define_scalar(True, dtype=DataType.Bool)
    T209, T210, T211 = fd.ops.sdpfa_bwd(
        T206, T21, T22, T23, T24, T25, S207, S208, T26, T27, None
    )
    T212 = fd.ops.permute(T211, dims=[0, 2, 1, 3])
    T217 = fd.ops.reshape(T212, new_shape=[1, 2048, 12288])
    T218 = fd.ops.permute(T209, dims=[0, 2, 1, 3])
    T223 = fd.ops.reshape(T218, new_shape=[1, 2048, 12288])
    T224 = fd.ops.permute(T210, dims=[0, 2, 1, 3])
    T229 = fd.ops.reshape(T224, new_shape=[1, 2048, 12288])
    T230 = fd.ops.cat([T223, T229, T217], dim=2)
    T234 = fd.ops.reshape(T230, new_shape=[2048, 36864])
    T235 = fd.ops.matmul(T234, T28)
    T240 = fd.ops.reshape(T235, new_shape=[1, 2048, 12288])
    T241 = fd.ops.permute(T234, dims=[1, 0])
    T245 = fd.ops.reshape(T29, new_shape=[2048, 12288])
    T246 = fd.ops.matmul(T241, T245)
    T247 = fd.ops.cast(T230, dtype=DataType.Float)
    T248 = fd.ops.sum(T247, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T249 = fd.ops.cast(T248, dtype=DataType.BFloat16)
    T250 = fd.ops.cast(T240, dtype=DataType.Float)
    T251 = fd.ops.sum(T250, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T252 = fd.ops.cast(T251, dtype=DataType.BFloat16)
    T253 = fd.ops.mul(T30, T250)
    T254 = fd.ops.mul(T31, T250)
    T255 = fd.ops.sum(T254, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T256 = fd.ops.cast(T255, dtype=DataType.BFloat16)
    T257 = fd.ops.mul(T32, T253)
    T258 = fd.ops.mul(T33, T253)
    T259 = fd.ops.sum(T258, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S260 = fd.define_scalar(1, dtype=DataType.Int)
    S261 = fd.define_scalar(2048, dtype=DataType.Int)
    S262 = fd.define_scalar(1, dtype=DataType.Int)
    T264 = fd.ops.broadcast_in_dim(T259, shape=[S260, S261, S262], broadcast_dims=[1])
    T265 = fd.ops.neg(T257)
    T266 = fd.ops.sum(T265, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S267 = fd.define_scalar(1, dtype=DataType.Int)
    S268 = fd.define_scalar(2048, dtype=DataType.Int)
    S269 = fd.define_scalar(1, dtype=DataType.Int)
    T271 = fd.ops.broadcast_in_dim(T266, shape=[S267, S268, S269], broadcast_dims=[1])
    S272 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T273 = fd.ops.mul(S272, T264)
    S274 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T275 = fd.ops.pow(T34, S274)
    T276 = fd.ops.mul(T273, T275)
    T277 = fd.ops.sum(T271, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S278 = fd.define_scalar(1, dtype=DataType.Int)
    S279 = fd.define_scalar(2048, dtype=DataType.Int)
    T281 = fd.ops.broadcast_in_dim(T277, shape=[S278, S279], broadcast_dims=[1])
    T282 = fd.ops.sum(T276, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    S283 = fd.define_scalar(1, dtype=DataType.Int)
    S284 = fd.define_scalar(2048, dtype=DataType.Int)
    T286 = fd.ops.broadcast_in_dim(T282, shape=[S283, S284], broadcast_dims=[1])
    S287 = fd.define_scalar(1, dtype=DataType.Int)
    S288 = fd.define_scalar(2048, dtype=DataType.Int)
    S289 = fd.define_scalar(1, dtype=DataType.Int)
    T291 = fd.ops.broadcast_in_dim(
        T281, shape=[S287, S288, S289], broadcast_dims=[0, 1]
    )
    S292 = fd.define_scalar(1, dtype=DataType.Int)
    S293 = fd.define_scalar(2048, dtype=DataType.Int)
    S294 = fd.define_scalar(12288, dtype=DataType.Int)
    T296 = fd.ops.broadcast_in_dim(
        T291, shape=[S292, S293, S294], broadcast_dims=[0, 1, 2]
    )
    S297 = fd.define_scalar(8.13802e-05, dtype=DataType.Double)
    T298 = fd.ops.mul(S297, T296)
    S299 = fd.define_scalar(1, dtype=DataType.Int)
    S300 = fd.define_scalar(2048, dtype=DataType.Int)
    S301 = fd.define_scalar(1, dtype=DataType.Int)
    T303 = fd.ops.broadcast_in_dim(
        T286, shape=[S299, S300, S301], broadcast_dims=[0, 1]
    )
    S304 = fd.define_scalar(1, dtype=DataType.Int)
    S305 = fd.define_scalar(2048, dtype=DataType.Int)
    S306 = fd.define_scalar(12288, dtype=DataType.Int)
    T308 = fd.ops.broadcast_in_dim(
        T303, shape=[S304, S305, S306], broadcast_dims=[0, 1, 2]
    )
    S309 = fd.define_scalar(1, dtype=DataType.Int)
    S310 = fd.define_scalar(2048, dtype=DataType.Int)
    S311 = fd.define_scalar(1, dtype=DataType.Int)
    T313 = fd.ops.broadcast_in_dim(T35, shape=[S309, S310, S311], broadcast_dims=[0, 1])
    S314 = fd.define_scalar(1, dtype=DataType.Int)
    S315 = fd.define_scalar(2048, dtype=DataType.Int)
    S316 = fd.define_scalar(12288, dtype=DataType.Int)
    T318 = fd.ops.broadcast_in_dim(
        T313, shape=[S314, S315, S316], broadcast_dims=[0, 1, 2]
    )
    S319 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T320 = fd.ops.mul(S319, T308)
    T321 = fd.ops.sub(T36, T318)
    T322 = fd.ops.mul(T320, T321)
    S323 = fd.define_scalar(12288.0, dtype=DataType.Double)
    S324 = fd.ops.reciprocal(S323)
    T325 = fd.ops.mul(T322, S324)
    T326 = fd.ops.add(T298, T325)
    T327 = fd.ops.add(T257, T326)
    T328 = fd.ops.add(T177, T327)
    T329 = fd.ops.cast(T328, dtype=DataType.BFloat16)
    fd.add_output(T57)
    fd.add_output(T59)
    fd.add_output(T96)
    fd.add_output(T98)
    fd.add_output(T101)
    fd.add_output(T105)
    fd.add_output(T197)
    fd.add_output(T199)
    fd.add_output(T246)
    fd.add_output(T249)
    fd.add_output(T252)
    fd.add_output(T256)
    fd.add_output(T329)


def test_transformer_backward(
    benchmark, disable_validation: bool, disable_benchmarking: bool
):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        transformer_backward_fusion(fd)

    inputs = [
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (12288, 49152), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 49152), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (49152, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.randn(12288, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 12288), (12288, 0, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.randn(2048, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 12288), (2048, 1, 0)
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor((1, 2048, 1), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((1, 2048), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (12288, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.randn(75472896, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 128), (75497472, 128, 36864, 1)
        ),
        torch.randn(75472896, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 128), (75497472, 128, 36864, 1)
        ),
        torch.randn(75472896, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 128), (75497472, 128, 36864, 1)
        ),
        torch.randn(25165824, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 96, 2048, 128), (25165824, 128, 12288, 1)
        ),
        torch.testing.make_tensor((1, 96, 2048), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
        torch.testing.make_tensor((), dtype=torch.int64, device="cpu"),
        torch.testing.make_tensor(
            (36864, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.bfloat16, device="cuda:0"
        ),
        torch.randn(12288, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 12288), (12288, 0, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.randn(2048, dtype=torch.float32, device="cuda:0").as_strided(
            (1, 2048, 12288), (2048, 1, 0)
        ),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
        torch.testing.make_tensor((1, 2048, 1), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((1, 2048), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor(
            (1, 2048, 12288), dtype=torch.float32, device="cuda:0"
        ),
    ]

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
