# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
T0 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
T1 = fd.define_tensor(symbolic_sizes=[-1], contiguous=[True], dtype=DataType.Float)
T2 = fd.define_tensor(
    symbolic_sizes=[-1, -1], contiguous=[True, True], dtype=DataType.Half
)
T3 = fd.ops.broadcast_in_dim(T0, output_shape=[1, 1024, 768], broadcast_dims=[2])
T4 = fd.ops.broadcast_in_dim(T1, output_shape=[1, 1024, 768], broadcast_dims=[2])
T5 = fd.ops.view(T2, original_shape=[1024, 768], new_shape=[1, 1024, 768])
T6 = fd.ops.cast(T5, dtype=DataType.Float)
S7 = fd.define_constant(0.500000)
T8 = fd.ops.mul(T6, S7)
S9 = fd.define_constant(0.707107)
T10 = fd.ops.mul(T6, S9)
T11 = fd.ops.erf(T10)
S12 = fd.define_constant(1.00000)
T13 = fd.ops.add(T11, S12)
T14 = fd.ops.mul(T8, T13)
T15 = fd.ops.cast(T14, dtype=DataType.Half)
T16 = fd.ops.cast(T15, dtype=DataType.Float)
T17, T18 = fd.ops.var_mean(T16, axes=[2], correction=0, keepdim=False)
T19 = fd.ops.broadcast_in_dim(T17, output_shape=[1, 1024, 1], broadcast_dims=[0, 1])
T20 = fd.ops.broadcast_in_dim(T18, output_shape=[1, 1024, 1], broadcast_dims=[0, 1])
S21 = fd.define_constant(1.00000e-05)
T22 = fd.ops.add(T19, S21)
T23 = fd.ops.broadcast_in_dim(
    T20, output_shape=[1, 1024, 768], broadcast_dims=[0, 1, 2]
)
T24 = fd.ops.rsqrt(T22)
T25 = fd.ops.sub(T16, T23)
T26 = fd.ops.broadcast_in_dim(
    T24, output_shape=[1, 1024, 768], broadcast_dims=[0, 1, 2]
)
T27 = fd.ops.mul(T25, T26)
T28 = fd.ops.mul(T27, T3)
T29 = fd.ops.add(T28, T4)
T30 = fd.ops.cast(T29, dtype=DataType.Float)
T31 = fd.ops.cast(T30, dtype=DataType.Half)
T32 = fd.ops.view(T31, original_shape=[1, 1024, 768], new_shape=[1024, 768])
fd.add_output(T5)
fd.add_output(T16)
fd.add_output(T20)
fd.add_output(T24)
fd.add_output(T32)
