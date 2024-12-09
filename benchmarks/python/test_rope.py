# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from .core import run_benchmark
import torch


# Mimic the Hugging Face implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L216
def rope_with_cat_fusion(
    fd: FusionDefinition,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    features_per_head: int,
) -> None:
    q = fd.define_tensor(
        shape=[batch_size, seq_len, num_heads, features_per_head],
        dtype=DataType.BFloat16,
    )
    cos = fd.define_tensor(
        shape=[seq_len, features_per_head],
        dtype=DataType.BFloat16,
    )
    sin = fd.define_tensor(
        shape=[seq_len, features_per_head],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.permute(q, dims=[0, 2, 1, 3])
    q_real = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, 0],
        end_indices=[batch_size, num_heads, seq_len, features_per_head // 2],
        strides=[1, 1, 1, 1],
    )
    q_image = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, features_per_head // 2],
        end_indices=[batch_size, num_heads, seq_len, features_per_head],
        strides=[1, 1, 1, 1],
    )

    # nvFuser has problems generating negation for bfloat.
    q_image = fd.ops.cast(q_image, dtype=DataType.Float)
    q_image = -q_image
    q_image = fd.ops.cast(q_image, dtype=DataType.BFloat16)

    q_rotated = fd.ops.cat([q_image, q_real], dim=-1)

    cos = fd.ops.broadcast_in_dim(
        cos, shape=[1, 1, seq_len, features_per_head], broadcast_dims=[2, 3]
    )
    sin = fd.ops.broadcast_in_dim(
        sin, shape=[1, 1, seq_len, features_per_head], broadcast_dims=[2, 3]
    )

    out = q * cos + q_rotated * sin
    out = fd.ops.cast(out, DataType.BFloat16)
    fd.add_output(out)


# Idea from @nikitaved: we split and concatenate the embeddings instead of `q`.
# The embeddings are constant that can be precomputed. So we pay the overhead
# of split and concatenation only once. The actual forward pass is merely
# elementwise+reduction surrounded by some meta ops.
def rope_without_cat_fusion(
    fd: FusionDefinition,
    batch_size: int,  # B
    seq_len: int,  # S
    num_heads: int,  # H
    features_per_head: int,  # F
) -> None:
    q = fd.define_tensor(
        shape=[batch_size, seq_len, num_heads, features_per_head],
        dtype=DataType.BFloat16,
    )
    # `cos_sin_matrix` is essentially a batch (of size S*F/2) of 2x2 matrices
    # laid out in a special way to keep computation simple.
    #
    # Using the notations in Figure 1 in https://arxiv.org/pdf/2104.09864.pdf,
    # cos_sin_matrix[0] contains the following:
    #
    #   cos(θ_1),   -sin(θ1)
    #   cos(θ_2),   -sin(θ2)
    #   ...
    #   cos(θ_F/2), -sin(θ_F/2)
    #   ------------------------
    #   sin(θ_1),   cos(θ_1)
    #   sin(θ_2),   cos(θ_2)
    #   ...
    #   sin(θ_F/2), cos(θ_F/2)
    #
    # cos_sin_matrix[i] is similar but each θ is multiplied by `i+1`.
    cos_sin_matrix = fd.define_tensor(
        shape=[seq_len, 2, features_per_head // 2, 2],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.reshape(
        q, new_shape=[batch_size, seq_len, num_heads, 2, features_per_head // 2]
    )
    q = fd.ops.permute(q, dims=[0, 2, 1, 4, 3])
    q = fd.ops.broadcast_in_dim(
        q,
        shape=[batch_size, num_heads, seq_len, 1, features_per_head // 2, 2],
        broadcast_dims=[0, 1, 2, 4, 5],
    )

    cos_sin_matrix = fd.ops.broadcast_in_dim(
        cos_sin_matrix,
        shape=[batch_size, num_heads, seq_len, 2, features_per_head // 2, 2],
        broadcast_dims=[2, 3, 4, 5],
    )

    out = fd.ops.sum(q * cos_sin_matrix, [-1])
    out = fd.ops.cast(out, DataType.BFloat16)
    out = fd.ops.reshape(
        out, new_shape=[batch_size, num_heads, seq_len, features_per_head]
    )
    fd.add_output(out)


@pytest.mark.parametrize("use_cat", [True, False], ids=["with_cat", "without_cat"])
def test_rope_benchmark(
    benchmark, use_cat: bool, disable_validation: bool, disable_benchmarking: bool
):
    batch_size = 32
    seq_len = 4096
    num_heads = 32
    features_per_head = 128

    # torch.manual_seed(0)
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        features_per_head,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    freqs = torch.randn(
        seq_len, features_per_head // 2, dtype=torch.bfloat16, device="cuda:0"
    )
    cos = freqs.cos()
    sin = freqs.sin()

    if use_cat:
        with FusionDefinition() as fd:
            rope_with_cat_fusion(fd, batch_size, seq_len, num_heads, features_per_head)
        inputs = [q, torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)]
    else:
        with FusionDefinition() as fd:
            rope_without_cat_fusion(
                fd, batch_size, seq_len, num_heads, features_per_head
            )
        # [S, F/2, 2]
        cos_and_minus_sin = torch.stack([cos, -sin], dim=-1)
        # [S, F/2, 2]
        sin_and_cos = torch.stack([sin, cos], dim=-1)
        # [S, 2, F/2, 2]
        cos_sin_matrix = torch.stack([cos_and_minus_sin, sin_and_cos], dim=1)
        inputs = [q, cos_sin_matrix]

    if not disable_validation:
        q_real, q_image = q.permute([0, 2, 1, 3]).split(features_per_head // 2, dim=-1)
        q_real = q_real.to(torch.float32)
        q_image = q_image.to(torch.float32)
        ref_out = torch.cat(
            [q_real * cos - q_image * sin, q_image * cos + q_real * sin], dim=-1
        ).to(torch.bfloat16)
        nvf_out = fd.execute(inputs)
        torch.testing.assert_close(nvf_out, [ref_out], atol=0, rtol=0)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


def llama_2_7b_hf_rope_fwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(shape=[2, 4096, 12288], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[4096, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T2 = fd.define_tensor(shape=[4096, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T9 = fd.ops.reshape(T0, new_shape=[2, 4096, 32, 3, 128])
    T10 = fd.ops.permute(T9, dims=[0, 2, 3, 1, 4])
    T29 = fd.ops.slice(T10, start_indices=[0, 0, 0, 0, 0], end_indices=[2, 32, 1, 4096, 128], strides=[1, 1, 1, 1, 1], manual_normalization=0)
    T48 = fd.ops.slice(T10, start_indices=[0, 0, 1, 0, 0], end_indices=[2, 32, 2, 4096, 128], strides=[1, 1, 1, 1, 1], manual_normalization=0)
    T54 = fd.ops.reshape(T29, new_shape=[2, 32, 4096, 128])
    T60 = fd.ops.reshape(T48, new_shape=[2, 32, 4096, 128])
    T76 = fd.ops.slice(T54, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T92 = fd.ops.slice(T54, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T93 = fd.ops.cast(T92, dtype=DataType.Float)
    T94 = fd.ops.neg(T93)
    T95 = fd.ops.cast(T94, dtype=DataType.BFloat16)
    T96 = fd.ops.cat([T95, T76], dim=-1, manual_padding=0)
    T101 = fd.ops.broadcast_in_dim(T1, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T106 = fd.ops.broadcast_in_dim(T2, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T112 = fd.ops.broadcast_in_dim(T101, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3])
    T113 = fd.ops.cast(T54, dtype=DataType.Float)
    T114 = fd.ops.cast(T112, dtype=DataType.Float)
    T115 = fd.ops.mul(T113, T114)
    T121 = fd.ops.broadcast_in_dim(T106, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3])
    T122 = fd.ops.cast(T96, dtype=DataType.Float)
    T123 = fd.ops.cast(T121, dtype=DataType.Float)
    T124 = fd.ops.mul(T122, T123)
    T125 = fd.ops.add(T115, T124)
    T126 = fd.ops.cast(T125, dtype=DataType.BFloat16)
    T142 = fd.ops.slice(T60, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T158 = fd.ops.slice(T60, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T159 = fd.ops.cast(T158, dtype=DataType.Float)
    T160 = fd.ops.neg(T159)
    T161 = fd.ops.cast(T160, dtype=DataType.BFloat16)
    T162 = fd.ops.cat([T161, T142], dim=-1, manual_padding=0)
    T163 = fd.ops.cast(T60, dtype=DataType.Float)
    T164 = fd.ops.mul(T163, T114)
    T165 = fd.ops.cast(T162, dtype=DataType.Float)
    T166 = fd.ops.mul(T165, T123)
    T167 = fd.ops.add(T164, T166)
    T168 = fd.ops.cast(T167, dtype=DataType.BFloat16)
    T184 = fd.ops.slice(T54, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 0], strides=[1, 1, 1, 1], manual_normalization=0)
    T185 = fd.ops.cat([T126, T184], dim=-1, manual_padding=0)
    T201 = fd.ops.slice(T60, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 0], strides=[1, 1, 1, 1], manual_normalization=0)
    T202 = fd.ops.cat([T168, T201], dim=-1, manual_padding=0)
    fd.add_output(T185)
    fd.add_output(T202)


def llama_2_7b_hf_rope_bwd(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[4096, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[2, 32, 4096, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T2 = fd.define_tensor(shape=[2, 32, 4096, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T3 = fd.define_tensor(shape=[4096, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T8 = fd.ops.broadcast_in_dim(T0, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T24 = fd.ops.slice(T1, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T30 = fd.ops.broadcast_in_dim(T8, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3])
    T46 = fd.ops.slice(T2, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T47 = fd.ops.cast(T24, dtype=DataType.Float)
    T48 = fd.ops.cast(T30, dtype=DataType.Float)
    T49 = fd.ops.cast(T46, dtype=DataType.Float)
    T50 = fd.ops.mul(T48, T47)
    T51 = fd.ops.mul(T48, T49)
    T52 = fd.ops.cast(T50, dtype=DataType.BFloat16)
    T53 = fd.ops.cast(T51, dtype=DataType.BFloat16)
    T69 = fd.ops.slice(T52, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T85 = fd.ops.slice(T53, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T86 = fd.ops.cast(T69, dtype=DataType.Float)
    T91 = fd.ops.broadcast_in_dim(T3, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T92 = fd.ops.cast(T85, dtype=DataType.Float)
    T93 = fd.ops.neg(T86)
    T99 = fd.ops.broadcast_in_dim(T91, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3])
    S100 = fd.define_scalar(0, dtype=DataType.Int)
    T106 = fd.ops.full(shape=[2, 32, 4096, 0], fill_value=S100, dtype=DataType.BFloat16)
    T107 = fd.ops.neg(T92)
    T108 = fd.ops.cast(T93, dtype=DataType.BFloat16)
    T109 = fd.ops.cast(T99, dtype=DataType.Float)
    S110 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T120 = fd.ops.pad(T106, [0, 128, 0, 0, 0, 0, 0, 0], S110)
    T121 = fd.ops.cast(T107, dtype=DataType.BFloat16)
    T137 = fd.ops.slice(T52, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S138 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T148 = fd.ops.pad(T108, [64, 0, 0, 0, 0, 0, 0, 0], S138)
    T149 = fd.ops.mul(T109, T47)
    T150 = fd.ops.cast(T120, dtype=DataType.Float)
    T166 = fd.ops.slice(T53, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S167 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T177 = fd.ops.pad(T121, [64, 0, 0, 0, 0, 0, 0, 0], S167)
    T178 = fd.ops.mul(T109, T49)
    S179 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T189 = fd.ops.pad(T137, [0, 64, 0, 0, 0, 0, 0, 0], S179)
    T190 = fd.ops.cast(T148, dtype=DataType.Float)
    T191 = fd.ops.add(T150, T149)
    S192 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T202 = fd.ops.pad(T166, [0, 64, 0, 0, 0, 0, 0, 0], S192)
    T203 = fd.ops.cast(T177, dtype=DataType.Float)
    T204 = fd.ops.add(T150, T178)
    T205 = fd.ops.cast(T189, dtype=DataType.Float)
    T206 = fd.ops.add(T191, T190)
    T207 = fd.ops.cast(T202, dtype=DataType.Float)
    T208 = fd.ops.add(T204, T203)
    T209 = fd.ops.add(T206, T205)
    T210 = fd.ops.add(T208, T207)
    T211 = fd.ops.cast(T209, dtype=DataType.BFloat16)
    T212 = fd.ops.cast(T210, dtype=DataType.BFloat16)
    S213 = fd.define_scalar(0, dtype=DataType.Int)
    T220 = fd.ops.full(shape=[2, 32, 1, 4096, 128], fill_value=S213, dtype=DataType.BFloat16)
    T227 = fd.ops.reshape(T211, new_shape=[2, 32, 1, 4096, 128])
    T234 = fd.ops.reshape(T212, new_shape=[2, 32, 1, 4096, 128])
    T235 = fd.ops.cat([T234, T227, T220], dim=2, manual_padding=0)
    T236 = fd.ops.permute(T235, dims=[0, 3, 1, 2, 4])
    T241 = fd.ops.reshape(T236, new_shape=[2, 4096, 12288])
    fd.add_output(T241)


def llama_3_8B_rope_fwd(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[2, 8192, 6144], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0]) 
    T1 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) 
    T2 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) 
    T9 = fd.ops.reshape(T0, new_shape=[2, 8192, 8, 6, 128])
    T10 = fd.ops.permute(T9, dims=[0, 2, 3, 1, 4]) 
    T29 = fd.ops.slice(T10, start_indices=[0, 0, 0, 0, 0], end_indices=[2, 8, 4, 8192, 128], strides=[1, 1, 1, 1, 1], manual_normalization=0)
    T48 = fd.ops.slice(T10, start_indices=[0, 0, 4, 0, 0], end_indices=[2, 8, 5, 8192, 128], strides=[1, 1, 1, 1, 1], manual_normalization=0)
    T55 = fd.ops.broadcast_in_dim(T48, shape=[2, 8, 4, 8192, 128], broadcast_dims=[0, 1, 2, 3, 4]) 
    T61 = fd.ops.reshape(T29, new_shape=[2, 32, 8192, 128])
    T67 = fd.ops.reshape(T55, new_shape=[2, 32, 8192, 128])
    T83 = fd.ops.slice(T61, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T99 = fd.ops.slice(T61, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T100 = fd.ops.cast(T99, dtype=DataType.Float)
    T101 = fd.ops.neg(T100)
    T102 = fd.ops.cast(T101, dtype=DataType.BFloat16)
    T103 = fd.ops.cat([T102, T83], dim=-1, manual_padding=0)
    T108 = fd.ops.broadcast_in_dim(T1, shape=[1, 8192, 128], broadcast_dims=[1, 2]) 
    T113 = fd.ops.broadcast_in_dim(T2, shape=[1, 8192, 128], broadcast_dims=[1, 2]) 
    T119 = fd.ops.broadcast_in_dim(T108, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]) 
    T120 = fd.ops.cast(T61, dtype=DataType.Float)
    T121 = fd.ops.cast(T119, dtype=DataType.Float)
    T122 = fd.ops.mul(T120, T121)
    T128 = fd.ops.broadcast_in_dim(T113, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]) 
    T129 = fd.ops.cast(T103, dtype=DataType.Float)
    T130 = fd.ops.cast(T128, dtype=DataType.Float)
    T131 = fd.ops.mul(T129, T130)
    T132 = fd.ops.add(T122, T131)
    T133 = fd.ops.cast(T132, dtype=DataType.BFloat16)
    T149 = fd.ops.slice(T67, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T165 = fd.ops.slice(T67, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T166 = fd.ops.cast(T165, dtype=DataType.Float)
    T167 = fd.ops.neg(T166)
    T168 = fd.ops.cast(T167, dtype=DataType.BFloat16)
    T169 = fd.ops.cat([T168, T149], dim=-1, manual_padding=0)
    T170 = fd.ops.cast(T67, dtype=DataType.Float)
    T171 = fd.ops.mul(T170, T121)
    T172 = fd.ops.cast(T169, dtype=DataType.Float)
    T173 = fd.ops.mul(T172, T130)
    T174 = fd.ops.add(T171, T173)
    T175 = fd.ops.cast(T174, dtype=DataType.BFloat16)
    T191 = fd.ops.slice(T61, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 0], strides=[1, 1, 1, 1], manual_normalization=0)
    T192 = fd.ops.cat([T133, T191], dim=-1, manual_padding=0)
    T208 = fd.ops.slice(T67, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 0], strides=[1, 1, 1, 1], manual_normalization=0)
    T209 = fd.ops.cat([T175, T208], dim=-1, manual_padding=0)
    fd.add_output(T192)
    fd.add_output(T209)


def llama_3_8B_rope_bwd(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) 
    T1 = fd.define_tensor(shape=[2, 32, 8192, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0]) 
    T2 = fd.define_tensor(shape=[2, 32, 8192, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0]) 
    T3 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0]) 
    T8 = fd.ops.broadcast_in_dim(T0, shape=[1, 8192, 128], broadcast_dims=[1, 2]) 
    T24 = fd.ops.slice(T1, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T30 = fd.ops.broadcast_in_dim(T8, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]) 
    T31 = fd.ops.cast(T24, dtype=DataType.Float)
    T32 = fd.ops.cast(T30, dtype=DataType.Float)
    T33 = fd.ops.mul(T32, T31)
    T34 = fd.ops.cast(T33, dtype=DataType.BFloat16)
    T50 = fd.ops.slice(T34, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T66 = fd.ops.slice(T2, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T67 = fd.ops.cast(T50, dtype=DataType.Float)
    T72 = fd.ops.broadcast_in_dim(T3, shape=[1, 8192, 128], broadcast_dims=[1, 2]) 
    T73 = fd.ops.cast(T66, dtype=DataType.Float)
    T74 = fd.ops.neg(T67)
    T80 = fd.ops.broadcast_in_dim(T72, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]) 
    S81 = fd.define_scalar(0, dtype=DataType.Int)
    T87 = fd.ops.full(shape=[2, 32, 8192, 0], fill_value=S81, dtype=DataType.BFloat16)
    T88 = fd.ops.mul(T32, T73)
    T89 = fd.ops.cast(T74, dtype=DataType.BFloat16)
    T90 = fd.ops.cast(T80, dtype=DataType.Float)
    S91 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T101 = fd.ops.pad(T87, [0, 128, 0, 0, 0, 0, 0, 0], S91)
    T102 = fd.ops.cast(T88, dtype=DataType.BFloat16)
    T118 = fd.ops.slice(T34, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S119 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T129 = fd.ops.pad(T89, [64, 0, 0, 0, 0, 0, 0, 0], S119)
    T130 = fd.ops.mul(T90, T31)
    T131 = fd.ops.cast(T101, dtype=DataType.Float)
    T147 = fd.ops.slice(T102, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    S148 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T158 = fd.ops.pad(T118, [0, 64, 0, 0, 0, 0, 0, 0], S148)
    T159 = fd.ops.cast(T129, dtype=DataType.Float)
    T160 = fd.ops.add(T131, T130)
    T161 = fd.ops.cast(T147, dtype=DataType.Float)
    T162 = fd.ops.cast(T158, dtype=DataType.Float)
    T163 = fd.ops.add(T160, T159)
    T164 = fd.ops.neg(T161)
    T165 = fd.ops.add(T163, T162)
    T166 = fd.ops.cast(T164, dtype=DataType.BFloat16)
    T167 = fd.ops.cast(T165, dtype=DataType.BFloat16)
    T183 = fd.ops.slice(T102, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S184 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T194 = fd.ops.pad(T166, [64, 0, 0, 0, 0, 0, 0, 0], S184)
    T195 = fd.ops.mul(T90, T73)
    T202 = fd.ops.reshape(T167, new_shape=[2, 8, 4, 8192, 128])
    S203 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T213 = fd.ops.pad(T183, [0, 64, 0, 0, 0, 0, 0, 0], S203)
    T214 = fd.ops.cast(T194, dtype=DataType.Float)
    T215 = fd.ops.add(T131, T195)
    T216 = fd.ops.cast(T202, dtype=DataType.Float)
    T217 = fd.ops.cast(T213, dtype=DataType.Float)
    T218 = fd.ops.add(T215, T214)
    T219 = fd.ops.sum(T216, dims=[2], keepdim=False, dtype=DataType.Null)
    T220 = fd.ops.add(T218, T217)
    T221 = fd.ops.cast(T219, dtype=DataType.BFloat16)
    T222 = fd.ops.cast(T220, dtype=DataType.BFloat16)
    S223 = fd.define_scalar(0, dtype=DataType.Int)
    T230 = fd.ops.full(shape=[2, 8, 1, 8192, 128], fill_value=S223, dtype=DataType.BFloat16)
    T237 = fd.ops.broadcast_in_dim(T221, shape=[2, 8, 1, 8192, 128], broadcast_dims=[0, 1, 3, 4])
    T244 = fd.ops.reshape(T222, new_shape=[2, 8, 4, 8192, 128])
    T245 = fd.ops.cat([T244, T237, T230], dim=2, manual_padding=0)
    T246 = fd.ops.permute(T245, dims=[0, 3, 1, 2, 4])
    T251 = fd.ops.reshape(T246, new_shape=[2, 8192, 6144])
    fd.add_output(T251)

# { 'name_benchmark' : {fn, [input0, input1, ...]} }
rope_configurations = {
  'llama_2_7b_hf_rope_fwd' : {llama_2_7b_hf_rope_fwd, [
        torch.testing.make_tensor((2, 4096, 12288), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((4096, 128), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((4096, 128), dtype=torch.bfloat16, device='cuda:0'),
    ]},
  'llama_2_7b_hf_rope_bwd' : {llama_2_7b_hf_rope_bwd, [
        torch.testing.make_tensor((4096, 128), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((2, 32, 4096, 128), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((2, 32, 4096, 128), dtype=torch.bfloat16, device='cuda:0'),
        torch.testing.make_tensor((4096, 128), dtype=torch.bfloat16, device='cuda:0'),
    ]},
  'llama_3_8B_rope_fwd' : {llama_3_8B_rope_fwd, [
    torch.testing.make_tensor((2, 8192, 6144), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    ]},
  'llama_3_8B_rope_bwd' : {llama_3_8B_rope_bwd, [
    torch.testing.make_tensor((8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2, 32, 8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2, 32, 8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 128), dtype=torch.bfloat16, device='cuda:0'),
    ]},
}


@pytest.mark.parametrize("rope_variation", ["llama_2_7b_hf_rope_fwd", "llama_2_7b_hf_rope_bwd"])
def test_rope_variantions_nvf_benchmark(
    benchmark,
    rope_variation: str,
    disable_benchmarking: bool,
):

    config = rope_configurations[benchmark_name]

    with FusionDefinition() as fd:
        config[0](fd)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, config[1])
