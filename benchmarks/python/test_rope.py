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
    T0 = fd.define_tensor(
        shape=[2, 4096, 12288],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T2 = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T9 = fd.ops.reshape(T0, new_shape=[2, 4096, 32, 3, 128])
    T10 = fd.ops.permute(T9, dims=[0, 2, 3, 1, 4])
    T29 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 0, 0, 0],
        end_indices=[2, 32, 1, 4096, 128],
        strides=[1, 1, 1, 1, 1],
        manual_normalization=0,
    )
    T48 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 1, 0, 0],
        end_indices=[2, 32, 2, 4096, 128],
        strides=[1, 1, 1, 1, 1],
        manual_normalization=0,
    )
    T54 = fd.ops.reshape(T29, new_shape=[2, 32, 4096, 128])
    T60 = fd.ops.reshape(T48, new_shape=[2, 32, 4096, 128])
    T76 = fd.ops.slice(
        T54,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T92 = fd.ops.slice(
        T54,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T93 = fd.ops.cast(T92, dtype=DataType.Float)
    T94 = fd.ops.neg(T93)
    T95 = fd.ops.cast(T94, dtype=DataType.BFloat16)
    T96 = fd.ops.cat([T95, T76], dim=-1, manual_padding=0)
    T101 = fd.ops.broadcast_in_dim(T1, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T106 = fd.ops.broadcast_in_dim(T2, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T112 = fd.ops.broadcast_in_dim(
        T101, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3]
    )
    T113 = fd.ops.cast(T54, dtype=DataType.Float)
    T114 = fd.ops.cast(T112, dtype=DataType.Float)
    T115 = fd.ops.mul(T113, T114)
    T121 = fd.ops.broadcast_in_dim(
        T106, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3]
    )
    T122 = fd.ops.cast(T96, dtype=DataType.Float)
    T123 = fd.ops.cast(T121, dtype=DataType.Float)
    T124 = fd.ops.mul(T122, T123)
    T125 = fd.ops.add(T115, T124)
    T126 = fd.ops.cast(T125, dtype=DataType.BFloat16)
    T142 = fd.ops.slice(
        T60,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T158 = fd.ops.slice(
        T60,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
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
    T184 = fd.ops.slice(
        T54,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T185 = fd.ops.cat([T126, T184], dim=-1, manual_padding=0)
    T201 = fd.ops.slice(
        T60,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T202 = fd.ops.cat([T168, T201], dim=-1, manual_padding=0)
    fd.add_output(T185)
    fd.add_output(T202)


def llama_2_7b_hf_rope_bwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[2, 32, 4096, 128],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[2, 32, 4096, 128],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[4096, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T8 = fd.ops.broadcast_in_dim(T0, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T24 = fd.ops.slice(
        T1,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T30 = fd.ops.broadcast_in_dim(
        T8, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3]
    )
    T46 = fd.ops.slice(
        T2,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T47 = fd.ops.cast(T24, dtype=DataType.Float)
    T48 = fd.ops.cast(T30, dtype=DataType.Float)
    T49 = fd.ops.cast(T46, dtype=DataType.Float)
    T50 = fd.ops.mul(T48, T47)
    T51 = fd.ops.mul(T48, T49)
    T52 = fd.ops.cast(T50, dtype=DataType.BFloat16)
    T53 = fd.ops.cast(T51, dtype=DataType.BFloat16)
    T69 = fd.ops.slice(
        T52,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T85 = fd.ops.slice(
        T53,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T86 = fd.ops.cast(T69, dtype=DataType.Float)
    T91 = fd.ops.broadcast_in_dim(T3, shape=[1, 4096, 128], broadcast_dims=[1, 2])
    T92 = fd.ops.cast(T85, dtype=DataType.Float)
    T93 = fd.ops.neg(T86)
    T99 = fd.ops.broadcast_in_dim(
        T91, shape=[2, 32, 4096, 128], broadcast_dims=[1, 2, 3]
    )
    S100 = fd.define_scalar(0, dtype=DataType.Int)
    T106 = fd.ops.full(shape=[2, 32, 4096, 0], fill_value=S100, dtype=DataType.BFloat16)
    T107 = fd.ops.neg(T92)
    T108 = fd.ops.cast(T93, dtype=DataType.BFloat16)
    T109 = fd.ops.cast(T99, dtype=DataType.Float)
    S110 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T120 = fd.ops.pad(T106, [0, 128, 0, 0, 0, 0, 0, 0], S110)
    T121 = fd.ops.cast(T107, dtype=DataType.BFloat16)
    T137 = fd.ops.slice(
        T52,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    S138 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T148 = fd.ops.pad(T108, [64, 0, 0, 0, 0, 0, 0, 0], S138)
    T149 = fd.ops.mul(T109, T47)
    T150 = fd.ops.cast(T120, dtype=DataType.Float)
    T166 = fd.ops.slice(
        T53,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 4096, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
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
    T220 = fd.ops.full(
        shape=[2, 32, 1, 4096, 128], fill_value=S213, dtype=DataType.BFloat16
    )
    T227 = fd.ops.reshape(T211, new_shape=[2, 32, 1, 4096, 128])
    T234 = fd.ops.reshape(T212, new_shape=[2, 32, 1, 4096, 128])
    T235 = fd.ops.cat([T234, T227, T220], dim=2, manual_padding=0)
    T236 = fd.ops.permute(T235, dims=[0, 3, 1, 2, 4])
    T241 = fd.ops.reshape(T236, new_shape=[2, 4096, 12288])
    fd.add_output(T241)


def llama_3_8B_rope_fwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[2, 8192, 6144],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[8192, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T2 = fd.define_tensor(
        shape=[8192, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T9 = fd.ops.reshape(T0, new_shape=[2, 8192, 8, 6, 128])
    T10 = fd.ops.permute(T9, dims=[0, 2, 3, 1, 4])
    T29 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 0, 0, 0],
        end_indices=[2, 8, 4, 8192, 128],
        strides=[1, 1, 1, 1, 1],
        manual_normalization=0,
    )
    T48 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 4, 0, 0],
        end_indices=[2, 8, 5, 8192, 128],
        strides=[1, 1, 1, 1, 1],
        manual_normalization=0,
    )
    T55 = fd.ops.broadcast_in_dim(
        T48, shape=[2, 8, 4, 8192, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T61 = fd.ops.reshape(T29, new_shape=[2, 32, 8192, 128])
    T67 = fd.ops.reshape(T55, new_shape=[2, 32, 8192, 128])
    T83 = fd.ops.slice(
        T61,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T99 = fd.ops.slice(
        T61,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T100 = fd.ops.cast(T99, dtype=DataType.Float)
    T101 = fd.ops.neg(T100)
    T102 = fd.ops.cast(T101, dtype=DataType.BFloat16)
    T103 = fd.ops.cat([T102, T83], dim=-1, manual_padding=0)
    T108 = fd.ops.broadcast_in_dim(T1, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T113 = fd.ops.broadcast_in_dim(T2, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T119 = fd.ops.broadcast_in_dim(
        T108, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]
    )
    T120 = fd.ops.cast(T61, dtype=DataType.Float)
    T121 = fd.ops.cast(T119, dtype=DataType.Float)
    T122 = fd.ops.mul(T120, T121)
    T128 = fd.ops.broadcast_in_dim(
        T113, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]
    )
    T129 = fd.ops.cast(T103, dtype=DataType.Float)
    T130 = fd.ops.cast(T128, dtype=DataType.Float)
    T131 = fd.ops.mul(T129, T130)
    T132 = fd.ops.add(T122, T131)
    T133 = fd.ops.cast(T132, dtype=DataType.BFloat16)
    T149 = fd.ops.slice(
        T67,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T165 = fd.ops.slice(
        T67,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
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
    T191 = fd.ops.slice(
        T61,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T192 = fd.ops.cat([T133, T191], dim=-1, manual_padding=0)
    T208 = fd.ops.slice(
        T67,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 0],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T209 = fd.ops.cat([T175, T208], dim=-1, manual_padding=0)
    fd.add_output(T192)
    fd.add_output(T209)


def llama_3_8B_rope_bwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[8192, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[2, 32, 8192, 128],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[2, 32, 8192, 128],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[8192, 128],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T8 = fd.ops.broadcast_in_dim(T0, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T24 = fd.ops.slice(
        T1,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T30 = fd.ops.broadcast_in_dim(
        T8, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]
    )
    T31 = fd.ops.cast(T24, dtype=DataType.Float)
    T32 = fd.ops.cast(T30, dtype=DataType.Float)
    T33 = fd.ops.mul(T32, T31)
    T34 = fd.ops.cast(T33, dtype=DataType.BFloat16)
    T50 = fd.ops.slice(
        T34,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T66 = fd.ops.slice(
        T2,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T67 = fd.ops.cast(T50, dtype=DataType.Float)
    T72 = fd.ops.broadcast_in_dim(T3, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T73 = fd.ops.cast(T66, dtype=DataType.Float)
    T74 = fd.ops.neg(T67)
    T80 = fd.ops.broadcast_in_dim(
        T72, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3]
    )
    S81 = fd.define_scalar(0, dtype=DataType.Int)
    T87 = fd.ops.full(shape=[2, 32, 8192, 0], fill_value=S81, dtype=DataType.BFloat16)
    T88 = fd.ops.mul(T32, T73)
    T89 = fd.ops.cast(T74, dtype=DataType.BFloat16)
    T90 = fd.ops.cast(T80, dtype=DataType.Float)
    S91 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T101 = fd.ops.pad(T87, [0, 128, 0, 0, 0, 0, 0, 0], S91)
    T102 = fd.ops.cast(T88, dtype=DataType.BFloat16)
    T118 = fd.ops.slice(
        T34,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    S119 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T129 = fd.ops.pad(T89, [64, 0, 0, 0, 0, 0, 0, 0], S119)
    T130 = fd.ops.mul(T90, T31)
    T131 = fd.ops.cast(T101, dtype=DataType.Float)
    T147 = fd.ops.slice(
        T102,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 8192, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
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
    T183 = fd.ops.slice(
        T102,
        start_indices=[0, 0, 0, 64],
        end_indices=[2, 32, 8192, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
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
    T230 = fd.ops.full(
        shape=[2, 8, 1, 8192, 128], fill_value=S223, dtype=DataType.BFloat16
    )
    T237 = fd.ops.broadcast_in_dim(
        T221, shape=[2, 8, 1, 8192, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T244 = fd.ops.reshape(T222, new_shape=[2, 8, 4, 8192, 128])
    T245 = fd.ops.cat([T244, T237, T230], dim=2, manual_padding=0)
    T246 = fd.ops.permute(T245, dims=[0, 3, 1, 2, 4])
    T251 = fd.ops.reshape(T246, new_shape=[2, 8192, 6144])
    fd.add_output(T251)


def hf_qwen2_rope_fwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 32768, 3584],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 32768, 512],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[1, 32768, 512],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[1, 32768, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[1, 32768, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T10 = fd.ops.reshape(T0, new_shape=[1, 32768, 28, 128])
    T11 = fd.ops.permute(T10, dims=[0, 2, 1, 3])
    T17 = fd.ops.reshape(T1, new_shape=[1, 32768, 4, 128])
    T18 = fd.ops.permute(T17, dims=[0, 2, 1, 3])
    T24 = fd.ops.reshape(T2, new_shape=[1, 32768, 4, 128])
    T25 = fd.ops.permute(T24, dims=[0, 2, 1, 3])
    T31 = fd.ops.broadcast_in_dim(
        T3, shape=[1, 1, 32768, 128], broadcast_dims=[0, 2, 3]
    )
    T37 = fd.ops.broadcast_in_dim(
        T4, shape=[1, 1, 32768, 128], broadcast_dims=[0, 2, 3]
    )
    T43 = fd.ops.broadcast_in_dim(
        T31, shape=[1, 28, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T44 = fd.ops.cast(T11, dtype=DataType.Float)
    T45 = fd.ops.cast(T43, dtype=DataType.Float)
    T46 = fd.ops.mul(T44, T45)
    T62 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 28, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T78 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 28, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T79 = fd.ops.cast(T78, dtype=DataType.Float)
    T80 = fd.ops.neg(T79)
    T81 = fd.ops.cast(T80, dtype=DataType.BFloat16)
    T82 = fd.ops.cat([T81, T62], dim=-1, manual_padding=0)
    T88 = fd.ops.broadcast_in_dim(
        T37, shape=[1, 28, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T89 = fd.ops.cast(T82, dtype=DataType.Float)
    T90 = fd.ops.cast(T88, dtype=DataType.Float)
    T91 = fd.ops.mul(T89, T90)
    T92 = fd.ops.add(T46, T91)
    T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
    T99 = fd.ops.broadcast_in_dim(
        T31, shape=[1, 4, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T100 = fd.ops.cast(T18, dtype=DataType.Float)
    T101 = fd.ops.cast(T99, dtype=DataType.Float)
    T102 = fd.ops.mul(T100, T101)
    T118 = fd.ops.slice(
        T18,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 4, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T134 = fd.ops.slice(
        T18,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 4, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T135 = fd.ops.cast(T134, dtype=DataType.Float)
    T136 = fd.ops.neg(T135)
    T137 = fd.ops.cast(T136, dtype=DataType.BFloat16)
    T138 = fd.ops.cat([T137, T118], dim=-1, manual_padding=0)
    T144 = fd.ops.broadcast_in_dim(
        T37, shape=[1, 4, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T145 = fd.ops.cast(T138, dtype=DataType.Float)
    T146 = fd.ops.cast(T144, dtype=DataType.Float)
    T147 = fd.ops.mul(T145, T146)
    T148 = fd.ops.add(T102, T147)
    T149 = fd.ops.cast(T148, dtype=DataType.BFloat16)
    T156 = fd.ops.broadcast_in_dim(
        T149, shape=[1, 4, 1, 32768, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T163 = fd.ops.broadcast_in_dim(
        T156, shape=[1, 4, 7, 32768, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T169 = fd.ops.reshape(T163, new_shape=[1, 28, 32768, 128])
    T176 = fd.ops.broadcast_in_dim(
        T25, shape=[1, 4, 1, 32768, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T183 = fd.ops.broadcast_in_dim(
        T176, shape=[1, 4, 7, 32768, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T189 = fd.ops.reshape(T183, new_shape=[1, 28, 32768, 128])
    fd.add_output(T93)
    fd.add_output(T169)
    fd.add_output(T189)


def hf_qwen2_rope_bwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 28, 32768, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 32768, 3584],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[1, 32768, 512],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[1, 32768, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[1, 28, 32768, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T5 = fd.define_tensor(
        shape=[1, 28, 32768, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T6 = fd.define_tensor(
        shape=[1, 32768, 128],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T13 = fd.ops.reshape(T0, new_shape=[1, 4, 7, 32768, 128])
    T14 = fd.ops.cast(T13, dtype=DataType.Float)
    T15 = fd.ops.sum(T14, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T16 = fd.ops.cast(T15, dtype=DataType.BFloat16)
    T23 = fd.ops.broadcast_in_dim(
        T16, shape=[1, 4, 1, 32768, 128], broadcast_dims=[1, 3, 4]
    )
    T24 = fd.ops.cast(T23, dtype=DataType.Float)
    T25 = fd.ops.sum(T24, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T31 = fd.ops.reshape(T1, new_shape=[1, 32768, 28, 128])
    T37 = fd.ops.reshape(T2, new_shape=[1, 32768, 4, 128])
    T43 = fd.ops.broadcast_in_dim(
        T3, shape=[1, 1, 32768, 128], broadcast_dims=[0, 2, 3]
    )
    T44 = fd.ops.cast(T25, dtype=DataType.BFloat16)
    T45 = fd.ops.permute(T31, dims=[0, 2, 1, 3])
    T46 = fd.ops.permute(T37, dims=[0, 2, 1, 3])
    T52 = fd.ops.broadcast_in_dim(
        T43, shape=[1, 28, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T58 = fd.ops.broadcast_in_dim(
        T44, shape=[1, 4, 32768, 128], broadcast_dims=[1, 2, 3]
    )
    T64 = fd.ops.broadcast_in_dim(
        T43, shape=[1, 4, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T80 = fd.ops.slice(
        T45,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 28, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T96 = fd.ops.slice(
        T46,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 4, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T97 = fd.ops.cast(T4, dtype=DataType.Float)
    T98 = fd.ops.cast(T52, dtype=DataType.Float)
    T99 = fd.ops.cast(T58, dtype=DataType.Float)
    T100 = fd.ops.cast(T64, dtype=DataType.Float)
    T101 = fd.ops.cast(T80, dtype=DataType.Float)
    T102 = fd.ops.cast(T96, dtype=DataType.Float)
    T103 = fd.ops.mul(T98, T97)
    T104 = fd.ops.mul(T100, T99)
    T105 = fd.ops.neg(T101)
    T106 = fd.ops.neg(T102)
    T107 = fd.ops.cast(T103, dtype=DataType.BFloat16)
    T108 = fd.ops.cast(T104, dtype=DataType.BFloat16)
    T124 = fd.ops.slice(
        T45,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 28, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T125 = fd.ops.cast(T105, dtype=DataType.BFloat16)
    T141 = fd.ops.slice(
        T46,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 4, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T142 = fd.ops.cast(T106, dtype=DataType.BFloat16)
    T158 = fd.ops.slice(
        T107,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 28, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T174 = fd.ops.slice(
        T108,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 4, 32768, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T181 = fd.ops.reshape(T5, new_shape=[1, 4, 7, 32768, 128])
    T182 = fd.ops.cat([T125, T124], dim=-1, manual_padding=0)
    T183 = fd.ops.cat([T142, T141], dim=-1, manual_padding=0)
    T184 = fd.ops.cast(T158, dtype=DataType.Float)
    T185 = fd.ops.cast(T174, dtype=DataType.Float)
    T186 = fd.ops.cast(T181, dtype=DataType.Float)
    T187 = fd.ops.cast(T45, dtype=DataType.Float)
    T188 = fd.ops.cast(T46, dtype=DataType.Float)
    T189 = fd.ops.cast(T182, dtype=DataType.Float)
    T190 = fd.ops.cast(T183, dtype=DataType.Float)
    T191 = fd.ops.neg(T184)
    T192 = fd.ops.neg(T185)
    T193 = fd.ops.sum(T186, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T194 = fd.ops.mul(T187, T97)
    T195 = fd.ops.mul(T188, T99)
    T196 = fd.ops.mul(T189, T97)
    T197 = fd.ops.mul(T190, T99)
    T203 = fd.ops.broadcast_in_dim(
        T6, shape=[1, 1, 32768, 128], broadcast_dims=[0, 2, 3]
    )
    T219 = fd.ops.slice(
        T107,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 28, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T220 = fd.ops.cast(T191, dtype=DataType.BFloat16)
    T236 = fd.ops.slice(
        T108,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 4, 32768, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T237 = fd.ops.cast(T192, dtype=DataType.BFloat16)
    T238 = fd.ops.cast(T193, dtype=DataType.BFloat16)
    T239 = fd.ops.sum(T194, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T240 = fd.ops.sum(T195, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T241 = fd.ops.sum(T196, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T242 = fd.ops.sum(T197, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T248 = fd.ops.broadcast_in_dim(
        T203, shape=[1, 28, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    S249 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T259 = fd.ops.pad(T219, [0, 64, 0, 0, 0, 0, 0, 0], S249)
    S260 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T270 = fd.ops.pad(T220, [64, 0, 0, 0, 0, 0, 0, 0], S260)
    T276 = fd.ops.broadcast_in_dim(
        T203, shape=[1, 4, 32768, 128], broadcast_dims=[0, 1, 2, 3]
    )
    S277 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T287 = fd.ops.pad(T236, [0, 64, 0, 0, 0, 0, 0, 0], S277)
    S288 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T298 = fd.ops.pad(T237, [64, 0, 0, 0, 0, 0, 0, 0], S288)
    T305 = fd.ops.broadcast_in_dim(
        T238, shape=[1, 4, 1, 32768, 128], broadcast_dims=[1, 3, 4]
    )
    T306 = fd.ops.cast(T239, dtype=DataType.BFloat16)
    T307 = fd.ops.cast(T240, dtype=DataType.BFloat16)
    T308 = fd.ops.cast(T241, dtype=DataType.BFloat16)
    T309 = fd.ops.cast(T242, dtype=DataType.BFloat16)
    T310 = fd.ops.cast(T248, dtype=DataType.Float)
    T311 = fd.ops.cast(T259, dtype=DataType.Float)
    T312 = fd.ops.cast(T270, dtype=DataType.Float)
    T313 = fd.ops.cast(T276, dtype=DataType.Float)
    T314 = fd.ops.cast(T287, dtype=DataType.Float)
    T315 = fd.ops.cast(T298, dtype=DataType.Float)
    T316 = fd.ops.cast(T305, dtype=DataType.Float)
    T322 = fd.ops.broadcast_in_dim(
        T306, shape=[1, 1, 32768, 128], broadcast_dims=[2, 3]
    )
    T328 = fd.ops.broadcast_in_dim(
        T307, shape=[1, 1, 32768, 128], broadcast_dims=[2, 3]
    )
    T334 = fd.ops.broadcast_in_dim(
        T308, shape=[1, 1, 32768, 128], broadcast_dims=[2, 3]
    )
    T340 = fd.ops.broadcast_in_dim(
        T309, shape=[1, 1, 32768, 128], broadcast_dims=[2, 3]
    )
    T341 = fd.ops.mul(T310, T97)
    T342 = fd.ops.add(T312, T311)
    T343 = fd.ops.mul(T313, T99)
    T344 = fd.ops.add(T315, T314)
    T345 = fd.ops.sum(T316, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T346 = fd.ops.cast(T322, dtype=DataType.Float)
    T347 = fd.ops.cast(T328, dtype=DataType.Float)
    T348 = fd.ops.cast(T334, dtype=DataType.Float)
    T349 = fd.ops.cast(T340, dtype=DataType.Float)
    T350 = fd.ops.add(T342, T341)
    T351 = fd.ops.add(T344, T343)
    T352 = fd.ops.cast(T345, dtype=DataType.BFloat16)
    T353 = fd.ops.add(T347, T346)
    T354 = fd.ops.add(T349, T348)
    T355 = fd.ops.cast(T350, dtype=DataType.BFloat16)
    T356 = fd.ops.cast(T351, dtype=DataType.BFloat16)
    T362 = fd.ops.broadcast_in_dim(
        T352, shape=[1, 4, 32768, 128], broadcast_dims=[1, 2, 3]
    )
    T363 = fd.ops.sum(T353, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T364 = fd.ops.sum(T354, dims=[0, 1], keepdim=False, dtype=DataType.Null)
    T365 = fd.ops.permute(T355, dims=[0, 2, 1, 3])
    T366 = fd.ops.permute(T356, dims=[0, 2, 1, 3])
    T367 = fd.ops.permute(T362, dims=[0, 2, 1, 3])
    T368 = fd.ops.cast(T363, dtype=DataType.BFloat16)
    T369 = fd.ops.cast(T364, dtype=DataType.BFloat16)
    T374 = fd.ops.reshape(T365, new_shape=[1, 32768, 3584])
    T379 = fd.ops.reshape(T366, new_shape=[1, 32768, 512])
    T384 = fd.ops.reshape(T367, new_shape=[1, 32768, 512])
    T389 = fd.ops.broadcast_in_dim(T368, shape=[1, 32768, 128], broadcast_dims=[1, 2])
    T394 = fd.ops.broadcast_in_dim(T369, shape=[1, 32768, 128], broadcast_dims=[1, 2])
    fd.add_output(T394)
    fd.add_output(T389)
    fd.add_output(T384)
    fd.add_output(T379)
    fd.add_output(T374)


def hf_phi3_rope_fwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[2, 4096, 9216],
        contiguity=[True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[48],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T2 = fd.define_tensor(
        shape=[1, 4096],
        contiguity=[None, True],
        dtype=DataType.Int,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T15 = fd.ops.slice(
        T0,
        start_indices=[0, 0, 0],
        end_indices=[2, 4096, 3072],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T28 = fd.ops.slice(
        T0,
        start_indices=[0, 0, 3072],
        end_indices=[2, 4096, 6144],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T41 = fd.ops.slice(
        T0,
        start_indices=[0, 0, 6144],
        end_indices=[2, 4096, 9216],
        strides=[1, 1, 1],
        manual_normalization=0,
    )
    T47 = fd.ops.reshape(T15, new_shape=[2, 4096, 32, 96])
    T48 = fd.ops.permute(T47, dims=[0, 2, 1, 3])
    T54 = fd.ops.reshape(T28, new_shape=[2, 4096, 32, 96])
    T55 = fd.ops.permute(T54, dims=[0, 2, 1, 3])
    T61 = fd.ops.reshape(T41, new_shape=[2, 4096, 32, 96])
    T62 = fd.ops.permute(T61, dims=[0, 2, 1, 3])
    T67 = fd.ops.broadcast_in_dim(T1, shape=[1, 48, 1], broadcast_dims=[1])
    T68 = fd.ops.cast(T67, dtype=DataType.Float)
    T73 = fd.ops.broadcast_in_dim(T68, shape=[1, 48, 1], broadcast_dims=[0, 1, 2])
    T78 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 4096], broadcast_dims=[0, 2])
    T79 = fd.ops.cast(T78, dtype=DataType.Float)
    T80 = fd.ops.matmul(T73, T79)
    T81 = fd.ops.permute(T80, dims=[0, 2, 1])
    T82 = fd.ops.cat([T81, T81], dim=-1, manual_padding=0)
    T83 = fd.ops.cos(T82)
    T84 = fd.ops.sin(T82)
    T85 = fd.ops.cast(T83, dtype=DataType.BFloat16)
    T86 = fd.ops.cast(T84, dtype=DataType.BFloat16)
    T92 = fd.ops.broadcast_in_dim(T85, shape=[1, 1, 4096, 96], broadcast_dims=[0, 2, 3])
    T98 = fd.ops.broadcast_in_dim(T86, shape=[1, 1, 4096, 96], broadcast_dims=[0, 2, 3])
    T104 = fd.ops.broadcast_in_dim(
        T92, shape=[2, 32, 4096, 96], broadcast_dims=[0, 1, 2, 3]
    )
    T105 = fd.ops.cast(T48, dtype=DataType.Float)
    T106 = fd.ops.cast(T104, dtype=DataType.Float)
    T107 = fd.ops.mul(T105, T106)
    T123 = fd.ops.slice(
        T48,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 48],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T139 = fd.ops.slice(
        T48,
        start_indices=[0, 0, 0, 48],
        end_indices=[2, 32, 4096, 96],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T140 = fd.ops.cast(T139, dtype=DataType.Float)
    T141 = fd.ops.neg(T140)
    T142 = fd.ops.cast(T141, dtype=DataType.BFloat16)
    T143 = fd.ops.cat([T142, T123], dim=-1, manual_padding=0)
    T149 = fd.ops.broadcast_in_dim(
        T98, shape=[2, 32, 4096, 96], broadcast_dims=[0, 1, 2, 3]
    )
    T150 = fd.ops.cast(T143, dtype=DataType.Float)
    T151 = fd.ops.cast(T149, dtype=DataType.Float)
    T152 = fd.ops.mul(T150, T151)
    T153 = fd.ops.add(T107, T152)
    T154 = fd.ops.cast(T153, dtype=DataType.BFloat16)
    T155 = fd.ops.cast(T55, dtype=DataType.Float)
    T156 = fd.ops.mul(T155, T106)
    T172 = fd.ops.slice(
        T55,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 48],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T188 = fd.ops.slice(
        T55,
        start_indices=[0, 0, 0, 48],
        end_indices=[2, 32, 4096, 96],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T189 = fd.ops.cast(T188, dtype=DataType.Float)
    T190 = fd.ops.neg(T189)
    T191 = fd.ops.cast(T190, dtype=DataType.BFloat16)
    T192 = fd.ops.cat([T191, T172], dim=-1, manual_padding=0)
    T193 = fd.ops.cast(T192, dtype=DataType.Float)
    T194 = fd.ops.mul(T193, T151)
    T195 = fd.ops.add(T156, T194)
    T196 = fd.ops.cast(T195, dtype=DataType.BFloat16)
    fd.add_output(T62)
    fd.add_output(T104)
    fd.add_output(T149)
    fd.add_output(T154)
    fd.add_output(T196)


def hf_phi3_rope_bwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[2, 32, 4096, 96],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[2, 32, 4096, 96],
        contiguity=[None, None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[2, 32, 4096, 96],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[2, 32, 4096, 96],
        contiguity=[None, None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[2, 32, 4096, 96],
        contiguity=[True, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T5 = fd.ops.cast(T0, dtype=DataType.Float)
    T6 = fd.ops.cast(T1, dtype=DataType.Float)
    T7 = fd.ops.cast(T2, dtype=DataType.Float)
    T8 = fd.ops.mul(T6, T5)
    T9 = fd.ops.mul(T6, T7)
    T10 = fd.ops.cast(T8, dtype=DataType.BFloat16)
    T11 = fd.ops.cast(T9, dtype=DataType.BFloat16)
    T27 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 48],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T43 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 0],
        end_indices=[2, 32, 4096, 48],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T44 = fd.ops.cast(T27, dtype=DataType.Float)
    T45 = fd.ops.cast(T43, dtype=DataType.Float)
    T46 = fd.ops.neg(T44)
    T47 = fd.ops.neg(T45)
    T63 = fd.ops.slice(
        T10,
        start_indices=[0, 0, 0, 48],
        end_indices=[2, 32, 4096, 96],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T64 = fd.ops.cast(T46, dtype=DataType.BFloat16)
    T80 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 48],
        end_indices=[2, 32, 4096, 96],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T81 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    S82 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T92 = fd.ops.pad(T63, [0, 48, 0, 0, 0, 0, 0, 0], S82)
    S93 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T103 = fd.ops.pad(T64, [48, 0, 0, 0, 0, 0, 0, 0], S93)
    S104 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T114 = fd.ops.pad(T80, [0, 48, 0, 0, 0, 0, 0, 0], S104)
    S115 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T125 = fd.ops.pad(T81, [48, 0, 0, 0, 0, 0, 0, 0], S115)
    T126 = fd.ops.cast(T3, dtype=DataType.Float)
    T127 = fd.ops.cast(T92, dtype=DataType.Float)
    T128 = fd.ops.cast(T103, dtype=DataType.Float)
    T129 = fd.ops.cast(T114, dtype=DataType.Float)
    T130 = fd.ops.cast(T125, dtype=DataType.Float)
    T131 = fd.ops.mul(T126, T5)
    T132 = fd.ops.add(T128, T127)
    T133 = fd.ops.mul(T126, T7)
    T134 = fd.ops.add(T130, T129)
    T135 = fd.ops.add(T132, T131)
    T136 = fd.ops.add(T134, T133)
    T137 = fd.ops.cast(T135, dtype=DataType.BFloat16)
    T138 = fd.ops.cast(T136, dtype=DataType.BFloat16)
    T139 = fd.ops.permute(T137, dims=[0, 2, 1, 3])
    T140 = fd.ops.permute(T4, dims=[0, 2, 1, 3])
    T141 = fd.ops.permute(T138, dims=[0, 2, 1, 3])
    T146 = fd.ops.reshape(T139, new_shape=[2, 4096, 3072])
    T151 = fd.ops.reshape(T140, new_shape=[2, 4096, 3072])
    T156 = fd.ops.reshape(T141, new_shape=[2, 4096, 3072])
    S157 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T165 = fd.ops.pad(T146, [3072, 3072, 0, 0, 0, 0], S157)
    S166 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T174 = fd.ops.pad(T151, [6144, 0, 0, 0, 0, 0], S166)
    S175 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T183 = fd.ops.pad(T156, [0, 6144, 0, 0, 0, 0], S175)
    T184 = fd.ops.cast(T165, dtype=DataType.Float)
    T185 = fd.ops.cast(T174, dtype=DataType.Float)
    T186 = fd.ops.cast(T183, dtype=DataType.Float)
    T187 = fd.ops.add(T185, T184)
    T188 = fd.ops.add(T187, T186)
    T189 = fd.ops.cast(T188, dtype=DataType.BFloat16)
    fd.add_output(T189)


def hf_mistral_nemo_rope_fwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 128000, 4096],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 128000, 1024],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T2 = fd.define_tensor(
        shape=[1, 128000, 1024],
        contiguity=[None, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[64],
        contiguity=[True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[0],
    )
    T4 = fd.define_tensor(
        shape=[1, 128000],
        contiguity=[None, True],
        dtype=DataType.Int,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T10 = fd.ops.reshape(T0, new_shape=[1, 128000, 32, 128])
    T11 = fd.ops.permute(T10, dims=[0, 2, 1, 3])
    T17 = fd.ops.reshape(T1, new_shape=[1, 128000, 8, 128])
    T18 = fd.ops.permute(T17, dims=[0, 2, 1, 3])
    T24 = fd.ops.reshape(T2, new_shape=[1, 128000, 8, 128])
    T25 = fd.ops.permute(T24, dims=[0, 2, 1, 3])
    T30 = fd.ops.broadcast_in_dim(T3, shape=[1, 64, 1], broadcast_dims=[1])
    T31 = fd.ops.cast(T30, dtype=DataType.Float)
    T36 = fd.ops.broadcast_in_dim(T31, shape=[1, 64, 1], broadcast_dims=[0, 1, 2])
    T41 = fd.ops.broadcast_in_dim(T4, shape=[1, 1, 128000], broadcast_dims=[0, 2])
    T42 = fd.ops.cast(T41, dtype=DataType.Float)
    T43 = fd.ops.matmul(T36, T42)
    T44 = fd.ops.permute(T43, dims=[0, 2, 1])
    T45 = fd.ops.cat([T44, T44], dim=-1, manual_padding=0)
    T46 = fd.ops.cos(T45)
    T47 = fd.ops.sin(T45)
    T48 = fd.ops.cast(T46, dtype=DataType.BFloat16)
    T49 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    T55 = fd.ops.broadcast_in_dim(
        T48, shape=[1, 1, 128000, 128], broadcast_dims=[0, 2, 3]
    )
    T61 = fd.ops.broadcast_in_dim(
        T49, shape=[1, 1, 128000, 128], broadcast_dims=[0, 2, 3]
    )
    T67 = fd.ops.broadcast_in_dim(
        T55, shape=[1, 32, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T68 = fd.ops.cast(T11, dtype=DataType.Float)
    T69 = fd.ops.cast(T67, dtype=DataType.Float)
    T70 = fd.ops.mul(T68, T69)
    T86 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 32, 128000, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T102 = fd.ops.slice(
        T11,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 32, 128000, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T103 = fd.ops.cast(T102, dtype=DataType.Float)
    T104 = fd.ops.neg(T103)
    T105 = fd.ops.cast(T104, dtype=DataType.BFloat16)
    T106 = fd.ops.cat([T105, T86], dim=-1, manual_padding=0)
    T112 = fd.ops.broadcast_in_dim(
        T61, shape=[1, 32, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T113 = fd.ops.cast(T106, dtype=DataType.Float)
    T114 = fd.ops.cast(T112, dtype=DataType.Float)
    T115 = fd.ops.mul(T113, T114)
    T116 = fd.ops.add(T70, T115)
    T117 = fd.ops.cast(T116, dtype=DataType.BFloat16)
    T123 = fd.ops.broadcast_in_dim(
        T55, shape=[1, 8, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T124 = fd.ops.cast(T18, dtype=DataType.Float)
    T125 = fd.ops.cast(T123, dtype=DataType.Float)
    T126 = fd.ops.mul(T124, T125)
    T142 = fd.ops.slice(
        T18,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 8, 128000, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T158 = fd.ops.slice(
        T18,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 8, 128000, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T159 = fd.ops.cast(T158, dtype=DataType.Float)
    T160 = fd.ops.neg(T159)
    T161 = fd.ops.cast(T160, dtype=DataType.BFloat16)
    T162 = fd.ops.cat([T161, T142], dim=-1, manual_padding=0)
    T168 = fd.ops.broadcast_in_dim(
        T61, shape=[1, 8, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T169 = fd.ops.cast(T162, dtype=DataType.Float)
    T170 = fd.ops.cast(T168, dtype=DataType.Float)
    T171 = fd.ops.mul(T169, T170)
    T172 = fd.ops.add(T126, T171)
    T173 = fd.ops.cast(T172, dtype=DataType.BFloat16)
    T180 = fd.ops.broadcast_in_dim(
        T173, shape=[1, 8, 1, 128000, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T187 = fd.ops.broadcast_in_dim(
        T180, shape=[1, 8, 4, 128000, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T193 = fd.ops.reshape(T187, new_shape=[1, 32, 128000, 128])
    T200 = fd.ops.broadcast_in_dim(
        T25, shape=[1, 8, 1, 128000, 128], broadcast_dims=[0, 1, 3, 4]
    )
    T207 = fd.ops.broadcast_in_dim(
        T200, shape=[1, 8, 4, 128000, 128], broadcast_dims=[0, 1, 2, 3, 4]
    )
    T213 = fd.ops.reshape(T207, new_shape=[1, 32, 128000, 128])
    fd.add_output(T55)
    fd.add_output(T61)
    fd.add_output(T117)
    fd.add_output(T193)
    fd.add_output(T213)


def hf_mistral_nemo_rope_bwd(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[1, 32, 128000, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T1 = fd.define_tensor(
        shape=[1, 1, 128000, 128],
        contiguity=[None, True, None, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T2 = fd.define_tensor(
        shape=[1, 32, 128000, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T3 = fd.define_tensor(
        shape=[1, 32, 128000, 128],
        contiguity=[None, True, True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 2, 1, 0],
    )
    T4 = fd.define_tensor(
        shape=[1, 1, 128000, 128],
        contiguity=[None, True, None, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[3, 1, 2, 0],
    )
    T11 = fd.ops.reshape(T0, new_shape=[1, 8, 4, 128000, 128])
    T12 = fd.ops.cast(T11, dtype=DataType.Float)
    T13 = fd.ops.sum(T12, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T14 = fd.ops.cast(T13, dtype=DataType.BFloat16)
    T21 = fd.ops.broadcast_in_dim(
        T14, shape=[1, 8, 1, 128000, 128], broadcast_dims=[1, 3, 4]
    )
    T22 = fd.ops.cast(T21, dtype=DataType.Float)
    T23 = fd.ops.sum(T22, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T24 = fd.ops.cast(T23, dtype=DataType.BFloat16)
    T30 = fd.ops.broadcast_in_dim(
        T1, shape=[1, 32, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T36 = fd.ops.broadcast_in_dim(
        T24, shape=[1, 8, 128000, 128], broadcast_dims=[1, 2, 3]
    )
    T42 = fd.ops.broadcast_in_dim(
        T1, shape=[1, 8, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    T43 = fd.ops.cast(T2, dtype=DataType.Float)
    T44 = fd.ops.cast(T30, dtype=DataType.Float)
    T45 = fd.ops.cast(T36, dtype=DataType.Float)
    T46 = fd.ops.cast(T42, dtype=DataType.Float)
    T47 = fd.ops.mul(T44, T43)
    T48 = fd.ops.mul(T46, T45)
    T49 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    T50 = fd.ops.cast(T48, dtype=DataType.BFloat16)
    T66 = fd.ops.slice(
        T49,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 32, 128000, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T82 = fd.ops.slice(
        T50,
        start_indices=[0, 0, 0, 0],
        end_indices=[1, 8, 128000, 64],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T89 = fd.ops.reshape(T3, new_shape=[1, 8, 4, 128000, 128])
    T90 = fd.ops.cast(T66, dtype=DataType.Float)
    T91 = fd.ops.cast(T82, dtype=DataType.Float)
    T92 = fd.ops.cast(T89, dtype=DataType.Float)
    T93 = fd.ops.neg(T90)
    T94 = fd.ops.neg(T91)
    T95 = fd.ops.sum(T92, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T111 = fd.ops.slice(
        T49,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 32, 128000, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T112 = fd.ops.cast(T93, dtype=DataType.BFloat16)
    T128 = fd.ops.slice(
        T50,
        start_indices=[0, 0, 0, 64],
        end_indices=[1, 8, 128000, 128],
        strides=[1, 1, 1, 1],
        manual_normalization=0,
    )
    T129 = fd.ops.cast(T94, dtype=DataType.BFloat16)
    T130 = fd.ops.cast(T95, dtype=DataType.BFloat16)
    T136 = fd.ops.broadcast_in_dim(
        T4, shape=[1, 32, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    S137 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T147 = fd.ops.pad(T111, [0, 64, 0, 0, 0, 0, 0, 0], S137)
    S148 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T158 = fd.ops.pad(T112, [64, 0, 0, 0, 0, 0, 0, 0], S148)
    T164 = fd.ops.broadcast_in_dim(
        T4, shape=[1, 8, 128000, 128], broadcast_dims=[0, 1, 2, 3]
    )
    S165 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T175 = fd.ops.pad(T128, [0, 64, 0, 0, 0, 0, 0, 0], S165)
    S176 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T186 = fd.ops.pad(T129, [64, 0, 0, 0, 0, 0, 0, 0], S176)
    T193 = fd.ops.broadcast_in_dim(
        T130, shape=[1, 8, 1, 128000, 128], broadcast_dims=[1, 3, 4]
    )
    T194 = fd.ops.cast(T136, dtype=DataType.Float)
    T195 = fd.ops.cast(T147, dtype=DataType.Float)
    T196 = fd.ops.cast(T158, dtype=DataType.Float)
    T197 = fd.ops.cast(T164, dtype=DataType.Float)
    T198 = fd.ops.cast(T175, dtype=DataType.Float)
    T199 = fd.ops.cast(T186, dtype=DataType.Float)
    T200 = fd.ops.cast(T193, dtype=DataType.Float)
    T201 = fd.ops.mul(T194, T43)
    T202 = fd.ops.add(T196, T195)
    T203 = fd.ops.mul(T197, T45)
    T204 = fd.ops.add(T199, T198)
    T205 = fd.ops.sum(T200, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T206 = fd.ops.add(T202, T201)
    T207 = fd.ops.add(T204, T203)
    T208 = fd.ops.cast(T205, dtype=DataType.BFloat16)
    T209 = fd.ops.cast(T206, dtype=DataType.BFloat16)
    T210 = fd.ops.cast(T207, dtype=DataType.BFloat16)
    T216 = fd.ops.broadcast_in_dim(
        T208, shape=[1, 8, 128000, 128], broadcast_dims=[1, 2, 3]
    )
    T217 = fd.ops.permute(T209, dims=[0, 2, 1, 3])
    T218 = fd.ops.permute(T210, dims=[0, 2, 1, 3])
    T219 = fd.ops.permute(T216, dims=[0, 2, 1, 3])
    T224 = fd.ops.reshape(T217, new_shape=[1, 128000, 4096])
    T229 = fd.ops.reshape(T218, new_shape=[1, 128000, 1024])
    T234 = fd.ops.reshape(T219, new_shape=[1, 128000, 1024])
    fd.add_output(T234)
    fd.add_output(T229)
    fd.add_output(T224)


# { 'name_benchmark' : (fn, [[sizes0, optional_strides0, dtype0], [sizes1, dtype1], ...]) }
rope_configurations = {
    "llama_2_7b_hf_rope_fwd": (
        llama_2_7b_hf_rope_fwd,
        [
            ((2, 4096, 12288), torch.bfloat16),
            ((4096, 128), torch.bfloat16),
            ((4096, 128), torch.bfloat16),
        ],
    ),
    "llama_2_7b_hf_rope_bwd": (
        llama_2_7b_hf_rope_bwd,
        [
            ((4096, 128), torch.bfloat16),
            ((2, 32, 4096, 128), torch.bfloat16),
            ((2, 32, 4096, 128), torch.bfloat16),
            ((4096, 128), torch.bfloat16),
        ],
    ),
    "llama_3_8B_rope_fwd": (
        llama_3_8B_rope_fwd,
        [
            ((2, 8192, 6144), torch.bfloat16),
            ((8192, 128), torch.bfloat16),
            ((8192, 128), torch.bfloat16),
        ],
    ),
    "llama_3_8B_rope_bwd": (
        llama_3_8B_rope_bwd,
        [
            ((8192, 128), torch.bfloat16),
            ((2, 32, 8192, 128), torch.bfloat16),
            ((2, 32, 8192, 128), torch.bfloat16),
            ((8192, 128), torch.bfloat16),
        ],
    ),
    "hf_qwen2_rope_fwd": (
        hf_qwen2_rope_fwd,
        [
            ((1, 32768, 3584), torch.bfloat16),
            ((1, 32768, 512), torch.bfloat16),
            ((1, 32768, 512), torch.bfloat16),
            ((1, 32768, 128), torch.bfloat16),
            ((1, 32768, 128), torch.bfloat16),
        ],
    ),
    "hf_qwen2_rope_bwd": (
        hf_qwen2_rope_bwd,
        [
            ((1, 28, 32768, 128), torch.bfloat16),
            ((1, 32768, 3584), torch.bfloat16),
            ((1, 32768, 512), torch.bfloat16),
            ((1, 32768, 128), torch.bfloat16),
            ((1, 28, 32768, 128), torch.bfloat16),
            ((1, 28, 32768, 128), torch.bfloat16),
            ((1, 32768, 128), torch.bfloat16),
        ],
    ),
    "hf_phi3_rope_fwd": (
        hf_phi3_rope_fwd,
        [
            ((2, 4096, 9216), torch.bfloat16),
            ((48,), torch.bfloat16),
            ((1, 4096), torch.int64),
        ],
    ),
    "hf_phi3_rope_bwd": (
        hf_phi3_rope_bwd,
        [
            ((2, 32, 4096, 96), torch.bfloat16),
            ((2, 32, 4096, 96), (0, 0, 96, 1), torch.bfloat16),
            ((2, 32, 4096, 96), (0, 0, 96, 1), torch.bfloat16),
            ((2, 32, 4096, 96), torch.bfloat16),
        ],
    ),
    "hf_mistral_nemo_rope_fwd": (
        hf_mistral_nemo_rope_fwd,
        [
            ((1, 128000, 4096), torch.bfloat16),
            ((1, 128000, 1024), torch.bfloat16),
            ((1, 128000, 1024), torch.bfloat16),
            ((64,), torch.bfloat16),
            ((1, 128000), torch.int64),
        ],
    ),
    "hf_mistral_nemo_rope_bwd": (
        hf_mistral_nemo_rope_bwd,
        [
            ((1, 32, 128000, 128), torch.bfloat16),
            ((1, 1, 128000, 128), torch.bfloat16),
            ((1, 32, 128000, 128), torch.bfloat16),
            ((1, 32, 128000, 128), torch.bfloat16),
            ((1, 1, 128000, 128), torch.bfloat16),
        ],
    ),
}


@pytest.mark.parametrize(
    "rope_variation",
    [
        "llama_2_7b_hf_rope_fwd",
        "llama_2_7b_hf_rope_bwd",
        "llama_3_8B_rope_fwd",
        "llama_3_8B_rope_bwd",
        "hf_qwen2_rope_fwd",
        "hf_qwen2_rope_bwd",
        "hf_ph3_rope_fwd",
        "hf_ph3_rope_bwd",
        "hf_mistral_nemo_rope_fwd",
        "hf_mistral_nemo_rope_bwd",
    ],
)
def test_rope_variations_nvf_benchmark(
    benchmark,
    rope_variation: str,
    disable_benchmarking: bool,
):
    config = rope_configurations[rope_variation]

    inputs = []
    for entry in config[1]:
        tensor = torch.testing.make_tensor(entry[0], dtype=entry[-1], device="cuda:0")
        inputs.append(
            tensor if len(entry) == 2 else tensor.as_strided(entry[0], entry[1])
        )

    with FusionDefinition() as fd:
        config[0](fd)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
