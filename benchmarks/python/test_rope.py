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


def test_rope_hf_qwen2_fwd(benchmark):
    """
    Replicates a RoPE from HuggingFace's Qwen2 network as Thunder defines it.
    """

    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 4096, 3584],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 4096, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 4096, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.ops.reshape(T0, new_shape=[1, 4096, 28, 128])
        T11 = fd.ops.permute(T10, dims=[0, 2, 1, 3])
        T17 = fd.ops.reshape(T1, new_shape=[1, 4096, 4, 128])
        T18 = fd.ops.permute(T17, dims=[0, 2, 1, 3])
        T24 = fd.ops.reshape(T2, new_shape=[1, 4096, 4, 128])
        T25 = fd.ops.permute(T24, dims=[0, 2, 1, 3])
        T31 = fd.ops.broadcast_in_dim(
            T3, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T37 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T43 = fd.ops.broadcast_in_dim(
            T31, shape=[1, 28, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T44 = fd.ops.cast(T11, dtype=DataType.Float)
        T45 = fd.ops.cast(T43, dtype=DataType.Float)
        T46 = fd.ops.mul(T44, T45)
        T62 = fd.ops.slice(
            T11,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 28, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T78 = fd.ops.slice(
            T11,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 28, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T79 = fd.ops.cast(T78, dtype=DataType.Float)
        T80 = fd.ops.neg(T79)
        T81 = fd.ops.cast(T80, dtype=DataType.BFloat16)
        T82 = fd.ops.cat([T81, T62], dim=-1, manual_padding=0)
        T88 = fd.ops.broadcast_in_dim(
            T37, shape=[1, 28, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T89 = fd.ops.cast(T82, dtype=DataType.Float)
        T90 = fd.ops.cast(T88, dtype=DataType.Float)
        T91 = fd.ops.mul(T89, T90)
        T92 = fd.ops.add(T46, T91)
        T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
        T99 = fd.ops.broadcast_in_dim(
            T31, shape=[1, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T100 = fd.ops.cast(T18, dtype=DataType.Float)
        T101 = fd.ops.cast(T99, dtype=DataType.Float)
        T102 = fd.ops.mul(T100, T101)
        T118 = fd.ops.slice(
            T18,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T134 = fd.ops.slice(
            T18,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T135 = fd.ops.cast(T134, dtype=DataType.Float)
        T136 = fd.ops.neg(T135)
        T137 = fd.ops.cast(T136, dtype=DataType.BFloat16)
        T138 = fd.ops.cat([T137, T118], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(
            T37, shape=[1, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T145 = fd.ops.cast(T138, dtype=DataType.Float)
        T146 = fd.ops.cast(T144, dtype=DataType.Float)
        T147 = fd.ops.mul(T145, T146)
        T148 = fd.ops.add(T102, T147)
        T149 = fd.ops.cast(T148, dtype=DataType.BFloat16)
        T156 = fd.ops.broadcast_in_dim(
            T149, shape=[1, 4, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T163 = fd.ops.broadcast_in_dim(
            T156, shape=[1, 4, 7, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T169 = fd.ops.reshape(T163, new_shape=[1, 28, 4096, 128])
        T176 = fd.ops.broadcast_in_dim(
            T25, shape=[1, 4, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T183 = fd.ops.broadcast_in_dim(
            T176, shape=[1, 4, 7, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T189 = fd.ops.reshape(T183, new_shape=[1, 28, 4096, 128])
        fd.add_output(T93)
        fd.add_output(T169)
        fd.add_output(T189)

    with FusionDefinition() as fd:
        nvfuser_fusion_id0(fd)

    inputs = (
        torch.randn((1, 4096, 3584), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 128), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 128), device="cuda", dtype=torch.bfloat16),
    )
    benchmark(fd.execute, inputs)


def test_rope_hf_qwen2_bwd(benchmark):
    def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[1, 4096, 3584],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T1 = fd.define_tensor(
            shape=[1, 4096, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T2 = fd.define_tensor(
            shape=[1, 4096, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T3 = fd.define_tensor(
            shape=[1, 4096, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 4096, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T10 = fd.ops.reshape(T0, new_shape=[1, 4096, 28, 128])
        T11 = fd.ops.permute(T10, dims=[0, 2, 1, 3])
        T17 = fd.ops.reshape(T1, new_shape=[1, 4096, 4, 128])
        T18 = fd.ops.permute(T17, dims=[0, 2, 1, 3])
        T24 = fd.ops.reshape(T2, new_shape=[1, 4096, 4, 128])
        T25 = fd.ops.permute(T24, dims=[0, 2, 1, 3])
        T31 = fd.ops.broadcast_in_dim(
            T3, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T37 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3]
        )
        T43 = fd.ops.broadcast_in_dim(
            T31, shape=[1, 28, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T44 = fd.ops.cast(T11, dtype=DataType.Float)
        T45 = fd.ops.cast(T43, dtype=DataType.Float)
        T46 = fd.ops.mul(T44, T45)
        T62 = fd.ops.slice(
            T11,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 28, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T78 = fd.ops.slice(
            T11,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 28, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T79 = fd.ops.cast(T78, dtype=DataType.Float)
        T80 = fd.ops.neg(T79)
        T81 = fd.ops.cast(T80, dtype=DataType.BFloat16)
        T82 = fd.ops.cat([T81, T62], dim=-1, manual_padding=0)
        T88 = fd.ops.broadcast_in_dim(
            T37, shape=[1, 28, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T89 = fd.ops.cast(T82, dtype=DataType.Float)
        T90 = fd.ops.cast(T88, dtype=DataType.Float)
        T91 = fd.ops.mul(T89, T90)
        T92 = fd.ops.add(T46, T91)
        T93 = fd.ops.cast(T92, dtype=DataType.BFloat16)
        T99 = fd.ops.broadcast_in_dim(
            T31, shape=[1, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T100 = fd.ops.cast(T18, dtype=DataType.Float)
        T101 = fd.ops.cast(T99, dtype=DataType.Float)
        T102 = fd.ops.mul(T100, T101)
        T118 = fd.ops.slice(
            T18,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4, 4096, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T134 = fd.ops.slice(
            T18,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4, 4096, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T135 = fd.ops.cast(T134, dtype=DataType.Float)
        T136 = fd.ops.neg(T135)
        T137 = fd.ops.cast(T136, dtype=DataType.BFloat16)
        T138 = fd.ops.cat([T137, T118], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(
            T37, shape=[1, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T145 = fd.ops.cast(T138, dtype=DataType.Float)
        T146 = fd.ops.cast(T144, dtype=DataType.Float)
        T147 = fd.ops.mul(T145, T146)
        T148 = fd.ops.add(T102, T147)
        T149 = fd.ops.cast(T148, dtype=DataType.BFloat16)
        T156 = fd.ops.broadcast_in_dim(
            T149, shape=[1, 4, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T163 = fd.ops.broadcast_in_dim(
            T156, shape=[1, 4, 7, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T169 = fd.ops.reshape(T163, new_shape=[1, 28, 4096, 128])
        T176 = fd.ops.broadcast_in_dim(
            T25, shape=[1, 4, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T183 = fd.ops.broadcast_in_dim(
            T176, shape=[1, 4, 7, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T189 = fd.ops.reshape(T183, new_shape=[1, 28, 4096, 128])
        fd.add_output(T93)
        fd.add_output(T169)
        fd.add_output(T189)

    with FusionDefinition() as fd:
        nvfuser_fusion_id0(fd)

    inputs = (
        torch.randn((1, 4096, 3584), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 512), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 128), device="cuda", dtype=torch.bfloat16),
        torch.randn((1, 4096, 128), device="cuda", dtype=torch.bfloat16),
    )
    benchmark(fd.execute, inputs)
