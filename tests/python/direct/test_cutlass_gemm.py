# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import pytest
import torch
from python.direct_utils import is_pre_blackwell
from nvfuser_direct import nvf_cutlass


@pytest.mark.skipif(
    is_pre_blackwell(), reason="Only supported on blackwell and newer devices."
)
@pytest.mark.skipif(
    not is_pre_blackwell_12(), reason="Does not support blackwell compute 12.0."
)
@pytest.mark.parametrize("config", [[1024, 128, 256], [32, 128, 256]])
@pytest.mark.parametrize("tokens_per_expert_neg_one", [[115, 144, 8], [5, 7, 9]])
@pytest.mark.parametrize("tensor_dtype", [torch.bfloat16, torch.float16])
def test_grouped_mm(
    config,
    tokens_per_expert_neg_one,
    tensor_dtype,
):
    # k dimension is multiple of 128 to avoid padding
    m, n, k = config
    tokens_per_expert = list(tokens_per_expert_neg_one)
    tokens_per_expert.append(m - sum(tokens_per_expert))
    g = len(tokens_per_expert)

    mat1 = torch.testing.make_tensor((m, k), dtype=tensor_dtype, device="cuda:0")
    mat2 = torch.testing.make_tensor((g, n, k), dtype=tensor_dtype, device="cuda:0")
    ab_strides = torch.full((g,), k, dtype=torch.int64, device="cuda:0")
    c_strides = torch.full((g,), n, dtype=torch.int64, device="cuda:0")

    offsets = torch.empty((g,), dtype=torch.int32, device="cuda:0")
    problem_sizes = torch.empty((g, 3), dtype=torch.int32, device="cuda:0")

    accumulated_tokens = 0
    # Use tokens_per_expert to calculate offsets into m dimension of input tensor.
    for i in range(g):
        offsets[i] = accumulated_tokens
        accumulated_tokens += tokens_per_expert[i]

        problem_sizes[i][0] = tokens_per_expert[i]
        problem_sizes[i][1] = n
        problem_sizes[i][2] = k

    out = nvf_cutlass.grouped_mm(
        mat1,
        mat2,
        ab_strides,
        c_strides,
        problem_sizes,
        offsets,
    )

    # Create pytorch expected output reference
    # For each expert, apply gemm. Slice the input matrix given the tokens_per_expert.
    # C[start:stop] = A[start:stop] @ B[expert].
    out_decomposed_ref = torch.empty(m, n, dtype=tensor_dtype, device="cuda:0")
    for i in range(g):
        l = offsets[i]
        if i == g - 1:
            r = m
        else:
            r = offsets[i + 1]
        # Get tokens for expert from activations
        #     mat1 [m, k] => mat1 [l:r, k]
        # Select expert from weights
        #     mat2 [g, n, k] => mat2 [i, n, k]
        # Transpose for matmul operation
        #     transpose(mat2 [i, n, k]) => mat2 [i, k, n]
        # Update output matrix with expert matmul operation
        #     out [l:r, n] = mat1[l:r, n] @ mat2[i, k, n]
        out_decomposed_ref[l:r] = torch.matmul(mat1[l:r], mat2[i].transpose(-1, -2))

    assert torch.allclose(out_decomposed_ref, out, atol=1e-2, rtol=1e-2)
