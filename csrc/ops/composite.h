// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir/interface_nodes.h>
#include <type.h>

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace nvfuser {

struct ForwardDropoutResult {
  TensorView* output = nullptr;
  TensorView* mask = nullptr;
};

TORCH_CUDA_CU_API ForwardDropoutResult dropout(TensorView* x, Val* prob);

TORCH_CUDA_CU_API ForwardDropoutResult
dropout(TensorView* x, Val* prob, Val* scale);

TORCH_CUDA_CU_API TensorView* dropout_backward(
    TensorView* dy,
    TensorView* mask,
    Val* scale);

struct LstmResult {
  TensorView* cell = nullptr;
  TensorView* hidden = nullptr;
};

TORCH_CUDA_CU_API LstmResult lstm(
    TensorView* prev_cell,
    TensorView* in_x,
    TensorView* forget_x,
    TensorView* cell_x,
    TensorView* out_x);

// Matmul functions are temporary internal functions for testing purposes only
// NOTE: These functions have the following restrictions:
// 1. M, N, and K dimensions must be multiples of 8
// 2. Tensors must be contiguously defined.
// 3. Inputs must be FP16/BF16
// 4. Heuristic support only exists for Ampere
TORCH_CUDA_CU_API TensorView* _matmul_nn(TensorView* a, TensorView* b);
TORCH_CUDA_CU_API TensorView* _matmul_nt(TensorView* a, TensorView* b);
TORCH_CUDA_CU_API TensorView* _matmul_tn(TensorView* a, TensorView* b);
TORCH_CUDA_CU_API TensorView* _matmul_tt(TensorView* a, TensorView* b);

TORCH_CUDA_CU_API TensorView* sign(TensorView* x);
TORCH_CUDA_CU_API Val* sign(Val* x);
TORCH_CUDA_CU_API TensorView* softplus(
    TensorView* x,
    Val* beta,
    Val* threshold);
TORCH_CUDA_CU_API TensorView* gelu(TensorView* x);
TORCH_CUDA_CU_API TensorView* gelu_backward(TensorView* dy, TensorView* x);
TORCH_CUDA_CU_API TensorView* tanh_gelu(TensorView* x);
TORCH_CUDA_CU_API TensorView* tanh_gelu_backward(TensorView* dy, TensorView* x);
TORCH_CUDA_CU_API TensorView* tanh_backward(TensorView* dy, TensorView* tanh_x);
TORCH_CUDA_CU_API TensorView* leaky_relu(TensorView* x, Val* negative_slope);

TORCH_CUDA_CU_API TensorView* view_as_real(TensorView* x);

} // namespace nvfuser
