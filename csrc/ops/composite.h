// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

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

NVF_API ForwardDropoutResult dropout(TensorView* x, Val* prob);

NVF_API ForwardDropoutResult dropout(TensorView* x, Val* prob, Val* scale);

NVF_API TensorView* dropout_backward(
    TensorView* dy,
    TensorView* mask,
    Val* scale);

struct LstmResult {
  TensorView* cell = nullptr;
  TensorView* hidden = nullptr;
};

NVF_API LstmResult lstm(
    TensorView* prev_cell,
    TensorView* in_x,
    TensorView* forget_x,
    TensorView* cell_x,
    TensorView* out_x);

// Matmul function which takes in tensors with the shapes
// A[M,K] B[K,N], but the tensors may have different layouts
// via strides. All restrictions from the matmul APIs also
// apply here.
TensorView* matmul(TensorView* a, TensorView* b);

NVF_API TensorView* sign(TensorView* x);
NVF_API Val* sign(Val* x);
TensorView* softplus(TensorView* x, Val* beta, Val* threshold);
NVF_API TensorView* gelu(TensorView* x);
NVF_API TensorView* gelu_backward(TensorView* dy, TensorView* x);
TensorView* tanh_gelu(TensorView* x);
TensorView* tanh_gelu_backward(TensorView* dy, TensorView* x);
TensorView* tanh_backward(TensorView* dy, TensorView* tanh_x);
TensorView* leaky_relu(TensorView* x, Val* negative_slope);

NVF_API TensorView* view_as_real(TensorView* x);

} // namespace nvfuser
