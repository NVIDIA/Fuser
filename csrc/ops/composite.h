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

NVF_API TensorView* triu(TensorView* tv, Val* offset);

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

// Linear functions which takes in two tensors of shapes input[* , in_features],
// weight[out_features, in_features] / [in_features] and an optional bias of
// shape [out_features] or 0D scalar. Bias can only be given if weight is a 2-D
// tensor.
TensorView* linear(TensorView* input, TensorView* weight, TensorView* bias);
// This is an implementation detail to reflect when linear is called
// without a bias. This calls the above function. We use this function
// since it simplifies creating a Python API which takes optional arguments.
// Other options include using lambdas or creating a new RecordFunctor for
// Linear.
TensorView* linear(TensorView* input, TensorView* weight);

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

// Matmul function which takes in tensors with the shapes
// A[*, M, K] / A[K] and B[*, K, N] / B[K], but the tensors may have different
// layouts via strides. This has the same functionality as torch.matmul
TensorView* matmul(TensorView* tv_a, TensorView* tv_b);

// Scaled Dot Product Flash Attention Forward Result
struct SdpfaFwdResult {
  TensorView* output = nullptr;
  TensorView* log_sumexp = nullptr;
  TensorView* philox_seed = nullptr;
  TensorView* philox_offset = nullptr;
};

// Scaled Dot Product Flash Attention Forward API.
// Returns the same output as at::_scaled_dot_product_flash_attention
SdpfaFwdResult sdpfa_fwd(
    TensorView* query,
    TensorView* key,
    TensorView* value,
    Val* dropout_p,
    Val* is_causal,
    Val* scale);

// Scaled Dot Product Flash Attention Backward Result
struct SdpfaBwdResult {
  TensorView* grad_query = nullptr;
  TensorView* grad_key = nullptr;
  TensorView* grad_value = nullptr;
};

// Scaled Dot Product Flash Attention Backward API.
// Returns the same output as at::_scaled_dot_product_flash_attention_backward
SdpfaBwdResult sdpfa_bwd(
    TensorView* grad_output,
    TensorView* query,
    TensorView* key,
    TensorView* value,
    TensorView* output,
    TensorView* log_sumexp,
    Val* dropout_p,
    Val* is_causal,
    TensorView* philox_seed,
    TensorView* philox_offset,
    Val* scale);

TensorView* embedding_fwd(
    TensorView* input,
    TensorView* weight,
    Val* padding_idx,
    Val* max_norm,
    Val* norm_type,
    Val* scale_grad_by_freq,
    Val* sparse);

} // namespace nvfuser
