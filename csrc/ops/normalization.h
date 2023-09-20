// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <ir/interface_nodes.h>
#include <type.h>

#include <tuple>
#include <vector>

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace nvfuser {

struct ForwardNormResult {
  TensorView* output = nullptr;
  TensorView* mean = nullptr;
  TensorView* invstd = nullptr;
};

struct BackwardNormResult {
  TensorView* grad_input = nullptr;
  TensorView* grad_weight = nullptr;
  TensorView* grad_bias = nullptr;
};

struct ForwardRMSNormResult {
  TensorView* output = nullptr;
  TensorView* invstd = nullptr;
};

struct BackwardRMSNormResult {
  TensorView* grad_input = nullptr;
  TensorView* grad_weight = nullptr;
};

struct VarMeanResult {
  TensorView* var = nullptr;
  TensorView* mean = nullptr;
};

} // namespace nvfuser

namespace std {

// Make these results behave like a std::tuple
using nvfuser::BackwardNormResult;
using nvfuser::BackwardRMSNormResult;
using nvfuser::ForwardNormResult;
using nvfuser::ForwardRMSNormResult;
using nvfuser::TensorView;
using nvfuser::VarMeanResult;

template <int i>
constexpr TensorView* get(const ForwardNormResult& results) {
  if (i == 0) {
    return results.output;
  }
  if (i == 1) {
    return results.mean;
  }
  if (i == 2) {
    return results.invstd;
  }
  return nullptr;
}

template <int i>
constexpr TensorView* get(const BackwardNormResult& results) {
  if (i == 0) {
    return results.grad_input;
  }
  if (i == 1) {
    return results.grad_weight;
  }
  if (i == 2) {
    return results.grad_bias;
  }
  return nullptr;
}

template <int i>
constexpr TensorView* get(const ForwardRMSNormResult& results) {
  if (i == 0) {
    return results.output;
  }
  if (i == 1) {
    return results.invstd;
  }
  return nullptr;
}

template <int i>
constexpr TensorView* get(const BackwardRMSNormResult& results) {
  if (i == 0) {
    return results.grad_input;
  }
  if (i == 1) {
    return results.grad_weight;
  }
  return nullptr;
}

template <int i>
constexpr TensorView* get(const VarMeanResult& results) {
  if (i == 0) {
    return results.var;
  }
  if (i == 1) {
    return results.mean;
  }
  return nullptr;
}

} // namespace std

namespace nvfuser {

TensorView* mean(TensorView* x, const std::vector<int>& dims, bool keepdim);

TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim);

TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim);

VarMeanResult variance_mean(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim);

TensorView* standard_deviation(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim);

TensorView* softmax(TensorView* x, int dim);

TensorView* softmax_backward(TensorView* dy, TensorView* y, const int dim);

TensorView* log_softmax(TensorView* x, int dim);

TensorView* log_softmax_backward(TensorView* dy, TensorView* y, const int dim);

ForwardNormResult layer_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    TensorView* bias,
    Val* eps);

ForwardNormResult layer_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    TensorView* bias,
    Val* eps);

ForwardRMSNormResult rms_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    Val* eps);

ForwardRMSNormResult rms_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    Val* eps);

BackwardNormResult layer_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* mean,
    TensorView* rstd,
    TensorView* weight,
    TensorView* bias,
    const std::vector<bool>& output_mask);

BackwardRMSNormResult rms_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* rstd,
    TensorView* weight,
    const std::vector<bool>& output_mask);

ForwardNormResult batch_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kTraining,
    Val* momentum,
    Val* eps,
    bool channels_last = false);

BackwardNormResult batch_norm_backward(
    TensorView* x,
    TensorView* dy,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last = false);

ForwardNormResult instance_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kUseInputStats, // kTraining?
    Val* momentum,
    Val* eps,
    bool channels_last = false);

BackwardNormResult instance_norm_backward(
    TensorView* x,
    TensorView* dy,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last = false);

} // namespace nvfuser
