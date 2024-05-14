// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <transform_view.h>

namespace nvfuser {

ForwardDropoutResult dropout(TensorView* x, Val* prob) {
  auto p1m = sub(IrBuilder::create<Val>(x->container(), 1.), prob);
  auto zero_check =
      add(eq(p1m, IrBuilder::create<Val>(x->container(), 0.)), p1m);
  auto scale = div(IrBuilder::create<Val>(x->container(), 1.), zero_check);
  return dropout(x, p1m, scale);
}

ForwardDropoutResult dropout(TensorView* x, Val* prob, Val* scale) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  NVF_ERROR(
      prob != nullptr && prob->getDataType().has_value() &&
          prob->getDataType().value() == DataType::Double,
      "Probability is not a valid Double.");
  NVF_ERROR(
      scale != nullptr && scale->getDataType().has_value() &&
          scale->getDataType().value() == DataType::Double,
      "Scale is not a valid Double.");

  auto rand_vals = rand_like(x);
  auto mask = lt(rand_vals, prob);
  auto apply_mask = mul(x, mask);
  auto y = mul(apply_mask, scale);

  return {y, mask};
}

TensorView* dropout_backward(TensorView* dy, TensorView* mask, Val* scale) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(mask != nullptr, "Mask is invalid");
  NVF_ERROR(
      scale != nullptr && scale->getDataType().has_value() &&
          scale->getDataType().value() == DataType::Double,
      "Scale is not a valid Double.");

  auto grad_mask = mul(dy, mask);
  auto dx = mul(grad_mask, scale);

  return dx;
}

TensorView* linear(TensorView* a, TensorView* b, TensorView* bias) {
  // TODO: Support 1+ dimensional A.
  NVF_CHECK(
      (a->nDims() == 2 && b->nDims() == 2),
      "Only 2-D Inputs and Weights are currently supported in Linear!");

  std::vector<bool> bcast_dims(a->nDims() + 1, false);
  // A: [M, Bcast, K]
  // B: [Bcast, N, K]
  bcast_dims.at(bcast_dims.size() - 2) = true;
  auto* tv0b = broadcast(a, bcast_dims);
  bcast_dims.at(bcast_dims.size() - 2) = false;
  bcast_dims.at(bcast_dims.size() - 3) = true;
  auto* tv1b = broadcast(b, bcast_dims);

  NVF_CHECK(
      a->getDataType().value() == b->getDataType().value(),
      "data types of inputs to matmul don't match");

  auto* output = fusedMultiplySum(tv0b, tv1b, {-1});
  if (bias) {
    NVF_CHECK(
        (bias->nDims() <= a->nDims()), "bias should be broadcastable to A");
    NVF_CHECK(
        a->getDataType().value() == bias->getDataType().value(),
        "bias doesn't match input/weight dtype");
    auto* bias_with_cast = maybeCastOp(output->getDataType().value(), bias);
    auto* bcast_bias = ops::maybeBroadcast({output, bias_with_cast})[1];
    auto* bias_output = add(output, bcast_bias);
    return maybeCastOp(a->getDataType().value(), bias_output);
  }
  return maybeCastOp(a->getDataType().value(), output);
}

TensorView* linear(TensorView* a, TensorView* b) {
  return linear(a, b, nullptr /*bias*/);
}

LstmResult lstm(
    TensorView* prev_cell,
    TensorView* in_x,
    TensorView* forget_x,
    TensorView* cell_x,
    TensorView* out_x) {
  NVF_ERROR(prev_cell != nullptr, "Previous cell state is invalid.");
  NVF_ERROR(in_x != nullptr, "In-gate input is invalid");
  NVF_ERROR(forget_x != nullptr, "Forget-gate input is invalid");
  NVF_ERROR(cell_x != nullptr, "Cell-gate input is invalid");
  NVF_ERROR(out_x != nullptr, "Out-gate input is invalid");

  const auto in_gate = sigmoid(in_x);
  const auto forget_gate = sigmoid(forget_x);
  const auto cell_gate = tanh(cell_x);
  const auto out_gate = sigmoid(out_x);

  const auto cell = add(mul(forget_gate, prev_cell), mul(in_gate, cell_gate));
  const auto hidden = mul(out_gate, tanh(cell));

  return {cell, hidden};
}

namespace {
template <typename T>
T* sign(T* x) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  auto zero = IrBuilder::create<Val>(x->container(), 0.);
  auto one = IrBuilder::create<Val>(x->container(), 1.);
  auto minus_one = IrBuilder::create<Val>(x->container(), -1.);
  auto sign = where(gt(x, zero), one, where(lt(x, zero), minus_one, zero));
  return castOp(x->getDataType().value(), sign);
}
} // namespace

TensorView* sign(TensorView* x) {
  return sign<TensorView>(x);
}

Val* sign(Val* x) {
  return sign<Val>(x);
}

TensorView* softplus(TensorView* x, Val* beta, Val* threshold) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  NVF_ERROR(beta != nullptr, "Beta is invalid.");
  NVF_ERROR(threshold != nullptr, "Threshold is not a valid Double.");

  auto op_beta = mul(x, beta);
  auto maybe_result = div(log1p(exp(op_beta)), beta);
  auto y = where(gt(op_beta, threshold), x, maybe_result);
  return y;
}

TensorView* gelu(TensorView* x) {
  NVF_ERROR(x != nullptr, "Input is invalid");

  auto kappa = IrBuilder::create<Val>(x->container(), M_SQRT1_2);
  auto half = IrBuilder::create<Val>(x->container(), 0.5);
  auto one = IrBuilder::create<Val>(x->container(), 1.);

  auto cdf = mul(half, add(one, erf(mul(x, kappa))));
  auto y = mul(x, cdf);
  return y;
}

TensorView* gelu_backward(TensorView* dy, TensorView* x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  const double kHalf = 0.5;

  auto cdf_1 = mul(x, IrBuilder::create<Val>(x->container(), M_SQRT1_2));
  auto cdf_2 = erf(cdf_1);
  auto cdf_3 = add(cdf_2, IrBuilder::create<Val>(x->container(), 1.));
  auto cdf_4 = mul(cdf_3, IrBuilder::create<Val>(x->container(), kHalf));

  auto pdf_1 = mul(x, x);
  auto pdf_2 = mul(pdf_1, IrBuilder::create<Val>(x->container(), -kHalf));
  auto pdf_3 = exp(pdf_2);

  auto out =
      addcmul(cdf_4, x, pdf_3, IrBuilder::create<Val>(x->container(), kAlpha));
  auto dx = mul(out, dy);
  return dx;
}

TensorView* tanh_gelu(TensorView* x) {
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_cube = mul(x, mul(x, x));

  auto inner_1 = mul(IrBuilder::create<Val>(x->container(), kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(IrBuilder::create<Val>(x->container(), kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto out =
      mul(x, add(IrBuilder::create<Val>(x->container(), 1.), tanh_inner));
  auto y = mul(IrBuilder::create<Val>(x->container(), 0.5), out);
  return y;
}

TensorView* tanh_gelu_backward(TensorView* dy, TensorView* x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_sq = mul(x, x);
  auto x_cube = mul(x, x_sq);

  auto inner_1 = mul(IrBuilder::create<Val>(x->container(), kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(IrBuilder::create<Val>(x->container(), kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto left = mul(IrBuilder::create<Val>(x->container(), 0.5), x);
  auto right = add(IrBuilder::create<Val>(x->container(), 1.), tanh_inner);

  auto left_derivative =
      mul(IrBuilder::create<Val>(x->container(), 0.5), right);

  auto tanh_inner_sq = mul(tanh_inner, tanh_inner);
  auto tanh_derivative =
      sub(IrBuilder::create<Val>(x->container(), 1.0), tanh_inner_sq);

  auto constant_mul_x_sq =
      mul(IrBuilder::create<Val>(x->container(), kBeta * 3 * kKappa), x_sq);
  auto inner_derivative =
      add(IrBuilder::create<Val>(x->container(), kBeta), constant_mul_x_sq);
  auto right_derivative = mul(left, mul(tanh_derivative, inner_derivative));

  auto dx = mul(dy, add(left_derivative, right_derivative));
  return dx;
}

TensorView* tanh_backward(TensorView* dy, TensorView* tanh_x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(tanh_x != nullptr, "Input is invalid");

  auto one = IrBuilder::create<Val>(tanh_x->container(), 1.);
  auto tanh_sq = mul(tanh_x, tanh_x);
  auto sub_tanh_sq = sub(one, tanh_sq);
  auto dx = mul(dy, sub_tanh_sq);
  return dx;
}

TensorView* leaky_relu(TensorView* x, Val* negative_slope) {
  NVF_ERROR(x != nullptr, "input is invalid.");
  NVF_ERROR(negative_slope != nullptr, "negative_slope is invalid");
  auto zero = IrBuilder::create<Val>(x->container(), 0.);
  return where(ge(x, zero), x, mul(negative_slope, x));
}

TensorView* view_as_real(TensorView* x) {
  auto input_type = x->getDataType().value();
  NVF_CHECK(
      isComplexType(input_type),
      "Operand of view_as_real must have complex type");

  auto vec_type = ArrayType{
      std::make_shared<DataType>(getTypeFromComplexType(input_type)), 2};
  auto tv_vector = bitCastOp(vec_type, x);
  return viewAsScalar(tv_vector);
}
namespace {

//! Create new output for matmul
static TensorView* newForMatmul(TensorView* tv_a, TensorView* tv_b) {
  auto orig_domain_a =
      TensorDomain::noReductions(tv_a->getMaybeRFactorDomain());
  auto orig_domain_b =
      TensorDomain::noReductions(tv_b->getMaybeRFactorDomain());

  auto ndims_a = orig_domain_a.size();
  auto ndims_b = orig_domain_b.size();

  // Matmul output size is same as the higher dimensional input size if both A/B
  // > 1D.
  auto ndims_out = std::max(ndims_a, ndims_b);
  if (std::min(ndims_a, ndims_b) == 1) {
    // If one of the inputs is 1D, the output size is 1 less than the higher
    // dimensional input size, since either M/N axis will be missing in the
    // output. For eg: [M, K] x [K] -> [M]
    ndims_out = std::max(ndims_a, ndims_b) - 1;
  }

  std::vector<IterDomain*> out_domain(ndims_out, nullptr);

  const std::vector<IterDomain*>& mapping_a = ops::mapMatmulOpIterDomains(
      orig_domain_a, MatmulRole::INPUT_A, ndims_out);
  const std::vector<IterDomain*>& mapping_b = ops::mapMatmulOpIterDomains(
      orig_domain_b, MatmulRole::INPUT_B, ndims_out);

  for (auto idx : c10::irange(ndims_out)) {
    std::vector<IterDomain*> input_ids;
    input_ids.reserve(2);
    if (mapping_a[idx] != nullptr) {
      input_ids.emplace_back(mapping_a[idx]);
    }
    if (mapping_b[idx] != nullptr) {
      input_ids.emplace_back(mapping_b[idx]);
    }
    out_domain[idx] = ops::newOutputIterDomain(input_ids);
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      out_domain, TensorDomain::getContiguityFilledWith(out_domain, true));

  return IrBuilder::create<TensorView>(td, tv_a->dtype());
}

} // namespace

TensorView* matmul(TensorView* tv_a, TensorView* tv_b) {
  NVF_CHECK(
      tv_a->nDims() > 0 && tv_b->nDims() > 0,
      "Expected inputs to be atleast 1D, got: ",
      tv_a->nDims(),
      " and ",
      tv_b->nDims());

  // Note: torch.matmul reference does not restrict the inputs to the same
  // dtype, but it fails for different input dtypes.
  //       This condition may potentially be modified. The following condition
  //       should change accordingly.
  NVF_CHECK(
      tv_a->dtype() == tv_b->dtype(),
      "Expected A and B dtypes to have the same dtype, got: ",
      tv_a->dtype(),
      " and ",
      tv_b->dtype());

  if (tv_a->nDims() == 1 && tv_b->nDims() == 1) {
    // Return the dot product instead of creating the MatmulOp.
    // Cast back the output if needed since torch.matmul maintains input dtype.
    return maybeCastOp(tv_a->dtype(), sum(mul(tv_a, tv_b), {0}));
  }

  // For all other cases, create a new MatmulOp
  TensorView* out = newForMatmul(tv_a, tv_b);
  IrBuilder::create<MatmulOp>(out, tv_a, tv_b);
  return out;
}

} // namespace nvfuser
