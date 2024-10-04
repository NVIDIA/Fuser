// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <transform_view.h>

namespace nvfuser {

ForwardDropoutResult dropout(TensorView* x, Val* prob) {
  auto p1m = sub(IrBuilder::createInContainer<Val>(x->container(), 1.), prob);
  auto zero_check =
      add(eq(p1m, IrBuilder::createInContainer<Val>(x->container(), 0.)), p1m);
  auto scale =
      div(IrBuilder::createInContainer<Val>(x->container(), 1.), zero_check);
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

namespace {

TensorView* newForLinear(
    TensorView* input,
    TensorView* weight,
    TensorView* bias) {
  auto input_domain = TensorDomain::noReductions(input->getLogicalDomain());
  auto weight_domain = TensorDomain::noReductions(weight->getLogicalDomain());

  // Output has a reduction axis rK if K is not bcast
  NVF_CHECK(
      input_domain.back()->isBroadcast() == weight_domain.back()->isBroadcast(),
      "K should be broadcast in both inputs and weights, or neither.");
  bool k_bcast = input_domain.back()->isBroadcast();
  size_t red_dims = k_bcast ? 0 : 1;

  // Linear: a = {*, in_features}, b = {out_features, in_features} /
  // {in_features}.The linear output is {*, (out_features), rK?}.
  // Reduction K is present only when K is not bcast.
  auto ndims_out =
      (input_domain.size() - 1) + (weight_domain.size() - 1) + red_dims;

  const std::vector<IterDomain*>& mapping_a =
      ops::mapLinearOpIterDomains(input_domain, 0, ndims_out, k_bcast);
  const std::vector<IterDomain*>& mapping_b =
      ops::mapLinearOpIterDomains(weight_domain, 1, ndims_out, k_bcast);
  std::vector<IterDomain*> mapping_bias(ndims_out, nullptr);
  if (bias != nullptr) {
    auto bias_domain = TensorDomain::noReductions(bias->getLogicalDomain());
    mapping_bias =
        ops::mapLinearOpIterDomains(bias_domain, 2, ndims_out, k_bcast);
  }

  std::vector<IterDomain*> out_domain(ndims_out, nullptr);

  for (auto idx : c10::irange(ndims_out - red_dims)) {
    out_domain[idx] = ops::newOutputIterDomain(
        {mapping_a.at(idx), mapping_b.at(idx), mapping_bias.at(idx)});
  }

  if (!k_bcast) {
    // Specify the iterdomain for K as reduction
    out_domain[ndims_out - 1] = ops::newOutputIterDomain(
        {mapping_a.back(), mapping_b.back()},
        /*force_iter_type=*/IterType::Reduction);
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      out_domain, TensorDomain::getContiguityFilledWith(out_domain, true));

  return IrBuilder::create<TensorView>(td, input->dtype());
}

} // namespace

TensorView* linear(TensorView* input, TensorView* weight, TensorView* bias) {
  auto input_ndims =
      TensorDomain::noReductions(input->getLogicalDomain()).size();
  NVF_CHECK(input_ndims > 0, "Input A must be at least 1D.");

  // `linear` previously supported 1D weight and 0D bias. The support was
  // however removed by #3073 to support sharded linear layers, yet-another
  // workaround of #2563. Otherwise, it would be unclear whether a 2D weight is
  // one device dimension plus a non-device or two non-devices.
  //
  // If needed, we can still support 1D weight and 0D bias in Thunder by
  // changing the thunder-to-nvFuser bridge to convert a 1D/0D linear to
  // unsqueeze followed by a 2D/1D linear followed by a squeeze. It'll likely
  // be the same speed because nvFuser treats squeezes and unsqueezes as meta
  // ops and run them on the host.
  auto weight_ndims =
      TensorDomain::noReductions(weight->getLogicalDomain()).size();
  NVF_CHECK(
      weight_ndims >= 2,
      "Input B must be at least 2D. The last two dimensions represent out "
      "features and in features. The extra, preceding dimensions are expected "
      "to be parallelized on DIDs during scheduling: ",
      weight);
  NVF_CHECK(
      input->dtype() == weight->dtype(),
      "Expected input and weight dtypes to have the same dtype, got: ",
      input->dtype(),
      " and ",
      weight->dtype());

  if (bias != nullptr) {
    NVF_CHECK(
        !TensorDomain::noReductions(bias->getLogicalDomain()).empty(),
        "Input bias must be at least 1D. The last dimension represents out "
        "features. The extra, preceding dimensions are expected to be "
        "parallelized on DIDs during scheduling: ",
        bias);
    NVF_CHECK(
        bias->dtype() == input->dtype(),
        "Expected bias to have the same dtype as A and B, got: ",
        bias->dtype(),
        " and ",
        input->dtype());
  }

  TensorView* out = newForLinear(input, weight, bias);
  IrBuilder::create<LinearOp>(out, input, weight, bias);
  return out;
}

TensorView* linear(TensorView* tv_a, TensorView* tv_b) {
  return linear(tv_a, tv_b, /*bias=*/nullptr);
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
  auto zero = IrBuilder::createInContainer<Val>(x->container(), 0.);
  auto one = IrBuilder::createInContainer<Val>(x->container(), 1.);
  auto minus_one = IrBuilder::createInContainer<Val>(x->container(), -1.);
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

  auto kappa = IrBuilder::createInContainer<Val>(x->container(), M_SQRT1_2);
  auto half = IrBuilder::createInContainer<Val>(x->container(), 0.5);
  auto one = IrBuilder::createInContainer<Val>(x->container(), 1.);

  auto cdf = mul(half, add(one, erf(mul(x, kappa))));
  auto y = mul(x, cdf);
  return y;
}

TensorView* gelu_backward(TensorView* dy, TensorView* x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  const double kHalf = 0.5;

  auto cdf_1 =
      mul(x, IrBuilder::createInContainer<Val>(x->container(), M_SQRT1_2));
  auto cdf_2 = erf(cdf_1);
  auto cdf_3 =
      add(cdf_2, IrBuilder::createInContainer<Val>(x->container(), 1.));
  auto cdf_4 =
      mul(cdf_3, IrBuilder::createInContainer<Val>(x->container(), kHalf));

  auto pdf_1 = mul(x, x);
  auto pdf_2 =
      mul(pdf_1, IrBuilder::createInContainer<Val>(x->container(), -kHalf));
  auto pdf_3 = exp(pdf_2);

  auto out = addcmul(
      cdf_4,
      x,
      pdf_3,
      IrBuilder::createInContainer<Val>(x->container(), kAlpha));
  auto dx = mul(out, dy);
  return dx;
}

TensorView* tanh_gelu(TensorView* x) {
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_cube = mul(x, mul(x, x));

  auto inner_1 =
      mul(IrBuilder::createInContainer<Val>(x->container(), kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 =
      mul(IrBuilder::createInContainer<Val>(x->container(), kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto out = mul(
      x,
      add(IrBuilder::createInContainer<Val>(x->container(), 1.), tanh_inner));
  auto y = mul(IrBuilder::createInContainer<Val>(x->container(), 0.5), out);
  return y;
}

TensorView* tanh_gelu_backward(TensorView* dy, TensorView* x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_sq = mul(x, x);
  auto x_cube = mul(x, x_sq);
  auto beta = IrBuilder::create<Val>(kBeta);
  auto kappa = IrBuilder::create<Val>(kKappa);
  auto one = IrBuilder::create<Val>(1.0);
  auto half = IrBuilder::create<Val>(0.5);

  auto inner = mul(beta, add(x, mul(kappa, x_cube)));
  auto tanh_inner = tanh(inner);

  auto left = mul(half, x);
  auto right = add(one, tanh_inner);

  auto left_derivative = mul(half, right);

  auto tanh_derivative = sub(one, mul(tanh_inner, tanh_inner));
  auto inner_derivative =
      mul(beta, add(one, mul(mul(IrBuilder::create<Val>(3.0), kappa), x_sq)));
  auto right_derivative = mul(mul(left, tanh_derivative), inner_derivative);

  auto dx = mul(dy, add(left_derivative, right_derivative));
  return dx;
}

TensorView* tanh_backward(TensorView* dy, TensorView* tanh_x) {
  NVF_ERROR(dy != nullptr, "Grad Output is invalid.");
  NVF_ERROR(tanh_x != nullptr, "Input is invalid");

  auto one = IrBuilder::createInContainer<Val>(tanh_x->container(), 1.);
  auto tanh_sq = mul(tanh_x, tanh_x);
  auto sub_tanh_sq = sub(one, tanh_sq);
  auto dx = mul(dy, sub_tanh_sq);
  return dx;
}

TensorView* leaky_relu(TensorView* x, Val* negative_slope) {
  NVF_ERROR(x != nullptr, "input is invalid.");
  NVF_ERROR(negative_slope != nullptr, "negative_slope is invalid");
  auto zero = IrBuilder::createInContainer<Val>(x->container(), 0.);
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
TensorView* newForMatmul(TensorView* tv_a, TensorView* tv_b) {
  auto orig_domain_a = TensorDomain::noReductions(tv_a->getLogicalDomain());
  auto orig_domain_b = TensorDomain::noReductions(tv_b->getLogicalDomain());

  auto ndims_a = orig_domain_a.size();
  auto ndims_b = orig_domain_b.size();

  auto b_kpos = orig_domain_b.size() > 1 ? ndims_b - 2 : ndims_b - 1;
  NVF_CHECK(
      orig_domain_a.back()->isBroadcast() ==
          orig_domain_b.at(b_kpos)->isBroadcast(),
      "K should be broadcast in both A and B, or neither.");

  // Output has a reduction axis rK if K is not bcast
  bool k_bcast = orig_domain_a.back()->isBroadcast();
  size_t red_dims = k_bcast ? 0 : 1;

  // Matmul output size is same as the higher dimensional input size if both A/B
  // > 1D, but with 1 additional IterType::Reduction axis rK if K is not
  // broadcast.
  auto ndims_out = std::max(ndims_a, ndims_b) + red_dims;
  if (std::min(ndims_a, ndims_b) == 1) {
    // If one of the inputs is 1D, the output size is the same as the higher
    // dimensional input size, since we will include a Reduction axis for K in
    // the output. For example: [iM, iK] x [iK] -> [iM, rK]
    ndims_out = std::max(ndims_a, ndims_b) - 1 + red_dims;
  }

  std::vector<IterDomain*> out_domain(ndims_out, nullptr);

  const std::vector<IterDomain*>& mapping_a =
      ops::mapMatmulOpIterDomains(orig_domain_a, 0, ndims_out);
  const std::vector<IterDomain*>& mapping_b =
      ops::mapMatmulOpIterDomains(orig_domain_b, 1, ndims_out);

  for (auto idx : c10::irange(ndims_out - red_dims)) {
    out_domain[idx] =
        ops::newOutputIterDomain({mapping_a.at(idx), mapping_b.at(idx)});
  }
  if (!k_bcast) {
    out_domain[ndims_out - 1] = ops::newOutputIterDomain(
        {mapping_a.back(), mapping_b.back()},
        /*force_iter_type=*/IterType::Reduction);
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

  // Create a new MatmulOp
  TensorView* out = newForMatmul(tv_a, tv_b);
  IrBuilder::create<MatmulOp>(out, tv_a, tv_b);
  return out;
}

SdpfaFwdResult sdpfa_fwd(
    TensorView* query,
    TensorView* key,
    TensorView* value,
    Val* dropout_p,
    Val* is_causal,
    Val* scale) {
  NVF_CHECK(
      query->dtype() == key->dtype() && query->dtype() == value->dtype(),
      "Expected query, key, and value to have the same dtype but got: ",
      query->dtype(),
      " ",
      key->dtype(),
      " ,and ",
      value->dtype());

  auto query_domain = TensorDomain::noReductions(query->getLogicalDomain());
  auto key_domain = TensorDomain::noReductions(key->getLogicalDomain());
  auto value_domain = TensorDomain::noReductions(value->getLogicalDomain());

  // Temporary handling of DID parallelization see
  // https://github.com/NVIDIA/Fuser/issues/2563
  bool has_device_dim = (query_domain.size() == 5);
  if (has_device_dim) {
    NVF_CHECK(
        query_domain[0]->isDeviceDim(),
        "Only suport DID parallelization on outermost axis");
    NVF_CHECK(
        key_domain[0]->isDeviceDim(),
        "Only suport DID parallelization on outermost axis");
    NVF_CHECK(
        value_domain[0]->isDeviceDim(),
        "Only suport DID parallelization on outermost axis");
  }

  auto concrete_query_size = TensorDomain::noDevices(query_domain).size();
  auto concrete_key_size = TensorDomain::noDevices(key_domain).size();
  auto concrete_value_size = TensorDomain::noDevices(value_domain).size();

  NVF_CHECK(
      concrete_query_size == 4 && concrete_key_size == 4 &&
          concrete_value_size == 4,
      "Expected query, key, and value to be 4D but got: ",
      concrete_query_size,
      " ",
      concrete_key_size,
      " ,and ",
      concrete_value_size);

  NVF_CHECK(
      !dropout_p || dropout_p->isScalar(),
      "Expected dropout to be a scalar double.");
  NVF_CHECK(
      !is_causal || is_causal->isScalar(),
      "Expected is_causal to be a scalar boolean.");
  NVF_CHECK(
      !scale || scale->isScalar(), "Expected scale to be a scalar double.");

  // Query: [DIDx(D)?,N,H,L,E], Key: [DIDx(D)?,N,H,S,E], Value:
  // [DIDx(D)?,N,H,S,Ev] Output: [DIDx(D)?,N,H,L,Ev] N, H are mapped for all
  // inputs to outputs. L is mapped from query to output. Ev is mapped from
  // value to output. Note: There is no mapping for S, E. This may change in the
  // future if we add additional reduction ids to the output.
  auto ndims_out = query_domain.size();

  // TensorView for attention output
  std::vector<IterDomain*> out_domain(ndims_out, nullptr);
  for (auto idx : c10::irange(ndims_out - 2)) {
    out_domain[idx] = ops::newOutputIterDomain(
        {query_domain.at(idx), key_domain.at(idx), value_domain.at(idx)});
  }
  out_domain[ndims_out - 2] =
      ops::newOutputIterDomain({query_domain.at(ndims_out - 2)});
  out_domain[ndims_out - 1] =
      ops::newOutputIterDomain({value_domain.at(ndims_out - 1)});

  TensorDomain* attn_td = IrBuilder::create<TensorDomain>(
      out_domain, TensorDomain::getContiguityFilledWith(out_domain, true));
  TensorView* output = IrBuilder::create<TensorView>(attn_td, query->dtype());

  // TensorView for log_sumexp [DIDx(D)?,N, H, L]
  std::vector<IterDomain*> log_sumexp_dom(ndims_out - 1, nullptr);
  for (auto idx : c10::irange(ndims_out - 2)) {
    log_sumexp_dom[idx] = ops::newOutputIterDomain(
        {query_domain.at(idx), key_domain.at(idx), value_domain.at(idx)});
  }
  log_sumexp_dom[ndims_out - 2] =
      ops::newOutputIterDomain({query_domain.at(ndims_out - 2)});
  TensorDomain* log_sumexp_td = IrBuilder::create<TensorDomain>(
      log_sumexp_dom,
      TensorDomain::getContiguityFilledWith(log_sumexp_dom, true));
  TensorView* log_sumexp =
      IrBuilder::create<TensorView>(log_sumexp_td, DataType::Float);

  // Scalar tensors of int64_t dtype.
  TensorView* philox_seed = TensorViewBuilder().dtype(DataType::Int).build();
  TensorView* philox_offset = TensorViewBuilder().dtype(DataType::Int).build();
  philox_seed->setCpuScalar(true);
  philox_offset->setCpuScalar(true);

  // Set default values for dropout_p (0.0), is_causal(false)
  if (dropout_p == nullptr) {
    dropout_p = IrBuilder::create<Val>(0.0, DataType::Double);
  }

  if (is_causal == nullptr) {
    is_causal = IrBuilder::create<Val>(false, DataType::Bool);
  }

  IrBuilder::create<SdpaFwdOp>(
      output,
      log_sumexp,
      philox_seed,
      philox_offset,
      query,
      key,
      value,
      dropout_p,
      is_causal,
      scale);
  return {output, log_sumexp, philox_seed, philox_offset};
}

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
    Val* scale) {
  NVF_CHECK(
      query->dtype() == key->dtype() && query->dtype() == value->dtype(),
      "Expected query, key, and value to have the same dtype but got: ",
      query->dtype(),
      " ",
      key->dtype(),
      " ,and ",
      value->dtype());

  auto query_domain = TensorDomain::noReductions(query->getLogicalDomain());
  auto key_domain = TensorDomain::noReductions(key->getLogicalDomain());
  auto value_domain = TensorDomain::noReductions(value->getLogicalDomain());

  // Temporary handling of DID parallelization see
  // https://github.com/NVIDIA/Fuser/issues/2563
  bool has_device_dim = (query_domain.size() == 5);
  if (has_device_dim) {
    auto check_first_is_did = [](const std::vector<IterDomain*>& ids) -> void {
      NVF_CHECK(
          ids[0]->isDeviceDim(),
          "Only support DID parallelization on outermost axis");
    };
    check_first_is_did(query_domain);
    check_first_is_did(key_domain);
    check_first_is_did(value_domain);
    check_first_is_did(grad_output->getLogicalDomain());
    check_first_is_did(output->getLogicalDomain());
  }

  auto concrete_query_size = TensorDomain::noDevices(query_domain).size();
  auto concrete_key_size = TensorDomain::noDevices(key_domain).size();
  auto concrete_value_size = TensorDomain::noDevices(value_domain).size();

  NVF_CHECK(
      concrete_query_size == 4 && concrete_key_size == 4 &&
          concrete_value_size == 4,
      "Expected query, key, and value to be 4D but got: ",
      concrete_query_size,
      " ",
      concrete_key_size,
      " ,and ",
      concrete_value_size);

  NVF_CHECK(
      !dropout_p || dropout_p->isScalar(),
      "Expected dropout to be a scalar double.");
  NVF_CHECK(
      !is_causal || is_causal->isScalar(),
      "Expected is_causal to be a scalar boolean.");
  NVF_CHECK(
      !scale || scale->isScalar(), "Expected scale to be a scalar double.");

  // Set default values for dropout_p (0.0), is_causal(false)
  if (dropout_p == nullptr) {
    dropout_p = IrBuilder::create<Val>(0.0, DataType::Double);
  }

  if (is_causal == nullptr) {
    is_causal = IrBuilder::create<Val>(false, DataType::Bool);
  }

  // Mark CPU scalar tensors.
  philox_seed->setCpuScalar(true);
  philox_offset->setCpuScalar(true);

  // Query: [N,H,L,E], Key: [N,H,S,E], Value: [N,H,S,Ev] Output: [N,H,L,Ev]
  TensorView* grad_query = ops::newOutputTV({query}, query->dtype());
  TensorView* grad_key = ops::newOutputTV({key}, key->dtype());
  TensorView* grad_value = ops::newOutputTV({value}, value->dtype());

  IrBuilder::create<SdpaBwdOp>(
      grad_query,
      grad_key,
      grad_value,
      grad_output,
      query,
      key,
      value,
      output,
      log_sumexp,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      scale);
  return {grad_query, grad_key, grad_value};
}

} // namespace nvfuser
