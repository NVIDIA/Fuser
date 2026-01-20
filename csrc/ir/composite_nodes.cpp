// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOptions.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/SymInt.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/options/embedding.h>

#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <ir/allocation_utils.h>
#include <ir/cloner.h>
#include <ir/composite_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <ops/arith.h>
#include <runtime/allocations.h>
#include <transform_iter.h>
#include <transform_rfactor.h>
#include <transform_view.h>
#if NVFUSER_CUTLASS_KERNEL_ENABLED
#include <nvf_cutlass.h>
#endif

namespace nvfuser {

MatmulOp::MatmulOp(IrBuilderPasskey passkey, Val* out, Val* in_a, Val* in_b)
    : Expr(passkey) {
  addOutput(out);
  addInput(in_a);
  addInput(in_b);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MatmulOp)

std::string MatmulOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = matmul(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << inB()->toString() << ")\n";
  return ss.str();
}

std::string MatmulOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> MatmulOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto a = inputs.at(0).as<at::Tensor>();
  const auto b = inputs.at(1).as<at::Tensor>();

  at::Tensor matmul_out;
  // aten::dot does not support meta device. Matmul lowers to dot for 1D @ 1D.
  const bool uses_dot = (a.dim() == 1 && b.dim() == 1);
  if (uses_dot && (a.is_meta() || b.is_meta())) {
    const auto out_scalar_type = at::result_type(a, b);
    auto out_opts = a.options().dtype(out_scalar_type).device(at::kMeta);
    matmul_out = at::empty({}, out_opts);
  } else {
    matmul_out = at::matmul(a, b);
  }

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    matmul_out = matmul_out.unsqueeze(rfactor_did_idx);
  }

  // Without InferContiguity, we mistakenly assume the output is contiguous.
  if (!isOptionEnabled(EnableOption::InferContiguity)) {
    const auto& [sizes, strides] = inferShapeAndContiguousStrides(out(), ee);
    auto meta_out = at::detail::empty_strided_meta(sizes, strides, a.dtype());

    if (meta_out.is_contiguous()) {
      return {matmul_out};
    }

    auto strided_matmul_out = at::empty_strided(sizes, strides, a.options());
    strided_matmul_out = strided_matmul_out.copy_(matmul_out);
    return {strided_matmul_out};
  }
  return {matmul_out};
}

LinearOp::LinearOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* bias)
    : Expr(passkey) {
  addOutput(out);
  addInput(in_a);
  addInput(in_b);

  if (bias != nullptr) {
    addInput(bias);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LinearOp)

std::string LinearOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = linear(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << inB()->toString();
  if (hasBias()) {
    indent(ss, indent_size + 1) << ",\n          " << bias()->toString();
  }
  indent(ss, indent_size + 1) << ")\n";
  return ss.str();
}

std::string LinearOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> LinearOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto in = inputs.at(0).as<at::Tensor>();
  auto weight = inputs.at(1).as<at::Tensor>();

  auto squeeze_device_dims = [](at::Tensor& t,
                                int64_t num_device_dims) -> void {
    // Record the initial shape for the error message.
    std::vector<int64_t> shape = t.sizes().vec();
    for ([[maybe_unused]] auto _ : arange(num_device_dims)) {
      NVF_CHECK(
          t.size(0) == 1,
          "When the weight is >2D, expect its preceding dimensions and "
          "the bias's preceding dimensions to "
          "be DID-parallel and therefore size-1: ",
          shape);
      t = t.squeeze(0);
    }
  };

  // The squeezes and unsqueezes are currently required to support a sharded
  // linear layer. Remove them after #2563.
  auto num_device_dims = weight.dim() - 2;
  squeeze_device_dims(weight, num_device_dims);

  at::Tensor out_tensor;
  if (hasBias()) {
    auto bias = inputs.at(2).as<at::Tensor>();
    squeeze_device_dims(bias, num_device_dims);
    out_tensor = at::linear(in, weight, bias);
  } else {
    out_tensor = at::linear(in, weight);
  }

  for ([[maybe_unused]] auto _ : arange(num_device_dims)) {
    out_tensor = out_tensor.unsqueeze(0);
  }

  // Handle rFactor DIDs similar to MatmulOp::evaluate.
  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    out_tensor = out_tensor.unsqueeze(rfactor_did_idx);
  }

  return {out_tensor};
}

SdpaFwdOp::SdpaFwdOp(
    IrBuilderPasskey passkey,
    TensorView* output,
    TensorView* log_sumexp,
    TensorView* philox_seed,
    TensorView* philox_offset,
    TensorView* query,
    TensorView* key,
    TensorView* value,
    TensorView* bias,
    TensorView* mask,
    Val* dropout_p,
    Val* is_causal,
    Val* scale)
    : Expr(passkey) {
  addOutput(output);
  addOutput(log_sumexp);
  addOutput(philox_seed);
  addOutput(philox_offset);

  addInput(query);
  addInput(key);
  addInput(value);
  addInput(dropout_p);
  addInput(is_causal);
  auto next_index = std::ssize(inputs());
  int64_t scale_input_index = -1;
  if (scale != nullptr) {
    scale_input_index = next_index++;
    addInput(scale);
  }
  int64_t bias_input_index = -1;
  if (bias != nullptr) {
    bias_input_index = next_index++;
    addInput(bias);
  }
  int64_t mask_input_index = -1;
  if (mask != nullptr) {
    mask_input_index = next_index++;
    addInput(mask);
  }

  addDataAttribute(scale_input_index);
  addDataAttribute(bias_input_index);
  addDataAttribute(mask_input_index);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SdpaFwdOp)

std::string SdpaFwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << attn_out()->toString() << "," << std::endl;
  indent(ss, indent_size) << logsumexp()->toString() << "," << std::endl;
  indent(ss, indent_size) << philox_seed()->toString() << "," << std::endl;
  indent(ss, indent_size) << philox_offset()->toString() << std::endl;
  indent(ss, indent_size + 1)
      << " = sdpa(" << query()->toString() << "," << std::endl;
  indent(ss, indent_size + 1)
      << "          " << key()->toString() << "," << std::endl;
  indent(ss, indent_size + 1)
      << "          " << value()->toString() << "," << std::endl;
  if (bias() != nullptr) {
    indent(ss, indent_size + 1)
        << "          bias=" << bias()->toString() << "," << std::endl;
  }
  if (mask() != nullptr) {
    indent(ss, indent_size + 1)
        << "          mask=" << mask()->toString() << "," << std::endl;
  }
  indent(ss, indent_size + 1)
      << "          dropout_p = " << dropout_p()->toInlineString() << ","
      << std::endl;
  indent(ss, indent_size + 1)
      << "          is_causal=" << is_causal()->toInlineString() << ","
      << std::endl;
  if (scale() != nullptr) {
    indent(ss, indent_size + 1)
        << "          scale=" << scale()->toInlineString() << "," << std::endl;
  }
  indent(ss, indent_size + 1) << ")" << std::endl;
  return ss.str();
}

std::string SdpaFwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
scaled_dot_product_attention_meta(at::Tensor query, at::Tensor value) {
  NVF_ERROR_EQ(query.dim(), 4);
  NVF_ERROR_EQ(value.dim(), 4);
  const auto batch_size = query.size(0);
  const auto num_heads = query.size(1);
  const auto seqlen_q = query.size(2);
  const auto out_head_dim = value.size(-1);

  auto out = at::empty(
      {batch_size, num_heads, seqlen_q, out_head_dim}, query.options());
  auto logsumexp = at::empty(
      {batch_size, num_heads, seqlen_q}, query.options().dtype(at::kFloat));
  // Produce defined meta tensors for philox outputs so downstream segments
  // can bind metadata and types correctly.
  const auto meta_u64 =
      at::TensorOptions().device(at::kMeta).dtype(at::kUInt64);
  // philox_seed/rng_state, see note:
  // https://github.com/pytorch/pytorch/blob/cdc8460f2c76f98ba30556e3f9358e857a2f22f0/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L773-L778
  auto rng_state = at::empty({2}, meta_u64);
  auto rng_offset = at::empty({}, meta_u64);

  return std::make_tuple(out, logsumexp, rng_state, rng_offset);
}

at::Tensor flattenBatchDims(at::Tensor t) {
  at::DimVector new_shape({-1});
  auto non_batch_dims = t.sizes().slice(t.dim() - 3);
  new_shape.append(non_batch_dims.begin(), non_batch_dims.end());
  return t.view(new_shape);
}

at::Tensor unflattenBatchDim(at::Tensor t, at::IntArrayRef batch_dims) {
  at::DimVector new_shape(batch_dims);
  auto non_batch_dims = t.sizes().slice(1);
  new_shape.append(non_batch_dims.begin(), non_batch_dims.end());
  return t.view(new_shape);
}
} // namespace

std::vector<PolymorphicValue> SdpaFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto query = inputs.at(0).as<at::Tensor>();
  auto key = inputs.at(1).as<at::Tensor>();
  auto value = inputs.at(2).as<at::Tensor>();
  auto bias = this->bias() != nullptr
      ? inputs.at(bias_input_index()).as<at::Tensor>()
      : at::Tensor();
  auto mask = this->mask() != nullptr
      ? inputs.at(mask_input_index()).as<at::Tensor>()
      : at::Tensor();
  const auto dropout_p = inputs.at(3).as<double>();
  const auto is_causal = inputs.at(4).as<bool>();
  const auto last_dim_size = query.size(-1);
  auto scale = this->scale() != nullptr
      ? inputs.at(scale_input_index()).as<double>()
      : 1.0 / std::sqrt(last_dim_size);

  auto batch_dims = query.sizes().slice(0, query.dim() - 3);
  NVF_CHECK_GE(batch_dims.size(), 1);
  query = flattenBatchDims(query);
  key = flattenBatchDims(key);
  value = flattenBatchDims(value);
  NVF_ERROR_EQ(query.dim(), 4);
  NVF_ERROR_EQ(key.dim(), 4);
  NVF_ERROR_EQ(value.dim(), 4);

  at::Tensor attn_bias = bias;
  if (mask.defined()) {
    NVF_CHECK_EQ(mask.dtype(), at::kBool);
    auto mask_bias =
        at::where(mask, 0.0f, -std::numeric_limits<float>::infinity())
            .to(query.dtype());
    if (attn_bias.defined()) {
      // Don't write `attn_bias += mask_bias` because the sum can be a larger
      // shape than `attn_bias`.
      attn_bias = attn_bias + mask_bias;
    } else {
      attn_bias = mask_bias;
    }
  }
  if (attn_bias.defined()) {
    // `attn_bias` is of shape [B, N, H, Q, K]. For triangle attention starting
    // nodes, B and N are adjacent in stride order and therefore can be
    // flattened with a `view`. For ending nodes, however, `B` and `N` are no
    // longer adjacent in stride order due to `mask` being transposed (see
    // test_alphafold3.py:test_triangle_attention).  `attn_bias` can't be
    // `flattenBatchDims`ed with a `view`. Therefore, `contiguous()` is
    // required.
    attn_bias = flattenBatchDims(attn_bias.contiguous());
  }

  // 4D SDPA
  auto [output, log_sumexp, philox_seed, philox_offset] = [&]() {
    if (query.is_meta()) {
      return scaled_dot_product_attention_meta(query, value);
    }

    if (attn_bias.defined()) {
      // This is a functional but suboptimal implementation for testing
      // triangle attention:
      // https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html.
      //
      // To accommodate 5D, we combine `bias` and `mask` into `attn_bias`, a
      // fully-allocated 5D tensor even though `bias` and `mask` have degenerate
      // dimensions. Then, we flatten all inputs to 4D before running
      // `at::_scaled_dot_product_attention_math`.
      //
      // This is suboptimal because:
      // 1. Broadcasting `bias` and `mask` outside the kernel wastes GPU memory.
      // 2. `at::_scaled_dot_product_attention_math` isn't fused. I
      // didn't use `at::_scaled_dot_product_efficient_attention` because it
      // comes with various sizes and strides constraints that make testing
      // hard.
      NVF_CHECK_EQ(
          dropout_p,
          0.0,
          "at::_scaled_dot_product_attention_math does not output rng_state "
          "and rng_offset for dropout backprop. So we only use it when "
          "dropout_p is 0.0.");
      auto philox_seed = at::empty({2}, query.options().dtype(at::kUInt64));
      auto philox_offset = at::empty({}, query.options().dtype(at::kUInt64));
      auto [out, log_sumexp] = at::_scaled_dot_product_attention_math(
          query,
          key,
          value,
          attn_bias,
          dropout_p,
          is_causal,
          /*dropout_mask=*/std::nullopt,
          scale);

      // at::_scaled_dot_product_attention_math produces a contiguous attention
      // output, but SdpaFwdOp requires the attention output to be in the same
      // layout as the query input:
      // https://github.com/NVIDIA/Fuser/blob/fe23484180f47f8ac27a3527fdbcef2ff1be2a66/csrc/preseg_passes/allocation_order_inference.cpp#L361-L362.
      // Therefore, we relayout the attention output according to attn_out()'s
      // allocation domain.
      NVF_ERROR(out.is_contiguous());
      const std::optional<Layout> out_layout = canonicalizeLayout(attn_out());
      NVF_CHECK(
          out_layout.has_value(),
          "Failed to canonicalize output layout of ",
          attn_out());
      const std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              attn_out()->getLogicalDomain(), out_layout->allocation_domain());
      NVF_ERROR(
          permutation.has_value(),
          "The allocation domain of a canonicalized layout of ",
          attn_out(),
          " is not a permutation of its logical domain.");
      out = unflattenBatchDim(out, batch_dims);
      out = out.permute(*permutation)
                .contiguous()
                .permute(ir_utils::inversePermutation(*permutation));
      out = flattenBatchDims(out);

      return std::make_tuple(out, log_sumexp, philox_seed, philox_offset);
    }

    NVF_ERROR(
        last_dim_size % 8 == 0,
        "Flash attention requires the last dimension to be a multiple of 8, "
        "but got: ",
        last_dim_size);

    auto
        [out,
         log_sumexp,
         cum_seq_q,
         cum_seq_k,
         query_seq_len,
         key_seq_len,
         philox_seed,
         philox_offset,
         debug_attn_mask] =
            at::_scaled_dot_product_flash_attention(
                query,
                key,
                value,
                dropout_p,
                is_causal,
                /*return_debug_mask=*/false,
                scale);

    return std::make_tuple(out, log_sumexp, philox_seed, philox_offset);
  }();

  if (batch_dims.size() > 1) {
    output = unflattenBatchDim(output, batch_dims);
    log_sumexp = unflattenBatchDim(log_sumexp, batch_dims);
  }

  // We ignore cum_seq_q/k outputs since they are undefined tensors for
  // non-nested tensors. We do not store query/key_seq_len since they can be
  // computed in non-nested tensor directly. debug_attn_mask is ignored
  // since `return_debug_mask=false`.
  return {output, log_sumexp, philox_seed, philox_offset};
}

SdpaBwdOp::SdpaBwdOp(
    IrBuilderPasskey passkey,
    TensorView* grad_query,
    TensorView* grad_key,
    TensorView* grad_value,
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
    Val* scale)
    : Expr(passkey) {
  addOutput(grad_query);
  addOutput(grad_key);
  addOutput(grad_value);
  addInput(grad_output);
  addInput(query);
  addInput(key);
  addInput(value);
  addInput(output);
  addInput(log_sumexp);
  addInput(dropout_p);
  addInput(is_causal);
  addInput(philox_seed);
  addInput(philox_offset);
  if (scale != nullptr) {
    addInput(scale);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SdpaBwdOp)

std::string SdpaBwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << grad_query()->toString() << ",\n";
  indent(ss, indent_size) << grad_key()->toString() << ",\n";
  indent(ss, indent_size) << grad_value()->toString() << "\n";
  indent(ss, indent_size + 1)
      << " = sdpa_bwd(" << grad_attn()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << query()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << key()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << value()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          " << attn_out()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          logsum_exp = " << logsumexp()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          dropout_p = " << dropout_p()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          is_causal = " << is_causal()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          philox_seed = " << philox_seed()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          philox_offset = " << philox_offset()->toString() << ",\n";
  if (scale() != nullptr) {
    indent(ss, indent_size + 1)
        << ",\n          scale = " << scale()->toInlineString();
  }
  indent(ss, indent_size + 1) << ")\n";
  return ss.str();
}

std::string SdpaBwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> SdpaBwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // Backward tensor inputs: grad_input, query, key, value, output,
  // logsumexp, max_q/k Temporary handling of DID parallelization. See
  // https://github.com/NVIDIA/Fuser/issues/2563
  auto query_domain =
      TensorDomain::noReductions(this->query()->getLogicalDomain());
  bool first_dim_is_did = query_domain.front()->isDeviceDim();
  auto out_grad = inputs[0].as<at::Tensor>();
  if (first_dim_is_did) {
    NVF_CHECK(out_grad.dim() == 5, "Expected 5D but found ", out_grad.sizes());
  } else {
    NVF_CHECK(out_grad.dim() == 4, "Expected 4D but found ", out_grad.sizes());
  }

  std::vector<at::Tensor> bwd_inputs;
  for (auto idx : arange(6)) {
    auto in_tensor = inputs.at(idx).as<at::Tensor>();
    // Removing the size 1 from sharded axis from tensors.
    if (first_dim_is_did) {
      in_tensor = in_tensor.squeeze(0);
    }
    bwd_inputs.push_back(in_tensor);
  }
  const auto dropout_p = inputs.at(6).as<double>();
  const auto is_causal = inputs.at(7).as<bool>();

  // Flash attention requires the last dimension to be padded to 8.
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L675-L677
  const auto last_dim_size = bwd_inputs[0].size(-1);
  NVF_ERROR(
      last_dim_size % 8 == 0,
      "Flash attention requires the last dimension to be a multiple of 8, but "
      "got: ",
      last_dim_size);
  // Conmpute scale using original size of last dimension
  double scale = inputs.size() > 10 ? inputs.back().as<double>()
                                    : 1.0 / std::sqrt(last_dim_size);

  // ATen reference:
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L680-L681
  // cum_seq_q/k are undefined tensors for non-nested input tensors.
  at::Tensor grad_query, grad_key, grad_value;
  if (bwd_inputs[0].is_meta()) {
    // Meta path: produce tensors with correct shapes/strides
    grad_query = at::empty_like(bwd_inputs[1]);
    grad_key = at::empty_like(bwd_inputs[2]);
    grad_value = at::empty_like(bwd_inputs[3]);
  } else {
    const auto philox_seed = inputs.at(8).as<at::Tensor>();
    const auto philox_offset = inputs.at(9).as<at::Tensor>();
    std::tie(grad_query, grad_key, grad_value) =
        at::_scaled_dot_product_flash_attention_backward(
            /*grad_output=*/bwd_inputs[0],
            /*query=*/bwd_inputs[1],
            /*key=*/bwd_inputs[2],
            /*value=*/bwd_inputs[3],
            /*output=*/bwd_inputs[4],
            /*logsumexp=*/bwd_inputs[5],
            /*cum_seq_q=*/at::Tensor(),
            /*cum_seq_k=*/at::Tensor(),
            // Note: ATen implementation expects max_q/max_k as scalars.
            /*max_q=*/bwd_inputs[1].size(2),
            /*max_k=*/bwd_inputs[2].size(2),
            /*dropout_p=*/dropout_p,
            /*is_causal=*/is_causal,
            /*philox_seed=*/philox_seed,
            /*philox_offset=*/philox_offset,
            /*scale=*/scale);
  }

  // Add device dimension back to outputs.
  if (first_dim_is_did) {
    grad_query = grad_query.unsqueeze(0);
    grad_key = grad_key.unsqueeze(0);
    grad_value = grad_value.unsqueeze(0);
  }

  return {grad_query, grad_key, grad_value};
}

EmbeddingFwdOp::EmbeddingFwdOp(
    IrBuilderPasskey passkey,
    TensorView* output,
    TensorView* input,
    TensorView* weight,
    Val* padding_idx,
    Val* max_norm,
    Val* norm_type,
    Val* scale_grad_by_freq,
    Val* sparse)
    : Expr(passkey) {
  addOutput(output);

  addInput(input);
  addInput(weight);
  addInput(norm_type);
  addInput(scale_grad_by_freq);
  addInput(sparse);
  if (padding_idx != nullptr) {
    addInput(padding_idx);
    addDataAttribute(true);
  } else {
    addDataAttribute(false);
  }
  if (max_norm != nullptr) {
    addInput(max_norm);
    addDataAttribute(true);
  } else {
    addDataAttribute(false);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EmbeddingFwdOp)

std::string EmbeddingFwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << ",\n";
  indent(ss, indent_size + 1) << " = embedding(" << in()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << weight()->toString() << ",\n";
  if (padding_idx() != nullptr) {
    indent(ss, indent_size + 1)
        << "          padding_idx = " << padding_idx()->toString() << ",\n";
  }
  if (max_norm() != nullptr) {
    indent(ss, indent_size + 1)
        << "          max_norm = " << max_norm()->toString() << ",\n";
  }
  indent(ss, indent_size + 1)
      << "          norm_type = " << norm_type()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          scale_grad_by_freq = "
      << scale_grad_by_freq()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          sparse = " << sparse()->toInlineString() << ")\n";
  return ss.str();
}

std::string EmbeddingFwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> EmbeddingFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto input = inputs.at(0).as<at::Tensor>();
  auto weight = inputs.at(1).as<at::Tensor>();
  auto norm_type = inputs.at(2).as<double>();
  auto scale_grad_by_freq = inputs.at(3).as<bool>();
  auto sparse = inputs.at(4).as<bool>();
  std::optional<int64_t> padding_idx = std::nullopt;
  if (has_padding_idx()) {
    padding_idx = inputs.at(5).as<int64_t>();
  }
  std::optional<double> max_norm = std::nullopt;
  if (has_max_norm()) {
    auto idx = 5 + has_padding_idx();
    max_norm = inputs.at(idx).as<double>();
  }

  // Meta-safe path: when either input or weight is a Meta tensor, avoid
  // calling into ATen embedding (which dispatches to index_select) and
  // instead synthesize the output shape and strides directly on Meta.
  if (input.is_meta() || weight.is_meta()) {
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(input.dim() + 1);
    for (int64_t d = 0; d < input.dim(); ++d) {
      out_sizes.push_back(input.size(d));
    }
    out_sizes.push_back(weight.size(1));
    auto out = at::empty(
        out_sizes, at::TensorOptions().device(at::kMeta).dtype(weight.dtype()));
    return {out};
  }

  namespace F = torch::nn::functional;
  return {F::embedding(
      input,
      weight,
      F::EmbeddingFuncOptions()
          .padding_idx(padding_idx)
          .max_norm(max_norm)
          .norm_type(norm_type)
          .scale_grad_by_freq(scale_grad_by_freq)
          .sparse(sparse))};
}

ArgsortOp::ArgsortOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    int64_t dim,
    bool descending,
    bool stable)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addDataAttribute(dim);
  addDataAttribute(descending);
  addDataAttribute(stable);
}

std::string ArgsortOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = argsort( "
                          << in()->toString() << ", dim = " << dim()
                          << ", descending = "
                          << (isDescending() ? "True" : "False")
                          << ", stable = " << (isStable() ? "True" : "False")
                          << " )\n";
  return ss.str();
}

std::string ArgsortOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ArgsortOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      inputs.size() == 1,
      "ArgsortOp expects 1 input but received ",
      inputs.size());

  const auto& in = inputs[0];
  NVF_ERROR(
      in.is<at::Tensor>(),
      "ArgsortOp expects tensor input but got ",
      in.type().name());

  // at::argsort signature is:
  // Tensor argsort(const Tensor &self, bool stable, int64_t dim, bool
  // descending)
  auto result =
      at::argsort(in.as<at::Tensor>(), isStable(), dim(), isDescending());

  return {result};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ArgsortOp)

TopKOp::TopKOp(
    IrBuilderPasskey passkey,
    Val* out_values,
    Val* out_indices,
    Val* in,
    Val* k,
    int64_t dim,
    bool largest,
    bool sorted)
    : Expr(passkey) {
  addOutput(out_values);
  addOutput(out_indices);
  addInput(in);
  addInput(k);
  addDataAttribute(dim);
  addDataAttribute(largest);
  addDataAttribute(sorted);
}

std::string TopKOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "( " << outValues()->toString() << ", "
                          << outIndices()->toString() << " ) = topk( "
                          << in()->toString() << ", " << k()->toString()
                          << ", dim = " << dim()
                          << ", largest = " << (isLargest() ? "True" : "False")
                          << ", sorted = " << (isSorted() ? "True" : "False")
                          << " )\n";
  return ss.str();
}

std::string TopKOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> TopKOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs[0];
  NVF_ERROR(
      in.is<at::Tensor>(),
      "TopKOp expects tensor input at position 0 but got ",
      in.type().name());

  const auto& k = inputs[1];
  NVF_ERROR(
      k.is<int64_t>(),
      "TopKOp expects int64_t input at position 1 as k but got ",
      k.type().name());

  // at::topk signature is:
  // std::tuple<Tensor, Tensor> topk(const Tensor &self, int64_t k, int64_t dim,
  // bool largest, bool sorted)
  auto result = at::topk(
      in.as<at::Tensor>(), k.as<int64_t>(), dim(), isLargest(), isSorted());

  // at::topk returns a tuple of (values, indices)
  return {std::get<0>(result), std::get<1>(result)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TopKOp)

GroupedMmaOp::GroupedMmaOp(
    IrBuilderPasskey passkey,
    Val* out_mat,
    Val* out_scale,
    Val* out_gamma,
    Val* mat1,
    Val* mat2,
    Val* offsets,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* bias,
    Val* beta)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  NVF_ERROR(offsets->isA<TensorView>(), "Offsets must be a TensorView");
  addOutput(out_mat);
  if (out_scale != nullptr) {
    NVF_ERROR(
        out_scale->isA<TensorView>(), "Output scale must be a TensorView");
    addOutput(out_scale);
  }
  if (out_gamma != nullptr) {
    NVF_ERROR(out_scale != nullptr, "Output gamma requires output scale");
    NVF_ERROR(
        out_gamma->isA<TensorView>(), "Output gamma must be a TensorView");
    addOutput(out_gamma);
  }
  addInput(mat1);
  addInput(mat2);
  addInput(offsets);

  int64_t offset = 3;
  int64_t scale_offset = -1;
  int64_t alpha_offset = -1;
  int64_t bias_offset = -1;
  int64_t beta_offset = -1;

  bool has_scale1 = scale1 != nullptr;
  if (has_scale1) {
    NVF_CHECK(
        scale1->isA<TensorView>(),
        "`scale1` must be a TensorView, but got: ",
        scale1);
    NVF_CHECK(scale2->isA<TensorView>(), "Scale2 must be a TensorView");
    addInput(scale1);
    addInput(scale2);
    scale_offset = offset;
    offset += 2;
  }

  bool has_alpha = alpha != nullptr;
  if (has_alpha) {
    NVF_CHECK(
        alpha->isA<TensorView>(),
        "`alpha` must be a TensorView, but got: ",
        alpha);
    addInput(alpha);
    alpha_offset = offset++;
  }

  bool has_bias = bias != nullptr;
  if (has_bias) {
    NVF_CHECK(
        bias->isA<TensorView>(),
        "`bias` must be a TensorView, but got: ",
        bias);
    addInput(bias);
    bias_offset = offset++;
  }

  bool has_beta = beta != nullptr;
  if (has_beta) {
    NVF_CHECK(
        beta->isA<TensorView>(),
        "`beta` must be a TensorView, but got: ",
        beta);
    addInput(beta);
    beta_offset = offset++;
  }

  addDataAttribute(scale_offset);
  addDataAttribute(alpha_offset);
  addDataAttribute(bias_offset);
  addDataAttribute(beta_offset);
}

std::string GroupedMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  if (outScale() != nullptr) {
    ss << ", " << outScale();
  }
  if (outGamma() != nullptr) {
    ss << ", " << outGamma();
  }
  ss << " = GroupedMmaOp(" << "mat1=" << matrix1() << ", "
     << "mat2=" << matrix2() << ", " << "offsets=" << offsets();
  if (hasScale()) {
    ss << ", " << "scale1=" << scale1() << ", " << "scale2=" << scale2();
  }
  if (hasAlpha()) {
    ss << ", " << "alpha=" << alpha();
  }
  if (hasBias()) {
    ss << ", " << "bias=" << bias();
  }
  if (hasBeta()) {
    ss << ", " << "beta=" << beta();
  }
  ss << ")\n";
  return ss.str();
}

std::string GroupedMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> GroupedMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // Meta-device fast path outside of torch version guard
  if (inputs.size() >= 3 && inputs[0].is<at::Tensor>() &&
      inputs[1].is<at::Tensor>() && inputs[2].is<at::Tensor>()) {
    const auto& mat1_meta = inputs[0].as<at::Tensor>();
    const auto& mat2_meta = inputs[1].as<at::Tensor>();
    const auto& offsets_meta = inputs[2].as<at::Tensor>();
    if (mat1_meta.is_meta() || mat2_meta.is_meta() || offsets_meta.is_meta()) {
      const int64_t num_groups = offsets_meta.numel();
      std::vector<int64_t> result_sizes;
      if (mat1_meta.dim() == 2 && mat2_meta.dim() == 2) {
        result_sizes = {num_groups, mat1_meta.size(0), mat2_meta.size(-1)};
      } else if (mat1_meta.dim() == 3 && mat2_meta.dim() == 2) {
        result_sizes = {mat1_meta.size(1), mat2_meta.size(-1)};
      } else if (mat1_meta.dim() == 2 && mat2_meta.dim() == 3) {
        result_sizes = {mat1_meta.size(0), mat2_meta.size(-1)};
      } else {
        NVF_THROW(
            "Expect ranks to be <2, 2>, <3, 2> or <2, 3>. Got: mat1 = ",
            mat1_meta.sizes(),
            " and mat2 = ",
            mat2_meta.sizes());
      }

      auto options = mat1_meta.options()
                         .device(c10::Device(c10::kMeta))
                         .dtype(data_type_to_aten(out()->dtype()));
      at::Tensor result = at::empty(result_sizes, options);

      if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
          rfactor_did_idx != -1) {
        result = result.unsqueeze(rfactor_did_idx);
      }

      return {result};
    }
  }

  NVF_ERROR(
      inputs[0].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 0 but got ",
      inputs[0].type().name());

  NVF_ERROR(
      inputs[1].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 1 but got ",
      inputs[1].type().name());

  NVF_ERROR(
      inputs[2].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 2 but got ",
      inputs[2].type().name());

  auto mat1 = inputs[0].as<at::Tensor>();
  auto mat2 = inputs[1].as<at::Tensor>();
  auto offsets = inputs[2].as<at::Tensor>();

  at::Tensor alpha;
  at::Tensor bias;
  at::Tensor beta;
  if (hasAlpha()) {
    int alpha_offset = alphaOffset();
    NVF_ERROR(
        inputs[alpha_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor alpha at position ",
        alpha_offset,
        " but got ",
        inputs[alpha_offset].type().name());
    alpha = inputs[alpha_offset].as<at::Tensor>();
  }
  if (hasBias()) {
    int bias_offset = biasOffset();
    NVF_ERROR(
        inputs[bias_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor bias at position ",
        bias_offset,
        " but got ",
        inputs[bias_offset].type().name());
    bias = inputs[bias_offset].as<at::Tensor>();
  }
  if (hasBeta()) {
    int beta_offset = betaOffset();
    NVF_ERROR(
        inputs[beta_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor beta at position ",
        beta_offset,
        " but got ",
        inputs[beta_offset].type().name());
    beta = inputs[beta_offset].as<at::Tensor>();
  }

  // This lambda returns the raw result, which will be postprocessed by the
  // caller, e.g., converting the data type and adding rFactor dimensions.
  auto result = [&]() -> at::Tensor {
    if (hasScale()) {
#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
      NVF_ERROR(
          inputs[3].is<at::Tensor>(),
          "GroupedMmaOp expects tensor input at position 3 but got ",
          inputs[3].type().name());
      NVF_ERROR(
          inputs[4].is<at::Tensor>(),
          "GroupedMmaOp expects tensor input at position 4 but got ",
          inputs[4].type().name());

      auto scale1 = inputs[scale1Offset()].as<at::Tensor>();
      auto scale2 = inputs[scale2Offset()].as<at::Tensor>();
      // Note: at::_scaled_grouped_mm requires k dimension to be the fastest on
      // both input matrices.
      auto mat1_k_last = mat1.contiguous();
      auto mat2_k_last = mat2.transpose(-1, -2).contiguous().transpose(-1, -2);

      // at::_scaled_grouped_mm limitation
      NVF_CHECK(
          scale1.size(-1) == 1 && scale2.size(-2) == 1,
          "Scale1 and scale2 must have size 1 at the k dimension. Got ",
          scale1.sizes(),
          " and ",
          scale2.sizes());
      // scale factor handling
      // see NOTE -- [ Grouped Matrix Multiplication semantics ]
      if (TensorDomain::noReductions(out()->getLogicalDomain()).size() == 3) {
        // case 1, aten API expects collapsed 1D scale with group dimension on
        // the slower side.
        scale1 = scale1.reshape(-1);
        scale2 = scale2.reshape(-1);
      } else {
        // case 2 and 3, aten doesn't allow broadcast on k dimension. squeeze k
        // out.
        scale1 = scale1.squeeze(-1);
        scale2 = scale2.squeeze(-2);
      }
      // undefined alpha, bias is not supported by aten API
      NVF_ERROR(!alpha.defined(), "alpha is not supported yet");
      NVF_ERROR(!beta.defined(), "beta is not supported yet");
      NVF_ERROR(!bias.defined(), "bias is not supported yet");
      // NOTE: at::_scaled_grouped_mm only supports bfloat16 as output at this
      // moment, otherwise we should have requested the output dtype directly
      // instead of casting the output afterwards.
      return at::_scaled_grouped_mm(
          mat1_k_last,
          mat2_k_last,
          scale1,
          scale2,
          offsets,
          /*bias=*/std::nullopt,
          /*alpha=*/std::nullopt,
          at::ScalarType::BFloat16);
#else
      NVF_THROW("at::_scaled_grouped_mm does not exist prior to PyTorch 2.8.");
#endif
    }

#if NVFUSER_CUTLASS_KERNEL_ENABLED
    const bool supported_by_cutlass_kernel =
        at::cuda::getCurrentDeviceProperties()->major == 10 &&
        mat1.dim() == 2 && mat2.dim() == 3 && mat1.is_contiguous() &&
        mat2.transpose(-1, -2).is_contiguous();
    if (supported_by_cutlass_kernel) {
      mat2 = mat2.transpose(-1, -2);
      NVF_ERROR(mat2.is_contiguous());
      // [m, k] x [g, n, k]; both contiguous
      const auto k = mat1.size(-1);
      const auto n = mat2.size(1);
      at::Tensor group_sizes = at::diff(
          offsets,
          /*n=*/1,
          /*dim=*/-1,
          /*prepend=*/
          at::zeros({1}, at::dtype(at::kInt).device(offsets.device())));
      at::Tensor ab_strides = at::full_like(offsets, k, at::dtype(at::kLong));
      at::Tensor c_strides = at::full_like(offsets, n, at::dtype(at::kLong));
      at::Tensor problem_sizes =
          at::stack({group_sizes, c_strides, ab_strides}, /*dim=*/-1)
              .to(at::kInt);
      offsets = at::cat(
          {at::zeros({1}, at::dtype(at::kInt).device(offsets.device())),
           offsets.slice(0, 0, -1)});
      return cutlass_kernels::grouped_mm(
          mat1, mat2, ab_strides, c_strides, problem_sizes, offsets);
    }
#endif

    // undefined bias is not supported by aten API
    NVF_ERROR(!alpha.defined(), "alpha is not supported yet");
    NVF_ERROR(!beta.defined(), "beta is not supported yet");
    NVF_ERROR(!bias.defined(), "bias is not supported yet");

    // Compute numbers of tokens per group from offsets.
    NVF_CHECK_EQ(
        c10::cuda::currentStreamCaptureStatusMayInitCtx(),
        c10::cuda::CaptureStatus::None,
        "GroupedMmaOp's fallback implementation below doesn't support CUDA "
        "graph capturing. The shapes of individual matmuls depend on "
        "`offsets`, which is data dependent.");
    at::Tensor offsets_cpu = offsets.cpu();
    NVF_ERROR_EQ(offsets_cpu.dtype(), at::kInt);
    const int* data_ptr = offsets_cpu.data_ptr<int>();
    const int64_t num_groups = offsets_cpu.numel();
    std::vector<int64_t> group_sizes(data_ptr, data_ptr + num_groups);
    for (int64_t i : arange(1, num_groups) | std::views::reverse) {
      group_sizes[i] -= group_sizes[i - 1];
    }

    std::vector<at::Tensor> group_mat1s;
    std::vector<at::Tensor> group_mat2s;
    at::Tensor result;
    std::vector<at::Tensor> group_outs;
    if (mat1.dim() == 2 && mat2.dim() == 2) {
      // [m, k] @ [k, n] => [g, m, n]
      group_mat1s = mat1.split(group_sizes, -1);
      group_mat2s = mat2.split(group_sizes, 0);
      result =
          at::empty({num_groups, mat1.size(0), mat2.size(-1)}, mat1.options());
      group_outs = result.unbind();
    } else if (mat1.dim() == 3 && mat2.dim() == 2) {
      // [g, m, k] @ [k, n] => [m, n]
      group_mat1s = mat1.unbind();
      group_mat2s = mat2.split(group_sizes, -1);
      result = at::empty({mat1.size(1), mat2.size(-1)}, mat1.options());
      group_outs = result.split(group_sizes, -1);
    } else if (mat1.dim() == 2 && mat2.dim() == 3) {
      // [m, k] @ [g, k, n] => [m, n]
      group_mat1s = mat1.split(group_sizes, 0);
      group_mat2s = mat2.unbind();
      result = at::empty({mat1.size(0), mat2.size(-1)}, mat1.options());
      group_outs = result.split(group_sizes, 0);
    } else {
      NVF_THROW(
          "Expect ranks to be <2, 2>, <3, 2> or <2, 3>. Got: mat1 = ",
          mat1.sizes(),
          " and mat2 = ",
          mat2.sizes());
    }

    for (auto [group_mat1, group_mat2, group_out] :
         zip(group_mat1s, group_mat2s, group_outs)) {
      at::matmul_out(group_out, group_mat1, group_mat2);
    }
    return result;
  }();

  // Post-processing
  result = result.to(data_type_to_aten(out()->dtype()));

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }

  return {result};
}

IterDomain* GroupedMmaOp::getKDimOfMatrix1() const {
  // mat1 is [g, m, k] or [m, k]
  const auto& logical_domain =
      TensorDomain::noReductions(matrix1()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

IterDomain* GroupedMmaOp::getKDimOfMatrix2() const {
  // mat2 is [g, k, n] or [k, n]
  const auto& logical_domain =
      TensorDomain::noReductions(matrix2()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

namespace {
IterDomain* returnFirstIfRankThree(const TensorView* tv) {
  const auto& logical_domain =
      TensorDomain::noReductions(tv->getLogicalDomain());
  if (logical_domain.size() == 3) {
    return logical_domain.at(0);
  } else {
    return nullptr;
  }
}
} // namespace

IterDomain* GroupedMmaOp::getGroupDimOfMatrix1() const {
  // matrix1 is [g, m, k] or [m, k]
  return returnFirstIfRankThree(matrix1());
}

IterDomain* GroupedMmaOp::getGroupDimOfMatrix2() const {
  // matrix2 is [g, k, n] or [k, n]
  return returnFirstIfRankThree(matrix2());
}

IterDomain* GroupedMmaOp::getGroupDimOfOutput() const {
  // output is [g, m, n] or [m, n]
  return returnFirstIfRankThree(out());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedMmaOp)

ScaledMmaOp::ScaledMmaOp(
    IrBuilderPasskey passkey,
    Val* out_mat,
    Val* out_scale,
    Val* out_gamma,
    Val* mat1,
    Val* mat2,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* bias,
    Val* beta)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  addOutput(out_mat);
  if (out_scale != nullptr) {
    NVF_ERROR(
        out_scale->isA<TensorView>(), "Output scale must be a TensorView");
    addOutput(out_scale);
  }
  if (out_gamma != nullptr) {
    NVF_ERROR(out_scale != nullptr, "Output gamma requires output scale");
    NVF_ERROR(
        out_gamma->isA<TensorView>(), "Output gamma must be a TensorView");
    addOutput(out_gamma);
  }
  addInput(mat1);
  addInput(mat2);
  NVF_ERROR(scale1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(scale2->isA<TensorView>(), "Second input must be a TensorView");
  addInput(scale1);
  addInput(scale2);

  int64_t offset = 4;
  int64_t alpha_offset = -1;
  int64_t bias_offset = -1;
  int64_t beta_offset = -1;

  bool has_alpha = alpha != nullptr;
  if (has_alpha) {
    NVF_CHECK(
        alpha->isA<TensorView>(),
        "`alpha` must be a TensorView, but got: ",
        alpha);
    addInput(alpha);
    alpha_offset = offset++;
  }

  bool has_bias = bias != nullptr;
  if (has_bias) {
    NVF_CHECK(
        bias->isA<TensorView>(),
        "`bias` must be a TensorView, but got: ",
        bias);
    addInput(bias);
    bias_offset = offset++;
  }

  bool has_beta = beta != nullptr;
  if (has_beta) {
    NVF_CHECK(
        beta->isA<TensorView>(),
        "`beta` must be a TensorView, but got: ",
        beta);
    addInput(beta);
    beta_offset = offset++;
  }

  // Store the offsets as attributes
  addDataAttribute(alpha_offset);
  addDataAttribute(bias_offset);
  addDataAttribute(beta_offset);
}

std::string ScaledMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  if (outScale() != nullptr) {
    ss << ", " << outScale();
  }
  if (outGamma() != nullptr) {
    ss << ", " << outGamma();
  }
  ss << " = ScaledMmaOp(";
  ss << "mat1=" << matrix1() << ", ";
  ss << "mat2=" << matrix2() << ", ";
  ss << "scale1=" << scale1() << ", ";
  ss << "scale2=" << scale2() << "";
  if (hasAlpha()) {
    ss << ", alpha=" << alpha();
  }
  if (hasBias()) {
    ss << ", bias=" << bias();
  }
  if (hasBeta()) {
    ss << ", beta=" << beta();
  }
  ss << ")\n";
  return ss.str();
}

std::string ScaledMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> ScaledMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  [[maybe_unused]] const auto& mat1 = inputs[0].as<at::Tensor>();
  [[maybe_unused]] const auto& mat2 = inputs[1].as<at::Tensor>();
  [[maybe_unused]] const auto& scale1 = inputs[2].as<at::Tensor>();
  [[maybe_unused]] const auto& scale2 = inputs[3].as<at::Tensor>();

  at::Tensor alpha;
  at::Tensor bias;
  at::Tensor beta;
  if (hasAlpha()) {
    int alpha_offset = alphaOffset();
    NVF_ERROR(
        inputs[alpha_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor alpha at position ",
        alpha_offset,
        " but got ",
        inputs[alpha_offset].type().name());
    alpha = inputs[alpha_offset].as<at::Tensor>();
  }
  if (hasBias()) {
    int bias_offset = biasOffset();
    NVF_ERROR(
        inputs[bias_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor bias at position ",
        bias_offset,
        " but got ",
        inputs[bias_offset].type().name());
    bias = inputs[bias_offset].as<at::Tensor>();
  }
  if (hasBeta()) {
    int beta_offset = betaOffset();
    NVF_ERROR(
        inputs[beta_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor beta at position ",
        beta_offset,
        " but got ",
        inputs[beta_offset].type().name());
    beta = inputs[beta_offset].as<at::Tensor>();
  }

  at::Tensor result;
#if NVFUSER_CUTLASS_KERNEL_ENABLED
  {
    at::ScalarType out_scalar_type = data_type_to_aten(out()->dtype());
    at::Tensor mat1_view = mat1;
    // nvfp4_scaled_mm expected layout
    at::Tensor mat2_view = mat2.t();

    DataType in_dtype = matrix1()->dtype();

    // NOTE: cutlass nvfp4 kernel doesn't support bias, beta or quantized output
    if (!bias.defined() && !beta.defined() && outputs().size() == 1) {
      bool cutlass_can_run = true;
      // NOTE: this felt ugly. I should go fix up the validate input
      try {
        cutlass_kernels::validateInputsNvfp4ScaledMm(
            mat1_view, mat2_view, scale1, scale2, alpha);
      } catch (...) {
        cutlass_can_run = false;
      }

      if (cutlass_can_run) {
        result = cutlass_kernels::nvfp4_scaled_mm(
            mat1_view,
            mat2_view,
            scale1,
            scale2,
            alpha,
            out_scalar_type,
            /*skip_checks=*/true);
      }
    }
  }
#endif

#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
  if (!result.defined()) {
    // TODO: interface with scaled matrix multiplication cutlass kernel. For
    // now, we'll fallback with aten kernels at::_scaled_mm has implementation
    // limitations:
    NVF_CHECK(!beta.defined(), "beta in ScaledMmaOp is not supported yet");
    NVF_CHECK(
        outScale() == nullptr,
        "output block scaling factor in ScaledMmaOp is not supported yet");
    NVF_CHECK(
        outGamma() == nullptr,
        "output global scaling factor in ScaledMmaOp is not supported yet");
    result = at::_scaled_mm(
        mat1,
        mat2,
        scale1,
        scale2,
        bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
        alpha.defined() ? std::optional<at::Tensor>(alpha) : std::nullopt,
        data_type_to_aten(out()->dtype()));
  }
#endif

  NVF_CHECK(result.defined(), "Couldn't find fallback kernel for scaled_mm");

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }

  return {result};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScaledMmaOp)

ScanOp::ScanOp(
    IrBuilderPasskey passkey,
    BinaryOpType op_type,
    Val* init,
    Val* out,
    Val* in,
    int64_t dim)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(init);
  addDataAttribute(op_type);
  addDataAttribute(dim);
}

std::string ScanOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString();
  ss << "\n";
  indent(ss, indent_size + 1) << " = scan(" << in()->toString() << ",\n";
  indent(ss, indent_size + 1) << "        dim=" << dim() << ",\n";
  indent(ss, indent_size + 1) << "        op_type=" << opType() << ",\n";
  indent(ss, indent_size + 1)
      << "        init=" << init()->toInlineString() << ")\n";
  return ss.str();
}

std::string ScanOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ScanOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto input = inputs.at(0).as<at::Tensor>();

  NVF_ERROR_EQ(inputs.size(), 1);

  // Meta-safe path: when input is a Meta tensor, avoid invoking ATen ops that
  // may not have Meta kernels (e.g., cummin). Instead, synthesize an output
  // tensor on Meta with the correct shape/strides and dtype.
  if (input.is_meta()) {
    const at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
    auto out_meta = at::empty(
        input.sizes(), at::TensorOptions().device(at::kMeta).dtype(out_dtype));
    return {out_meta};
  }

  at::Tensor out_t;
  switch (opType()) {
    case BinaryOpType::Add:
      out_t = at::cumsum(input, dim());
      break;
    case BinaryOpType::FMax:
    case BinaryOpType::Max:
      out_t = std::get<0>(at::cummax(input, dim()));
      break;
    case BinaryOpType::FMin:
    case BinaryOpType::Min:
      out_t = std::get<0>(at::cummin(input, dim()));
      break;
    case BinaryOpType::Mul:
      out_t = at::cumprod(input, dim());
      break;
    default:
      NVF_THROW("Unhandled opType() ", opType());
  }

  at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
  if (out_t.dtype() != out_dtype) {
    out_t = out_t.to(out_dtype);
  }

  return {out_t};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScanOp)

CutlassNvfp4GroupedMmaOp::CutlassNvfp4GroupedMmaOp(
    IrBuilderPasskey passkey,
    Val* out_mat,
    Val* mat1,
    Val* mat2,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* problem_sizes,
    Val* expert_offsets,
    Val* sf_offsets)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  NVF_ERROR(scale1->isA<TensorView>(), "Scale1 must be a TensorView");
  NVF_ERROR(scale2->isA<TensorView>(), "Scale2 must be a TensorView");
  NVF_ERROR(alpha->isA<TensorView>(), "Alpha must be a TensorView");
  NVF_ERROR(
      problem_sizes->isA<TensorView>(), "Problem sizes must be a TensorView");
  NVF_ERROR(
      expert_offsets->isA<TensorView>(), "Expert offsets must be a TensorView");
  NVF_ERROR(sf_offsets->isA<TensorView>(), "SF offsets must be a TensorView");

  addOutput(out_mat);
  addInput(mat1);
  addInput(mat2);
  addInput(scale1);
  addInput(scale2);
  addInput(alpha);
  addInput(problem_sizes);
  addInput(expert_offsets);
  addInput(sf_offsets);
}

std::string CutlassNvfp4GroupedMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  ss << " = CutlassNvfp4GroupedMmaOp(";
  ss << "mat1=" << matrix1() << ", ";
  ss << "mat2=" << matrix2() << ", ";
  ss << "scale1=" << scale1() << ", ";
  ss << "scale2=" << scale2() << ", ";
  ss << "alpha=" << alpha() << ", ";
  ss << "problem_sizes=" << problemSizes() << ", ";
  ss << "expert_offsets=" << expertOffsets() << ", ";
  ss << "sf_offsets=" << scalingFactorOffsets() << ")\n";
  return ss.str();
}

std::string CutlassNvfp4GroupedMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> CutlassNvfp4GroupedMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& mat1 = inputs[0].as<at::Tensor>();
  const auto& mat2 = inputs[1].as<at::Tensor>();
  const auto& scale1 = inputs[2].as<at::Tensor>();
  const auto& scale2 = inputs[3].as<at::Tensor>();
  const auto& alpha = inputs[4].as<at::Tensor>();
  const auto& problem_sizes = inputs[5].as<at::Tensor>();
  const auto& expert_offsets = inputs[6].as<at::Tensor>();
  const auto& sf_offsets = inputs[7].as<at::Tensor>();

  // Meta-device fast path outside of torch version guard
  if (mat1.is_meta() || mat2.is_meta() || scale1.is_meta() ||
      scale2.is_meta() || alpha.is_meta() || problem_sizes.is_meta() ||
      expert_offsets.is_meta() || sf_offsets.is_meta()) {
    // For nvfp4_scaled_grouped_mm, the output shape is [M, N]
    // where M = mat1.size(0) and N = mat2.size(2).
    // Note: CutlassNvfp4GroupedMmaOp expects mat2 to be [G, K/2, N] (packed) at
    // runtime and transposes it before calling into CUTLASS.
    std::vector<int64_t> result_sizes = {mat1.size(0), mat2.size(2)};

    at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
    auto options =
        mat1.options().device(c10::Device(c10::kMeta)).dtype(out_dtype);
    at::Tensor result = at::empty(result_sizes, options);

    if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
        rfactor_did_idx != -1) {
      result = result.unsqueeze(rfactor_did_idx);
    }

    return {result};
  }

#if NVFUSER_CUTLASS_KERNEL_ENABLED
  NVF_CHECK(
      mat1.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
      mat2.scalar_type() == at::ScalarType::Float4_e2m1fn_x2);

  // Validate problem_sizes tensor
  NVF_CHECK(problem_sizes.dim() == 2, "problem_sizes must be a 2D tensor");
  NVF_CHECK(
      problem_sizes.size(1) == 3,
      "problem_sizes must have shape (num_experts, 3)");
  int num_experts = problem_sizes.size(0);

  at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
  const auto options =
      at::TensorOptions().device(mat1.device()).dtype(out_dtype);

  // Calculate proper stride tensors for the cutlass kernel
  // ab_strides: stride information for input matrices A and B
  // c_strides: stride information for output matrix C
  // Note: mat1 is packed fp4x2.
  int k = mat1.size(1) * 2;
  int n = mat2.size(2);
  auto ab_strides =
      at::empty({num_experts}, options.dtype(at::ScalarType::Long));
  auto c_strides =
      at::empty({num_experts}, options.dtype(at::ScalarType::Long));
  // FIXME: this could be done outside and provided as input to avoid two kernel
  // launches.
  ab_strides.fill_(k);
  c_strides.fill_(n);

  // Call the cutlass kernel, note that it expect g,n,k layout on mat2.
  at::Tensor result = cutlass_kernels::nvfp4_scaled_grouped_mm(
      mat1.view(at::ScalarType::Float4_e2m1fn_x2),
      mat2.transpose(-1, -2).view(at::ScalarType::Float4_e2m1fn_x2),
      scale1,
      scale2,
      alpha,
      ab_strides,
      c_strides,
      problem_sizes,
      expert_offsets,
      sf_offsets,
      out_dtype);

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }
  return {result};
#else
  NVF_THROW("CutlassNvfp4GroupedMmaOp requires CUTLASS kernels to be enabled");
#endif
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CutlassNvfp4GroupedMmaOp)

PreprocessGroupedMatmulInputSf::PreprocessGroupedMatmulInputSf(
    IrBuilderPasskey passkey,
    Val* output,
    Val* input,
    Val* input_offsets,
    Val* output_offsets,
    BlockScalingFactorLayout layout,
    Val* k,
    Val* g,
    Val* row_idx,
    Val* col_idx)
    : Expr(passkey) {
  addInput(input);
  addInput(input_offsets);
  addInput(output_offsets);
  addInput(k);
  addInput(g);
  addOutput(output);
  addDataAttribute(layout);
  if (row_idx != nullptr) {
    addAttribute(row_idx);
  }
  if (col_idx != nullptr) {
    addAttribute(col_idx);
  }
}

std::string PreprocessGroupedMatmulInputSf::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = preprocessGroupedMatmulInputSf(\n";
  indent_size++;
  indent(ss, indent_size) << "input = " << in()->toString() << ",\n";
  indent(ss, indent_size) << "input_offsets = " << inputOffsets()->toString()
                          << ",\n";
  indent(ss, indent_size) << "output_offsets = " << outputOffsets()->toString()
                          << ",\n";
  indent(ss, indent_size) << "layout = "
                          << (layout() == BlockScalingFactorLayout::Block128x4
                                  ? "Block128x4"
                                  : "Unknown")
                          << "\n";
  indent_size--;
  indent(ss, indent_size) << ")\n";
  return ss.str();
}

std::string PreprocessGroupedMatmulInputSf::toInlineString(
    int indent_size) const {
  NVF_CHECK(false, "PreprocessGroupedMatmulInputSf can not be printed inline");
}

std::vector<PolymorphicValue> PreprocessGroupedMatmulInputSf::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("PreprocessGroupedMatmulInputSf evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PreprocessGroupedMatmulInputSf)

// Details:
// Currently output_scales is the first input in the constructor even though
// it's the second output. This is because if it's the second output then we hit
// a bug in indexing. The stack trace can be seen here:
// https://gist.github.com/protonu/dc35024c1291625b2b7ce87baa39e2ae
// This happens when creating UnswitchPredicate, probably in the call to
// TensorIndexer::getPredicates. The incorrect predicate_domains for the tv
// in the call to getPredicateDomains.
BlockQuantizationOp::BlockQuantizationOp(
    IrBuilderPasskey passkey,
    Val* output_scales,
    Val* output,
    Val* input,
    Val* logical_index,
    Val* global_scale,
    int64_t block_size,
    bool swizzled_scales)
    : Expr(passkey) {
  addOutput(output);
  addOutput(output_scales);
  addInput(input);
  if (global_scale) {
    addInput(global_scale);
  }
  addAttribute(logical_index);
  addDataAttribute(block_size);
  addDataAttribute(swizzled_scales);
}

std::string BlockQuantizationOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "(" << blockScales()->toString() << ",\n "
                          << quantizedOutput()->toString() << ")\n"
                          << " = block_quantize(" << in()->toString() << ")\n";
  return ss.str();
}

std::string BlockQuantizationOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "BlockQuantizationOp can not be printed inline");
}

std::vector<PolymorphicValue> BlockQuantizationOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("BlockQuantizationOp evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockQuantizationOp)

} // namespace nvfuser
