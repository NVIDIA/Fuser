// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using Sizes = std::vector<int64_t>;
using SDPAParamType = std::tuple<std::optional<Sizes>, DataType>;
using SDPATest = NVFuserFixtureParamTest<SDPAParamType>;

constexpr int64_t n = 16, h = 32, l = 64, s = 128, e = 64;

// Note: Flash Attention is only supported on Ampere and above.

TEST(SDPATest, NonCausalAttnConcrete) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(k_shape, DataType::Half);
  auto tvv = makeConcreteTensor(k_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto tvattn = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);
  fusion->addOutput(tvattn.output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(k_shape, options);
  at::Tensor v = at::randn(v_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_outputs = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);
  ;

  FusionExecutor fe;
  fusion->aliasOutputToInput(
      fusion->outputs()[0], /*input=*/nullptr, AllocationType::Evaluate);
  fe.compileFusion(fusion.get(), {q, k, v});
  auto out = fe.runFusion({q, k, v});

  testValidate(
      fusion.get(),
      out,
      {q, k, v},
      {std::get<0>(aten_outputs)},
      __LINE__,
      __FILE__);
}

TEST(SDPATest, NonCausalAttnSymbolic) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});

  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(k_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto tvattn = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);
  fusion->addOutput(tvattn.output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(k_shape, options);
  at::Tensor v = at::randn(v_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_outputs = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);
  ;

  FusionExecutor fe;
  fusion->aliasOutputToInput(
      fusion->outputs()[0], /*input=*/nullptr, AllocationType::Evaluate);
  fe.compileFusion(fusion.get(), {q, k, v});
  auto out = fe.runFusion({q, k, v});

  testValidate(
      fusion.get(),
      out,
      {q, k, v},
      {std::get<0>(aten_outputs)},
      __LINE__,
      __FILE__);
}

TEST(SDPATest, CausalAttn) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});

  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(k_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto tvattn = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      /*scale=*/IrBuilder::create<Val>(1e-3));
  fusion->addOutput(tvattn.output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(k_shape, options);
  at::Tensor v = at::randn(v_shape, options);

  auto aten_outputs = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/true,
      /*return_debug_mask=*/false,
      /*scale=*/1e-3);
  ;

  FusionExecutor fe;
  fusion->aliasOutputToInput(
      fusion->outputs()[0], /*input=*/nullptr, AllocationType::Evaluate);
  fe.compileFusion(fusion.get(), {q, k, v});
  auto out = fe.runFusion({q, k, v});

  testValidate(
      fusion.get(),
      out,
      {q, k, v},
      {std::get<0>(aten_outputs)},
      __LINE__,
      __FILE__);
}
} // namespace nvfuser
