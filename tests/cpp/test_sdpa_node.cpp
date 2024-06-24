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
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/optimization_pass.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class SDPATest : public NVFuserTest {
 protected:
  SDPATest() : optimization_guard_(false), allocation_order_guard_(false) {}

 private:
  // Note: `MoveSplitCat` and `AllocationDomain` preseg passes use ID model.
  // `SdpaFwdOp` currently does not work with ID model since it requires all
  // sibling outputs to have the same root domain.
  //  This will be modified in a future PR.
  preseg_passes::OptimizationPassGuard<preseg_passes::MoveSplitCatPass>
      optimization_guard_;
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      allocation_order_guard_;
};

constexpr int64_t n = 16, h = 32, l = 64, s = 128, e = 64;

// Note: Flash Attention is only supported on Ampere and above.

TEST_F(SDPATest, NonCausalAttnConcrete) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(k_shape, DataType::Half);
  auto tvv = makeConcreteTensor(v_shape, DataType::Half);

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

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({q, k, v});
  EXPECT_TRUE(at::allclose(out[0], std::get<0>(aten_outputs)));
}

TEST_F(SDPATest, NonCausalAttnSymbolic) {
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

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({q, k, v});
  EXPECT_TRUE(at::allclose(out[0], std::get<0>(aten_outputs)));
}

TEST_F(SDPATest, CausalAttn) {
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

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({q, k, v});
  EXPECT_TRUE(at::allclose(out[0], std::get<0>(aten_outputs)));
}

TEST_F(SDPATest, PairwiseRootDomainMap) {
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

  // Verify mapping between Q,K,V and attention output
  std::vector<TensorView*> producer_tvs{tvq, tvk, tvv};
  for (auto role : {AttnRole::Q, AttnRole::K, AttnRole::V}) {
    auto producer_tv = producer_tvs[(int)role];
    auto pairwise_map = PairwiseRootDomainMap(producer_tv, tvattn.output)
                            .mapProducerToConsumer();

    auto mappingExists = [&pairwise_map](
                             IterDomain* p_id, IterDomain* c_id) -> bool {
      return pairwise_map.find(p_id) != pairwise_map.end() &&
          pairwise_map[p_id] == c_id;
    };

    // Mapping for N, H exists from Q/K/V to output.
    for (auto idx : c10::irange(2)) {
      EXPECT_TRUE(
          mappingExists(producer_tv->axis(idx), tvattn.output->axis(idx)));
    }
    // Mapping for L exists between Q and output.
    if (role == AttnRole::Q) {
      EXPECT_TRUE(mappingExists(producer_tv->axis(2), tvattn.output->axis(2)));
    } else {
      EXPECT_FALSE(mappingExists(producer_tv->axis(2), tvattn.output->axis(2)));
    }
    // Mapping for Ev exists between V and output.
    if (role == AttnRole::V) {
      EXPECT_TRUE(mappingExists(producer_tv->axis(3), tvattn.output->axis(3)));
    } else {
      EXPECT_FALSE(mappingExists(producer_tv->axis(3), tvattn.output->axis(3)));
    }
  }
}

TEST_F(SDPATest, NonCausalAttnSymbolicBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});
  std::vector<int64_t> attn_shape({n, h, l, e});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(k_shape, options);
  at::Tensor v = at::randn(v_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto
      [output,
       log_sumexp,
       cum_seq_q,
       cum_seq_k,
       query_seq_len,
       key_seq_len,
       philox_seed,
       philox_offset,
       debug_attn_mask] = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);

  auto tv_grad_output = makeSymbolicTensor(attn_shape, DataType::Half);
  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(k_shape, DataType::Half);
  auto tv_output = makeSymbolicTensor(attn_shape, DataType::Half);
  auto tv_logsumexp = makeSymbolicTensor({n, h, l}, DataType::Float);
  auto tv_cumq = makeSymbolicTensor(1, DataType::Null);
  auto tv_cumk = makeSymbolicTensor(1, DataType::Null);
  auto tv_seed = makeSymbolicTensor({}, DataType::Int);
  tv_seed->setCpuScalar(true);
  auto tv_offset = makeSymbolicTensor({}, DataType::Int);
  tv_offset->setCpuScalar(true);

  fusion->addInput(tv_grad_output);
  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);
  fusion->addInput(tv_output);
  fusion->addInput(tv_logsumexp);
  fusion->addInput(tv_cumq);
  fusion->addInput(tv_cumk);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      tv_cumq,
      tv_cumk,
      /*max_q=*/IrBuilder::create<Val>(*query_seq_len.maybe_as_int()),
      /*max_k=*/IrBuilder::create<Val>(*key_seq_len.maybe_as_int()),
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  fusion->addOutput(tvgrad.grad_query);
  fusion->addOutput(tvgrad.grad_key);
  fusion->addOutput(tvgrad.grad_value);

  at::Tensor grad_out = at::randn(attn_shape, options);

  std::vector<c10::IValue> sdpa_bwd_inputs = {grad_out, q, k, v, output, log_sumexp, cum_seq_q, cum_seq_k, philox_seed, philox_offset};
  FusionExecutor fe;
  std::for_each(
    fusion->outputs().begin(),
    fusion->outputs().end(),
    [&](Val* out){fusion->aliasOutputToInput(
      out, /*input=*/nullptr, AllocationType::Evaluate);});
  
  fe.compileFusion(fusion.get(), sdpa_bwd_inputs);
  auto out = fe.runFusion(sdpa_bwd_inputs);

}
} // namespace nvfuser
