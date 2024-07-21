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
#include <multidevice/device_mesh.h>
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

auto addSdpaFwdOutputs = [](Fusion* fusion, SdpfaFwdResult output) {
  fusion->addOutput(output.output);
  fusion->addOutput(output.log_sumexp);
  fusion->addOutput(output.query_seq_len);
  fusion->addOutput(output.key_seq_len);
  fusion->addOutput(output.philox_seed);
  fusion->addOutput(output.philox_offset);
  fusion->addOutput(output.debug_attn_mask);
};

using AtenSdpaOut = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>;
auto validateSdpaFwdOutputs = [](std::vector<at::Tensor> nvf_out,
                                 AtenSdpaOut aten_out) {
  auto
      [attn,
       log_sumexp,
       cum_seq_q,
       cum_seq_k,
       query_seq_len,
       key_seq_len,
       philox_seed,
       philox_offset,
       debug_attn_mask] = aten_out;
  // nvf_out = {attn, log_sumexp, philox_seed, philox_offset, debug_attn_mask} 
  // Since, dropout_p = 0.0 to validate outputs,
  // philox_seed and philox_offset are uninitialized empty tensors with garbage
  // values for this case, so we skip validating those values.
  EXPECT_TRUE(at::allclose(nvf_out[0], attn));
  EXPECT_TRUE(at::allclose(nvf_out[1], log_sumexp));
  EXPECT_EQ(nvf_out[2].item<int64_t>(), query_seq_len);
  EXPECT_EQ(nvf_out[3].item<int64_t>(), key_seq_len);
  EXPECT_TRUE(at::equal(nvf_out[6], debug_attn_mask));
};

TEST_F(SDPATest, NonCausalAttnConcrete) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(kv_shape, DataType::Half);
  auto tvv = makeConcreteTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto output = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);
  addSdpaFwdOutputs(fusion.get(), output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_out = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);

  FusionExecutorCache fec(std::move(fusion));
  auto nvf_out = fec.runFusionWithInputs({q, k, v});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}

TEST_F(SDPATest, NonCausalAttnSymbolic) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});

  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(kv_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto output = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);
  addSdpaFwdOutputs(fusion.get(), output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_out = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);

  FusionExecutorCache fec(std::move(fusion));
  auto nvf_out = fec.runFusionWithInputs({q, k, v});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}

TEST_F(SDPATest, CausalAttn) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});

  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(kv_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto output = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      /*scale=*/IrBuilder::create<Val>(1e-3));
  addSdpaFwdOutputs(fusion.get(), output);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  auto aten_out = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/true,
      /*return_debug_mask=*/false,
      /*scale=*/1e-3);

  FusionExecutorCache fec(std::move(fusion));
  auto nvf_out = fec.runFusionWithInputs({q, k, v});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}

TEST_F(SDPATest, PairwiseLogicalDomainMap) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});

  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(kv_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto output = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      /*scale=*/IrBuilder::create<Val>(1e-3));
  addSdpaFwdOutputs(fusion.get(), output);

  // Verify mapping between Q,K,V and all output
  // Producers:
  //   query = [N, H, L, E]
  //   key = [N, H, S, E]
  //   value = [N, H, S, Ev]
  // Consumers:
  //   output = [N, H, L, Ev]
  //   logsumexp = [N, H, L]
  std::vector<TensorView*> producer_tvs{tvq, tvk, tvv};
  for (auto role : {AttnRole::Q, AttnRole::K, AttnRole::V}) {
    auto producer_tv = producer_tvs[(int)role];

    for (Val* consumer : fusion->outputs()) {
      auto consumer_tv = consumer->as<TensorView>();
      auto pairwise_map = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
                              .mapProducerToConsumer();
      auto mappingExists = [&pairwise_map](
                               IterDomain* p_id, IterDomain* c_id) -> bool {
        return pairwise_map.find(p_id) != pairwise_map.end() &&
            pairwise_map[p_id] == c_id;
      };

      auto consumer_root = consumer_tv->getMaybeRootDomain();
      for (auto idx : c10::irange(consumer_tv->nDims())) {
        // Mapping for N, H exists from Q/K/V to any output.
        if (idx < 2) {
          EXPECT_TRUE(
              mappingExists(producer_tv->axis(idx), consumer_root.at(idx)));
        }
        // Mapping for L exists between Q and output, log_sumexp.
        if (idx == 2 && role == AttnRole::Q) {
          EXPECT_TRUE(mappingExists(producer_tv->axis(2), consumer_root.at(2)));
        }
        // Mapping for Ev exists between V and output.
        if (idx == 3 && role == AttnRole::V) {
          EXPECT_TRUE(mappingExists(producer_tv->axis(3), consumer_root.at(3)));
        }
      }
    }
  }
}

TEST_F(SDPATest, NonCausalAttnConcreteBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  at::manual_seed(0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});
  std::vector<int64_t> attn_shape({n, h, l, e});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  double dropout_p = 0.2;
  bool is_causal = false;
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
       debug_attn_mask] =
          at::_scaled_dot_product_flash_attention(
              q,
              k,
              v,
              /*dropout_p=*/dropout_p,
              /*is_causal=*/is_causal,
              /*return_debug_mask=*/false,
              scale);

  auto tv_grad_output = makeConcreteTensor(attn_shape, DataType::Half);
  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(kv_shape, DataType::Half);
  auto tvv = makeConcreteTensor(kv_shape, DataType::Half);
  auto tv_output = makeConcreteTensor(attn_shape, DataType::Half);
  auto tv_logsumexp = makeConcreteTensor({n, h, l}, DataType::Float);
  auto tv_maxq = makeConcreteTensor({}, DataType::Int);
  tv_maxq->setCpuScalar(true);
  auto tv_maxk = makeConcreteTensor({}, DataType::Int);
  tv_maxk->setCpuScalar(true);
  auto tv_seed = makeConcreteTensor({}, DataType::Int);
  tv_seed->setCpuScalar(true);
  auto tv_offset = makeConcreteTensor({}, DataType::Int);
  tv_offset->setCpuScalar(true);

  fusion->addInput(tv_grad_output);
  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);
  fusion->addInput(tv_output);
  fusion->addInput(tv_logsumexp);
  fusion->addInput(tv_maxq);
  fusion->addInput(tv_maxk);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      tv_maxq,
      tv_maxk,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  fusion->addOutput(tvgrad.grad_query);
  fusion->addOutput(tvgrad.grad_key);
  fusion->addOutput(tvgrad.grad_value);

  at::Tensor grad_out = at::randn(attn_shape, options);

  std::vector<c10::IValue> sdpa_bwd_inputs = {
      grad_out,
      q,
      k,
      v,
      output,
      log_sumexp,
      // max_q/k are represented as CPU scalar tensors in nvFuser and integers
      // in ATen.
      at::scalar_tensor(*query_seq_len.maybe_as_int(), at::dtype(at::kLong)),
      at::scalar_tensor(*key_seq_len.maybe_as_int(), at::dtype(at::kLong)),
      philox_seed,
      philox_offset};

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs(sdpa_bwd_inputs);

  auto [ref_grad_query, ref_grad_key, ref_grad_value] =
      at::_scaled_dot_product_flash_attention_backward(
          grad_out,
          q,
          k,
          v,
          output,
          log_sumexp,
          cum_seq_q,
          cum_seq_k,
          /*max_q=*/*query_seq_len.maybe_as_int(),
          /*max_k=*/*key_seq_len.maybe_as_int(),
          /*dropout_p=*/dropout_p,
          /*is_causal=*/is_causal,
          philox_seed,
          philox_offset,
          /*scale=*/scale);

  testValidate(
      fec.fusion(),
      out,
      sdpa_bwd_inputs,
      {ref_grad_query, ref_grad_key, ref_grad_value},
      __LINE__,
      __FILE__);
}

TEST_F(SDPATest, NonCausalAttnSymbolicBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  at::manual_seed(0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});
  std::vector<int64_t> attn_shape({n, h, l, e});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  double dropout_p = 0.2;
  bool is_causal = false;
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
       debug_attn_mask] =
          at::_scaled_dot_product_flash_attention(
              q,
              k,
              v,
              dropout_p,
              is_causal,
              /*return_debug_mask=*/false,
              scale);

  auto tv_grad_output = makeSymbolicTensor(attn_shape, DataType::Half);
  auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
  auto tvk = makeSymbolicTensor(kv_shape, DataType::Half);
  auto tvv = makeSymbolicTensor(kv_shape, DataType::Half);
  auto tv_output = makeSymbolicTensor(attn_shape, DataType::Half);
  auto tv_logsumexp = makeSymbolicTensor({n, h, l}, DataType::Float);
  auto tv_maxq = makeSymbolicTensor({}, DataType::Int);
  tv_maxq->setCpuScalar(true);
  auto tv_maxk = makeSymbolicTensor({}, DataType::Int);
  tv_maxk->setCpuScalar(true);
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
  fusion->addInput(tv_maxq);
  fusion->addInput(tv_maxk);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      tv_maxq,
      tv_maxk,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  fusion->addOutput(tvgrad.grad_query);
  fusion->addOutput(tvgrad.grad_key);
  fusion->addOutput(tvgrad.grad_value);

  at::Tensor grad_out = at::randn(attn_shape, options);

  std::vector<c10::IValue> sdpa_bwd_inputs = {
      grad_out,
      q,
      k,
      v,
      output,
      log_sumexp,
      // max_q/k are represented as CPU scalar tensors in nvFuser and integers
      // in ATen.
      at::scalar_tensor(*query_seq_len.maybe_as_int(), at::dtype(at::kLong)),
      at::scalar_tensor(*key_seq_len.maybe_as_int(), at::dtype(at::kLong)),
      philox_seed,
      philox_offset};

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs(sdpa_bwd_inputs);

  auto [ref_grad_query, ref_grad_key, ref_grad_value] =
      at::_scaled_dot_product_flash_attention_backward(
          grad_out,
          q,
          k,
          v,
          output,
          log_sumexp,
          cum_seq_q,
          cum_seq_k,
          /*max_q=*/*query_seq_len.maybe_as_int(),
          /*max_k=*/*key_seq_len.maybe_as_int(),
          dropout_p,
          is_causal,
          philox_seed,
          philox_offset,
          /*scale=*/scale);

  testValidate(
      fec.fusion(),
      out,
      sdpa_bwd_inputs,
      {ref_grad_query, ref_grad_key, ref_grad_value},
      __LINE__,
      __FILE__);
}

// Test SDPA is segmented correctly. See issue #2517:
// https://github.com/NVIDIA/Fuser/issues/2517
TEST_F(SDPATest, AttnProgram) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> kv_shape({n, h, s, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(kv_shape, DataType::Half);
  auto tvv = makeConcreteTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  tvq = add(tvq, tvq);
  tvq = castOp(DataType::Half, tvq);
  auto tvattn = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);

  TensorView* tvout = add(tvattn.output, tvattn.output);
  fusion->addOutput(tvout);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options);
  at::Tensor k = at::randn(kv_shape, options);
  at::Tensor v = at::randn(kv_shape, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_outputs = at::_scaled_dot_product_flash_attention(
      q + q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);
  auto expected_out = (std::get<0>(aten_outputs).to(at::kFloat)) * 2.0;

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({q, k, v});
  EXPECT_TRUE(at::allclose(out[0], expected_out));
}

TEST_F(SDPATest, AttnFwdBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> q_shape({n, h, l, e});
  std::vector<int64_t> k_shape({n, h, s, e});
  std::vector<int64_t> v_shape({n, h, s, e});
  std::vector<int64_t> attn_shape({n, h, l, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(k_shape, DataType::Half);
  auto tvv = makeConcreteTensor(v_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  auto sdpa_fwd_out = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);

  // Set query_seq_len, key_seq_len as CPU scalars
  sdpa_fwd_out.query_seq_len->setCpuScalar(true);
  sdpa_fwd_out.key_seq_len->setCpuScalar(true);
  sdpa_fwd_out.philox_seed->setCpuScalar(true);
  sdpa_fwd_out.philox_offset->setCpuScalar(true);

  auto tv_grad_output = makeConcreteTensor(attn_shape, DataType::Half);
  fusion->addInput(tv_grad_output);

  auto sdpa_grad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      sdpa_fwd_out.output,
      sdpa_fwd_out.log_sumexp,
      sdpa_fwd_out.query_seq_len,
      sdpa_fwd_out.key_seq_len,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      sdpa_fwd_out.philox_seed,
      sdpa_fwd_out.philox_offset,
      /*scale=*/nullptr);

  fusion->addOutput(sdpa_fwd_out.output);
  fusion->addOutput(sdpa_grad.grad_query);
  fusion->addOutput(sdpa_grad.grad_key);
  fusion->addOutput(sdpa_grad.grad_value);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options).set_requires_grad(true);
  at::Tensor k = at::randn(k_shape, options).set_requires_grad(true);
  at::Tensor v = at::randn(v_shape, options).set_requires_grad(true);
  at::Tensor grad_out = at::randn(attn_shape, options);

  FusionExecutorCache fec(std::move(fusion));
  auto nvf_out = fec.runFusionWithInputs({q, k, v, grad_out});

  auto attn = at::scaled_dot_product_attention(
      q,
      k,
      v,
      /*attn_mask=*/std::nullopt,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*scale=*/std::nullopt);
  q.retain_grad();
  k.retain_grad();
  v.retain_grad();
  attn.backward(grad_out);

  testValidate(
      fec.fusion(),
      nvf_out,
      {q, k, v, grad_out},
      {attn, q.grad(), k.grad(), v.grad()},
      __LINE__,
      __FILE__);
}

// TODO: Remove/update when https://github.com/NVIDIA/Fuser/issues/2563 is
// resolved.
TEST_F(SDPATest, Sharded_SdpaFwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  constexpr int64_t d = 4;
  auto mesh = DeviceMesh::createForNumDevices(d);
  std::vector<int64_t> q_shape({d, n, h / d, l, e});
  std::vector<int64_t> kv_shape({d, n, h / d, s, e});

  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(kv_shape, DataType::Half);
  auto tvv = makeConcreteTensor(kv_shape, DataType::Half);

  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);

  for (TensorView* tv : {tvq, tvk, tvv}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  auto output = sdpfa_fwd(
      tvq,
      tvk,
      tvv,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);

  addSdpaFwdOutputs(fusion.get(), output);
  for (TensorView* tv : {output.output, output.log_sumexp}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn({n, h / d, l, e}, options);
  at::Tensor k = at::randn({n, h / d, s, e}, options);
  at::Tensor v = at::randn({n, h / d, s, e}, options);

  double scale = 1.0 / std::sqrt(e);
  auto aten_out = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale);

  FusionExecutorCache fec(std::move(fusion));
  auto nvf_out =
      fec.runFusionWithInputs({q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}
} // namespace nvfuser
