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
#include <preseg_passes/propagate_shardings.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using SDPATest = NVFuserTest;

constexpr int64_t n = 16, h = 32, l = 64, s = 128, e = 64;

// Note: Flash Attention is only supported on Ampere and above.

namespace {
void addSdpaFwdOutputs(Fusion* fusion, SdpfaFwdResult output) {
  fusion->addOutput(output.output);
  fusion->addOutput(output.log_sumexp);
  fusion->addOutput(output.philox_seed);
  fusion->addOutput(output.philox_offset);
}
} // namespace

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
auto validateSdpaFwdOutputs = [](KernelArgumentHolder nvf_out,
                                 AtenSdpaOut aten_out, AtenSdpaOut aten_out_meta) {
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
  // nvf_out = {attn, log_sumexp, philox_seed, philox_offset}.
  // Since, dropout_p = 0.0 to validate outputs,
  // philox_seed and philox_offset are uninitialized empty tensors with
  // garbage values for this case, so we skip validating those values.
  NVF_CHECK(at::allclose(nvf_out[0].as<at::Tensor>(), attn));
  NVF_CHECK(at::allclose(nvf_out[1].as<at::Tensor>(), log_sumexp));

  auto
      [attn_meta,
       log_sumexp_meta,
       cum_seq_q_meta,
       cum_seq_k_meta,
       query_seq_len_meta,
       key_seq_len_meta,
       philox_seed_meta,
       philox_offset_meta,
       debug_attn_mask_meta] = aten_out_meta;
  EXPECT_EQ(attn.sizes(), attn_meta.sizes());
  EXPECT_EQ(log_sumexp.sizes(), log_sumexp_meta.sizes());
  EXPECT_EQ(cum_seq_q.sizes(), cum_seq_q_meta.sizes());
  EXPECT_EQ(cum_seq_k.sizes(), cum_seq_k_meta.sizes());
  EXPECT_EQ(philox_seed.sizes(), philox_seed_meta.sizes());
  EXPECT_EQ(philox_offset.sizes(), philox_offset_meta.sizes());
  EXPECT_EQ(debug_attn_mask.sizes(), debug_attn_mask_meta.sizes());
  EXPECT_EQ(attn.strides(), attn_meta.strides());
  EXPECT_EQ(log_sumexp.strides(), log_sumexp_meta.strides());
  EXPECT_EQ(cum_seq_q.strides(), cum_seq_q_meta.strides());
  EXPECT_EQ(cum_seq_k.strides(), cum_seq_k_meta.strides());
  EXPECT_EQ(philox_seed.strides(), philox_seed_meta.strides());
  EXPECT_EQ(philox_offset.strides(), philox_offset_meta.strides());
  EXPECT_EQ(debug_attn_mask.strides(), debug_attn_mask_meta.strides());
};

// Check SDPAFwdOp mapping in IdModel and ComputeAtMap.
void checkSdpaFwdMapping(Fusion* fusion, Expr* op) {
  IdModel id_model(fusion, /*build_graphs=*/false);
  const ValGraph& vg = id_model.buildExactGraph();
  vg.validateConsistency();

  ComputeAtMap compute_at_map(fusion);

  auto sdpa_op = dynamic_cast<SdpaFwdOp*>(op);
  ASSERT_TRUE(sdpa_op != nullptr);

  /* SdpaFwdOp:
  Consumers:
  output = [N, H, L, Ev]
  logsumexp = [N, H, L]
  Producers:
  query = [N, H, L, E]
  key = [N, H, S, E]
  value = [N, H, S, Ev]
  Note: S, E are not mapped together in the producers and do not have any
  mapping to the consumer.
  */
  std::vector<TensorView*> producer_tvs{
      sdpa_op->query(), sdpa_op->key(), sdpa_op->value()};
  std::vector<TensorView*> consumer_tvs{
      sdpa_op->attn_out(), sdpa_op->logsumexp()};

  for (auto producer : producer_tvs) {
    for (auto consumer : consumer_tvs) {
      std::vector<IterDomain*> producer_ids = producer->getLogicalDomain();
      std::vector<IterDomain*> consumer_ids = consumer->getMaybeRootDomain();

      size_t num_device_dim = producer_ids.at(0)->isDeviceDim() ? 1 : 0;

      // Idx=0: producer_ids[0], consumer_ids[0] = N
      // Idx=1: producer_ids[1], consumer_ids[1] = H
      // Idx=2: producer_ids[2]=L/S, consumer_ids [2] = L
      // Idx=3: prodcuer_ids[3] = E/Ev, consumer_idx[3] = Ev

      for (auto idx : arange(consumer_ids.size())) {
        if (idx < (2 + num_device_dim)) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        } else if (
            idx == (2 + num_device_dim) && producer->sameAs(sdpa_op->query())) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        } else if (
            idx == (3 + num_device_dim) && producer->sameAs(sdpa_op->value())) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        }
      }
    }
  }
}

// Check SDPABwdOp mapping in IdModel and ComputeAtMap.
void checkSdpaBwdMapping(Fusion* fusion, Expr* op) {
  IdModel id_model(fusion, /*build_graphs=*/false);
  const ValGraph& vg = id_model.buildExactGraph();
  vg.validateConsistency();

  ComputeAtMap compute_at_map(fusion);

  auto sdpa_op = dynamic_cast<SdpaBwdOp*>(op);
  ASSERT_TRUE(sdpa_op != nullptr);

  /* SdpaBwdOp:
    Consumers:
    grad_query = [N, H, L, E]
    grad_key = [N, H, S, E]
    grad_value = [N, H, S, Ev]
    Producers:
    grad_output = [N, H, L, Ev]
    query = [N, H, L, E]
    key = [N, H, S, E]
    value = [N, H, S, Ev]
    output = [N, H, L, Ev]
    logsumexp = [N, H, L]
  */
  // Do not try to map from rng_state.
  std::vector<TensorView*> producer_tvs{
      sdpa_op->grad_attn(),
      sdpa_op->query(),
      sdpa_op->key(),
      sdpa_op->value(),
      sdpa_op->attn_out(),
      sdpa_op->logsumexp()};
  std::vector<TensorView*> consumer_tvs{
      sdpa_op->grad_query(), sdpa_op->grad_key(), sdpa_op->grad_value()};
  for (TensorView* producer : producer_tvs) {
    for (TensorView* consumer : consumer_tvs) {
      std::vector<IterDomain*> producer_ids = producer->getLogicalDomain();
      std::vector<IterDomain*> consumer_ids = consumer->getMaybeRootDomain();

      size_t num_device_dim = producer_ids.at(0)->isDeviceDim() ? 1 : 0;

      // Idx=0: producer_ids[0], consumer_ids[0] = N
      // Idx=1: producer_ids[1], consumer_ids[1] = H
      // Idx=2: producer_ids[2], consumer_ids [2] = L/S
      // Idx=3: producer_ids[3], consumer_ids[3] = E/Ev

      bool producer_has_s = producer->sameAs(sdpa_op->key()) ||
          producer->sameAs(sdpa_op->value());
      bool consumer_has_s = consumer->sameAs(sdpa_op->grad_key()) ||
          consumer->sameAs(sdpa_op->grad_value());

      bool producer_has_e = producer->sameAs(sdpa_op->query()) ||
          producer->sameAs(sdpa_op->key());
      bool consumer_has_e = consumer->sameAs(sdpa_op->grad_query()) ||
          consumer->sameAs(sdpa_op->grad_key());
      for (auto idx : arange(producer_ids.size())) {
        if (idx < 2 + num_device_dim) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        } else if (
            idx == (2 + num_device_dim) && (producer_has_s == consumer_has_s)) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        } else if (
            idx == (3 + num_device_dim) && (producer_has_e == consumer_has_e)) {
          checkMapped(vg, producer_ids.at(idx), consumer_ids.at(idx));
          EXPECT_TRUE(compute_at_map.areMapped(
              producer_ids.at(idx),
              consumer_ids.at(idx),
              IdMappingMode::EXACT));
        }
      }
    }
  }
}

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

  checkSdpaFwdMapping(fusion.get(), output.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs({q, k, v});

  ExpressionEvaluator ee;
  ee.bind(tvq, q.to(at::kMeta));
  ee.bind(tvk, k.to(at::kMeta));
  ee.bind(tvv, v.to(at::kMeta));
  AtenSdpaOut aten_out_meta = {
      ee.evaluate(fusion->outputs().at(0)),
      ee.evaluate(fusion->outputs().at(1)),
      ee.evaluate(fusion->outputs().at(2)),
      ee.evaluate(fusion->outputs().at(3)),
      ee.evaluate(fusion->outputs().at(4)),
      ee.evaluate(fusion->outputs().at(5)),
      ee.evaluate(fusion->outputs().at(6)),
      ee.evaluate(fusion->outputs().at(7)),
      ee.evaluate(fusion->outputs().at(8)),
  };
  validateSdpaFwdOutputs(nvf_out, aten_out, aten_out_meta);
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

  checkSdpaFwdMapping(fusion.get(), output.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs({q, k, v});
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

  checkSdpaFwdMapping(fusion.get(), output.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs({q, k, v});
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

    for (TensorView* consumer_tv : {output.output, output.log_sumexp}) {
      auto pairwise_map = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
                              .mapProducerToConsumer();
      auto mappingExists = [&pairwise_map](
                               IterDomain* p_id, IterDomain* c_id) -> bool {
        return pairwise_map.find(p_id) != pairwise_map.end() &&
            pairwise_map[p_id] == c_id;
      };

      auto consumer_root = consumer_tv->getMaybeRootDomain();
      for (auto idx : arange(consumer_tv->nDims())) {
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
  auto [tv_seed, tv_offset] = createSdpaRngTvs();

  fusion->addInput(tv_grad_output);
  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);
  fusion->addInput(tv_output);
  fusion->addInput(tv_logsumexp);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  fusion->addOutput(tvgrad.grad_query);
  fusion->addOutput(tvgrad.grad_key);
  fusion->addOutput(tvgrad.grad_value);

  checkSdpaBwdMapping(fusion.get(), tvgrad.grad_query->definition());

  at::Tensor grad_out = at::randn(attn_shape, options);

  KernelArgumentHolder sdpa_bwd_inputs = {
      grad_out, q, k, v, output, log_sumexp, philox_seed, philox_offset};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs(sdpa_bwd_inputs);

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
      executor_cache.fusion(),
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
  auto [tv_seed, tv_offset] = createSdpaRngTvs();

  fusion->addInput(tv_grad_output);
  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);
  fusion->addInput(tv_output);
  fusion->addInput(tv_logsumexp);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  fusion->addOutput(tvgrad.grad_query);
  fusion->addOutput(tvgrad.grad_key);
  fusion->addOutput(tvgrad.grad_value);

  checkSdpaBwdMapping(fusion.get(), tvgrad.grad_query->definition());

  at::Tensor grad_out = at::randn(attn_shape, options);

  KernelArgumentHolder sdpa_bwd_inputs = {
      grad_out, q, k, v, output, log_sumexp, philox_seed, philox_offset};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs(sdpa_bwd_inputs);

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
      executor_cache.fusion(),
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

  checkSdpaFwdMapping(fusion.get(), tvattn.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs({q, k, v});
  EXPECT_TRUE(at::allclose(out[0].as<at::Tensor>(), expected_out));
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

  auto tv_grad_output = makeConcreteTensor(attn_shape, DataType::Half);
  fusion->addInput(tv_grad_output);

  auto sdpa_grad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      sdpa_fwd_out.output,
      sdpa_fwd_out.log_sumexp,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      sdpa_fwd_out.philox_seed,
      sdpa_fwd_out.philox_offset,
      /*scale=*/nullptr);

  fusion->addOutput(sdpa_fwd_out.output);
  fusion->addOutput(sdpa_grad.grad_query);
  fusion->addOutput(sdpa_grad.grad_key);
  fusion->addOutput(sdpa_grad.grad_value);

  checkSdpaFwdMapping(fusion.get(), sdpa_fwd_out.output->definition());
  checkSdpaBwdMapping(fusion.get(), sdpa_grad.grad_query->definition());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn(q_shape, options).set_requires_grad(true);
  at::Tensor k = at::randn(k_shape, options).set_requires_grad(true);
  at::Tensor v = at::randn(v_shape, options).set_requires_grad(true);
  at::Tensor grad_out = at::randn(attn_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs({q, k, v, grad_out});

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
      executor_cache.fusion(),
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

  checkSdpaFwdMapping(fusion.get(), output.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs(
      {q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}

// TODO: Remove/update when https://github.com/NVIDIA/Fuser/issues/2563 is
// resolved.
TEST_F(SDPATest, Sharded_SdpaBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  constexpr int64_t d = 4;
  auto mesh = DeviceMesh::createForNumDevices(d);
  std::vector<int64_t> q_shape({d, n, h / d, l, e});
  std::vector<int64_t> kv_shape({d, n, h / d, s, e});
  std::vector<int64_t> attn_shape({d, n, h / d, l, e});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor q = at::randn({n, h / d, l, e}, options);
  at::Tensor k = at::randn({n, h / d, s, e}, options);
  at::Tensor v = at::randn({n, h / d, s, e}, options);

  constexpr double dropout_p = 0.2;
  constexpr bool is_causal = false;
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

  auto tv_grad_output = makeConcreteTensor(attn_shape, DataType::Half);
  auto tvq = makeConcreteTensor(q_shape, DataType::Half);
  auto tvk = makeConcreteTensor(kv_shape, DataType::Half);
  auto tvv = makeConcreteTensor(kv_shape, DataType::Half);
  auto tv_output = makeConcreteTensor(attn_shape, DataType::Half);
  auto tv_logsumexp = makeConcreteTensor({d, n, h / d, l}, DataType::Float);
  auto [tv_seed, tv_offset] = createSdpaRngTvs();

  fusion->addInput(tv_grad_output);
  fusion->addInput(tvq);
  fusion->addInput(tvk);
  fusion->addInput(tvv);
  fusion->addInput(tv_output);
  fusion->addInput(tv_logsumexp);
  fusion->addInput(tv_seed);
  fusion->addInput(tv_offset);

  for (TensorView* tv :
       {tvq, tvk, tvv, tv_grad_output, tv_output, tv_logsumexp}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  auto tvgrad = sdpfa_bwd(
      tv_grad_output,
      tvq,
      tvk,
      tvv,
      tv_output,
      tv_logsumexp,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      tv_seed,
      tv_offset,
      /*scale=*/nullptr);

  for (TensorView* tv :
       {tvgrad.grad_query, tvgrad.grad_key, tvgrad.grad_value}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
    fusion->addOutput(tv);
  }

  checkSdpaBwdMapping(fusion.get(), tvgrad.grad_query->definition());

  at::Tensor grad_out = at::randn({n, h / d, l, e}, options);

  KernelArgumentHolder sdpa_bwd_inputs = {
      grad_out.unsqueeze(0),
      q.unsqueeze(0),
      k.unsqueeze(0),
      v.unsqueeze(0),
      output.unsqueeze(0),
      log_sumexp.unsqueeze(0),
      philox_seed,
      philox_offset};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs(sdpa_bwd_inputs);

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
      executor_cache.fusion(),
      out,
      sdpa_bwd_inputs,
      {ref_grad_query.unsqueeze(0),
       ref_grad_key.unsqueeze(0),
       ref_grad_value.unsqueeze(0)},
      __LINE__,
      __FILE__);
}

TEST_F(SDPATest, ComputeAt) {
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

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  checkSdpaFwdMapping(fusion.get(), output.output->definition());

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs(
      {q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)});
  validateSdpaFwdOutputs(nvf_out, aten_out);
}

} // namespace nvfuser
