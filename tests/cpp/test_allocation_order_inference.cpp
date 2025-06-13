// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <runtime/executor.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::ElementsAre;

using AllocationOrderInferenceTest = NVFuserTest;

std::vector<int64_t> getAllocationOrder(TensorView* tv) {
  std::optional<std::vector<int64_t>> permutation =
      ir_utils::computePermutation(
          tv->getLogicalDomain(), tv->getMaybeAllocationDomain());
  return permutation.value();
}

TEST_F(AllocationOrderInferenceTest, BroadcastOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 =
      broadcast(tv0, {true, false, false, true, false, true, false, true});
  fusion.addOutput(tv2); // (0, 2, 3, 1) -> (0, 3, 5, 7, 1, 4, 6, 2)
  auto tv3 = broadcast(tv1, {true, false, true, true});
  fusion.addOutput(tv3);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(0, 3, 5, 7, 1, 4, 6, 2));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, UnaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = relu(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv1), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, BinaryOpPropagationOneTV) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv0);
  auto s1 = IrBuilder::create<Val>(1L);
  // Testing propagation between tensor and a scalar
  auto tv2 = add(tv0, s1);
  fusion.addOutput(tv2);
  // Testing propagation between tensor and a scalar
  auto tv3 = add(s1, tv0);
  fusion.addOutput(tv3);
  auto s4 = IrBuilder::create<Val>(3L);
  // binary op between scalars
  auto s5 = add(s1, s4);
  auto tv6 = add(tv0, s5);
  fusion.addOutput(tv6);
  auto tv7 = add(s5, tv0);
  fusion.addOutput(tv7);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(0, 2, 3, 1));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(0, 2, 3, 1));
  EXPECT_THAT(getAllocationOrder(tv6), ElementsAre(0, 2, 3, 1));
  EXPECT_THAT(getAllocationOrder(tv7), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, BinaryOpPropagationTwoTV) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Testing propagation between two tensors
  auto tv0 = makeSymbolicTensor({-1, 1, 1, -1});
  fusion.addInput(tv0);
  // tv1 has more non-broadcast iter domain and dominates output memory format
  auto tv1 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);
  auto tv3 = add(tv1, tv0);
  fusion.addOutput(tv3);

  std::vector<IterDomain*> tv0_format = {
      tv0->axis(0), tv0->axis(2), tv0->axis(1), tv0->axis(3)};
  tv0->setAllocationDomain(tv0_format, true);
  std::vector<IterDomain*> tv1_format = {
      tv1->axis(1), tv1->axis(0), tv1->axis(2), tv1->axis(3)};
  tv1->setAllocationDomain(tv1_format, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(1, 0, 2, 3));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(1, 0, 2, 3));
}

TEST_F(AllocationOrderInferenceTest, BinaryOpPropagationWithBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Testing propagation between two tensors
  // tv0 has more non-broadcast iter domain and dominates output memory format
  auto tv0 = makeSymbolicTensor({1, -1, -1, -1});
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor({-1, 1, 1, 1});
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  // since tv0->axis(0) is a broadcast, and tv2->axis(0) is not exact map. The
  // mapping would skip tv->axis(0) and continue mapping for the rest of iter
  // domains. tv2 will have output allocation order as {0, 3, 2, 1}.
  std::vector<IterDomain*> tv0_alloc = {
      tv0->axis(3), tv0->axis(2), tv0->axis(0), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_alloc, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(0, 3, 2, 1));
}

TEST_F(AllocationOrderInferenceTest, TensorFactoryBinaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, 1});
  fusion.addInput(tv0);
  auto s1 = IrBuilder::create<Val>(16L);
  auto s2 = IrBuilder::create<Val>(32L);
  auto fill_value = IrBuilder::create<Val>(1.0);
  // factory method
  auto tv1 = full({s1, s2}, fill_value, DataType::Float);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);
  auto tv3 = add(tv1, tv0);
  fusion.addOutput(tv3);

  std::vector<IterDomain*> tv0_c_last = {tv0->axis(1), tv0->axis(0)};
  tv0->setAllocationDomain(tv0_c_last, true);

  // tv1 is tensor created by factory method, its layout shouldn't be propagated
  // to output
  std::vector<IterDomain*> tv1_c_last = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_c_last, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(1, 0));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(1, 0));
}

TEST_F(AllocationOrderInferenceTest, TensorEmptyAllocationOrderPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, 1});
  fusion.addInput(tv0);
  auto s1 = IrBuilder::create<Val>(16L);
  auto s2 = IrBuilder::create<Val>(32L);
  auto fill_value = IrBuilder::create<Val>(1.0);
  // factory method
  auto tv1 = full({s1, s2}, fill_value, DataType::Float);
  auto tv2 = full({s1, s2}, fill_value, DataType::Float);
  // tv3 is produced by two tv from factory methods, where both have empty
  // allocation order this test is to verify that empty allocation order does
  // propagates across binary operations
  auto tv3 = add(tv1, tv2);
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  std::vector<IterDomain*> tv0_c_last = {tv0->axis(1), tv0->axis(0)};
  tv0->setAllocationDomain(tv0_c_last, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv4), ElementsAre(1, 0));
}

TEST_F(AllocationOrderInferenceTest, TernaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv2);
  auto tv3 = gt(tv0, IrBuilder::create<Val>(0.0));
  fusion.addOutput(tv3);
  auto tv4 = where(tv3, tv1, tv2);
  fusion.addOutput(tv4);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);
  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);
  std::vector<IterDomain*> tv2_nhwc = {
      tv2->axis(0), tv2->axis(2), tv2->axis(3), tv2->axis(1)};
  tv2->setAllocationDomain(tv2_nhwc, true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(0, 2, 3, 1));
  EXPECT_THAT(getAllocationOrder(tv4), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, ReductionOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
  std::vector<IterDomain*> tv0_order = {
      tv0->axis(1), tv0->axis(2), tv0->axis(3), tv0->axis(0)};
  tv0->setAllocationDomain(tv0_order, true); // stride order: {1, 2, 3, 0}
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor({-1, 1}); // stride order: {0, 1}
  fusion.addInput(tv1);
  // Instead of propagating stride order: {1, 2, 3, 0}
  // The end result is {2, 1, 3, 0} because we skip mapping from Iteration id to
  // reduction id. See Note [ Allocation Order Mapping ] sharp-edge 0 for
  // details.
  // TODO: restore behavior after issue:
  // https://github.com/NVIDIA/Fuser/issues/2202
  auto tv2 = sum(tv0, {1});
  fusion.addOutput(tv2);
  // ditto. stride order here is {2, 1, 0} instead of {1, 2, 0}
  auto tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);
  // tv3 dominates the propagation since it has more non-broadcast dimension
  auto tv4 = add(tv1, tv3); // stride order: {1, 0}
  fusion.addOutput(tv4);
  // tv5's new broadcast dimension are placed as outermost in allocation domain,
  // it dropped the reduction dimension from tv3 and preserves other
  // alloc_domain order from tv3. Hence tv5 has stride order: {0, 3, 2, 1}
  auto tv5 = broadcast(tv3, {true, false, false, true});
  fusion.addOutput(tv5);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
#if true
  // permutation here is strange because in propagation we are preserving
  // reduction iter domain in its position in logical domain See issue:
  // https://github.com/NVIDIA/Fuser/issues/2202
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(2, 1, 3, 0));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(2, 1, 0));
#else
  EXPECT_THAT(getAllocationOrder(tv2), ElementsAre(1, 2, 3, 0));
  EXPECT_THAT(getAllocationOrder(tv3), ElementsAre(1, 2, 0));
#endif
  EXPECT_THAT(getAllocationOrder(tv4), ElementsAre(1, 0));
  EXPECT_THAT(getAllocationOrder(tv5), ElementsAre(0, 3, 2, 1));
}

TEST_F(AllocationOrderInferenceTest, EnableInRuntime) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(4);
  fusion->addInput(tv0);
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor in_tensor = at::randn({2, 4, 8, 8}, options);
  at::Tensor in_nhwc =
      in_tensor.as_strided({2, 4, 8, 8}, {4 * 8 * 8, 1, 4 * 8, 4});
  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({in_nhwc});
  auto ref_out = in_nhwc.relu();

  EXPECT_TRUE(cg_outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));
  EXPECT_TRUE(ref_out.allclose(cg_outputs[0].as<at::Tensor>()));
}

TEST_F(AllocationOrderInferenceTest, QkvSplitSdpaForward) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t b = 2, s = 1024, h = 12, e = 768;

  auto* in = makeContigConcreteTensor({b, s, h * e * 3}, DataType::Half);
  fusion.addInput(in);
  std::vector<TensorView*> chunks = chunk(in, 3, -1);
  for (auto*& chunk : chunks) {
    chunk = reshape(chunk, {b, s, h * e}, {b, s, h, e});
    chunk = transpose(chunk, 1, 2);
  }
  SdpfaFwdResult outs = sdpfa_fwd(
      chunks[0],
      chunks[1],
      chunks[2],
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      /*scale=*/nullptr);
  fusion.addOutput(outs.output);
  fusion.addOutput(outs.log_sumexp);

  preseg_passes::OptimizationPass<
      preseg_passes::MarkAliasesPreparePass>::runPass(&fusion);
  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(outs.output), ElementsAre(0, 2, 1, 3));
  EXPECT_THAT(getAllocationOrder(outs.log_sumexp), ElementsAre(0, 1, 2));
}

TEST_F(AllocationOrderInferenceTest, SdpaBackward) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t b = 2, s = 1024, h = 12, e = 768;

  auto* o_grad = makeConcreteTensor({b, h, s, e}, DataType::Half);
  auto* q = makeConcreteTensor({b, h, s, e}, DataType::Half);
  auto* k = makeConcreteTensor({b, h, s, e}, DataType::Half);
  auto* v = makeConcreteTensor({b, h, s, e}, DataType::Half);
  auto* o = makeConcreteTensor({b, h, s, e}, DataType::Half);
  auto* lse = makeConcreteTensor({b, h, s}, DataType::Float);

  auto [seed, offset] = createSdpaRngTvs();

  fusion.addInput(o_grad);
  fusion.addInput(q);
  fusion.addInput(k);
  fusion.addInput(v);
  fusion.addInput(o);
  fusion.addInput(lse);
  fusion.addInput(seed);
  fusion.addInput(offset);

  o_grad->setAllocationDomain(
      {o_grad->axis(0), o_grad->axis(2), o_grad->axis(1), o_grad->axis(3)},
      true);
  for (auto* tv : {q, k, v}) {
    tv->setAllocationDomain(
        {tv->axis(0), tv->axis(2), tv->axis(1), tv->axis(3)},
        {true, false, true, true});
  }
  o->setAllocationDomain(
      {o->axis(0), o->axis(2), o->axis(1), o->axis(3)}, true);

  auto grads = sdpfa_bwd(
      o_grad,
      q,
      k,
      v,
      o,
      lse,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      seed,
      offset,
      /*scale=*/nullptr);

  fusion.addOutput(grads.grad_query);
  fusion.addOutput(grads.grad_key);
  fusion.addOutput(grads.grad_value);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(grads.grad_query), ElementsAre(0, 2, 1, 3));
  EXPECT_THAT(getAllocationOrder(grads.grad_key), ElementsAre(0, 2, 1, 3));
  EXPECT_THAT(getAllocationOrder(grads.grad_value), ElementsAre(0, 2, 1, 3));
}

TEST_F(AllocationOrderInferenceTest, SdpaFwdWithDID) { 
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t b = 2, s = 1024, h = 12, e = 768, d = 2;

  auto* q = makeContigConcreteTensor({b, s, h, e}, DataType::Half);
  auto* k = makeContigConcreteTensor({b, s, h, e}, DataType::Half);
  auto* v = makeContigConcreteTensor({b, s, h, e}, DataType::Half);

  fusion.addInput(q);
  fusion.addInput(k);
  fusion.addInput(v);

  for (auto* tv : {q, k, v}) {
    tv->reorder({0, 2, 1, 3});
    tv->outer_split(1, d);
    tv->axis(1)->parallelize(ParallelType::DIDx);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  SdpfaFwdResult outs = sdpfa_fwd(
      q,
      k,
      v,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(true),
      /*scale=*/nullptr);
  fusion.addOutput(outs.output);
  fusion.addOutput(outs.log_sumexp);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(outs.output), ElementsAre(0, 2, 1, 3));
  EXPECT_THAT(getAllocationOrder(outs.log_sumexp), ElementsAre(0, 1, 2));
  
}

TEST_F(AllocationOrderInferenceTest, UnaryOpWithDID) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = relu(tv0);
  fusion.addOutput(tv1);

  tv0->reorder({1, 0});
  tv0->outer_split(0, 2);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  preseg_passes::OptimizationPass<preseg_passes::AllocationDomainPass>::runPass(
      &fusion);
  EXPECT_THAT(getAllocationOrder(tv1), ElementsAre(1, 0));
}

} // namespace nvfuser
