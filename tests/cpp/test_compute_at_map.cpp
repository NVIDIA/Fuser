// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ComputeAtMapTest = NVFuserTest;

TEST_F(ComputeAtMapTest, FusionMappingRelation) {
  // See https://github.com/csarofeen/pytorch/pull/1960
  // and https://github.com/csarofeen/pytorch/pull/2113
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({1, 1});
  TensorView* tv1 = makeConcreteTensor({-1, 1, 1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false, false});
  auto tv4 = add(tv3, tv1);

  fusion->addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-2);

  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  ComputeAtMap ca_map(fusion);

  auto tv4_inner_node = tv4->axis(0)->definition()->input(1)->as<IterDomain>();
  NVF_CHECK(
      ca_map.areMapped(tv2->axis(0), tv4_inner_node, IdMappingMode::EXACT));
  NVF_CHECK(ca_map.areMapped(
      tv2->axis(0), tv4_inner_node, IdMappingMode::PERMISSIVE));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({2, 1, 1}, options);

  KernelExecutor ke;
  ke.compile(fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Repro of issue #4238
TEST_F(ComputeAtMapTest, UnregisteredAlmostExactExpr) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto i0 = tv1->axis(0);

  auto i1 = i0->cloneWithoutRFactor();
  auto b0 = IterDomainBuilder(fusion.zeroVal(), fusion.oneVal())
                .iter_type(IterType::Broadcast)
                .build();
  auto b1 = IterDomainBuilder(fusion.zeroVal(), fusion.oneVal())
                .iter_type(IterType::Broadcast)
                .build();

  auto i2 = IterDomain::merge(i1, b0);
  IrBuilder::create<Merge>(i0, i2, b1);

  tv1->setLoopDomain({i2, b1});

  // i1 and b0 are used to generate i2 and are connected to i0, but
  // since the loop domain is {i2, b1}, they are not included in
  // tv1. Thus, while the merge of i1 and b0 is a trivial merge, it
  // should not be considered when building the almost-exact
  // graph.
  ComputeAtMap ca_map(&fusion);
}

} // namespace nvfuser
