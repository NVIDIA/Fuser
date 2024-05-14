// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using CaRootDomainMapTest = NVFuserTest;

namespace {

void checkIdMapped(
    ComputeAtRootDomainMap& root_map,
    TensorView* v0,
    IterDomain* id0,
    TensorView* v1,
    IterDomain* id1,
    bool should_map) {
  if (should_map) {
    NVF_CHECK(
        root_map.canMap(v0->domain(), id0, v1->domain(), id1),
        "Should be mappable: ",
        id0,
        " of ",
        v0,
        " and ",
        id1,
        " of ",
        v1);
  } else {
    NVF_CHECK(
        !root_map.canMap(v0->domain(), id0, v1->domain(), id1),
        "Should not be mappable: ",
        id0,
        " of ",
        v0,
        " and ",
        id1,
        " of ",
        v1);
  }
}

void checkIdMapped(
    TensorView* v0,
    const std::vector<IterDomain*>& root0,
    const std::vector<bool> should_map0,
    TensorView* v1,
    const std::vector<IterDomain*>& root1,
    const std::vector<bool> should_map1) {
  ComputeAtRootDomainMap map;
  map.build();
  NVF_ERROR(root0.size() == should_map0.size());
  NVF_ERROR(root1.size() == should_map1.size());
  size_t idx0 = 0;
  for (const auto i : c10::irange(root0.size())) {
    size_t idx1 = 0;
    for (const auto j : c10::irange(root1.size())) {
      if (should_map0[i] && should_map1[j] && idx0 == idx1) {
        checkIdMapped(map, v0, root0[i], v1, root1[j], true);
      } else {
        checkIdMapped(map, v0, root0[i], v1, root1[j], false);
      }
      if (should_map1[j])
        ++idx1;
    }
    if (should_map0[i])
      ++idx0;
  }
}

void checkIdMapped(
    TensorView* v0,
    const std::vector<IterDomain*>& root0,
    TensorView* v1,
    const std::vector<IterDomain*>& root1) {
  checkIdMapped(
      v0,
      root0,
      std::vector<bool>(root0.size(), true),
      v1,
      root1,
      std::vector<bool>(root1.size(), true));
}

} // namespace

TEST_F(CaRootDomainMapTest, FusionRootMappingBasic_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv3 = broadcast(tv0, {true, false, false});
  auto tv4 = broadcast(tv1, {false, true, false});
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {false, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, false, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {false, true},
      tv1,
      tv1->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {false, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, false, true});
  checkIdMapped(tv3, tv3->getRootDomain(), tv4, tv4->getRootDomain());
  checkIdMapped(tv3, tv3->getRootDomain(), tv5, tv5->getRootDomain());
  checkIdMapped(tv4, tv4->getRootDomain(), tv5, tv5->getRootDomain());
}

TEST_F(CaRootDomainMapTest, FusionRootMappingRfactor_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [I,I]
  TensorView* tv0 = makeSymbolicTensor(2);
  // [I,I,I]
  TensorView* tv1 = makeSymbolicTensor(3);

  //[I,I,R]
  auto tv2 = sum(tv1, {2});
  auto tv3 = add(tv2, tv0);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv3);

  // scheduling:
  //[B,I,R0,R1=128], root = [B,I,R]
  tv2->split(2, 128);

  // root=[B,I,Irf], rfactor=[B,I,Irf,Rrf]
  auto tv4 = tv2->rFactor({3});

  checkIdMapped(tv1, tv1->getRootDomain(), tv4, tv4->getRootDomain());
  checkIdMapped(
      tv4,
      tv4->getRFactorDomain(),
      {true, true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true, false},
      tv3,
      tv3->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, true, false},
      tv3,
      tv3->getRootDomain(),
      {true, true});
  checkIdMapped(tv0, tv0->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv1,
      tv1->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv2,
      tv2->getRootDomain(),
      {true, true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRFactorDomain(),
      {true, true, false, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true, false});
}

TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  // The second dimension cannot be mapped as it would require recomputation.
  checkIdMapped(tv0, tv0->getRootDomain(), tv1, tv1->getRootDomain());
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(tv2, tv2->getRootDomain(), tv3, tv3->getRootDomain());
}

TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  auto tv3 = tv1->rFactor({-2});

  checkIdMapped(tv0, tv0->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv3,
      tv3->getMaybeRFactorDomain(),
      {true, false, true},
      tv1,
      tv1->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  tv1->split(-1, 4);
  auto tv4 = tv1->rFactor({-2});

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getMaybeRFactorDomain(),
      {true, false, true},
      tv1,
      tv1->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(tv2, tv2->getRootDomain(), tv3, tv3->getRootDomain());
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
}

// Reproducer of issue #749
TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency5_CUDA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, true});
}

// Similar to RootMappingReductionDependency5 but with rFactor
TEST_F(CaRootDomainMapTest, FusionRootMappingReductionDependency6_CUDA_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  tv2->split(1, 4);
  auto tv6 = tv2->rFactor({-1});

  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv6,
      tv6->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv6,
      tv6->getMaybeRFactorDomain(),
      {true, true, false},
      tv2,
      tv2->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, true},
      tv4,
      tv4->getRootDomain(),
      {true, true});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, true},
      tv5,
      tv5->getRootDomain(),
      {true, true});
}

TEST_F(
    CaRootDomainMapTest,
    FusionRootMappingMultipleBroadcastWithNoCommonConsumer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv0, {true, false});
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  // If there is no common consumer, there is no recomputation constraint.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {false, true});
}

TEST_F(CaRootDomainMapTest, FusionRootMappingBroadcastNonUniqueSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);
  auto tv3 = broadcast(tv0, {false, true});
  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);
  auto tv5 = add(tv2, tv3);
  fusion.addOutput(tv5);

  // Broadcast domains can be used with multiple domains with
  // different sizes. In this test, the broadcast domain of tv3 has
  // two consumers, tv4 and tv5, which may have different sizes. Each
  // of the consumers is used with the broadcast domain of tv3, but
  // the two consumers may not have the same size, it is not possible
  // to map those domains.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv2,
      tv2->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv2,
      tv2->getRootDomain(),
      {true, false},
      tv3,
      tv3->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, false},
      tv4,
      tv4->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv3,
      tv3->getRootDomain(),
      {true, false},
      tv5,
      tv5->getRootDomain(),
      {true, false});
  checkIdMapped(
      tv4,
      tv4->getRootDomain(),
      {true, false},
      tv5,
      tv5->getRootDomain(),
      {true, false});
}

TEST_F(CaRootDomainMapTest, FusionRootMappingBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  // tv0[I0]
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {true, false});
  // tv1[B1, I0]
  auto tv2 = broadcast(tv1, {true, false, false});
  // tv2[B2, B1, I0]
  fusion.addOutput(tv2);

  // In this case, tv1 and tv2 has one and two broadcast domains,
  // respectively. It is the second broadcast domain that is mapped to
  // the broadcast of tv1.
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv1,
      tv1->getRootDomain(),
      {false, true});
  checkIdMapped(
      tv1,
      tv1->getRootDomain(),
      {true, true},
      tv2,
      tv2->getRootDomain(),
      {false, true, true}); // Not {true, false, true}
  checkIdMapped(
      tv0,
      tv0->getRootDomain(),
      {true},
      tv2,
      tv2->getRootDomain(),
      {false, false, true});
}

// Repro of issue #1950
TEST_F(CaRootDomainMapTest, FusionRootMappingRepro1950_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);
  auto tv2 = makeSymbolicTensor(3);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = set(tv0);
  auto tv4 = mul(tv1, tv3);
  auto tv5 = mul(tv1, tv2);
  auto tv6 = mul(tv5, tv3);
  auto tv7 = sum(tv6, {2});
  auto tv8 = broadcast(tv7, {false, false, true});
  auto tv9 = mul(tv3, tv8);

  // Issue #1950 was caused by a particular traversal ordering based
  // on the output tensor ordering as below
  fusion.addOutput(tv9);
  fusion.addOutput(tv5);
  fusion.addOutput(tv4);

  ComputeAtRootDomainMap root_map;
  root_map.build();

  checkIdMapped(root_map, tv4, tv4->axis(-1), tv9, tv9->axis(-1), false);
}

// Step-1 to fix https://github.com/NVIDIA/Fuser/issues/1631
// Needs to check consumer mapped with reduction inputs in
// isReductionOutputMapped(), otherwise compute at root domain map
// is wrong then leads to wrong compute at position and finally
// expr sort failed.
// After fix, there are two persistent buffers and can be further
// reduced to one with a following step-2 to fix the issue in resolution
// points detection.
TEST_F(CaRootDomainMapTest, FusionRootMappingConsumerMappedWithReductionInput) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = castOp(DataType::Float, tv0);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  // manual project tv2 to input, so tv7 is mapped with tv2.
  auto tv5 = castOp(DataType::Float, tv0);
  auto tv6 = div(tv5, tv4);
  auto tv7 = set(tv1);
  auto tv8 = add(tv7, tv6);
  auto tv9 = add(tv2, tv7);
  fusion->addOutput(tv8);
  fusion->addOutput(tv9);

  // |--5------------------|
  // 0 -> 2 -> r3 -> b4 -> 6 -> 8    9
  // 1-------> 7----------------|
  //           |---------------------|
  //      |--------------------------|
  // tv7 has two consumers, tv8 and tv9.
  // tv8 is a consumer of the reduction output.
  // If tv9 is mapped with tv2, we can't map tv8 and tv9 because tv9 is in the
  // pre-reduction set through tv2 and tv8 is in the post-reduction set.
  ComputeAtRootDomainMap root_map;
  root_map.build();
  checkIdMapped(root_map, tv2, tv2->axis(1), tv9, tv9->axis(1), true);
  checkIdMapped(root_map, tv7, tv7->axis(1), tv8, tv8->axis(1), false);
  checkIdMapped(root_map, tv7, tv7->axis(1), tv9, tv9->axis(1), false);
}

} // namespace nvfuser
