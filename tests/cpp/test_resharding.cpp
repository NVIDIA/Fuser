// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <fusion_segmenter.h>
#include <host_ir/lower_to_communication.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <runtime/executor_kernel_arg.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using testing::Each;
using testing::IsEmpty;
using testing::IsFalse;
using testing::ResultOf;

using ReshardingTest = NVFuserTest;

TEST_F(ReshardingTest, SplitingView) {
  const int b = 2, s = 3, h = 96, e = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeContigConcreteTensor({-1, -1, h * e});
  TensorView* out = reshape(
      in,
      {size(in, 0),
       size(in, 1),
       IrBuilder::create<Val>(h),
       IrBuilder::create<Val>(e)});
  fusion.addInput(in);
  fusion.addOutput(out);

  const int d = 2;
  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(2, d);
    tv->axis(2)->parallelize(ParallelType::DIDx);
  }

  at::Tensor in_tensor = at::randn({b, s, h * e / d}, at::Device(at::kCUDA));
  KernelArgumentHolder args({in_tensor});
  DynamicTransform::concretizeFusion(&fusion, args);

  EXPECT_THAT(fusion.exprs(), Each(ResultOf(isResharding, IsFalse())));
}

TEST_F(ReshardingTest, MergingView) {
  const int b = 2, s = 3, h = 96, e = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeContigConcreteTensor({-1, -1, h, e});
  TensorView* out = flatten(in, /*start_dim=*/2);
  fusion.addInput(in);
  fusion.addOutput(out);

  const int d = 2;
  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(2, d);
    tv->axis(2)->parallelize(ParallelType::DIDx);
  }

  at::Tensor in_tensor = at::randn({b, s, h / d, e}, at::Device(at::kCUDA));
  KernelArgumentHolder args({in_tensor});
  DynamicTransform::concretizeFusion(&fusion, args);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1});
  TensorView* out = set(in);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  TensorView* out = set(in);
  in->setDeviceMesh({0, 1});
  out->setDeviceMesh({0, 2});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_DifferentParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  TensorView* out = set(in);
  out->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_SameMesh_SameParallelType) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = set(in);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  TensorView* out = sum(in, {0});

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  TensorView* out = sum(in, {0});

  in->setDeviceMesh({0, 1});
  out->setDeviceMesh({0, 1, 2});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_ParallelizeDifferentAxes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {0});
  out->axis(1)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_UnshardedAxis) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {1});

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_AllReduce) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {0});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Add_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y}) {
    tv->setDeviceMesh({0, 1});
  }
  z->setDeviceMesh({0, 1, 2});

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_OnlyOutputParallelized) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  z->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_OnlyInputsParallelized) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  for (auto* tv : {x, y}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_SameMesh_SameParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_InputsParallelizedDifferently) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1, 2});
  }
  for (auto* tv : {x, z}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_Broadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(2);
  x->setDeviceMesh({0, 1});
  TensorView* y = makeContigTensor(1);
  y->setDeviceMesh({0, 1});
  y = broadcast(y, {true, false});
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Matmul_NoResharding) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(2);
  TensorView* y = makeContigTensor(2);
  TensorView* z = matmul(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  for (auto* tv : {x, z}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Matmul_Resharding) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(2); // [iM, iK]
  TensorView* y = makeContigTensor(2); // [iK, iN]
  TensorView* z = matmul(x, y); // [iM, iN, rk]

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  x->axis(0)->parallelize(ParallelType::DIDx);
  y->axis(1)->parallelize(ParallelType::DIDx);
  z->axis(0)->parallelize(ParallelType::DIDx);

  // iN is sharded in y but not in z.
  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Allgather) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  TensorView* in = makeContigTensor(2);
  in->setDeviceMesh(DeviceMesh::createForNumDevices(kNumDevices));
  TensorView* out = set(in);

  in->split(0, kNumDevices, /*inner_split=*/false);
  in->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, ReduceScatter) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  const auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = makeContigConcreteTensor({kNumDevices, 2, kNumDevices * 3});
  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* out = sum(in, {0});
  out->split(-1, kNumDevices, /*inner_split=*/false);
  out->axis(-2)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Allreduce) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  const auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = makeContigConcreteTensor({kNumDevices, 2, 3});
  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* allreduce = sum(in, {0});

  TensorView* out = add(allreduce, allreduce);

  EXPECT_TRUE(isResharding(allreduce->definition()));
  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Broadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  const auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = makeContigTensor(2);
  TensorView* out = broadcast(in, {true, false, false});

  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
  }
  out->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, ReshardingSqueeze) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  const auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({true, std::nullopt})
                       .shape({-1, 1})
                       .build();
  in->setDeviceMesh(mesh);
  TensorView* out = squeeze(in, {1});

  in->merge(0);
  in->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

// Currently, simplifyExpr doesn't recognize that `0 <= x < 1` ==> `x == 0`.
TEST_F(ReshardingTest, DISABLED_NonreshardingSqueeze) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t kNumDevices = 2;
  const auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({true, std::nullopt})
                       .shape({-1, 1})
                       .build();
  in->setDeviceMesh(mesh);
  TensorView* out = squeeze(in, {1});

  in->merge(0);
  in->axis(0)->parallelize(ParallelType::DIDx);
  out->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, InsertResharding_Before) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = makeSymbolicTensor(3);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(c);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh0);
  c->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  c = outputs[0]->as<TensorView>();
  std::vector<TensorView*> inputs(c->definition()->inputs().size());
  for (auto i : arange(c->definition()->inputs().size())) {
    inputs[i] = c->definition()->input(i)->as<TensorView>();
  }
  EXPECT_TRUE(getTvsWithDifferentSharding(c, inputs).empty());
}

TEST_F(ReshardingTest, InsertResharding_After) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  fusion.addInput(a);
  fusion.addOutput(b);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  b = outputs[0]->as<TensorView>();
  Expr* expr = b->definition();
  EXPECT_TRUE(expr->isA<LoadStoreOp>());
  EXPECT_EQ(expr->as<LoadStoreOp>()->opType(), LoadStoreOpType::Set);
  std::vector<TensorView*> tvs = {expr->inputs()[0]->as<TensorView>()};
  EXPECT_THAT(getTvsWithDifferentSharding(a, tvs), IsEmpty());
}

TEST_F(ReshardingTest, InsertShardedAxisReordering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addOutput(c);

  DeviceMesh mesh({0, 1});
  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);

  b->axis(1)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  int num_inner_reshardings = 0;
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr) && !isCommunicationLayoutCompliant(expr)) {
      num_inner_reshardings++;
    }
  }
  EXPECT_GT(num_inner_reshardings, 0);

  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(&fusion);
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr)) {
      EXPECT_TRUE(isCommunicationLayoutCompliant(expr));
    }
  }
}

using ReshardingSelectOpTest = NVFuserTest;

TEST_F(ReshardingSelectOpTest, NonResharding) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* tv0 = makeContigTensor(3);
  auto* tv1 = select(
      tv0, /*dim=*/0, /*index=*/IrBuilder::create<Val>(0, DataType::Int));

  DeviceMesh mesh({0});
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);

  tv0->axis(1)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_FALSE(isResharding(tv1->definition()));
}

TEST_F(ReshardingSelectOpTest, ReshardinSelectIntoDeviceDim) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* tv0 = makeContigTensor(3);
  auto* tv1 = select(
      tv0, /*dim=*/0, /*index=*/IrBuilder::create<Val>(0, DataType::Int));

  DeviceMesh mesh({0});
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);

  tv0->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(tv1->definition()));
}

TEST_F(ReshardingSelectOpTest, ReshardingSelectIntoNonDeviceDim) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* tv0 = makeContigTensor(3);
  auto* tv1 = select(
      tv0, /*dim=*/0, /*index=*/IrBuilder::create<Val>(0, DataType::Int));

  DeviceMesh mesh({0});
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);

  tv0->axis(1)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(tv1->definition()));
}

} // namespace nvfuser
