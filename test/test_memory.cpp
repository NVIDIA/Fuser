// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class TMATest : public NVFuserTest {
  // TODO: assert hopper
};

TEST_F(TMATest, Store1D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv3->axis(0)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, Store2D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv3->axis(0)->parallelize(ParallelType::Bulk);
  tv3->axis(1)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, Store3D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv3->axis(0)->parallelize(ParallelType::Bulk);
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(2)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, Store4D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv3->axis(0)->parallelize(ParallelType::Bulk);
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, Store5D_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(5);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv3->axis(0)->parallelize(ParallelType::Bulk);
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);
  tv3->axis(4)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser