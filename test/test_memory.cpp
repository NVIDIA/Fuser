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

// Tuning guide:
// https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator

// TODO:
// Example usage of TMA
// https://github.com/NVIDIA/cutlass/blob/87349d349605c1e24366fcbe8f04d0141dcb617b/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp#L311-L414
// https://github.com/NVIDIA/cutlass/blob/87349d349605c1e24366fcbe8f04d0141dcb617b/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp#L516-L545
// https://github.com/NVIDIA/cutlass/blob/87349d349605c1e24366fcbe8f04d0141dcb617b/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp

// TODO: smem fence?

// https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/

// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma.hpp

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

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  // tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024}, options);
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

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(1, 4);
    tv->axis(1)->parallelize(ParallelType::BIDx);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDy);
  }
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
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

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(2, 4);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->split(1, 4);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDz);
  }
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);
  tv3->axis(5)->parallelize(ParallelType::Bulk);
  tv2->axis(5)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16, 16}, options);
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

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(3, 4);
    tv->axis(3)->parallelize(ParallelType::TIDx);
    tv->split(2, 4);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->split(1, 4);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDz);
  }
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);
  tv3->axis(5)->parallelize(ParallelType::Bulk);
  tv3->axis(7)->parallelize(ParallelType::Bulk);
  tv2->axis(7)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16, 16, 16}, options);
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

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(4, 4);
    tv->axis(4)->parallelize(ParallelType::TIDx);
    tv->split(3, 4);
    tv->axis(3)->parallelize(ParallelType::TIDy);
    tv->split(2, 4);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    tv->split(1, 4);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDz);
  }
  tv3->axis(1)->parallelize(ParallelType::Bulk);
  tv3->axis(3)->parallelize(ParallelType::Bulk);
  tv3->axis(5)->parallelize(ParallelType::Bulk);
  tv3->axis(7)->parallelize(ParallelType::Bulk);
  tv3->axis(9)->parallelize(ParallelType::Bulk);
  tv2->axis(9)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16, 16, 16, 16}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser