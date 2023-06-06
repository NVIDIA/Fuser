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

// TODO: smem fence?

// https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/


TEST_F(TMATest, Store1DNoSwizzle_CUDA) {
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024 * 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser