/*
> [!NOTE]
> This file is both a [cpp](../../tests/cpp/tutorial_ldmatrix_stmatrix.cpp) and a Markdown.
> You may see some strange symbols in the rendered Markdown.
> It is difficult to avoid them. But they should not affect reading.
> All the unit tests displayed here are executable from the `test_tutorial`
> binary

<!--*/
#pragma GCC diagnostic ignored "-Wcomment"
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
/*-->

To see prints in the test, set verbose to `true`:<!-- */ //-->\
```cpp
constexpr static bool verbose = false; /*
```

# LdMatrix and StMatrix Support in NVFuser
<!--*/
#include <sstream>
#include <string>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

#define NOT_IMPLEMENTED GTEST_SKIP() << "Not implemented yet";

namespace nvfuser {

using HopperLdStMatrixTutorial = HopperBase;

/* -->

The scheduleLdStMatrix function creates and schedules an abstract tensor for a
TensorView with a ldmatrix or stmatrix definition. The layout is based on the
register accumulation layout for wgmma and the hard-coded index supported by
the indexing pass. <!-- */ //-->\
```cpp
AbstractTensor scheduleLdStMatrix(TensorView* tv) {
  // Assume the input TensorView is block tiled. e.g., The last two iterDomains
  // are the warp tile except for k dimension.
  // The CTA tile is (128, 256).
  // The Warp tile is (64, 256).
  // The TMA box is (64, 64).
  // The LdStMatrix.x4 tile is (16, 16).
  // The core matrix for wgmma and LdStMatrix is (8, 8).

  AbstractTensor abstract_tensor(tv->getLoopDomain());
  // (GM, GN, cta_m(2), cta_n(1), m(64), n(256))

  // Split by TMA shared memory box
  abstract_tensor.split(-1, 64);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), m(64), ni(64))

  // Split by (16, 16) matrix for LdStMatrix.x4
  abstract_tensor.split(-2, 16);
  abstract_tensor.split(-1, 16);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mi(16), nii(16))

  // Split (16, 16) matrix into four (8, 8) sub-matrices
  abstract_tensor.split(-2, 8);
  abstract_tensor.split(-1, 8);

  // Each register handles two adjacent elements.
  abstract_tensor.split(-1, 2);

  // The four (8, 8) sub-matrices are traversed in this order to follow the
  // register layout for wgmma accumulator matrix.
  // *****************
  // *       *       *
  // *       *       *
  // *   0   *   2   *
  // *       *       *
  // *       *       *
  // *****************
  // *       *       *
  // *       *       *
  // *   1   *   3   *
  // *       *       *
  // *       *       *
  // *****************
  abstract_tensor.reorder({{-5, -2}, {-4, -5}, {-2, -4}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mii(8), niiio(4),
  // niio(2), mio(2), niiii(2))

  // For an (16, 16) matrix, each register will hold 8 values. The LdStMatrix
  // instruction will load or store these values with a single instruction. We
  // remove this serial for-loop from the kernel by merging the last three
  // iterDomains together and then applying ParallelType::Vectorize.
  abstract_tensor.merge(-2, -1);
  abstract_tensor.merge(-2, -1);
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mii(8), niiio(4), (niio
  // * mio * niiii)(8))

  // Reorder iterDomains so the serial IterDomain for (CTA_N / TMA_N) and
  // (TMA_N and LDST_N) are adjacent.
  abstract_tensor.reorder({{-5, -4}, {-4, -5}});

  // Four LdStMatrix.x4 instructions are issued simultaneously to process
  // (64, 16) tile. Merge mio, miii, and niiio iterDomains together.
  abstract_tensor.merge(-4, -3);
  abstract_tensor.merge(-3, -2);
  // (GM, GN, cta_m(2), cta_n(1), no(4), nio(4), (mo * mii * niiio)(128), (niio
  // * mio * niiii)(8))

  // Hard-coded shared memory index expects a single serial IterDomain
  abstract_tensor.merge(-4, -3);
  return abstract_tensor;
} /*
```

Create LdStMatrixSet example<!-- */ //-->\
```cpp
TEST_F(HopperLdStMatrixTutorial, LdStMatrixSet) {
  const auto dtype = DataType::BFloat16;

  // Fusion Overview:
  // tv0 - None (global memory)
  // tv0_smem - Tma load with swizzle 128B (shared memory)
  // tv0_reg - LdMatrix (registers)
  // tv1_smem - StMatrix (shared memory)
  // tv1 - Tma store with swizzle 128B (global memory)

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  // ===========================================================================

  // Constants
  constexpr int64_t cta_m = 128;
  constexpr int64_t cta_n = 256;
  constexpr int64_t warp_m = 64;
  constexpr int64_t warp_n = 256;
  constexpr int64_t ldst_matrix_tile_m = 16;
  constexpr int64_t ldst_matrix_tile_n = 16;
  fusion.manage("ldst_matrix_m_tile", ldst_matrix_tile_m);
  fusion.manage("ldst_matrix_n_tile", ldst_matrix_tile_n);
  fusion.manage("ldst_matrix_m_smem", warp_m);
  fusion.manage("ldst_matrix_n_smem", warp_n);

  // ===========================================================================

  // The definition for tv0_smem is tma load, which moves data from shared to
  // global memory.
  TensorView* tv0_smem = tv0->cacheAfter();
  tv0_smem->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_smem->setMemoryType(MemoryType::Shared);

  // TODO Add ldmatrix support
  // The definition for tv0_reg is ldmatrix, which moves data from shared memory
  // to registers.
  TensorView* tv0_reg = tv0_smem->cacheAfter();
  tv0_reg->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrix);

  // The definition for tv1_smem is stmatrix, which moves data from registers to
  // shared memory.
  TensorView* tv1_smem = tv1->cacheBefore();
  tv1_smem->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::StMatrix);
  tv1_smem->setMemoryType(MemoryType::Shared);

  // The definition for tv1 is tma store, which moves data from shared to global
  // memory.
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // ===========================================================================

  // Create 2D block tile
  // (M, N)
  tv1->split(0, cta_m);
  tv1->split(-1, cta_n);
  tv1->reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(128), cta_n(256))

  tv1->split(-2, warp_m);
  tv1->split(-1, warp_n);
  tv1->reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), warp_m(64), warp_n(256))

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv1);

  // Schedule output from TMA Load
  MmaInputSmemSwizzle input_swizzle =
      mma_utils::tmaSwizzleSharedMemory(tv0_smem);
  mma_utils::MmaSwizzler::scheduleTMALoadForMma(tv0_smem, input_swizzle);

  // Schedule global memory output from TMA Store
  MmaInputSmemSwizzle output_swizzle =
      mma_utils::tmaSwizzleSharedMemory(tv1_smem);
  mma_utils::scheduleTMAStoreForMmaOutput(tv1, output_swizzle);

  // ===========================================================================

  // NOTE: When using a custom allocation domain, all iterDomains to the left
  // of the computeAt position must exist in the loop domain. The utility
  // function for applying swizzle to TMA LoadStoreOp creates the appropriate
  // TMA Box. Creating the same TMA Box in the loop domain via AbstractTensor
  // allows for inlining iterDomains that are not identical, causing an
  // assertion in indexing pass.

  // Move data from tv0_reg to tv1_smem using StMatrix
  AbstractTensor tv1_smem_abstract_tensor =
      scheduleLdStMatrix(tv1_smem);
  // Create tma store allocation domain with swizzle
  if (output_swizzle != MmaInputSmemSwizzle::None) {
    mma_utils::scheduleTMAStoreForMmaOutput(tv1_smem, output_swizzle);
  }
  tv1_smem->setLoopDomain(tv1_smem_abstract_tensor.as<IterDomain*>());
  // (GM(BDX), GN(BDY), cta_m(2), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128), (niio * mio * niiii)(8))

  // Use ParallelType::TIDx to launch four StMatrix.x4 in parallel.
  // Use ParallelType::Vectorize because StMatrix.x4 stores eight elements per
  // thread per operation.
  tv1_smem->axis(-2)->parallelize(ParallelType::TIDx);
  tv1_smem->axis(-1)->parallelize(ParallelType::Vectorize);
  // (GM(BDX), GN(BDY), cta_m(2)(TDY), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128)(TDX), (niio * mio * niiii)(8)(V))

  // ===========================================================================

  // Move data from tv0_reg to tv1_smem using LdMatrix
  AbstractTensor tv0_reg_abstract_tensor =
      scheduleLdStMatrix(tv0_reg);
  tv0_reg->setLoopDomain(tv0_reg_abstract_tensor.as<IterDomain*>());
  // (GM(BDX), GN(BDY), cta_m(2), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128), (niio * mio * niiii)(8))

  // Set allocation domain according to loop domain
  tv0_reg->setAllocationDomain(
      tv0_reg->getLoopDomain(), /*new_contiguity=*/true);

  // Use ParallelType::TIDx to launch four LdMatrix.x4 in parallel.
  // Use ParallelType::Vectorize because LdMatrix.x4 stores eight elements per
  // thread per operation.
  tv0_reg->axis(-2)->parallelize(ParallelType::TIDx);
  tv0_reg->axis(-1)->parallelize(ParallelType::Vectorize);
  // (GM(BDX), GN(BDY), cta_m(2)(TDY), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128)(TDX), (niio * mio * niiii)(8)(V))

  // ===========================================================================

  inlineMost();

  // ===========================================================================

  constexpr int dim0 = 8192, dim1 = 8192;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_tv0});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  auto cg_outputs = ke.run({at_tv0});

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), at_tv0));
} /*
```

<!--*/
} // namespace nvfuser
// \-->
