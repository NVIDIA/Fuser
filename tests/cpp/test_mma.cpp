// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <exceptions.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/inlining.h>
#include <algorithm>
#include <iterator>
#include <unordered_map>

namespace nvfuser {

namespace debugging {

// Utilities for debugging MMA ops

// Set a tensor as identity, for example
//  [1, 0, 0]
//  [0, 1, 0]
//  [0, 0, 1]
//  [0, 0, 0]
//  [0, 0, 0]
// This is helpful for debugging because mathematically, an identity matrix
// multiplies any matrix to itself. For example, if you are seeing a wrong
// result, but you don't know if it's because of the input B's memory format is
// not scheduled correctly, you can set the input A to identity and print the
// output. By reading the output, you can tell how the memory layout of input B
// looks like.
void setAsIdentity(at::Tensor tensor) {
  tensor.zero_();
  for (auto i : c10::irange(tensor.size(0))) {
    for (auto j : c10::irange(tensor.size(1))) {
      if (i == j) {
        tensor[i][j] = 1;
      }
    }
  }
}

// Set a tensor as a range, for example
//  [0, 1, 2]
//  [3, 4, 5]
//  [6, 7, 8]
// This makes the tensor easier to read if you print it out.
void setAsARange(at::Tensor tensor) {
  tensor.zero_();
  for (auto i : c10::irange(tensor.size(0))) {
    for (auto j : c10::irange(tensor.size(1))) {
      tensor[i][j] = i * tensor.size(1) + j;
    }
  }
}

} // namespace debugging

using MmaTestParams = std::tuple<MmaMacro, PrimDataType>;

class MmaTest : public NVFuserFixtureParamTest<MmaTestParams> {
 protected:
  MmaMacro macro;
  PrimDataType dtype;

  void SetUp() override {
    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());

    if (isTuring(macro) && cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }

    if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    }

    NVFuserTest::SetUp();
  }
};

std::vector<at::Tensor> scheduleCompileAndRun(
    Fusion* fusion,
    TensorView* tva,
    TensorView* tvb,
    std::pair<at::Tensor, at::Tensor> inputs,
    int64_t dim_to_reduce,
    MmaMacro macro,
    bool propagate_backwards) {
  fusion->addInput(tva);
  fusion->addInput(tvb);

  // Just doing a gmem->register copy
  auto tv0 = set(tva);

  // Just doing a gmem->register copy
  auto tv1 = set(tvb);

  // Dim to reduce is 1 for [M, K, N] and 2 for [M, N, K].
  auto tv2 = fusedMultiplySum(tv0, tv1, {dim_to_reduce});
  fusion->addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  // In this test we don't handle input a (tv0) having
  // an allocation domain.
  NVF_CHECK(
      !tva->hasAllocation(),
      "tva cannot have an allocation domain in this test");

  if (tvb->hasAllocation()) {
    // Get the permutation that describes the difference
    // between the logical domain and allocation domain.
    auto b_permutation =
        ir_utils::computePermutation(
            tvb->getLogicalDomain(), tvb->getAllocationDomain())
            .value();

    // Reorder the ouput of Mma.
    tv2->reorder(b_permutation);

    // We have to propage the changes we made to then output back to the inputs
    // of the Mma Op. Just for the purpose of demonstration we also show how
    // it's equivalent to applying the transform to the input of the Mma
    // directly.
    if (propagate_backwards) {
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          tv2, -1, {});
    } else {
      tv1->reorder(b_permutation);
    }
  }

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] or [M, K, N] -> [N, M, K]
  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);
  tv1->applyMmaSwizzle(MmaOperand::B);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv1->merge(1);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  FusionExecutor fe;
  fe.compileFusion(
      fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  return fe.runFusion({inputs.first, inputs.second});
}

TEST_P(MmaTest, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto M = getM(macro);
  auto N = getN(macro);
  auto K = getK(macro);

  auto tv0 = makeConcreteTensor({M, 1, K}, dtype);
  auto tv1 = makeConcreteTensor({1, N, K}, dtype);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto a_input = at::randn({M, 1, K}, options);
  auto b_input = at::randn({1, N, K}, options);

  auto cg_outputs = scheduleCompileAndRun(
      &fusion,
      tv0,
      tv1,
      {a_input, b_input},
      2 /*dim to reduce [M, N, K]*/,
      macro,
      false /* propagate backwards*/);

  auto tref = a_input.squeeze()
                  .to(at::kFloat)
                  .matmul(b_input.squeeze().t().to(at::kFloat));

  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(MmaTest, SingleTileWithStridedInput) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto M = getM(macro);
  auto N = getN(macro);
  auto K = getK(macro);

  auto tv0 = makeConcreteTensor({M, K, 1}, dtype);
  auto tv1 = makeConcreteTensor({1, K, N}, dtype);
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(2), tv1->axis(1)}, true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto a_input = at::randn({M, K, 1}, options);
  auto b_input = at::randn({1, K, N}, options);
  b_input = b_input.as_strided(b_input.sizes(), {N * K, 1, K});

  auto cg_outputs = scheduleCompileAndRun(
      &fusion,
      tv0,
      tv1,
      {a_input, b_input},
      1 /*dim to reduce [M, K, N]*/,
      macro,
      false /* propagate backwards*/);

  auto tref =
      a_input.squeeze().to(at::kFloat).matmul(b_input.squeeze().to(at::kFloat));

  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));

  // Clear the fusion and try propagating changes to the mma output.
  fusion.clear();
  tv0 = makeConcreteTensor({M, K, 1}, dtype);
  tv1 = makeConcreteTensor({1, K, N}, dtype);
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(2), tv1->axis(1)}, true);
  cg_outputs = scheduleCompileAndRun(
      &fusion,
      tv0,
      tv1,
      {a_input, b_input},
      1 /*dim to reduce [M, N, K]*/,
      macro,
      true /* propagate backwards*/);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

auto all_dtypes = testing::Values(DataType::Half, DataType::BFloat16);

std::string testName(const testing::TestParamInfo<MmaTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  os << toString(macro) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    Turing,
    MmaTest,
    testing::Combine(
        testing::Values(
            MmaMacro::Turing_16_8_8,
            MmaMacro::Turing_16_8_16,
            MmaMacro::Turing_16_16_16),
        testing::Values(DataType::Half)),
    testName);

INSTANTIATE_TEST_SUITE_P(
    Ampere,
    MmaTest,
    testing::Combine(
        testing::Values(MmaMacro::Ampere_16_8_16, MmaMacro::Ampere_16_16_16),
        all_dtypes),
    testName);

// For smem mma input tensors, the schedule does not matter, we just naively
// parallelize it so the test runs faster.
void naivelyParallelize(TensorView* tv) {
  while (tv->nDims() > 1) {
    tv->merge(0);
  }
  tv->split(0, 128);
  tv->axis(1)->parallelize(ParallelType::TIDx);
}

using HopperMmaRSTestParams =
    std::tuple<MmaMacro, PrimDataType, MmaLayout, MmaInputSmemSwizzle>;

class HopperRS : public HopperBase,
                 public ::testing::WithParamInterface<HopperMmaRSTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_b;

  void SetUp() override {
    HopperBase::SetUp();

    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_b = std::get<3>(GetParam());
  }
};

TEST_P(HopperRS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv1);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

// There are only three possible swizzle modes for smem operands of Hopper MMA:
// 32 byte, 64 byte, and 128 byte. Depending on the layout and the macro, the
// inner size may be smaller than the swizzle size. For example, if the macro is
// M64_N8_K16, and the layout is TT, then K is the inner dim, so the inner size
// is 16 items, that is, 32 bytes. If the swizzle mode is 128 byte, then the
// inner size is only 1/4 of the swizzle size. In the SingleTile test, we will
// just pad the inner dim to match the swizzle size, which is a 4x waste of smem
// space. In this test, instead of padding the inner dim, we will use four tiles
// to cover the entire swizzle size, so there is no waste of smem space. Note
// that composing four tiles to form a single swizzle pattern means that the
// memory layout of these four tiles will be interleaved with each other. The
// kernel we are getting is like this:
//
// For TN layout where the inner dimension is a reduction:
//   load operand B from gmem to smem;
//   accumulator = 0;
//   for i in tiles:
//     load operand A from gmem to register;
//     accumulator += A * B;
//   store accumulator to gmem;
//
// For TT layout where the inner dimension is not a reduction:
//   load operand B from gmem to smem;
//   for i in tiles:
//     load operand A from gmem to register;
//     accumulator = 0;
//     accumulator += A * B;
//     store accumulator to gmem;
TEST_P(HopperRS, FullSwizzle) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto swizzle_size = getBytesFromSwizzle(swizzle_b) / dataTypeSize(dtype);
  auto inner_size = layout == MmaLayout::TT ? getN(macro) : getK(macro);

  if (swizzle_size / inner_size <= 1) {
    GTEST_SKIP()
        << "Already tested in SingleTile, not interested in testing it again";
  }

  if (swizzle_size % inner_size != 0) {
    GTEST_SKIP()
        << "We will be using swizzle size as CTA tile size, so it must be divisible";
  }

  const auto k_axis = layout == MmaLayout::TT ? 1 : 2;

  auto shapes = layout == MmaLayout::TT
      ? matmulAtInputShape3DHopperRS(
            getM(macro), swizzle_size, getK(macro), layout)
      : matmulAtInputShape3DHopperRS(
            getM(macro), getN(macro), swizzle_size, layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = fusedMultiplySum(tv0, tv1, {k_axis});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0); // n, m, k

  // Split the inner dimension by the inner size, and reorder the outer
  // of the split to dim 0.
  if (layout == MmaLayout::TN) {
    // inner is K, and K has multiple tiles
    tv0->split(2, inner_size);
    tv0->reorder({{-2, 0}});
    // ko, n, m, ki
  } else {
    // inner is N, and N has multiple tiles
    tv0->split(0, inner_size);
    // no, ni, m, k
  }

  // Now, the inner 2 dimensions are a single MMA tile
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(2);
  tv0->merge(2);
  tv0->axis(2)->parallelize(ParallelType::TIDx);

  // Just schedule tv1 the same way as in SingleTile. Note that although
  // the schedule are the same, the memory layout of tv1 is different.
  // For example, assume that the inner size is 16, and the swizzle size is 64.
  // For the case of SingleTile, the input tensor size will be 16, so the inner
  // dimension will be split as:
  //   1, 64 = split(16, 64)
  // For the case of FullSwizzle, the input tensor size will be 64, so the inner
  // dimension will be split as:
  //   1, 64 = split(64, 64)
  tv1->applyMmaSwizzle(swizzle_b);
  naivelyParallelize(tv1);

  // Split the inner dimension by the inner size, and reorder the outer
  // of the split to dim 0.
  tv2c->split(-1, inner_size);
  tv2c->reorder({{-2, 0}});

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }

  // Create a dummy broadcasting IterDomain to denote that this instruction
  // is a collective operation over 128 threads. This is a newly created
  // broadcasting IterDomain and is not connected to other IterDomains in the
  // TensorDomain. The reason for doing so is because the MMA instruction is
  // really a collective operation over 128 threads, and by definition there is
  // no per-thread assignment like "this thread works on this part of the
  // tensor". It is actually all threads working on all data. For this reason,
  // the threadIdx.x should not appear anywhere in the index of the tensor.
  tv2c->broadcast(1, 128);
  tv2c->axis(1)->parallelize(ParallelType::TIDx);

  if (layout == MmaLayout::TT) {
    // If TN, then the inner dim is K, which is also the reduction dimension.
    // For this case, K does not exist in tv2, so nothing to split.
    // If TT, then the inner dim is N, which is not the reduction dimension.
    // For this case, N exists in tv2, so we need to split it.
    tv2->split(-1, inner_size);
    tv2->reorder({{-2, 0}});
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Inline gmem->register load into the MMA expression at 1.
  // The shared loop is the loop over multiple tiles.
  tv0->inlineAt(1);
  // If TN, then the dim with multiple tiles is K, then the shared loop is a
  // reduction loop. This reduction loop does not exist in the register->gmem
  // store, so nothing to inline.
  // If TT, then the dim with multiple tiles is N, then the shared loop is not a
  // reduction loop. This shared loop exists in the register->gmem store, so we
  // will inline the MMA expression into the register->gmem store.
  if (layout == MmaLayout::TT) {
    tv2c->inlineAt(1);
  }

  auto inputs =
      (layout == MmaLayout::TT ? matmulAtInput3DHopperRS(
                                     getM(macro),
                                     swizzle_size,
                                     getK(macro),
                                     layout,
                                     data_type_to_aten(dtype))
                               : matmulAtInput3DHopperRS(
                                     getM(macro),
                                     getN(macro),
                                     swizzle_size,
                                     layout,
                                     data_type_to_aten(dtype)));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});

  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

// Same as SingleTile, except that the core matrices of B are stored
// in a transposed way in smem. This is useful for testing if we are
// inferring strides of core matrices correctly.
TEST_P(HopperRS, SingleTileTransposed) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  tv1->applyMmaSwizzle(swizzle_b);

  // ****************************************************
  // This is where this test is different from SingleTile
  auto alloc = tv1->getAllocationDomain();
  std::swap(alloc[0], alloc[1]);
  tv1->setAllocationDomain(alloc, true);
  // ****************************************************

  naivelyParallelize(tv1);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(HopperRS, SingleTileWithTMALoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, dtype);
  auto tv1 = makeContigConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  matmul_utils::moveInnerBroadcastLeft(tv1);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(HopperRS, SingleTileWithTMALoadStore) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, dtype);
  auto tv1 = makeContigConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  auto tv3 = set(tv2);
  tv2->setMemoryType(MemoryType::Shared);
  tv3->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  fusion.addOutput(tv3);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  matmul_utils::moveInnerBroadcastLeft(tv1);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  EXPECT_TRUE(tv2c->getMemoryType() == MemoryType::Local);
  EXPECT_TRUE(tv2->getMemoryType() == MemoryType::Shared);
  EXPECT_TRUE(tv3->getMemoryType() == MemoryType::Global);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  mma_utils::scheduleTMAStoreForMmaOutput(tv3, getM(macro), getN(macro));

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(HopperRS, MultipleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t num_tiles = 2;

  auto shapes = matmulAtInputShape3DHopperRS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout);

  int64_t inner_tile_size = layout == MmaLayout::TT ? getN(macro) : getK(macro);
  int64_t inner_size = num_tiles * inner_tile_size;
  int64_t swizzle_size = getBytesFromSwizzle(swizzle_b) / dataTypeSize(dtype);
  bool instruction_tile_span_multiple_swizzle = inner_size > swizzle_size;
  bool span_uneven_swizzle = inner_tile_size % swizzle_size != 0 &&
      swizzle_size % inner_tile_size != 0;

  if (instruction_tile_span_multiple_swizzle && span_uneven_swizzle) {
    GTEST_SKIP()
        << "This test stores smem inputs on the inner dimension densely, "
           "which is not compatible with this macro and swizzle mode "
           "because TensorCore instructions span multiple swizzle patterns unevenly.";
  }

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  // Split by tile
  tv0->split(2, getK(macro));
  tv0->split(1, getM(macro));
  tv0->split(0, getN(macro));
  // [No, Ni, Mo, Mi, Ko, Ki] -> // [Mo, No, Ko, Ni, Mi, Ki]
  tv0->reorder({{2, 0}, {1, -3}, {-2, 2}});
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(4);
  tv0->merge(4);
  tv0->axis(4)->parallelize(ParallelType::TIDx);

  // Note that we do not split tv1 by tile. We just directly swizzle the entire
  // CTA. This means, the location of core matrices of each instruction will be
  // discontiguous. For example, something like this:
  //  it0 it0 it1 it1
  //  it0 it0 it1 it1
  //  it2 it2 it3 it3
  //  it2 it2 it3 it3
  // where itX refers to "instruction tile X".
  //
  // Being discontiguous is not a problem, as long as different core matrices
  // are stored in a strided manner, and we will be able to infer the correct
  // stride.
  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv1);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    // Split by tile
    tv2c->split(-3, getM(macro));
    tv2c->split(-2, getN(macro));
    tv2c->split(-1, getK(macro));
    // [Mo, Mi, No, Ni, Ko, Ki] -> [Mo, No, Ko, Mi, Ni, Ki]
    tv2c->reorder({{-5, -3}, {-3, -2}});
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    // Split by tile
    tv2->split(-2, getM(macro));
    tv2->split(-1, getN(macro));
    // [Mo, Mi, No, Ni] -> [Mo, No, Mi, Ni]
    tv2->reorder({{-3, -2}});
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Inline most. Register will only have one tile.
  inlineMost();

  auto inputs = matmulAtInput3DHopperRS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout,
      data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion(
      {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

std::string testNameHopperRS(
    const testing::TestParamInfo<HopperMmaRSTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  auto swizzle_b = std::get<3>(info.param);
  os << toString(macro) << "_" << toString(layout) << "_" << toString(swizzle_b)
     << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperRS,
    testing::Combine(
        kAllHopperMacros,
        all_dtypes,
        testing::Values(MmaLayout::TT, MmaLayout::TN),
        kAllSmemSwizzleModes),
    testNameHopperRS);

using HopperMmaSSTestParams = std::tuple<
    MmaMacro,
    PrimDataType,
    MmaLayout,
    MmaInputSmemSwizzle,
    MmaInputSmemSwizzle>;

class HopperSS : public HopperBase,
                 public ::testing::WithParamInterface<HopperMmaSSTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_a;
  MmaInputSmemSwizzle swizzle_b;

  void SetUp() override {
    HopperBase::SetUp();

    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_a = std::get<3>(GetParam());
    swizzle_b = std::get<4>(GetParam());
  }
};

TEST_P(HopperSS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // Bring related dims to innermost, that is:
  // - Reorder tv0 as [1, M, K] or [1, K, M]
  // - Reorder tv1 as [1, N, K] or [1, K, N]
  matmul_utils::moveInnerBroadcastLeft(tv0);
  matmul_utils::moveInnerBroadcastLeft(tv1);

  tv0->applyMmaSwizzle(swizzle_a);
  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

// Same as SingleTile, except that the core matrices of A and B are stored
// in a transposed way in smem. This is useful for testing if we are
// inferring strides of core matrices correctly.
TEST_P(HopperSS, SingleTileTransposed) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // Bring related dims to innermost, that is:
  // - Reorder tv0 as [1, M, K] or [1, K, M]
  // - Reorder tv1 as [1, N, K] or [1, K, N]
  matmul_utils::moveInnerBroadcastLeft(tv0);
  matmul_utils::moveInnerBroadcastLeft(tv1);

  tv0->applyMmaSwizzle(swizzle_a);
  tv1->applyMmaSwizzle(swizzle_b);

  // ****************************************************
  // This is where this test is different from SingleTile
  auto alloc0 = tv0->getAllocationDomain();
  std::swap(alloc0[0], alloc0[1]);
  tv0->setAllocationDomain(alloc0, true);
  auto alloc1 = tv1->getAllocationDomain();
  std::swap(alloc1[0], alloc1[1]);
  tv1->setAllocationDomain(alloc1, true);
  // ****************************************************

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

// See the note in HopperRS.FullSwizzle for the explanation of this test.
TEST_P(HopperSS, FullSwizzle) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  bool m_is_inner = layout == MmaLayout::NT || layout == MmaLayout::NN;
  auto swizzle_size_a = getBytesFromSwizzle(swizzle_a) / dataTypeSize(dtype);
  auto inner_size_a = m_is_inner ? getM(macro) : getK(macro);
  bool multiple_a = swizzle_size_a / inner_size_a > 1;

  bool n_is_inner = layout == MmaLayout::TT || layout == MmaLayout::NT;
  auto swizzle_size_b = getBytesFromSwizzle(swizzle_b) / dataTypeSize(dtype);
  auto inner_size_b = n_is_inner ? getN(macro) : getK(macro);
  bool multiple_b = swizzle_size_b / inner_size_b > 1;

  if (!multiple_a && !multiple_b) {
    GTEST_SKIP()
        << "Already tested in SingleTile, not interested in testing it again";
  }

  if ((multiple_a && swizzle_size_a % inner_size_a != 0) ||
      (multiple_b && swizzle_size_b % inner_size_b != 0)) {
    GTEST_SKIP()
        << "We will be using swizzle size as CTA tile size, so it must be divisible";
  }

  int64_t m = (multiple_a && m_is_inner) ? swizzle_size_a : getM(macro);
  int64_t n = (multiple_b && n_is_inner) ? swizzle_size_b : getN(macro);
  int64_t k1 = (multiple_a && !m_is_inner) ? swizzle_size_a : getK(macro);
  int64_t k2 = (multiple_b && !n_is_inner) ? swizzle_size_b : getK(macro);

  if (k1 != k2) {
    GTEST_SKIP()
        << "This test assumes the CTA tile size of A and B must be the same";
  }

  auto shapes = matmulAtInputShape3DHopperSS(m, n, k1, layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // Bring related dims to innermost, that is:
  // - Reorder tv0 as [1, M, K] or [1, K, M]
  // - Reorder tv1 as [1, N, K] or [1, K, N]
  matmul_utils::moveInnerBroadcastLeft(tv0);
  matmul_utils::moveInnerBroadcastLeft(tv1);

  // Just schedule tv0 and tv1 the same way as in SingleTile. Note that although
  // the schedule are the same, the memory layout is different.
  // For example, assume that the inner size is 16, and the swizzle size is 64.
  // For the case of SingleTile, the input tensor size will be 16, so the inner
  // dimension will be split as:
  //   1, 64 = split(16, 64)
  // For the case of FullSwizzle, the input tensor size will be 64, so the inner
  // dimension will be split as:
  //   1, 64 = split(64, 64)
  tv0->applyMmaSwizzle(swizzle_a);
  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

  // [M, N, K]
  int64_t inline_pos = 0;
  if (multiple_a && m_is_inner) {
    tv2c->split(-3, getM(macro));
    tv2->split(-2, getM(macro));
    inline_pos++;
  }
  if (multiple_b && n_is_inner) {
    tv2c->split(-2, getN(macro));
    tv2c->reorder({{-3, -4}});
    tv2->split(-1, getN(macro));
    tv2->reorder({{-2, -3}});
    inline_pos++;
  }
  if ((multiple_a && !m_is_inner) || (multiple_b && !n_is_inner)) {
    tv2c->split(-1, getK(macro));
    tv2c->reorder({{-2, -4}});
  }
  // [Mo, No, Ko, Mi, Ni, Ki]

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  tv2c->inlineAt(inline_pos);

  auto inputs =
      matmulAtInput3DHopperSS(m, n, k1, layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(HopperSS, SingleTileWithTMALoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, dtype);
  auto tv1 = makeContigConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv0->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  matmul_utils::moveInnerBroadcastLeft(tv0);
  matmul_utils::moveInnerBroadcastLeft(tv1);
  tv0->applyMmaSwizzleForTMALoad(swizzle_a);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  // For smem mma input tensors, the schedule does not matter, we just naively
  // parallelize it so the test runs faster.
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    // Note: according to internal doc "Representing ldmatrix", we need both a
    // read domain and a write domain to correctly represent MmaOp. Without this
    // new mechanism, there is no correct loop domain, and the only choices are
    // either we want to represent the smem read well, or represent the register
    // write well. We choose to represent the smem read well here. Likely, this
    // means we will not be able to have multiple tiles in register, but we can
    // workaround this by always inlining the MmaOp most. We should fix this
    // after we implemented the new read/write domain mechanism.
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  auto inputs = matmulAtInput3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

TEST_P(HopperSS, MultipleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t num_tiles = 2;

  auto shapes = matmulAtInputShape3DHopperSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout);

  const char* skip_reason =
      "This test stores smem inputs on the inner dimension densely, "
      "which is not compatible with this macro and swizzle mode "
      "because TensorCore instructions span multiple swizzle patterns unevenly.";

  {
    // Check if need to skip due to unsupported memory layout for A
    int64_t inner_tile_size = layout == MmaLayout::TT || layout == MmaLayout::TN
        ? getK(macro)
        : getM(macro);
    int64_t inner_size = num_tiles * inner_tile_size;
    int64_t swizzle_size = getBytesFromSwizzle(swizzle_a) / dataTypeSize(dtype);
    bool instruction_tile_span_multiple_swizzle = inner_size > swizzle_size;
    bool span_uneven_swizzle = inner_tile_size % swizzle_size != 0 &&
        swizzle_size % inner_tile_size != 0;

    if (instruction_tile_span_multiple_swizzle && span_uneven_swizzle) {
      GTEST_SKIP() << skip_reason;
    }
  }

  {
    // Check if need to skip due to unsupported memory layout for B
    int64_t inner_tile_size = layout == MmaLayout::TT || layout == MmaLayout::NT
        ? getN(macro)
        : getK(macro);
    int64_t inner_size = num_tiles * inner_tile_size;
    int64_t swizzle_size = getBytesFromSwizzle(swizzle_b) / dataTypeSize(dtype);
    bool instruction_tile_span_multiple_swizzle = inner_size > swizzle_size;
    bool span_uneven_swizzle = inner_tile_size % swizzle_size != 0 &&
        swizzle_size % inner_tile_size != 0;

    if (instruction_tile_span_multiple_swizzle && span_uneven_swizzle) {
      GTEST_SKIP() << skip_reason;
    }
  }

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // Bring related dims to innermost, that is:
  // - Reorder tv0 as [1, M, K] or [1, K, M]
  // - Reorder tv1 as [1, N, K] or [1, K, N]
  matmul_utils::moveInnerBroadcastLeft(tv0);
  matmul_utils::moveInnerBroadcastLeft(tv1);

  // Note that we do not split tv0 and tv1 by tile. We just directly swizzle the
  // entire CTA. This means, the location of core matrices of each instruction
  // will be discontiguous. For example, something like this:
  //  it0 it0 it1 it1
  //  it0 it0 it1 it1
  //  it2 it2 it3 it3
  //  it2 it2 it3 it3
  // where itX refers to "instruction tile X".
  //
  // Being discontiguous is not a problem, as long as different core matrices
  // are stored in a strided manner, and we will be able to infer the correct
  // stride.
  tv0->applyMmaSwizzle(swizzle_a);
  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

  {
    // Split by tile
    tv2c->split(-3, getM(macro));
    tv2c->split(-2, getN(macro));
    tv2c->split(-1, getK(macro));
    // [Mo, Mi, No, Ni, Ko, Ki] -> [Mo, No, Ko, Mi, Ni, Ki]
    tv2c->reorder({{-5, -3}, {-3, -2}});
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
    tv2c->axis(-1)->parallelize(ParallelType::Mma);
    tv2c->axis(-2)->parallelize(ParallelType::Mma);
    tv2c->axis(-3)->parallelize(ParallelType::Mma);
  }
  {
    // Split by tile
    tv2->split(-2, getM(macro));
    tv2->split(-1, getN(macro));
    // [Mo, Mi, No, Ni] -> [Mo, No, Mi, Ni]
    tv2->reorder({{-3, -2}});
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  inlineMost();

  auto inputs = matmulAtInput3DHopperSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout,
      data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

std::string testNameHopperSS(
    const testing::TestParamInfo<HopperMmaSSTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  auto swizzle_a = std::get<3>(info.param);
  auto swizzle_b = std::get<4>(info.param);
  os << toString(macro) << "_" << toString(layout) << "_" << toString(swizzle_a)
     << "_" << toString(swizzle_b) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperSS,
    testing::Combine(
        kAllHopperMacros,
        all_dtypes,
        kAllSupportedMmaLayout,
        kAllSmemSwizzleModes,
        kAllSmemSwizzleModes),
    testNameHopperSS);

} // namespace nvfuser
