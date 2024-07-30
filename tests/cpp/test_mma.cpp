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
#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
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
  moveInnerBroadcastLeft(tv0);
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

std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DHopperRS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

std::pair<at::Tensor, at::Tensor> matmulAtInput3DHopperRS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DHopperRS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

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

  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv1);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

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

  // const auto m_axis = 0;
  // const auto n_axis = layout == MmaLayout::TT ? 2 : 1;
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

  moveInnerBroadcastLeft(tv0); // n, m, k
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
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(2);
  tv0->merge(2);
  tv0->axis(2)->parallelize(ParallelType::TIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv1);

  tv2c->split(-1, inner_size);
  tv2c->reorder({{-2, 0}});
  tv2c->axis(1)->parallelize(ParallelType::Mma);
  tv2c->axis(2)->parallelize(ParallelType::Mma);
  tv2c->axis(3)->parallelize(ParallelType::Mma);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  tv2c->broadcast(1, 128);
  tv2c->axis(1)->parallelize(ParallelType::TIDx);

  if (layout == MmaLayout::TT) {
    tv2->split(-1, inner_size);
    tv2->reorder({{-2, 0}});
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  tv0->inlineAt(1);
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

  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  moveInnerBroadcastLeft(tv1);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

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

  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  moveInnerBroadcastLeft(tv1);
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
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
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

TEST_P(HopperRS, SingleTileWithTMALoadOuterDimNotSplit) {
  if (layout == MmaLayout::TT) {
    GTEST_SKIP() << "Skipping test as we only handle TN layout in this test";
  }

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

  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  // In this case we don't split the outer dimension, thus having
  // fewer TMA loads.
  tv1->applyMmaSwizzleForTMALoad(swizzle_b, /* don't split outer dim*/ false);

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

std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DHopperSS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    case MmaLayout::NT:
      return {{K, M, 1}, {K, 1, N}};
    case MmaLayout::NN:
      return {{1, K, M}, {N, K, 1}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

std::pair<at::Tensor, at::Tensor> matmulAtInput3DHopperSS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DHopperSS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

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
  moveInnerBroadcastLeft(tv0);
  moveInnerBroadcastLeft(tv1);

  // Hopper tensor core assumes K major, so we are using !transpose_a here.
  tv0->applyMmaSwizzle(swizzle_a);
  tv1->applyMmaSwizzle(swizzle_b);

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

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

  moveInnerBroadcastLeft(tv0);
  moveInnerBroadcastLeft(tv1);
  tv0->applyMmaSwizzleForTMALoad(swizzle_a);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  // For smem mma input tensors, the schedule does not matter, we just naively
  // parallelize it so the test runs faster.
  tv0->axis(1)->parallelize(ParallelType::TIDx);
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
