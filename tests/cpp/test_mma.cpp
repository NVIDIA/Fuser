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
#include <bit>
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
  for (auto i : arange(tensor.size(0))) {
    for (auto j : arange(tensor.size(1))) {
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
  for (auto i : arange(tensor.size(0))) {
    for (auto j : arange(tensor.size(1))) {
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

KernelArgumentHolder scheduleCompileAndRun(
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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
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

  KernelExecutor ke;
  ke.compile(
      fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  return ke.run({inputs.first, inputs.second});
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

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
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

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));

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
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

auto all_dtypes = std::vector<PrimDataType>{DataType::Half, DataType::BFloat16};

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
        testing::ValuesIn(all_dtypes)),
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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
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

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

using HopperMmaRSStMatrixTestParams = std::tuple<
    MmaMacro,
    PrimDataType,
    MmaLayout,
    MmaInputSmemSwizzle,
    std::vector<int64_t>>;

class HopperRSStmatrix
    : public HopperBase,
      public ::testing::WithParamInterface<HopperMmaRSStMatrixTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_b;
  std::vector<int64_t> tile_sizes;

  void SetUp() override {
    HopperBase::SetUp();
    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_b = std::get<3>(GetParam());
    tile_sizes = std::get<4>(GetParam());
  }
};

TEST_P(HopperRSStmatrix, SingleTileWithTMALoadStoreStMatrix) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  int64_t tile_m = tile_sizes.at(0);
  int64_t tile_n = tile_sizes.at(1);

  if (getM(macro) % tile_m || getN(macro) % tile_n) {
    GTEST_SKIP() << "skipping test as output is not divisible by tile size";
  }

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

  int axes = 0;
  switch (layout) {
    case MmaLayout::TT:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }

  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  auto tv3 = castOp(dtype, tv2);

  auto tv4 = set(tv3);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  fusion.addOutput(tv4);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv3c = tv3->cacheBefore();

  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StMatrix);

  matmul_utils::moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  // This is a temporary way to pass this information
  // to the custom index generator for stmatrix.
  // TODO: remove the need for fusion managed cache.
  fusion.manage("ldst_matrix_m_tile", tile_m);
  fusion.manage("ldst_matrix_n_tile", tile_n);
  fusion.manage("ldst_matrix_m_smem", getM(macro));
  fusion.manage("ldst_matrix_n_smem", getN(macro));

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  matmul_utils::moveInnerBroadcastLeft(tv1);
  tv1->applyMmaSwizzleForTMALoad(swizzle_b);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2->reorder({{-1, -2}});
  }

  EXPECT_TRUE(tv3c->getMemoryType() == MemoryType::Local);
  EXPECT_TRUE(tv3->getMemoryType() == MemoryType::Shared);
  EXPECT_TRUE(tv4->getMemoryType() == MemoryType::Global);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setAllocationDomain(s.as<IterDomain*>(), true);

    tv2->axis(-1)->parallelize(ParallelType::Mma);
    tv2->axis(-2)->parallelize(ParallelType::Mma);
    tv2->axis(-3)->parallelize(ParallelType::Mma);
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3c->getLoopDomain());
    tv3c->setLoopDomain(s.as<IterDomain*>());
    tv3c->setAllocationDomain(s.as<IterDomain*>(), true);
  }

  MmaInputSmemSwizzle swizzle = mma_utils::tmaSwizzleSharedMemory(tv3);
  if (swizzle != MmaInputSmemSwizzle::None && tile_n == 8) {
    GTEST_SKIP() << "IdModel does not support stmatrix.x2";
  }

  AbstractTensor stmatrix_abstract =
      mma_utils::scheduleLdStMatrixSharedMemory(tv3, tile_m, tile_n);
  std::vector<IterDomain*> stmatrix = stmatrix_abstract.as<IterDomain*>();
  stmatrix.at(stmatrix.size() - 2)->parallelize(ParallelType::TIDx);
  stmatrix.at(stmatrix.size() - 1)->parallelize(ParallelType::Vectorize);
  tv3->setAlternateLoopDomain(stmatrix);

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3->getLoopDomain());

    if (swizzle != MmaInputSmemSwizzle::None) {
      mma_utils::scheduleTMAStoreForMmaOutput(tv3, swizzle);
    }

    tv3->setLoopDomain(s.as<IterDomain*>());
  }
  mma_utils::scheduleLdStMatrixForMmaOutput(tv3, tile_m, tile_n);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);

  mma_utils::scheduleTMAStoreForMmaOutput(tv4, swizzle);

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
                  inputs.first.squeeze().to(at::kFloat),
                  inputs.second.squeeze().to(at::kFloat),
                  layout)
                  .to(data_type_to_aten(dtype));

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-1, 1e-1));
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
    HopperRSStmatrix,
    testing::Combine(
        testing::ValuesIn(kAllHopperMacros),
        testing::Values(DataType::Half),
        testing::Values(MmaLayout::TN),
        testing::Values(MmaInputSmemSwizzle::None),
        testing::Values(
            // M, N
            std::vector<int64_t>{16, 8},
            std::vector<int64_t>{16, 16})));

auto mmaHopperRSParamsGenerator() {
  // A very simple PRNG:
  // https://en.wikipedia.org/wiki/Lehmer_random_number_generator
  uint32_t lcg_parkmiller = 1;
  // Only select 1/dilute of the params, 1 means not diluting
  const uint32_t dilute = std::stoi(getNvFuserEnv("MMA_TEST_DILUTE", "8"));
  std::vector<HopperMmaRSTestParams> params;
  std::unordered_set<MmaMacro> macros;
  std::unordered_set<PrimDataType> dtypes;
  std::unordered_set<MmaLayout> layouts;
  std::unordered_set<MmaInputSmemSwizzle> swizzle_bs;
  for (auto macro : kAllHopperMacros) {
    for (auto dtype : all_dtypes) {
      for (auto layout : {MmaLayout::TT, MmaLayout::TN}) {
        for (auto swizzle_b : kAllSmemSwizzleModes) {
          bool new_macro = macros.insert(macro).second;
          bool new_dtype = dtypes.insert(dtype).second;
          bool new_layout = layouts.insert(layout).second;
          bool new_swizzle_b = swizzle_bs.insert(swizzle_b).second;
          // Ensure that we test all possible values for each parameter
          // at least once
          if (new_macro || new_dtype || new_layout || new_swizzle_b) {
            params.push_back(std::make_tuple(macro, dtype, layout, swizzle_b));
          } else {
            if (lcg_parkmiller % dilute == 0) {
              params.push_back(
                  std::make_tuple(macro, dtype, layout, swizzle_b));
            }
            lcg_parkmiller = (uint64_t)lcg_parkmiller * 48271 % 0x7fffffff;
          }
        }
      }
    }
  }
  return testing::ValuesIn(params);
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperRS,
    mmaHopperRSParamsGenerator(),
    testNameHopperRS);

using MmaSSTestParams = std::tuple<
    MmaMacro,
    PrimDataType,
    MmaLayout,
    MmaInputSmemSwizzle,
    MmaInputSmemSwizzle>;

template <typename Base>
class SSTest : public Base,
               public ::testing::WithParamInterface<MmaSSTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_a;
  MmaInputSmemSwizzle swizzle_b;

  void SetUp() override {
    Base::SetUp();

    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_a = std::get<3>(GetParam());
    swizzle_b = std::get<4>(GetParam());
  }
};

using HopperSS = SSTest<HopperBase>;

TEST_P(HopperSS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes =
      matmulAtInputShape3DSS(getM(macro), getN(macro), getK(macro), layout);

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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
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

  auto inputs = matmulAtInput3DSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

// Same as SingleTile, except that the core matrices of A and B are stored
// in a transposed way in smem. This is useful for testing if we are
// inferring strides of core matrices correctly.
TEST_P(HopperSS, SingleTileTransposed) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes =
      matmulAtInputShape3DSS(getM(macro), getN(macro), getK(macro), layout);

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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
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

  auto inputs = matmulAtInput3DSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

const char* skipIfSwizzleNotCompatibleWithTiling(
    int64_t num_tiles,
    MmaLayout layout,
    MmaMacro macro,
    MmaInputSmemSwizzle swizzle_a,
    MmaInputSmemSwizzle swizzle_b) {
  constexpr int64_t dtype_size = 2;
  const char* skip_reason =
      "This test stores smem inputs on the inner dimension densely, "
      "which is not compatible with this macro and swizzle mode "
      "because TensorCore instructions span multiple swizzle patterns "
      "unevenly.";

  {
    // Check if need to skip due to unsupported memory layout for A
    int64_t inner_tile_size = layout == MmaLayout::TT || layout == MmaLayout::TN
        ? getK(macro)
        : getM(macro);
    int64_t inner_size = num_tiles * inner_tile_size;
    int64_t swizzle_size = getBytesFromSwizzle(swizzle_a) / dtype_size;
    bool instruction_tile_span_multiple_swizzle = inner_size > swizzle_size;
    bool span_uneven_swizzle = inner_tile_size % swizzle_size != 0 &&
        swizzle_size % inner_tile_size != 0;

    if (instruction_tile_span_multiple_swizzle && span_uneven_swizzle) {
      return skip_reason;
    }
  }

  {
    // Check if need to skip due to unsupported memory layout for B
    int64_t inner_tile_size = layout == MmaLayout::TT || layout == MmaLayout::NT
        ? getN(macro)
        : getK(macro);
    int64_t inner_size = num_tiles * inner_tile_size;
    int64_t swizzle_size = getBytesFromSwizzle(swizzle_b) / dtype_size;
    bool instruction_tile_span_multiple_swizzle = inner_size > swizzle_size;
    bool span_uneven_swizzle = inner_tile_size % swizzle_size != 0 &&
        swizzle_size % inner_tile_size != 0;

    if (instruction_tile_span_multiple_swizzle && span_uneven_swizzle) {
      return skip_reason;
    }
  }
  return nullptr;
}

TEST_P(HopperSS, MultipleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t num_tiles = 2;

  const char* skip_reason = skipIfSwizzleNotCompatibleWithTiling(
      num_tiles, layout, macro, swizzle_a, swizzle_b);
  if (skip_reason) {
    GTEST_SKIP() << skip_reason;
  }

  auto shapes = matmulAtInputShape3DSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout);

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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
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

  auto inputs = matmulAtInput3DSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout,
      data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

std::string testNameSS(const testing::TestParamInfo<MmaSSTestParams>& info) {
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

auto mmaSSParamsGenerator(const auto& all_macros) {
  // A very simple PRNG:
  // https://en.wikipedia.org/wiki/Lehmer_random_number_generator
  uint32_t lcg_parkmiller = 1;
  // Only select 1/dilute of the params, 1 means not diluting
  const uint32_t dilute = std::stoi(getNvFuserEnv("MMA_TEST_DILUTE", "8"));
  std::vector<MmaSSTestParams> params;
  std::unordered_set<MmaMacro> macros;
  std::unordered_set<PrimDataType> dtypes;
  std::unordered_set<MmaLayout> layouts;
  std::unordered_set<MmaInputSmemSwizzle> swizzle_as;
  std::unordered_set<MmaInputSmemSwizzle> swizzle_bs;
  for (auto macro : all_macros) {
    for (auto dtype : all_dtypes) {
      for (auto layout : kAllSupportedMmaLayout) {
        for (auto swizzle_a : kAllSmemSwizzleModes) {
          for (auto swizzle_b : kAllSmemSwizzleModes) {
            bool new_macro = macros.insert(macro).second;
            bool new_dtype = dtypes.insert(dtype).second;
            bool new_layout = layouts.insert(layout).second;
            bool new_swizzle_a = swizzle_as.insert(swizzle_a).second;
            bool new_swizzle_b = swizzle_bs.insert(swizzle_b).second;
            // Ensure that we test all possible values for each parameter
            // at least once
            if (new_macro || new_dtype || new_layout || new_swizzle_a ||
                new_swizzle_b) {
              params.push_back(
                  std::make_tuple(macro, dtype, layout, swizzle_a, swizzle_b));
            } else {
              if (lcg_parkmiller % dilute == 0) {
                params.push_back(std::make_tuple(
                    macro, dtype, layout, swizzle_a, swizzle_b));
              }
              lcg_parkmiller = (uint64_t)lcg_parkmiller * 48271 % 0x7fffffff;
            }
          }
        }
      }
    }
  }
  return testing::ValuesIn(params);
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperSS,
    mmaSSParamsGenerator(kAllHopperMacros),
    testNameSS);

using Blackwell1CTAM64SS = SSTest<BlackwellBase>;
using Blackwell1CTAM128SS = SSTest<BlackwellBase>;
using Blackwell2CTAM128SS = SSTest<BlackwellBase>;
using Blackwell2CTAM128SS = SSTest<BlackwellBase>;

TEST_P(Blackwell1CTAM128SS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes =
      matmulAtInputShape3DSS(getM(macro), getN(macro), getK(macro), layout);

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

  auto tv3 = set(tv2); // tmem -> register
  auto tv4 = set(tv3); // register -> gmem

  fusion.addOutput(tv4);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

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
    tv2->axis(0)->parallelize(ParallelType::Mma);
    tv2->axis(1)->parallelize(ParallelType::Mma);
    tv2->axis(2)->parallelize(ParallelType::Mma);
    tv2->setMemoryType(MemoryType::Tensor);
    tv2->setTMemDimSepPos(1);
  }
  {
    tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);
    for (auto tv : {tv3, tv4}) {
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }
  }

  auto inputs = matmulAtInput3DSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

TEST_P(Blackwell1CTAM128SS, MultipleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t num_tiles = 2;

  const int64_t num_cols = getN(macro) * num_tiles * num_tiles;

  if (num_cols > 512) {
    GTEST_SKIP() << "Skipping test due to excessive number of columns";
  }

  const char* skip_reason = skipIfSwizzleNotCompatibleWithTiling(
      num_tiles, layout, macro, swizzle_a, swizzle_b);
  if (skip_reason) {
    GTEST_SKIP() << skip_reason;
  }

  // TODO: currently we are not doing tiling on K dimension because we are not
  // generating the correct predicate correctly. The correct way to do MMA is to
  // let one thread submit the instruction, but currently all threads are
  // submitting the instruction, so we are doing duplicate computation. If there
  // are multiple tiles on K dimension, the result will be incorrect because we
  // are adding the same number multiple times.
  auto shapes = matmulAtInputShape3DSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout);

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
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto mma_result = tv2->cacheBefore();
  mma_result->setMemoryType(MemoryType::Tensor);

  auto register_result = tv2->cacheBefore();
  register_result->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdTMem);

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

  // Schedule the MMA result
  {
    // Split by tile
    mma_result->split(-3, getM(macro));
    mma_result->split(-2, getN(macro));
    mma_result->split(-1, getK(macro));
    // [Mo, Mi, No, Ni, Ko, Ki] -> [Mo, No, Ko, Mi, Ni, Ki]
    mma_result->reorder({{-5, -3}, {-3, -2}});
    mma_result->axis(-1)->parallelize(ParallelType::Mma);
    mma_result->axis(-2)->parallelize(ParallelType::Mma);
    mma_result->axis(-3)->parallelize(ParallelType::Mma);
    // Set the allocation domain
    // [Mo, No, Ko, Mi, Ni, Ki] -> [Mi, |, Mo, No, Ko, Ni, Ki]
    // Note that we intentionally reorder both the loop domain and the
    // allocation domain. The reason for reordering the loop domain is to
    // prevent `inlineMost` from inlining the MmaOp. The reason for not
    // wanting to inline is because we want to test the case where the
    // TMem has multiple tiles, so that we know our indexing is correct.
    mma_result->reorder({{3, 0}});
    mma_result->setAllocationDomain(mma_result->getLoopDomain(), true);
    mma_result->setTMemDimSepPos(1);
  }

  // Schedule TMem load and gmem store
  for (auto tv : {register_result, tv2}) {
    // Split by tile
    tv->split(-2, getM(macro));
    tv->split(-1, getN(macro));
    // [Mo, Mi, No, Ni] -> [Mo, No, Mi, Ni]
    tv->reorder({{-3, -2}});
    tv->axis(-2)->parallelize(ParallelType::TIDx);
  }
  if (std::has_single_bit(static_cast<uint64_t>(getN(macro)))) {
    // We can only vectorize if N is a power of 2
    register_result->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  inlineMost();

  auto inputs = matmulAtInput3DSS(
      num_tiles * getM(macro),
      num_tiles * getN(macro),
      num_tiles * getK(macro),
      layout,
      data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    Blackwell1CTAM128SS,
    mmaSSParamsGenerator(kAllBlackwell1CTAM128Macros),
    testNameSS);

} // namespace nvfuser
