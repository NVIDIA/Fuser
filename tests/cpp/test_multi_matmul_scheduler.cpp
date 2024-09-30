// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <scheduler/matmul_heuristic_plugin_api.h>
#include <scheduler/mma_utils.h>
#include <scheduler/multi_matmul.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <memory>

namespace nvfuser {

using MultiMatmulSchedulerMatchTestParams = std::tuple<
    bool, // a_m_inner
    bool, // b_k_inner
    int64_t, // vec_size_a
    int64_t, // vec_size_b
    int64_t, // vec_size_epilogue
    bool, // smem_epilogue
    int64_t, // splitk_factor
    bool, // cta_order_col_major
    int64_t // grid_swizzle_factor
    >;

class MultiMatmulSchedulerMatchTest
    : public NVFuserFixtureParamTest<MultiMatmulSchedulerMatchTestParams> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  MultiMatmulSchedulerMatchTest() {
    fusion_ptr_ = std::make_unique<Fusion>();
    fusion = fusion_ptr_.get();
    fusion_guard_ptr_ = std::make_unique<FusionGuard>(fusion);

    std::tie(
        a_m_inner,
        b_k_inner,
        vec_size_a,
        vec_size_b,
        vec_size_epilogue,
        smem_epilogue,
        splitk_factor,
        cta_order_col_major,
        grid_swizzle_factor) = GetParam();

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(256, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.supported_vec_size = {vec_size_a, vec_size_b, vec_size_epilogue};
    mparams.tile_sizes = gemm_tile;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
    mparams.async_gmem_load_operands =
        mparams.circular_buffer_options.smem_circular_buffer_stage > 1;
    mparams.use_smem_epilogue = smem_epilogue;
    mparams.splitk_factor = splitk_factor;
    mparams.cta_order = cta_order_col_major
        ? MatmulParams::TileRasterizationOrder::ColumnMajor
        : MatmulParams::TileRasterizationOrder::RowMajor;
    mparams.grid_swizzle_factor = grid_swizzle_factor;
  }

  void SetUp() {
    NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  }

  void TearDown() {
    if (!IsSkipped()) {
      compareSchedules();
    }
  }

  // Get A and B in shapes [M, K] and [K, N] with allocation domains set.
  std::pair<TensorView*, TensorView*> getInputTVs() {
    auto tv0 = makeContigTensor(2, DataType::Half);
    auto tv1 = makeContigTensor(2, DataType::Half);

    if (a_m_inner) {
      tv0->setAllocationDomain({tv0->axis(1), tv0->axis(0)}, true);
    }

    if (b_k_inner) {
      tv1->setAllocationDomain({tv1->axis(1), tv1->axis(0)}, true);
    }
    return {tv0, tv1};
  }

  // Get A and B in shapes [Batch, M, K] and [Batch, K, N] with allocation
  // domains set.
  std::pair<TensorView*, TensorView*> getBatchInputTVs() {
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);

    if (a_m_inner) {
      tv0->setAllocationDomain(
          {tv0->axis(0), tv0->axis(2), tv0->axis(1)}, true);
    }

    if (b_k_inner) {
      tv1->setAllocationDomain(
          {tv1->axis(0), tv1->axis(2), tv1->axis(1)}, true);
    }
    return {tv0, tv1};
  }

  // Get A and B in already-broadcasted shapes [M, K, 1] and [1, K, N],
  // possibly with allocation domains sset
  std::pair<TensorView*, TensorView*> getBroadcastInputTVs() {
    auto tv0 = makeContigConcreteTensor({-1, 1, -1}, DataType::Half);
    auto tv1 = makeContigConcreteTensor({1, -1, -1}, DataType::Half);

    if (a_m_inner) {
      tv0->setAllocationDomain(
          {tv0->axis(2), tv0->axis(1), tv0->axis(0)}, true);
    }

    if (b_k_inner) {
      tv1->setAllocationDomain(
          {tv1->axis(2), tv1->axis(1), tv1->axis(0)}, true);
    }
    return {tv0, tv1};
  }

  std::pair<at::Tensor, at::Tensor> getInputTensors(
      int M,
      int N,
      int K,
      bool a_m_inner,
      bool b_k_inner,
      bool broadcasted = false) {
    const auto options =
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0 /*device*/);
    auto t0 = a_m_inner ? at::randn({M, K}, options).as_strided({M, K}, {1, M})
                        : at::randn({M, K}, options);
    auto t1 = b_k_inner ? at::randn({K, N}, options).as_strided({K, N}, {1, K})
                        : at::randn({K, N}, options);
    if (broadcasted) {
      t0 = t0.unsqueeze(-1);
      t1 = t1.unsqueeze(0);
    }
    return {t0, t1};
  }

  // Recursively compare scalar values
  void compareScalars(Val* v_orig, Val* v_new) {
    std::stringstream suffix_ss;
    suffix_ss << " when comparing new scalar " << v_new->toInlineString()
              << " to original scalar " << v_orig->toInlineString();
    std::string suffix = suffix_ss.str();
    EXPECT_TRUE(v_orig->isScalar()) << suffix;
    EXPECT_TRUE(v_new->isScalar()) << suffix;

    EXPECT_EQ(v_new->isConst(), v_orig->isConst()) << suffix;
    if (v_new->isConst() && v_orig->isConst()) {
      EXPECT_EQ(v_new->value(), v_orig->value()) << suffix;
      return;
    }

    EXPECT_EQ(v_new->definition() == nullptr, v_orig->definition() == nullptr);
    if (v_orig->definition() == nullptr) {
      if (Val* v_orig_cloned = cloner_->clone(v_orig)) {
        EXPECT_TRUE(v_orig_cloned == v_new) << suffix;
      }
    } else if (
        v_new->definition() != nullptr && v_orig->definition() != nullptr) {
      Expr* def_orig = v_orig->definition();
      Expr* def_new = v_new->definition();
      EXPECT_TRUE(def_orig->sameOp(def_new)) << suffix;
      EXPECT_EQ(def_orig->attributes().size(), def_new->attributes().size())
          << suffix;
      EXPECT_EQ(def_orig->inputs().size(), def_new->inputs().size()) << suffix;
      for (size_t i : c10::irange(def_orig->inputs().size())) {
        if (!testing::Test::HasFailure() && i < def_orig->inputs().size() &&
            i < def_new->inputs().size()) {
          compareScalars(def_orig->input(i), def_new->input(i));
        }
      }
    }
  }

  void compareIDs(IterDomain* id_orig, IterDomain* id_new) {
    std::stringstream suffix_ss;
    suffix_ss << " when comparing new IterDomain " << id_new->toString()
              << " to original IterDomain " << id_orig->toString();
    std::string suffix = suffix_ss.str();
    EXPECT_EQ(id_orig->getIterType(), id_new->getIterType()) << suffix;
    // ParallelType checking is disabled altogether for now. We now check that
    // the compiled code matches which verifies that the loop groups have the
    // same ParallelType.
    /*
    if (id_orig->isParallelized()) {
      // In some cases the new scheduler parallelizes IDs that were not
      // previously parallelized. This is OK as long as the generated kernels
      // match, since inlined dimensions don't need to all be parallelized.
      // However, we do want to ensure that at least as many dimensions get
      // parallelized in the new scheduler, so if id_orig is parallelized, we
      // should match it.
      EXPECT_EQ(id_orig->getParallelType(), id_new->getParallelType())
          << suffix;
    }
    */
    EXPECT_EQ(id_orig->hasExpandedExtent(), id_new->hasExpandedExtent())
        << suffix;
    compareScalars(
        id_orig->getMaybeExpandedExtent(), id_new->getMaybeExpandedExtent());

    // Print suffix whenever a test fails before recursing
    ASSERT_FALSE(testing::Test::HasFailure()) << suffix;

    EXPECT_EQ(id_new->definition() == nullptr, id_orig->definition() == nullptr)
        << suffix;
    if (id_orig->definition() == nullptr) {
      // TODO: reinstate this check if we decide to check ParallelType again
      /*
      if (Val* id_orig_cloned = cloner_->clone(id_orig)) {
        EXPECT_TRUE(id_orig_cloned->sameAs(id_new)) << suffix;
      }
      */
    } else {
      Expr* def_orig = id_orig->definition();
      Expr* def_new = id_new->definition();
      EXPECT_TRUE(def_orig->sameOp(def_new))
          << "def_orig\n  " << def_orig->toString()
          << "is not the same as def_new\n  " << def_new->toString() << suffix;
      EXPECT_EQ(def_orig->attributes().size(), def_new->attributes().size())
          << suffix;
      EXPECT_EQ(def_orig->inputs().size(), def_new->inputs().size()) << suffix;
      for (size_t i : c10::irange(def_orig->inputs().size())) {
        if (!testing::Test::HasFailure() && i < def_orig->inputs().size() &&
            i < def_new->inputs().size()) {
          compareIDs(
              def_orig->input(i)->as<IterDomain>(),
              def_new->input(i)->as<IterDomain>());
        }
      }
    }
    // Print suffix after recursing for context
    ASSERT_FALSE(testing::Test::HasFailure()) << suffix;
  }

  void compareTVs(TensorView* tv_orig, TensorView* tv_new) {
    std::stringstream suffix_ss;
    suffix_ss << " when comparing new TensorView\n  " << tv_new->toString()
              << "\nto original TensorView\n  " << tv_orig->toString();
    std::string suffix = suffix_ss.str();

    EXPECT_EQ(tv_new->hasSwizzleOp(), tv_orig->hasSwizzleOp());
    EXPECT_EQ(tv_orig->shouldPromoteReuse(), tv_new->shouldPromoteReuse());
    EXPECT_EQ(tv_orig->getMemoryType(), tv_new->getMemoryType());

    EXPECT_EQ(tv_new->getComputeAtPosition(), tv_orig->getComputeAtPosition());
    EXPECT_EQ(tv_orig->isCircularBuffered(), tv_new->isCircularBuffered());
    if (tv_orig->isCircularBuffered() && tv_new->isCircularBuffered()) {
      EXPECT_EQ(tv_orig->circularBufferDepth(), tv_new->circularBufferDepth());
    }

    // Inspect loop domain
    ASSERT_EQ(tv_new->nDims(), tv_orig->nDims()) << suffix;
    for (size_t i : c10::irange(tv_orig->nDims())) {
      IterDomain* id_orig = tv_orig->axis((int64_t)i);
      IterDomain* id_new = tv_new->axis((int64_t)i);
      compareIDs(id_orig, id_new);
      // Print transforms after failure recursing for context
      if (testing::Test::HasFailure()) {
        std::cout << "ORIG: " << tv_orig->toString() << std::endl;
        tv_orig->printTransforms();
        std::cout << "NEW: " << tv_new->toString() << std::endl;
        tv_new->printTransforms();
      }
      ASSERT_FALSE(testing::Test::HasFailure()) << suffix;
    }

    // Inspect allocation domain
    ASSERT_EQ(tv_new->hasAllocation(), tv_orig->hasAllocation()) << suffix;
    if (tv_orig->hasAllocation()) {
      const std::vector<IterDomain*>& alloc_orig =
          tv_orig->getAllocationDomain();
      const std::vector<IterDomain*>& alloc_new = tv_new->getAllocationDomain();
      ASSERT_EQ(alloc_new.size(), alloc_orig.size()) << suffix;
      for (size_t i : c10::irange(alloc_orig.size())) {
        compareIDs(alloc_orig[i], alloc_new[i]);
        // Print transforms after failure recursing for context
        if (testing::Test::HasFailure()) {
          std::cout << "ORIG: " << tv_orig->toString() << std::endl;
          tv_orig->printTransforms();
          std::cout << "NEW: " << tv_new->toString() << std::endl;
          tv_new->printTransforms();
        }
        ASSERT_FALSE(testing::Test::HasFailure()) << suffix;
      }
    }

    // Inspect definition. If it's a LoadStoreOp, check that the op type and
    // cache_op match
    EXPECT_EQ(
        tv_new->definition() == nullptr, tv_orig->definition() == nullptr);
    if (auto* lsop_orig = dynamic_cast<LoadStoreOp*>(tv_orig->definition())) {
      auto lsop_new = dynamic_cast<LoadStoreOp*>(tv_new->definition());
      ASSERT_TRUE(lsop_new != nullptr) << suffix;
      EXPECT_EQ(lsop_new->opType(), lsop_orig->opType());
      EXPECT_EQ(lsop_new->cacheOp(), lsop_orig->cacheOp());
    } else if (
        auto* rop_orig = dynamic_cast<ReductionOp*>(tv_orig->definition())) {
      auto rop_new = dynamic_cast<ReductionOp*>(tv_new->definition());
      EXPECT_EQ(
          rop_new->serialGridReductionRequested(),
          rop_orig->serialGridReductionRequested());
    }

    // Print suffix whenever a test fails before recursing
    ASSERT_FALSE(testing::Test::HasFailure()) << suffix;

    Expr* def_orig = tv_orig->definition();
    Expr* def_new = tv_new->definition();
    EXPECT_TRUE(def_orig->sameOp(def_new))
        << "def_orig\n  " << def_orig->toString()
        << "is not the same as def_new\n  " << def_new->toString() << suffix;
    EXPECT_EQ(def_orig->attributes().size(), def_new->attributes().size())
        << suffix;
    EXPECT_EQ(def_orig->inputs().size(), def_new->inputs().size()) << suffix;
    for (size_t i : c10::irange(def_orig->inputs().size())) {
      if (i >= def_new->inputs().size()) {
        break;
      }
      auto* tv_inp_orig = dynamic_cast<TensorView*>(def_orig->input(i));
      auto* tv_inp_new = dynamic_cast<TensorView*>(def_new->input(i));
      EXPECT_EQ(tv_inp_orig != nullptr, tv_inp_new != nullptr) << suffix;

      if (!testing::Test::HasFailure() && tv_inp_orig != nullptr &&
          !tv_inp_orig->isFusionInput()) {
        // Skip comparing input TVs since their schedules don't affect the
        // generated kernel
        EXPECT_FALSE(tv_inp_new->isFusionInput());
        compareTVs(tv_inp_orig, tv_inp_new);
      }
    }

    // Print suffix whenever a test fails in recursive call
    ASSERT_FALSE(testing::Test::HasFailure()) << suffix;
  }

  void compareSchedules() {
    // clone fusion for scheduling with original matmul scheduler
    Fusion new_fusion;
    cloner_ = std::make_unique<IrCloner>(Fusion::copy(fusion, &new_fusion));

    // Schedule fusion with original matmul scheduler
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(fusion, &mparams);

    // Schedule cloned fusion with new scheduler
    scheduleMultipleMatmuls(&new_fusion, &mparams);

    // Find tensors to compare. Note that these, and all producer tensors will
    // be checked.
    auto getTensorsToCompare = [](Fusion* fusion) {
      std::vector<TensorView*> tvs;

      // returning all TV outputs means we will check all TVs in the fusion due
      // to recursion
      for (Val* v : fusion->outputs()) {
        tvs.push_back(v->as<TensorView>());
      }

      return tvs;
    };
    std::vector<TensorView*> orig_compare_tvs = getTensorsToCompare(fusion);
    std::vector<TensorView*> new_compare_tvs = getTensorsToCompare(&new_fusion);

    // Compare each TensorView
    NVF_ERROR(new_compare_tvs.size() == orig_compare_tvs.size());
    for (size_t i : c10::irange(new_compare_tvs.size())) {
      compareTVs(orig_compare_tvs[i], new_compare_tvs[i]);
    }

    // If there are no errors up to this point, then check that the generated
    // kernels match
    if (!testing::Test::HasFailure()) {
      FusionExecutor fe_orig, fe_new;
      fe_orig.compileFusion(fusion);
      fe_new.compileFusion(&new_fusion);
      ASSERT_EQ(fe_new.kernelString(), fe_orig.kernelString());
    }
  }

 protected:
  MatmulParams mparams;
  Fusion* fusion = nullptr;
  bool a_m_inner = false, b_k_inner = false;
  int64_t vec_size_a, vec_size_b, vec_size_epilogue;
  bool smem_epilogue;
  int64_t splitk_factor;
  bool cta_order_col_major;
  int64_t grid_swizzle_factor;

 private:
  std::unique_ptr<Fusion> fusion_ptr_;
  std::unique_ptr<FusionGuard> fusion_guard_ptr_;

  // This is used to clone from original to new Fusion using the unscheduled
  // Fusion.
  std::unique_ptr<IrCloner> cloner_;
};

TEST_P(MultiMatmulSchedulerMatchTest, SimpleMatmul) {
  auto [tv0, tv1] = getInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);

  fusion->addOutput(tv2);
}

// In this example the inputs are already broadcasted to [M K N]
TEST_P(MultiMatmulSchedulerMatchTest, SimpleMatmulBroadcastedInputs) {
  auto [tv0, tv1] = getBroadcastInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);
}

TEST_P(MultiMatmulSchedulerMatchTest, TransposeToLinear) {
  auto [tv0, tv1] = getInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = transpose(tv1); // [N, K]
  auto tv3 = linear(tv0, tv2);

  fusion->addOutput(tv3);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulBias0d) {
  auto [tv0, tv1] = getInputTVs();

  auto tv2 = makeContigTensor(0, DataType::Half);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = transpose(tv1); // [N, K]
  auto tv4 = linear(tv0, tv3, tv2);

  fusion->addOutput(tv4);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulBias1d) {
  if (mparams.use_smem_epilogue && mparams.splitk_factor == 1) {
    GTEST_SKIP() << "Skipping case that does not compile with either scheduler."
                 << " See https://github.com/NVIDIA/Fuser/issues/2979";
  }

  auto [tv0, tv1] = getInputTVs();

  auto tv2 = makeContigTensor(1, DataType::Half);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = transpose(tv1); // [N, K]
  auto tv4 = linear(tv0, tv3, tv2);

  fusion->addOutput(tv4);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulFloatBias1d) {
  if (mparams.use_smem_epilogue && mparams.splitk_factor == 1) {
    GTEST_SKIP() << "Skipping case that does not compile with either scheduler."
                 << " See https://github.com/NVIDIA/Fuser/issues/2979";
  }

  auto [tv0, tv1] = getInputTVs();

  auto tv2 = makeContigTensor(1, DataType::Float);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = transpose(tv1); // [N, K]
  auto tv4 = linear(tv0, tv3);
  // The "linear" op requires bias to have same dtype as A and B, so instead we
  // add it manually ourselves here.
  auto tv5 = add(tv4, tv2);

  fusion->addOutput(tv5);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulBias2d) {
  auto [tv0, tv1] = getInputTVs();

  auto tv2 = makeContigTensor(2, DataType::Half);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = transpose(tv1); // [N, K]
  auto tv4 = linear(tv0, tv3, tv2);

  fusion->addOutput(tv4);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulSinEpilogue) {
  auto [tv0, tv1] = getInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);
  auto tv3 = sin(tv2);

  fusion->addOutput(tv3);
}

TEST_P(MultiMatmulSchedulerMatchTest, MatmulSinPrologue) {
  auto [tv0, tv1] = getInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = sin(tv1);
  auto tv3 = castOp(tv1->dtype(), tv2);
  auto tv4 = matmul(tv0, tv3);

  fusion->addOutput(tv4);
}

TEST_P(MultiMatmulSchedulerMatchTest, BatchMatmul) {
  auto [tv0, tv1] = getBatchInputTVs();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);

  fusion->addOutput(tv2);
}

// Copied from DistributedMatmulTest.MulSum_LayoutTN_NoComms
TEST_P(MultiMatmulSchedulerMatchTest, MultiDeviceMulSum) {
  int num_devices = 1;
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  int M = 256, N = 64, K = 64;
  int Mo = num_devices;
  int Mi = M / Mo;
  std::vector<int> a_shape = {Mo, Mi, K};
  std::vector<int> b_shape = {N, K};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
  TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
  TensorView* a_b = broadcast(a, {false, false, true, false}); // (Mo,Mi,b,K)
  TensorView* b_b = broadcast(b, {true, true, false, false}); // (b,b,N,K)
  TensorView* ab = mul(a_b, b_b); // (Mo,Mi,N,K)
  TensorView* c = sum(ab, {-1}); // (Mo,Mi,N,r)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, a_b, b_b, ab, c};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);
  // TODO: If c's allocation domain isn't set, it will fail validation at
  // csrc/device_lower/validation.cpp:419, Vectorized dim for consumer has to be
  // from a contiguous inner most position.
  c->setAllocationDomain(c->getLoopDomain(), true);
}

std::string printMatchTestParams(
    const testing::TestParamInfo<MultiMatmulSchedulerMatchTestParams>& info) {
  std::ostringstream os;
  os << "amin" << std::get<0>(info.param);
  os << "bkin" << std::get<1>(info.param);
  os << "va" << std::get<2>(info.param);
  os << "vb" << std::get<3>(info.param);
  os << "vepi" << std::get<4>(info.param);
  os << "smep" << std::get<5>(info.param);
  os << "sk" << std::get<6>(info.param);
  os << "col" << std::get<7>(info.param);
  os << "swiz" << std::get<8>(info.param);
  return os.str();
}

// Test combinations that mostly affect operand loading
INSTANTIATE_TEST_SUITE_P(
    Operands,
    MultiMatmulSchedulerMatchTest,
    testing::Combine(
        testing::Bool(), // a_m_inner
        testing::Bool(), // b_k_inner
        testing::Values(8, 2), // vec_size_a
        testing::Values(8, 2), // vec_size_b
        testing::Values(4), // vec_size_epilogue
        testing::Values(false), // use_smem_epilogue
        testing::Values(1, 2), // splitk_factor
        testing::Bool(), // cta_order_col_major
        testing::Values(1, 2) // grid_swizzle_factor
        ),
    printMatchTestParams);

// Test combinations that mostly affect epilogue
INSTANTIATE_TEST_SUITE_P(
    Epilogue,
    MultiMatmulSchedulerMatchTest,
    testing::Combine(
        testing::Values(false), // a_m_inner
        testing::Values(false), // b_k_inner
        testing::Values(8), // vec_size_a
        testing::Values(8), // vec_size_b
        testing::Values(4, 2), // vec_size_epilogue
        testing::Bool(), // use_smem_epilogue
        testing::Values(1, 2), // splitk_factor
        testing::Values(false), // cta_order_col_major
        testing::Values(1) // grid_swizzle_factor
        ),
    printMatchTestParams);

} // namespace nvfuser
