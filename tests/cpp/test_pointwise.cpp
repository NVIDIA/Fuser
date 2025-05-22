// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/domain_map.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using PointwiseTest = NVFuserTest;

namespace {

int64_t getVecSizeForPointwise(const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented());
  const PointwiseParams* params = runtime->schedulerHeuristics()
                                      ->heuristicsList()
                                      .at(0)
                                      ->as<PointwiseParams>();
  return params->vectorization_factor;
}

bool hasVectorizationCache(TensorView* tv) {
  NVF_CHECK(tv->isFusionInput());
  NVF_CHECK(tv->uses().size() == 1);
  auto set_expr = dynamic_cast<LoadStoreOp*>(tv->uses().at(0));
  NVF_CHECK(set_expr != nullptr && set_expr->opType() == LoadStoreOpType::Set);
  auto cached_input = set_expr->out()->as<TensorView>();
  NVF_CHECK(cached_input, "expects input to be cached");

  for (const auto* id : cached_input->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      return true;
    }
  }
  return false;
}

class DomainMapUnitTest : public scheduler_tools::DomainMap {
 public:
  DomainMapUnitTest(Fusion* fusion) : scheduler_tools::DomainMap(fusion) {};
  bool testTargetCoverage(TensorView* target_tv, TensorView* reference_tv)
      const {
    return areAllTargetIdsCoveredBy(target_tv, reference_tv);
  }
};

} // namespace

TEST_F(PointwiseTest, VectorizeStrideContiguity2D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(2).contiguity({false, true}).build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguity3D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(3).contiguity({false, true, true}).build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguity5D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int>> sizes_and_vec{
      {9, 17, 1}, {9, 10, 2}, {9, 16, 4}};

  for (auto tup : sizes_and_vec) {
    auto size1 = std::get<0>(tup);
    auto size2 = std::get<1>(tup);
    auto vec = std::get<2>(tup);
    at::Tensor t0 = at::randn({4, size1, 12345, size2, 3}, options)
                        .narrow(1, 0, 8)
                        .narrow(3, 0, 4);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

// Test that vectorization is properly computed when base pointer is not aligned
// at 16 bytes. This can happen if a tensor is sliced then passed as input.
// See https://github.com/NVIDIA/Fuser/pull/2118
TEST_F(PointwiseTest, VectorizeStrideMisalignedBase) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int, int>> sizes_strides_align_and_vec{
      {4, 4, 4, 4, 4},
      {4, 4, 4, 2, 2},
      {4, 4, 4, 1, 1},
      {4, 4, 2, 4, 2},
      {4, 4, 2, 1, 1},
      {4, 2, 4, 4, 2},
      {4, 2, 4, 1, 1},
      {2, 4, 4, 4, 2},
      {2, 4, 4, 1, 1},
      {2, 2, 2, 4, 2},
      {2, 2, 2, 1, 1}};

  for (auto tup : sizes_strides_align_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto align = std::get<3>(tup);
    auto vec = std::get<4>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {
        stride1, (int64_t)stride2 * 12345, (int64_t)stride2, 3, 1};
    // Create a strided input that is misaligned by "align" elements
    //  First, find required size of align=0 tensor. Allocate this much plus
    //  align elements. Then slice and view as aligned tensor.
    int64_t alloc_size = 1l;
    for (auto i : arange(shape.size())) {
      alloc_size += (shape.at(i) - 1) * stride.at(i);
    }
    alloc_size += align;
    at::Tensor flat = at::randn({alloc_size}, options);
    at::Tensor t0 = flat.as_strided(shape, stride, /*storage_offset=*/align);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);
    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguitySelfOverlapping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int>> sizes_strides_and_vec{
      {4, 4, 4, 4},
      {4, 4, 2, 2},
      {4, 2, 4, 2},
      {2, 4, 4, 2},
      {4, 4, 1, 1},
      {4, 1, 4, 1},
      {1, 4, 4, 1},
      {2, 2, 2, 2},
      {2, 2, 1, 1},
      {2, 1, 2, 1},
      {1, 2, 2, 1}};

  for (auto tup : sizes_strides_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto vec = std::get<3>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {
        stride1, (int64_t)stride2 * 12345, (int64_t)stride2, 3, 1};
    at::Tensor t0 = at::empty_strided(shape, stride, options);
    t0.random_();
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);
    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeAllocationDomain) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({2, 0, 1})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0, DataType::Float));
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(2), tv1->axis(1)}, true);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::empty_strided({1024, 128, 25}, {128 * 25, 1, 128}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// All inputs & outputs share the same allocation domain permutation from root
// domain, but intermediate tv2 isn't specified a stride order. There's also a
// broadcast IterDomain on tv1, which is tricky for vectorization analysis to
// figure out which axes should be excluded from the computation of
// vectorization factor.
TEST_F(PointwiseTest, Issue1567VectorizeAllocationDomain) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({2, 0, 1})
                        .build();
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .shape({1, -1, 1})
                        .contiguity({std::nullopt, std::nullopt, true})
                        .strideOrder({2, 0, 1})
                        .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0, DataType::Float));
  tv3->setAllocationDomain({tv3->axis(0), tv3->axis(2), tv3->axis(1)}, true);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::empty_strided({1024, 128, 25}, {128 * 25, 1, 128}, options);
  at::Tensor t1 = at::empty_strided({1, 128, 1}, {128, 1, 128}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase0) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, std::nullopt})
                        .shape({-1, -1, 1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 2, 1}, options);
  at::Tensor t1 = at::randn({1024, 2, 512}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results =
      scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1}, false);
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_FALSE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, std::nullopt, true})
                        .shape({-1, 1, -1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 1, 2}, options);
  at::Tensor t1 = at::randn({1024, 512, 2}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, std::nullopt, true})
                        .shape({-1, 1, -1})
                        .build();
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({1, 2, 0})
                        .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 1, 2}, options);
  at::Tensor t1 = at::empty_strided({1024, 512, 2}, {2, 2048, 1}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, VIssue1567ectorizationFactorAnalysisCase3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({std::nullopt, true, true})
                        .shape({1, -1, -1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1024, 2}, options);
  at::Tensor t1 = at::randn({512, 1024, 2}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

namespace {
Fusion createPointwiseFusion(bool shard, int sharded_dim = -1) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // Sharded fusion needs to add an additional sharded axis.
  TensorView* tv0 = makeContigTensor(shard ? 4 : 3);
  TensorView* tv1 = makeContigTensor(2);
  auto tv2 = add(tv0, tv0);
  std::vector<bool> bcast_mask;
  if (shard) {
    bcast_mask = {false, true, false, false};
    bcast_mask[sharded_dim] = true;
  } else {
    bcast_mask = {true, false, false};
  }
  TensorView* tv3 = broadcast(tv1, bcast_mask);
  TensorView* tv4 = add(tv2, tv3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv4);

  if (shard) {
    DeviceMesh mesh = DeviceMesh::createForNumDevices(4);
    for (TensorView* tv : {tv0, tv2, tv3, tv4}) {
      tv->setDeviceMesh(mesh);
      tv->axis(sharded_dim)->parallelize(ParallelType::DIDx);
    }
    tv1->setDeviceMesh(mesh);
  }
  return fusion;
}
} // namespace

// Check that (1) a sharded pointwise fusion returns the same
// pointwise scheduling parameters as its equivalent
// unsharded fusion and (2) the output is correct.
TEST_F(PointwiseTest, ShardedPointwise) {
  int64_t sharded_dim = 0;
  std::vector<std::vector<int64_t>> input_sizes = {
      {16, 8, 48},
      {2, 512, 4096},
      {2048, 512, 16},
      {65536, 512, 16},
      {512, 3, 65536},
  };
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  for (auto input_size : input_sizes) {
    at::Tensor t0 = at::randn(input_size, options);
    at::Tensor t1 = at::randn({input_size[1], input_size[2]}, options);

    KernelArgumentHolder sharded_inputs = {t0.unsqueeze(sharded_dim), t1};

    auto pwise_scheduler =
        SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);

    Fusion sharded_fusion = createPointwiseFusion(true, sharded_dim);
    SchedulerRuntimeInfo sharded_runtime_info(&sharded_fusion, sharded_inputs);
    auto sharded_params = pwise_scheduler->computeHeuristics(
        &sharded_fusion, sharded_runtime_info);
    auto sharded_pparams = sharded_params->as<PointwiseParams>();

    Fusion unsharded_fusion = createPointwiseFusion(false);
    SchedulerRuntimeInfo unsharded_runtime_info(&unsharded_fusion, {t0, t1});
    auto unsharded_params = pwise_scheduler->computeHeuristics(
        &unsharded_fusion, unsharded_runtime_info);
    auto unsharded_pparams = unsharded_params->as<PointwiseParams>();

    // Note: occasionally one of the compile parameter index types is int64_t
    // instead of int which causes PointwiseParams::sameAs to return false,
    // despite the pointwise specific parameters being identical, so we just
    // explicitly check pointwise schedule params.
    EXPECT_EQ(sharded_pparams->break_point, unsharded_pparams->break_point);
    EXPECT_EQ(sharded_pparams->split_block, unsharded_pparams->split_block);
    EXPECT_EQ(
        sharded_pparams->split_grid_y_dim, unsharded_pparams->split_grid_y_dim);
    EXPECT_EQ(
        sharded_pparams->vectorization_factor,
        unsharded_pparams->vectorization_factor);
    EXPECT_EQ(
        sharded_pparams->flip_grid_binding,
        unsharded_pparams->flip_grid_binding);

    pwise_scheduler->schedule(&sharded_fusion, sharded_params.get());
    KernelExecutor ke;
    ke.compile(&sharded_fusion, sharded_inputs, sharded_params->lparams);
    auto cg_outputs = ke.run(sharded_inputs, {}, sharded_params->lparams);
    testValidate(
        &sharded_fusion, cg_outputs, sharded_inputs, __LINE__, __FILE__);
  }
}

// Repro of issue #657
TEST_F(PointwiseTest, VectorizeWithBroadcastAndReshape1) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Sizes don't matter as long as they are large enough to trigger
  // vectorization
  std::vector<int64_t> shape1{1024, 1024};
  std::vector<int64_t> shape2{1024, 1024, 4};
  std::vector<int64_t> shape3{1024 * 1024 * 4};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv0);

  auto tv1 = makeContigConcreteTensor(shape2);
  fusion->addInput(tv1);

  auto tv2 = broadcast(tv0, {false, false, true});
  fusion->addOutput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = reshape(tv3, shape2, shape3);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
}

// Repro of issue #657
TEST_F(PointwiseTest, VectorizeWithBroadcastAndReshape2) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Sizes don't matter as long as they are large enough to trigger
  // vectorization
  std::vector<int64_t> shape1{1024, 1024};
  std::vector<int64_t> shape2{1024, 1024, 4};
  std::vector<int64_t> shape3{1024 * 1024 * 4};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv0);

  auto tv1 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv1);

  auto tv2 = makeContigConcreteTensor(shape2);
  fusion->addInput(tv2);

  auto tv3 = broadcast(tv0, {false, false, true});
  fusion->addOutput(tv3);

  auto tv4 = add(tv3, tv2);

  auto tv5 = broadcast(tv1, {false, false, true});

  auto tv6 = add(tv4, tv5);

  auto tv7 = reshape(tv6, shape2, shape3);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
}

TEST_F(PointwiseTest, VectorizeWithExpandedBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t kTensorSize = 65536;
  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Half)
                       .shape({2, kTensorSize})
                       .expanded({true, false})
                       .build();
  in->setAllocationDomain({in->axis(1), in->axis(0)}, true);
  TensorView* out = add(in, in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto in_tensor =
      at::randn({kTensorSize}, options).as_strided({2, kTensorSize}, {0, 1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_GT(getVecSizeForPointwise(executor_cache), 1);
}

using VectUnrollFactors = std::tuple<int64_t, int64_t, int64_t>;
using PointwiseParamsTest = NVFuserFixtureParamTest<VectUnrollFactors>;
TEST_P(PointwiseParamsTest, UnrollOnTopOfVectorize) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  fusion->addOutput(tv3);

  int dim0 = 1024;
  int dim1 = 2048;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim1}, options);

  // Generate heuristics
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0, t1});
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion.get(), runtime_info);
  auto pparams = heuristic_params->as<PointwiseParams>();

  // Modify heuristics to enforce unroll on top of vectorization

  // Set unroll factors from test parameters
  auto [vect_factor, unroll_inner, unroll_outer] = GetParam();
  pparams->unroll_factor_inner = unroll_inner;
  pparams->unroll_factor_outer = unroll_outer;
  pparams->vectorization_factor = vect_factor;

  // Schedule, compile, run, validate
  scheduler_instance->schedule(fusion.get(), pparams);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, pparams->lparams);
  auto cg_outputs = ke.run({t0, t1}, {}, pparams->lparams);
  const auto& lparams = ke.lastLaunchParams();
  ASSERT_EQ(lparams.gdimy(), dim0 / unroll_outer);
  ASSERT_EQ(
      lparams.gdimx(), dim1 / vect_factor / lparams.bdimx() / unroll_inner);
  testValidate(fusion.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    PointwiseParamsTest,
    ::testing::Combine(
        testing::Values(1, 4), // vectorization factors
        testing::Values(1, 2), // inner unroll factors
        testing::Values(1, 2) // outer unroll factors
        ),
    [](const testing::TestParamInfo<VectUnrollFactors>& info) -> std::string {
      std::stringstream ss;
      ss << "vect_" << std::get<0>(info.param);
      ss << "_inner_unroll_" << std::get<1>(info.param);
      ss << "_outer_unroll_" << std::get<2>(info.param);
      return sanitizeTestName(ss.str());
    });

namespace {
int64_t getUnrollFactor(int64_t n_inputs_factor, int64_t computation_factor) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t required_bytes_per_thread =
      scheduler_utils::getRequiredBytesInFlight() /
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  constexpr int64_t vect_bytes = 16L;
  int64_t unroll_factor = std::max(1L, required_bytes_per_thread / vect_bytes);
  if (unroll_factor > 1) {
    unroll_factor *= computation_factor;
    unroll_factor /= n_inputs_factor;
  }
  return unroll_factor;
}

} // namespace

// Test pointwise heuristics.
// current heuristics does fully unroll when have more than 8 waves
// of blocks. For device with high bandwidh, unroll factor is 2 when
// there is only one input tensor, scaled up by computation factor and
// scaled down by number of input tensors.
TEST_F(PointwiseTest, Heuristicst1Compute1Unroll2) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(1, 1);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);

  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_inner, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Heuristicst1Compute2Unroll4) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(1, 2);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // outer bcast tv is not counted
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  auto tv4 = exp(tv3);
  auto tv5 = reciprocal(tv4);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim1}, options);

  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_outer, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, HeuristicsInput2Compute4Unroll4) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(2, 4);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = tanh(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0, dim1}, options);
  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_inner, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, VectorizePadLoweringPermuted) {
  // Pointwise scheduler applies permutation to restore contiguous memory access
  // on reference TV. Vectorization validation requires vectorized operations to
  // preserve the allocation domain of their inputs. This test checks that PadOp
  // propagates the allocation domain properly.
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  // input is permuted
  auto tv0 = TensorViewBuilder()
                 .shape({1024, 1024})
                 .dtype(DataType::Float)
                 .contiguity(true)
                 .strideOrder({0, 1})
                 .build();
  fusion.addInput(tv0);
  auto tv1 = pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(4L)});
  auto tv2 = relu(tv1);
  fusion.addOutput(tv2);
  // output is permuted
  tv2->setAllocationDomain({tv2->axis(1), tv2->axis(0)}, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 =
      at::randn({1024 * 1024}, options).as_strided({1024, 1024}, {1, 1024});

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {t0}).outputs;
  // check that we vectorize 4
  bool found_vectorize = false;
  for (auto id : fusion.outputs().at(0)->as<TensorView>()->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      EXPECT_EQ(id->extent()->evaluate(), 4);
      found_vectorize = true;
      break;
    }
  }
  EXPECT_TRUE(found_vectorize);
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg0) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);
  // tv2 {i0, b2, i1}
  auto tv2 = broadcast(tv1, {false, true, false});
  // tv3 {i0, b3{1 ex 4}, i1}
  auto tv3 = expand(
      tv2,
      {tv2->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       tv2->axis(2)->extent()});
  // NOTE hat currently expand doesn't introduce an iter domain operation, so
  // we don't see that i4 is produced by realizing the expanded extent of b3{1
  // ex 4} tv4 {i0, i4*i1}
  auto tv4 = reshape(tv3, {2, 4, 3}, {2, 12});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv4 is not covered by tv1, because the expanded ID i4 participates in
  // transformation
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv1));

  // tv3 is not covered by tv1, because the missing ID b3{1 ex 4} is concretized
  // as i4, which is not mapped on tv1
  EXPECT_FALSE(domain_map.testTargetCoverage(tv3, tv1));

  // tv1 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv1, tv4));

  // tv1 is not a valid reference
  EXPECT_FALSE(domain_map.isValidReference(tv1));

  // tv4 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 7}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0});
  testValidate(fusion, cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i2, i0, i1}
  TensorView* tv1 = makeContigTensor(3);
  fusion->addInput(tv1);
  // tv2 {i0*i1}
  auto tv2 = reshape(tv0, {2, 4}, {8});
  fusion->addOutput(tv2);

  // tv3 {b3, i0, i1}
  auto tv3 = broadcast(tv0, {true, false, false});
  // tv4 {i2, i0, i1}
  auto tv4 = add(tv1, tv3);
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv4 is not covered by tv2, because it misses i2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv2));

  // tv2 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  // tv2 is not a valid reference
  EXPECT_FALSE(domain_map.isValidReference(tv2));

  // tv4 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({3, 2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);
  // tv2 {i0, b2, i1}
  auto tv2 = broadcast(tv1, {false, true, false});
  // tv3 {i0, b3{1 ex 4}, i1}
  auto tv3 = expand(
      tv2,
      {tv2->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       tv2->axis(2)->extent()});
  fusion->addOutput(tv3);

  DomainMapUnitTest domain_map(fusion);
  // tv3 is covered by tv1, because the missing ID b3{1 ex 4} is broadcast and
  // doesn't get resolved to a concrete broadcast ID.
  EXPECT_TRUE(domain_map.testTargetCoverage(tv3, tv1));

  // tv1 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv1, tv3));

  // tv1 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv1));

  // tv3 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv3));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 7}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0});
  testValidate(fusion, cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapFactory) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv1 {i1}
  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv1);

  // tv2 {b2, b3, i1}
  auto tv2 = broadcast(tv0, {true, true, false});
  // NOTE tv1 will be broadcasted to {b2, i0, i1} before the add.
  // tv3 {b2, i0, i1}
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto size_val = IrBuilder::create<Val>(4.0, DataType::Int);
  auto one_val = IrBuilder::create<Val>(1, DataType::Int);
  // factory method creates an iter domain out of thin air
  // tv4 {i4{4}, b4, i1}
  auto tv4 = ones({size_val, one_val, tv0->axis(0)->extent()}, DataType::Float);
  // tv5 {i4{4}, i0, i1}
  auto tv5 = mul(tv2, tv4);
  fusion->addOutput(tv5);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is not covered by tv3, because it's missing i4{4}
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv3));
  // tv1 is not covered by tv4, since it's missing i0
  EXPECT_FALSE(domain_map.testTargetCoverage(tv1, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv3));
  // tv5 has the same IDs as tv4, and is not a valid reference.
  EXPECT_FALSE(domain_map.isValidReference(tv5));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({25}, {1}, options);
  at::Tensor t1 = at::empty_strided({7, 25}, {25, 1}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  SegmentedFusion* segmented_fusion = runtime->fusionSegments();
  // This fusion currently cannot be scheduled as a single kernel. It is
  // expected to be segmented as: g{(pointwise)
  //   inputs: tv0, tv1
  //   outputs: tv2, tv3
  //   tv2 = broadcast(tv0)
  //   tv3 = add (tv2, broadcast(tv1))
  // }
  //
  // g{(pointwise)
  //   inputs: tv2
  //   outputs: tv5
  //   tv4 = full({4, 1, i0})
  //   tv5 = mul(tv2, tv4)
  // }
  EXPECT_EQ(segmented_fusion->groups().size(), 2);

  for (SegmentedGroup* group : segmented_fusion->groups()) {
    const std::vector<Expr*>& exprs = group->exprs();

    size_t num_full = std::count_if(exprs.begin(), exprs.end(), [](Expr* expr) {
      return expr->isA<FullOp>();
    });
    if (num_full != 0) {
      // this is the segment contains the factory op.
      EXPECT_EQ(exprs.size(), 2);
      EXPECT_EQ(num_full, 1);
      auto binary_op_iter =
          std::find_if(exprs.begin(), exprs.end(), [](Expr* expr) {
            return expr->isA<BinaryOp>();
          });
      EXPECT_EQ(
          (*binary_op_iter)->as<BinaryOp>()->getBinaryOpType(),
          BinaryOpType::Mul);
      Fusion* group_fusion = group->getFusion();
      // validate that we have a valid reference in the segmented fusion
      DomainMapUnitTest group_dm(group_fusion);
      EXPECT_EQ(group_fusion->outputs().size(), 1);
      EXPECT_TRUE(group_dm.isValidReference(
          group_fusion->outputs()[0]->as<TensorView>()));
    } else {
      // validate segmentation has the correct ops
      EXPECT_EQ(exprs.size(), 3);
      EXPECT_EQ(
          std::count_if(
              exprs.begin(),
              exprs.end(),
              [](Expr* expr) { return expr->isA<BroadcastOp>(); }),
          2);
      EXPECT_EQ(
          std::count_if(
              exprs.begin(),
              exprs.end(),
              [](Expr* expr) { return expr->isA<BinaryOp>(); }),
          1);
      Fusion* group_fusion = group->getFusion();
      auto output_add = std::find_if(
          group_fusion->outputs().begin(),
          group_fusion->outputs().end(),
          [](Val* val) { return val->definition()->isA<BinaryOp>(); });
      EXPECT_TRUE(output_add != group_fusion->outputs().end());
      DomainMapUnitTest group_dm(group_fusion);
      // validate that the segmented fusion choose the add output as the
      // reference
      EXPECT_TRUE(group_dm.isValidReference((*output_add)->as<TensorView>()));
    }
  }

  testValidate(fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapPad0) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {b1, i0}
  TensorView* tv0 = TensorViewBuilder().shape({1, -1}).build();
  fusion->addInput(tv0);
  // tv1 {i2, b1, i0}
  TensorView* tv1 = TensorViewBuilder().shape({-1, 1, -1}).build();
  fusion->addInput(tv1);
  // tv2 {i2, b1, i0}
  auto tv2 = add(tv1, tv0);
  fusion->addOutput(tv2);
  // i3 = resize(b1 + 4 + 4)
  // tv3 {i3, i0}
  auto tv3 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  // tv4 {i3*i0}
  auto tv4 = reshape(tv3, {9, 5}, {45});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is covered by tv2, because i3 is produced by b1
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv2));
  // tv2 is not covered by tv4, it's missing i2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv4));
  EXPECT_TRUE(domain_map.isValidReference(tv2));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({1, 5}, {5, 1}, options);
  at::Tensor t1 = at::empty_strided({7, 1, 5}, {5, 5, 1}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapPad1) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {b1, i0}
  TensorView* tv0 = TensorViewBuilder().shape({1, -1}).build();
  fusion->addInput(tv0);
  // tv1 {i2, i3, i4, b5}
  TensorView* tv1 = TensorViewBuilder().shape({-1, -1, -1, 1}).build();
  fusion->addInput(tv1);

  // tv2 {b6, b7, b1, i0}
  auto tv2 = broadcast(tv0, {true, true, false, false});
  // tv3 {i2, i3, i4, i0}
  auto tv3 = add(tv1, tv2);
  fusion->addOutput(tv3);
  // i8 = resize(b1 + 4 + 4)
  // tv4 {i8, i0}
  auto tv4 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is covered by tv3, because i8 is produced by b1, a broadcast dimension
  // concretized as i4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv3));
  // tv3 is not covered by tv4, it's missing i2 and i3
  EXPECT_FALSE(domain_map.testTargetCoverage(tv3, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv4));
  EXPECT_TRUE(domain_map.isValidReference(tv3));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({1, 5}, {5, 1}, options);
  at::Tensor t1 = at::empty_strided({2, 3, 4, 1}, {12, 4, 1, 1}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapSlice0) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i1, i0}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i1, i0}
  // use concrete tensor to avoid need of concretization
  TensorView* tv1 = makeContigConcreteTensor({2, 4});
  fusion->addInput(tv1);

  // b3 = resize(i0 + 0 - 3)
  // tv2 {i1, b2}
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv2);
  // tv3 {i1, i0}
  auto tv3 = add(tv0, tv1);
  // tv4 {i1*i0}
  auto tv4 = reshape(tv3, {2, 4}, {8});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv2 and tv4 has the same source IDs, since b3 = resize(i0 + 0 - 3)
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv2));
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_TRUE(domain_map.isValidReference(tv2));
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapSlice1) {
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i2, i1, i0}
  TensorView* tv0 = makeContigTensor(3);
  fusion->addInput(tv0);
  // tv1 {i1, i0}
  // use concrete tensor to avoid need of concretization
  TensorView* tv1 = makeContigConcreteTensor({2, 4});
  fusion->addInput(tv1);

  // b3 = resize(i0 + 0 - 3)
  // tv2 {i1, b3}
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv2);
  // tv3 {i2, i1, i0}
  auto tv3 = add(tv0, tv1);
  // tv4 {i2, i1*i0}
  auto tv4 = reshape(tv3, {2, 2, 4}, {2, 8});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // i2 is missing in tv2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv2));
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv2));
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 4}, options);
  at::Tensor t1 = at::randn({2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, DomainMapBroadcastIssue3653) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({2, 4, 8});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({2});
  fusion.addInput(tv1);

  auto tv2 = reshape(tv0, {2, 4, 8}, {2, 32});
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = add(tv2, tv3);

  // tv4 covers source IDs {2, 4, 8}.
  fusion.addOutput(tv4);
  // meanwhile, tv3's broadcast ID map through permissive to `32`, which is not
  // directly contained by tv4's source IDs. This test ensures that we project
  // the mapped ID back to its source IDs and correctly schedule this fusion as
  // a single kernel.
  fusion.addOutput(tv3);

  DomainMapUnitTest domain_map(fusion_ptr.get());
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4, 8}, options);
  auto t1 = at::randn({2}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented());

  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, InnerDimAllocationTransformationOnProducer) {
  for (bool inner_split : {true, false}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* in = makeContigTensor(2);
    in->split(1, 4, inner_split); // outer split
    in->setAllocationDomain(in->getLoopDomain(), true);
    TensorView* out = cos(in);
    fusion->addInput(in);
    fusion->addOutput(out);
    EXPECT_EQ(
        scheduler_utils::getInputsOutputsWithInnerDim(out, true, false).size(),
        2);
  }
}

TEST_F(PointwiseTest, InnerDimAllocationTransformationOnConsumer) {
  for (bool inner_split : {true, false}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* in = makeContigTensor(2);
    TensorView* out = cos(in);
    out->split(1, 4, inner_split);
    out->setAllocationDomain(out->getLoopDomain(), true);
    fusion->addInput(in);
    fusion->addOutput(out);

    // input and output have mapping inner dim.
    EXPECT_EQ(
        scheduler_utils::getInputsOutputsWithInnerDim(out, true, false).size(),
        2);
  }
}

TEST_F(
    PointwiseTest,
    InnerDimAllocationTransformationOnBothConsumerAndProducer) {
  for (bool in_inner_split : {true, false}) {
    for (bool out_inner_split : {true, false}) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());
      TensorView* in = makeContigTensor(2);
      in->split(1, 4, in_inner_split);
      in->setAllocationDomain(in->getLoopDomain(), true);
      TensorView* out = cos(in);
      out->split(1, 4, out_inner_split);
      out->setAllocationDomain(out->getLoopDomain(), true);
      fusion->addInput(in);
      fusion->addOutput(out);

      // input and output have mapping inner dim.
      EXPECT_EQ(
          scheduler_utils::getInputsOutputsWithInnerDim(out, true, false)
              .size(),
          2);
    }
  }
}

} // namespace nvfuser
