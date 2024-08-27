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
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using PointwiseTest = NVFuserTest;

namespace {

int64_t getVecSizeForPointwise(const FusionExecutorCache& fec) {
  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented());
  const PointwiseParams& params =
      runtime->schedulerHeuristics()->heuristicsList().at(0)->pointwiseParams();
  if (!params.vectorize) {
    return 1;
  }
  return params.unroll_factor;
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

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(fec), vec);

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

  FusionExecutorCache fec(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(fec), vec);

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

  FusionExecutorCache fec(std::move(fusion_ptr));

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
    auto cg_outputs = fec.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(fec), vec);

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

  FusionExecutorCache fec(std::move(fusion_ptr));

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
    for (auto i : c10::irange(shape.size())) {
      alloc_size += (shape.at(i) - 1) * stride.at(i);
    }
    alloc_size += align;
    at::Tensor flat = at::randn({alloc_size}, options);
    at::Tensor t0 = flat.as_strided(shape, stride, /*storage_offset=*/align);
    auto cg_outputs = fec.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(fec), vec);
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

  FusionExecutorCache fec(std::move(fusion_ptr));

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
    auto cg_outputs = fec.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(fec), vec);
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

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::empty_strided({1024, 128, 25}, {128 * 25, 1, 128}, options);
  auto cg_outputs = fec.runFusionWithInputs({t0});
  EXPECT_EQ(getVecSizeForPointwise(fec), 4);
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
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
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
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_FALSE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
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
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
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

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 1, 2}, options);
  at::Tensor t1 = at::empty_strided({1024, 512, 2}, {2, 2048, 1}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
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

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1024, 2}, options);
  at::Tensor t1 = at::randn({512, 1024, 2}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
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
    Fusion sharded_fusion = createPointwiseFusion(true, sharded_dim);
    Fusion unsharded_fusion = createPointwiseFusion(false);
    at::Tensor t0 = at::randn(input_size, options);
    at::Tensor t1 = at::randn({input_size[1], input_size[2]}, options);
    std::vector<c10::IValue> sharded_inputs = {t0.unsqueeze(sharded_dim), t1};
    auto params = getPointwiseHeuristics(&sharded_fusion, sharded_inputs);
    auto unsharded_params = getPointwiseHeuristics(&unsharded_fusion, {t0, t1});
    // Note: occasionally one of the compile parameter index types is int64_t
    // instead of int which causes PointwiseParams::sameAs to return false,
    // despite the pointwise specific parameters being identical, so we just
    // explicitly check pointwise schedule params.
    EXPECT_EQ(params->vectorize, unsharded_params->vectorize);
    EXPECT_EQ(params->break_point, unsharded_params->break_point);
    EXPECT_EQ(params->split_block, unsharded_params->split_block);
    EXPECT_EQ(params->split_grid_y_dim, unsharded_params->split_grid_y_dim);
    EXPECT_EQ(params->unroll_factor, unsharded_params->unroll_factor);
    EXPECT_EQ(params->flip_grid_binding, unsharded_params->flip_grid_binding);

    auto lparams = schedulePointwise(&sharded_fusion, sharded_inputs);
    FusionExecutor fe;
    fe.compileFusion(&sharded_fusion, sharded_inputs, lparams);
    auto cg_outputs = fe.runFusion(sharded_inputs, lparams);
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
  at::manual_seed(0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

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
  at::manual_seed(0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
}

} // namespace nvfuser
