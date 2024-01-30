// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <kernel_cache.h>
#include <ir/interface_nodes.h>
#include <fusion.h>
#include <test/utils.h>
#include <test/validator.h>
#include <ops/all_ops.h>

namespace nvfuser {

using PointwiseTest = NVFuserTest;

namespace {

size_t getVecSizeForPointwise(FusionExecutorCache& fec) {
  auto most_recent_params =
      fec.getMostRecentKernelRuntime()->getMostRecentExecutorLog().params;
  const auto* params = dynamic_cast<PointwiseParams*>(most_recent_params.get());
  NVF_ERROR(
      params != nullptr,
      "`fec`'s contained fusion didn't trigger the pointwise scheduler.");
  if (params->vectorize) {
    return params->unroll_factor;
  }
  return 1;
}

bool hasVectorization(TensorView* tv) {
  for (const auto* id : tv->getLeafDomain()) {
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
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(fec), (size_t)vec);

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
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(fec), (size_t)vec);

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
  fec.profile(true);

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

    EXPECT_EQ(getVecSizeForPointwise(fec), (size_t)vec);

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
  fec.profile(true);

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
    EXPECT_EQ(getVecSizeForPointwise(fec), (size_t)vec);
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
  fec.profile(true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({1024, 128, 25}, {128*25, 1, 128}, options);
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
  at::Tensor t0 = at::empty_strided({1024, 128, 25}, {128*25, 1, 128}, options);
  at::Tensor t1 = at::empty_strided({1, 128, 1}, {128, 1, 128}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  // NOTE: force pointwise scheduler here just for testing purpose
  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_EQ(params->vectorize, true);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_TRUE(hasVectorization(tv0));
  EXPECT_TRUE(hasVectorization(tv1));
  EXPECT_TRUE(hasVectorization(tv2));

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
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .build();
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

  EXPECT_EQ(params->vectorize, true);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_FALSE(hasVectorization(tv0));
  EXPECT_TRUE(hasVectorization(tv1));
  EXPECT_TRUE(hasVectorization(tv2));

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
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .build();
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

  EXPECT_EQ(params->vectorize, true);
  EXPECT_EQ(params->unroll_factor, 2);
  EXPECT_TRUE(hasVectorization(tv0));
  EXPECT_TRUE(hasVectorization(tv1));
  EXPECT_TRUE(hasVectorization(tv2));

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
  fec.profile(true);

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

  EXPECT_EQ(params->vectorize, true);
  EXPECT_EQ(params->unroll_factor, 4);
  EXPECT_TRUE(hasVectorization(tv0));
  EXPECT_TRUE(hasVectorization(tv1));
  EXPECT_TRUE(hasVectorization(tv2));

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
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

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

  EXPECT_EQ(params->vectorize, true);
  EXPECT_EQ(params->unroll_factor, 2);
  EXPECT_TRUE(hasVectorization(tv0));
  EXPECT_TRUE(hasVectorization(tv1));
  EXPECT_TRUE(hasVectorization(tv2));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
