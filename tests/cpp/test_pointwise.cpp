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
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

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

bool hasVectorizationCache(TensorView* tv) {
  NVF_CHECK(tv->isFusionInput());
  NVF_CHECK(tv->uses().size() == 1);
  auto set_expr = dynamic_cast<LoadStoreOp*>(tv->uses().at(0));
  NVF_CHECK(set_expr != nullptr && set_expr->opType() == LoadStoreOpType::Set);
  auto cached_input = set_expr->out()->as<TensorView>();
  NVF_CHECK(cached_input, "expects input to be cached");

  for (const auto* id : cached_input->getLeafDomain()) {
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
  fec.profile(true);

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

  EXPECT_TRUE(params->vectorize);
  EXPECT_EQ(params->unroll_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// params: use 1D scheduler, sharded dimenson
class ShardedPointwiseTest
    : public NVFuserTest,
      public testing::WithParamInterface<std::tuple<bool, int>> {};
   
TEST_P(ShardedPointwiseTest, DID_Compatible) {
  auto [use_1D_scheduler, sharded_dim] = GetParam();
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeContigTensor(4);
  TensorView* tv1 = makeContigTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv0);
  auto tv3 = broadcast(tv1, {true, true, false, false});
  auto tv4 = add(tv2, tv3);
  fusion->addOutput(tv4);

  DeviceMesh mesh = DeviceMesh::createForNumDevices(4);
  for (TensorView* tv : {tv0, tv2, tv3, tv4}) {
    tv->setDeviceMesh(mesh);
    tv->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  }
  tv1->setDeviceMesh(mesh);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  // Trigger the 1D scheduler by using small input size.
  std::vector<int64_t> input_size(3);
  if (use_1D_scheduler) {
    input_size = {16, 8, 24};
  } else {
    input_size = {1024, 32, 64};
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::randn(input_size, options).unsqueeze(sharded_dim);
  at::Tensor t1 = at::randn({input_size[1], input_size[2]}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  auto params = getPointwiseHeuristics(fusion, aten_inputs);
  auto lparams = schedulePointwise(fusion, aten_inputs);
  FusionExecutor fe;
  fe.compileFusion(fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  EXPECT_EQ(use_1D_scheduler, params->break_point==0);
  testValidate(fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  // Create a single device fusion and verify the same scheduling parameters are used for
  // given the same single device fusion. 
  auto unsharded_fusion_ptr = std::make_unique<Fusion>();
  auto unsharded_fusion = unsharded_fusion_ptr.get();
  FusionGuard unsharded_fg(unsharded_fusion);

  TensorView* utv0 = makeContigTensor(3);
  TensorView* utv1 = makeContigTensor(2);

  unsharded_fusion->addInput(utv0);
  unsharded_fusion->addInput(utv1);
  auto utv2 = add(utv0, utv0);
  auto utv3 = broadcast(utv1, {true, false, false});
  auto utv4 = add(utv2, utv3);
  unsharded_fusion->addOutput(utv4);

  FusionExecutorCache fec2(std::move(unsharded_fusion_ptr));
  fec2.profile(true);

  std::vector<c10::IValue> unsharded_aten_inputs = {t0.squeeze(sharded_dim), t1};
  auto unsharded_params = getPointwiseHeuristics(unsharded_fusion, unsharded_aten_inputs);
  EXPECT_TRUE(params->sameAs(unsharded_params));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ShardedPointwiseTest,
    testing::Combine(testing::Bool(), testing::Values(0, 1)),
    [](const testing::TestParamInfo<std::tuple<bool, int>>& info)
        -> std::string {
      bool use_1D_scheduler;
      int sharded_dim;
      std::tie(use_1D_scheduler, sharded_dim) = info.param;
      std::ostringstream os;
      os << (use_1D_scheduler ? "1D_scheduler" : "2D_scheduler")
         << "_input_sharded_along_dim_" << sharded_dim;
      return os.str();
    });

} // namespace nvfuser
