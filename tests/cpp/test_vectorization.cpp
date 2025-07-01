// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// #include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/vectorize_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fstream>

namespace nvfuser {

namespace {

void checkMappedVal(
    const std::unordered_map<TensorView*, Val*>& map,
    TensorView* tv_target,
    int64_t val) {
  auto iter = map.find(tv_target);
  EXPECT_TRUE(iter != map.end());
  if (iter != map.end()) {
    EXPECT_EQ(iter->second->evaluate(), val);
  }
}

} // namespace

using VectorizationAnalysisTest = NVFuserTest;

// Simple pad test
TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeFastestDimensionP2C) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  // positive resize (+2, +2)
  auto inner_pos =
      pad(tv0, {IrBuilder::create<Val>(2L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos, 2));
  fusion.addOutput(inner_pos);

  // positive uneven resize (+4, +2)
  auto inner_pos_uneven =
      pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos_uneven, 2));
  fusion.addOutput(inner_pos_uneven);

  // positive large resize (+32, +32)
  auto inner_pos_large =
      pad(tv0, {IrBuilder::create<Val>(32L), IrBuilder::create<Val>(32L)});
  // projected extent is 16
  expection_list.emplace_back(std::make_pair(inner_pos_large, 16));
  fusion.addOutput(inner_pos_large);

  // negative resize (-2, -2)
  auto inner_neg =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-2L)});
  expection_list.emplace_back(std::make_pair(inner_neg, 2));
  fusion.addOutput(inner_neg);

  // negative uneven resize (-2, -4)
  auto inner_neg_uneven =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-4L)});
  expection_list.emplace_back(std::make_pair(inner_neg_uneven, 2));
  fusion.addOutput(inner_neg_uneven);

  // negative large resize to zero (-8, -8)
  auto inner_neg_large =
      pad(tv0, {IrBuilder::create<Val>(-8L), IrBuilder::create<Val>(-8L)});
  // output id with extent 0 cannot be vectorized
  expection_list.emplace_back(std::make_pair(inner_neg_large, 0));
  fusion.addOutput(inner_neg_large);

  // uneven resize (-2, 4)
  auto inner_uneven =
      pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(4L)});
  expection_list.emplace_back(std::make_pair(inner_uneven, 2));
  fusion.addOutput(inner_uneven);

  // one side resize (0, 4)
  auto inner_one_size =
      pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(4L)});
  // resize extent of 0 wouldn't affect vectorization factor
  expection_list.emplace_back(std::make_pair(inner_one_size, 4));
  fusion.addOutput(inner_one_size);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();

  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

// Simple pad test
TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeFastestDimensionC2P) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 8});
  fusion.addInput(tv0);
  // positive resize (+24, +24)
  auto tv1 =
      pad(tv0, {IrBuilder::create<Val>(24L), IrBuilder::create<Val>(24L)});
  fusion.addOutput(tv1);

  // negative resize to zero (-4, -4)
  auto tv2 =
      pad(tv0, {IrBuilder::create<Val>(-4), IrBuilder::create<Val>(-4L)});
  fusion.addOutput(tv2);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_tv1 =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv1, tv1->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_tv1, tv0, 8);
  checkMappedVal(projected_extent_map_from_tv1, tv2, 0);

  // because tv2's fastest dimension is resized to 0
  std::unordered_map<TensorView*, Val*> projected_extent_map_from_tv2 =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv2, tv2->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_tv2, tv0, 0);
  checkMappedVal(projected_extent_map_from_tv2, tv1, 0);
}

TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeMiddleDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  // positive resize (+2, +2)
  auto middle_pos =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(2L),
           IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(middle_pos, 2 * 16));
  fusion.addOutput(middle_pos);

  // negative resize (-2, -2)
  auto middle_neg =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-2L),
           IrBuilder::create<Val>(-2L)});
  expection_list.emplace_back(std::make_pair(middle_neg, 2 * 16));
  fusion.addOutput(middle_neg);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

TEST_F(
    VectorizationAnalysisTest,
    ContigInnerDimsMapperResizeMultipleDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8, 32});
  fusion.addInput(tv0);

  // the inner most dimension of resize would participate in vectorization
  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(8L),
           IrBuilder::create<Val>(8L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  fusion.addOutput(tv1);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_producer =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_producer, tv1, 8);

  std::unordered_map<TensorView*, Val*> projected_extent_map_from_consumer =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv1, tv1->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  checkMappedVal(projected_extent_map_from_consumer, tv0, 8);
}

TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeStacked) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  auto tv0 = makeContigConcreteTensor({4, 8, 36});
  fusion.addInput(tv0);
  // resize on different dimension
  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-4L),
           IrBuilder::create<Val>(-4L)});
  auto tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-2L),
           IrBuilder::create<Val>(-2L)});
  // only the inner most resize is included in vectorization analysis
  expection_list.emplace_back(std::make_pair(tv2, 2 * 36));
  fusion.addOutput(tv2);

  // resize on the same dimension, squeeze size to zero
  auto tv3 =
      pad(tv0, {IrBuilder::create<Val>(-9L), IrBuilder::create<Val>(-9L)});
  auto tv4 =
      pad(tv3, {IrBuilder::create<Val>(-9L), IrBuilder::create<Val>(-9L)});
  // output id with extent 0 cannot be vectorized
  expection_list.emplace_back(std::make_pair(tv4, 0));
  fusion.addOutput(tv4);

  // resize on the same dimension
  auto tv5 =
      pad(tv0, {IrBuilder::create<Val>(-6L), IrBuilder::create<Val>(-6L)});
  auto tv6 = pad(tv5, {IrBuilder::create<Val>(9L), IrBuilder::create<Val>(9L)});
  // two resize operation would stack
  expection_list.emplace_back(std::make_pair(tv6, 3));
  fusion.addOutput(tv6);

  std::unordered_map<TensorView*, Val*> projected_extent_map =
      vectorize_helper::ContiguousInnerDimensionsMapper::map(
          tv0, tv0->getLogicalDomain())
          .getTvToContigMergeOfInnerSizeMap();
  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, tv, val);
  }
}

TEST_F(NVFuserTest, MultipleVectorize) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  TensorView* tv1 = makeContigTensor(1);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv3 = add(tv0, tv1);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({40960}, options);
  at::Tensor t1 = at::randn({40960}, options);
  auto t2 = t0 + t1;

  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.profile(true);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime1 = executor_cache.getMostRecentKernelRuntime();
  auto log1 =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  NVF_CHECK(log1 != nullptr);
  NVF_CHECK(log1->vectorization_factor > 1);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  t0 = at::randn({40964}, options);
  t1 = at::randn({40964}, options);
  t2 = t0 + t1;

  outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime2 = executor_cache.getMostRecentKernelRuntime();
  auto log2 =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  NVF_CHECK(log2 != nullptr);
  NVF_CHECK(log2->vectorization_factor > 1);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  t0 = at::randn({40962}, options);
  t1 = at::randn({40962}, options);
  t2 = t0 + t1;

  outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime3 = executor_cache.getMostRecentKernelRuntime();
  auto log3 =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  NVF_CHECK(log3 != nullptr);
  NVF_CHECK(log3->vectorization_factor > 1);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {t2}, __LINE__, __FILE__);

  NVF_CHECK(runtime1 == runtime2);
  NVF_CHECK(runtime1 != runtime3);
}

TEST_F(NVFuserTest, VectorizeSimple) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(3);

  fusion.addInput(tv0);

  auto tv1 = unaryOp(UnaryOpType::Sin, tv0);

  fusion.addOutput(tv1);

  auto tv0_cache = tv0->cacheAfter();

  tv1->cacheBefore();

  tv1->merge(0);
  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv1, 2);

  tv0_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::empty({2, 6, 32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  at::Tensor aten_output = aten_input.sin();

  testValidate(
      &fusion, cg_outputs, {aten_input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleVectorizeUnroll) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();
  tv3->cacheBefore();

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);

  // Split by n_threads
  tv3->split(1, 2);
  tv3->split(0, 3);
  tv3->split(0, 1);

  // [bidx, unswitch, unroll{2}, tidx, vectorize{2}]

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(3)->parallelize(ParallelType::TIDx);

  tv3->reorder({{4, 2}});
  // [bidx, unswitch, vectorize{2}, unroll{2}, tidx]

  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);

  tv0_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(2)->parallelize(ParallelType::Vectorize);
  tv3->axis(2)->parallelize(ParallelType::Vectorize);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);
  tv1->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t1 = at::randn({64, 2, 128}, options);
  at::Tensor t2 = at::rand_like(t1);
  at::Tensor output = at::empty_like(t1);

  KernelExecutor ke;
  ke.compile(&fusion, {t1, t2});
  ke.run({t1, t2}, {output});

  at::Tensor tv2_ref = t2 + 2.0;
  at::Tensor output_ref = t1 + tv2_ref;

  NVF_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, Vectorization1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2048;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, Vectorization2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    // Vectorize the wrong dimension
    tv->axis(-2)->parallelize(ParallelType::Vectorize);
  }

  KernelExecutor ke;
  // Make sure compilation fails
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.compile(&fusion));
}

// TODO: Re-enable once vectorization validation is fixed
TEST_F(NVFuserTest, Vectorization3) {
  GTEST_SKIP();
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->split(1, 16);
  tv2->split(1, 64);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto c0 = tv0->cacheAfter();
  auto c1 = tv1->cacheAfter();
  tv2->cacheBefore();

  c0->computeAt(tv2, -2);
  c1->computeAt(tv2, -2);

  std::vector<TensorView*> vectorized_tvs = {c0, c1, tv2};
  for (auto tv : vectorized_tvs) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2049;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0, t1}));

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run(
      {t0.index({"...", at::indexing::Slice(1)}),
       t1.index({"...", at::indexing::Slice(1)})}));

  t0 = at::randn({bx, 2048}, options).index({"...", at::indexing::Slice(4)});
  t1 = at::randn({bx, 2048}, options).index({"...", at::indexing::Slice(4)});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, VectorizationRFactor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);

  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);

  auto tv3 = sum(tv2, {-1});

  fusion.addOutput(tv3);

  tv3->split(-1, 128 * 4);
  tv3->split(-1, 4);
  // Reduce outer dim first
  auto tv4 = tv3->rFactor({-3, -1});
  // Tv3 will reduce threads

  auto tv6 = tv0->cacheAfter();
  auto tv7 = tv1->cacheAfter();

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  tv6->axis(-1)->parallelize(ParallelType::Vectorize);
  tv7->axis(-1)->parallelize(ParallelType::Vectorize);

  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 128;
  const int by = 2048;
  at::Tensor t0 = at::randn({bx, by}, options);
  at::Tensor t1 = at::randn({bx, by}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto aten_output = t0.add(t1).sum(1);
  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);

  auto t3 = t0.add(t1).sum(1);

  testValidate(&fusion, cg_outputs, {t0, t1}, {t3}, __LINE__, __FILE__);
}

// Repro of an issue found in PR #733. Previously the runtime
// validation of strides of vectorized tensors issued a false positive
TEST_F(NVFuserTest, VectorizationStrideValidation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({2, 1, 3});
  const std::vector<int64_t> expanded_shape({2, 5, 3});

  auto tv0 = TensorViewBuilder()
                 .ndims(shape.size())
                 .shape(expanded_shape)
                 .contiguity({false, std::nullopt, true})
                 .expanded({false, true, false})
                 .build();
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0)->merge(0);
  tv1->split(0, 2);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options).expand({-1, 5, -1});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // This previously triggered a false positive error with the stride
  // validation
  auto cg_outputs = ke.run({t0});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().equal(t0));
}

// From dtype, to dtype, vectorization factor
using VectorizationCastParams = std::tuple<DataType, DataType, int64_t>;

class VectorizationCastTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<VectorizationCastParams> {
 public:
  void SetUp() override {
    std::tie(dtype_from, dtype_to, vectorization_factor) = GetParam();
  }

 protected:
  DataType dtype_from;
  DataType dtype_to;
  int64_t vectorization_factor;
};

TEST_P(VectorizationCastTest, CastKernel) {
  if (dtype_from == DataType::Float8_e8m0fnu ||
      dtype_to == DataType::Float8_e8m0fnu ||
      dtype_from == DataType::Float4_e2m1fn ||
      dtype_to == DataType::Float4_e2m1fn) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(10, 0);
  }
  if (dtype_from == DataType::Float8_e4m3fn ||
      dtype_from == DataType::Float8_e5m2 ||
      dtype_to == DataType::Float8_e4m3fn ||
      dtype_to == DataType::Float8_e5m2) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  }
  if (dtype_from == DataType::BFloat16 || dtype_to == DataType::BFloat16) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({vectorization_factor}, dtype_from);
  fusion.addInput(tv0);
  auto tv1 = castOp(dtype_to, tv0);
  fusion.addOutput(tv1);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheBefore();

  for (auto tv : {tv0_cache, tv1_cache, tv1}) {
    tv->axis(0)->parallelize(ParallelType::Vectorize);
  }

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(dtype_from))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({vectorization_factor}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
auto all_vectorization_cast_params = ::testing::Values(
    // cvt.frnd2{.relu}{.satfinite}.f16x2.f32
    VectorizationCastParams(DataType::Float, DataType::Half, 2),
    // Two cvt.frnd2{.relu}{.satfinite}.f16x2.f32
    VectorizationCastParams(DataType::Float, DataType::Half, 4),

    // cvt.frnd2{.relu}{.satfinite}.bf16x2.f32
    VectorizationCastParams(DataType::Float, DataType::BFloat16, 2),
    // Two cvt.frnd2{.relu}{.satfinite}.bf16x2.f32
    VectorizationCastParams(DataType::Float, DataType::BFloat16, 4));

// Not enabled yet, just because it will be implemented in a later PR.
auto disabled_vectorization_cast_params = ::testing::Values(
    // cvt.rn.satfinite{.relu}.f8x2type.f32
    VectorizationCastParams(DataType::Float, DataType::Float8_e4m3fn, 2),
    VectorizationCastParams(DataType::Float, DataType::Float8_e5m2, 2),
    // Two cvt.rn.satfinite{.relu}.f8x2type.f32
    VectorizationCastParams(DataType::Float, DataType::Float8_e4m3fn, 4),
    VectorizationCastParams(DataType::Float, DataType::Float8_e5m2, 4),

    // cvt.rn.satfinite{.relu}.f8x2type.f16x2
    VectorizationCastParams(DataType::Half, DataType::Float8_e4m3fn, 2),
    VectorizationCastParams(DataType::Half, DataType::Float8_e5m2, 2),
    // Two cvt.rn.satfinite{.relu}.f8x2type.f16x2
    VectorizationCastParams(DataType::Half, DataType::Float8_e4m3fn, 4),
    VectorizationCastParams(DataType::Half, DataType::Float8_e5m2, 4),

    // cvt.rn.{.relu}.f16x2.f8x2type
    VectorizationCastParams(DataType::Float8_e4m3fn, DataType::Half, 2),
    VectorizationCastParams(DataType::Float8_e5m2, DataType::Half, 2),
    // Two cvt.rn.{.relu}.f16x2.f8x2type
    VectorizationCastParams(DataType::Float8_e4m3fn, DataType::Half, 4),
    VectorizationCastParams(DataType::Float8_e5m2, DataType::Half, 4),

    // cvt.rn.satfinite{.relu}.f4x2type.f32
    VectorizationCastParams(DataType::Float, DataType::Float4_e2m1fn, 2),
    // Two cvt.rn.satfinite{.relu}.f4x2type.f32
    VectorizationCastParams(DataType::Float, DataType::Float4_e2m1fn, 4),

    // cvt.rn{.relu}.f16x2.f4x2type
    VectorizationCastParams(DataType::Float4_e2m1fn, DataType::Half, 2),
    // Two cvt.rn{.relu}.f16x2.f4x2type
    VectorizationCastParams(DataType::Float4_e2m1fn, DataType::Half, 4),

    // cvt.frnd3{.satfinite}.ue8m0x2.f32
    VectorizationCastParams(DataType::Float, DataType::Float8_e8m0fnu, 2),
    // Two cvt.frnd3{.satfinite}.ue8m0x2.f32
    VectorizationCastParams(DataType::Float, DataType::Float8_e8m0fnu, 4),

    // cvt.frnd3{.satfinite}.ue8m0x2.bf16x2
    VectorizationCastParams(DataType::BFloat16, DataType::Float8_e8m0fnu, 2),
    // Two cvt.frnd3{.satfinite}.ue8m0x2.bf16x2
    VectorizationCastParams(DataType::BFloat16, DataType::Float8_e8m0fnu, 4),

    // cvt.rn.bf16x2.ue8m0x2
    VectorizationCastParams(DataType::Float8_e8m0fnu, DataType::BFloat16, 2),
    // Two cvt.rn.bf16x2.ue8m0x2
    VectorizationCastParams(DataType::Float8_e8m0fnu, DataType::BFloat16, 4));

std::string vectorizeCastTestName(
    testing::TestParamInfo<VectorizationCastParams> info) {
  std::stringstream ss;
  ss << "from_" << std::get<0>(info.param) << "_to_" << std::get<1>(info.param)
     << "_Vectorize_" << std::get<2>(info.param);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    VectorizationCastTest,
    all_vectorization_cast_params,
    vectorizeCastTestName);

} // namespace nvfuser
