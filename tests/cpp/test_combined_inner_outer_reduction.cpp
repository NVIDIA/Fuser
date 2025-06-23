// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;
using testing::UnorderedElementsAre;

// tuple of data type, batch size (outer dim), hidden size (inner dim)
using CombinedSchedulerParams = std::tuple<DataType, int64_t, int64_t>;

class CombinedSchedulerTest
    : public NVFuserFixtureParamTest<CombinedSchedulerParams> {
 protected:
  void SetUp() override {
    NVFuserFixtureParamTest<CombinedSchedulerParams>::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_P(CombinedSchedulerTest, LayerNormBackward) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto [dtype, batch_size, hidden_size] = GetParam();

  std::vector<int64_t> norm_shape{hidden_size};
  std::vector<int64_t> input_shape{batch_size, hidden_size};
  std::vector<int64_t> outer_shape{batch_size, 1};
  bool fp16_or_bf16 = dtype == DataType::Half || dtype == DataType::BFloat16;
  auto grad_out = makeContigTensor(input_shape.size(), dtype);
  auto input = makeContigTensor(input_shape.size(), dtype);
  auto mean =
      makeConcreteTensor(outer_shape, fp16_or_bf16 ? DataType::Float : dtype);
  auto rstd =
      makeConcreteTensor(outer_shape, fp16_or_bf16 ? DataType::Float : dtype);
  auto weight = makeContigTensor(norm_shape.size(), dtype);
  auto bias = makeContigTensor(norm_shape.size(), dtype);
  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(mean);
  fusion->addInput(rstd);
  fusion->addInput(weight);
  fusion->addInput(bias);

  if (fp16_or_bf16) {
    grad_out = castOp(DataType::Float, grad_out);
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }
  auto layer_norm_results = layer_norm_backward(
      grad_out,
      input,
      norm_shape,
      mean,
      rstd,
      weight,
      bias,
      {true, true, true});

  if (fp16_or_bf16) {
    layer_norm_results.grad_input =
        castOp(dtype, layer_norm_results.grad_input);
    layer_norm_results.grad_bias = castOp(dtype, layer_norm_results.grad_bias);
    layer_norm_results.grad_weight =
        castOp(dtype, layer_norm_results.grad_weight);
  }

  fusion->addOutput(layer_norm_results.grad_input);
  fusion->addOutput(layer_norm_results.grad_weight);
  fusion->addOutput(layer_norm_results.grad_bias);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor aten_grad_out = at::randn(input_shape, options);
  at::Tensor aten_input = at::randn(input_shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  at::Tensor aten_bias = at::randn(norm_shape, options);

  constexpr float kEps = 1e-5;
  auto aten_results = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);
  auto aten_output = std::get<0>(aten_results);
  auto aten_mean = std::get<1>(aten_results);
  auto aten_rstd = std::get<2>(aten_results);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {
      aten_grad_out, aten_input, aten_mean, aten_rstd, aten_weight, aten_bias};
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto aten_gradients = at::native_layer_norm_backward(
      aten_grad_out,
      aten_input,
      norm_shape,
      aten_mean,
      aten_rstd,
      aten_weight,
      aten_bias,
      {true, true, true});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      args,
      {std::get<0>(aten_gradients),
       std::get<1>(aten_gradients),
       std::get<2>(aten_gradients)},
      __LINE__,
      __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    CombinedSchedulerTest,
    ::testing::Combine(
        // aten doesn't support complex data types
        testing::ValuesIn(getFloatingDataTypes(/*include_complex=*/false)),
        testing::Values(216), // batch size
        testing::Values(
            3,
            32,
            96,
            576,
            768,
            1024,
            1280,
            1600,
            1984,
            1987,
            65536)), // hidden size
    [](const testing::TestParamInfo<CombinedSchedulerParams>& info)
        -> std::string {
      std::stringstream ss;
      ss << "dtype_" << std::get<0>(info.param);
      ss << "_batch_" << std::get<1>(info.param);
      ss << "_hidden_" << std::get<2>(info.param);
      return sanitizeTestName(ss.str());
    });

// This case is to test the correctness of the combined inner and outer
// scheduler, if link_inner_outer = true, the inner and outer reductions are
// linked, otherwise the two outer reductions are linked. In either case, the
// fusion should be segmented since the current combined scheduler assumes there
// is no shared consumer between inter reductions and outer reductions and among
// tensors in outer reductions.
TEST_F(CombinedSchedulerTest, SharedConsumer) {
  auto runTest = [](const std::vector<int64_t>& batch_shape,
                    const std::vector<int64_t>& norm_shape,
                    DataType dtype,
                    bool link_inner_outer) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    std::vector<int64_t> input_shape(batch_shape);
    std::copy(
        norm_shape.begin(), norm_shape.end(), std::back_inserter(input_shape));

    const size_t kM = input_shape.size();
    const size_t kN = norm_shape.size();
    const size_t kOuterNumDims = kM - kN;
    std::vector<int64_t> outer_shape;
    for (const auto idx : arange(kOuterNumDims)) {
      outer_shape.push_back(input_shape[idx]);
    }
    for (const auto idx : arange(kOuterNumDims, kM)) {
      // just to avoid unused variable warning
      outer_shape.push_back(1 + idx - idx);
    }

    auto grad_out = makeContigTensor(input_shape.size(), dtype);
    auto input = makeContigTensor(input_shape.size(), dtype);
    auto mean = makeConcreteTensor(
        outer_shape, dtype == DataType::Half ? DataType::Float : dtype);
    auto rstd = makeConcreteTensor(
        outer_shape, dtype == DataType::Half ? DataType::Float : dtype);
    auto weight = makeContigTensor(norm_shape.size(), dtype);
    auto bias = makeContigTensor(norm_shape.size(), dtype);
    fusion.addInput(grad_out);
    fusion.addInput(input);
    fusion.addInput(mean);
    fusion.addInput(rstd);
    fusion.addInput(weight);
    fusion.addInput(bias);

    if (dtype == DataType::Half) {
      grad_out = castOp(DataType::Float, grad_out);
      input = castOp(DataType::Float, input);
      weight = castOp(DataType::Float, weight);
      bias = castOp(DataType::Float, bias);
    }

    auto layer_norm_results = layer_norm_backward(
        grad_out,
        input,
        norm_shape,
        mean,
        rstd,
        weight,
        bias,
        {true, true, true});

    if (dtype == DataType::Half) {
      layer_norm_results.grad_input =
          castOp(dtype, layer_norm_results.grad_input);
      layer_norm_results.grad_bias =
          castOp(dtype, layer_norm_results.grad_bias);
      layer_norm_results.grad_weight =
          castOp(dtype, layer_norm_results.grad_weight);
    }
    // link inner and outer reduction or outer and outer reduction
    auto out_linked = link_inner_outer
        ? add(layer_norm_results.grad_input, layer_norm_results.grad_weight)
        : add(layer_norm_results.grad_bias, layer_norm_results.grad_weight);

    if (!link_inner_outer) {
      auto out_linked_scale = mul(out_linked, IrBuilder::create<Val>(0.5));
      fusion.addOutput(out_linked_scale);
    } else {
      fusion.addOutput(out_linked);
    }

    fusion.addOutput(layer_norm_results.grad_input);
    fusion.addOutput(layer_norm_results.grad_weight);
    fusion.addOutput(layer_norm_results.grad_bias);

    auto maybe_fp16_options = at::TensorOptions()
                                  .dtype(data_type_to_aten(dtype))
                                  .device(at::kCUDA, 0);
    at::Tensor aten_grad_out = at::randn(input_shape, maybe_fp16_options);
    at::Tensor aten_input = at::randn(input_shape, maybe_fp16_options);
    at::Tensor aten_weight = at::randn(norm_shape, maybe_fp16_options);
    at::Tensor aten_bias = at::randn(norm_shape, maybe_fp16_options);
    auto at_weight = c10::optional<at::Tensor>(aten_weight);
    auto at_bias = c10::optional<at::Tensor>(aten_bias);

    const float kEps = 1e-5;
    auto aten_results =
        at::native_layer_norm(aten_input, norm_shape, at_weight, at_bias, kEps);
    auto aten_output = std::get<0>(aten_results);
    auto aten_mean = std::get<1>(aten_results);
    auto aten_rstd = std::get<2>(aten_results);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    KernelArgumentHolder args = {
        aten_grad_out,
        aten_input,
        aten_mean,
        aten_rstd,
        aten_weight,
        aten_bias};
    auto cg_outputs = executor_cache.runFusionWithInputs(args);

    auto aten_gradients = at::native_layer_norm_backward(
        aten_grad_out.to(at::kDouble),
        aten_input.to(at::kDouble),
        norm_shape,
        aten_mean.to(at::kDouble),
        aten_rstd.to(at::kDouble),
        c10::optional<at::Tensor>(aten_weight.to(at::kDouble)),
        c10::optional<at::Tensor>(aten_bias.to(at::kDouble)),
        {true, true, true});

    auto aten_out_linked = link_inner_outer
        ? std::get<0>(aten_gradients) + std::get<1>(aten_gradients)
        : std::get<1>(aten_gradients) + std::get<2>(aten_gradients);
    if (!link_inner_outer) {
      aten_out_linked = aten_out_linked.mul(0.5);
    }
    bool is_segmented =
        executor_cache.getMostRecentKernelRuntime()->isSegmented();
    NVF_CHECK(is_segmented, "Fusion is not segmented");

    testValidate(
        &fusion,
        cg_outputs,
        args,
        {aten_out_linked,
         std::get<0>(aten_gradients),
         std::get<1>(aten_gradients),
         std::get<2>(aten_gradients)},
        __LINE__,
        __FILE__);
  };

  DataType dtype = DataType::Float;
  std::vector<int64_t> batch_shape = {8192};
  std::vector<int64_t> norm_shape = {2048};
  runTest(batch_shape, norm_shape, dtype, true);
  runTest(batch_shape, norm_shape, dtype, false);
}

// This case is to test the correctness of the combined inner and outer
// scheduler. One tensor is using the inner reduction results and outer
// reduction results. should be segmented.
TEST_F(CombinedSchedulerTest, SharedProducer) {
  auto runTest = [](const std::vector<int64_t>& batch_shape,
                    const std::vector<int64_t>& norm_shape,
                    DataType dtype,
                    int case_id) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    std::vector<int64_t> input_shape(batch_shape);
    std::copy(
        norm_shape.begin(), norm_shape.end(), std::back_inserter(input_shape));

    const size_t kM = input_shape.size();
    const size_t kN = norm_shape.size();
    const size_t kOuterNumDims = kM - kN;
    std::vector<int64_t> outer_shape;
    for (const auto idx : arange(kOuterNumDims)) {
      outer_shape.push_back(input_shape[idx]);
    }
    for (const auto idx : arange(kOuterNumDims, kM)) {
      // just to avoid unused variable warning
      outer_shape.push_back(1 + idx - idx);
    }

    auto grad_out = makeContigTensor(input_shape.size(), dtype);
    auto input = makeContigTensor(input_shape.size(), dtype);
    auto mean = makeConcreteTensor(
        outer_shape, dtype == DataType::Half ? DataType::Float : dtype);
    auto rstd = makeConcreteTensor(
        outer_shape, dtype == DataType::Half ? DataType::Float : dtype);
    auto weight = makeContigTensor(norm_shape.size(), dtype);
    auto bias = makeContigTensor(norm_shape.size(), dtype);
    fusion.addInput(grad_out);
    fusion.addInput(input);
    fusion.addInput(mean);
    fusion.addInput(rstd);
    fusion.addInput(weight);
    fusion.addInput(bias);

    if (dtype == DataType::Half) {
      grad_out = castOp(DataType::Float, grad_out);
      input = castOp(DataType::Float, input);
      weight = castOp(DataType::Float, weight);
      bias = castOp(DataType::Float, bias);
    }

    auto layer_norm_results = layer_norm_backward(
        grad_out,
        input,
        norm_shape,
        mean,
        rstd,
        weight,
        bias,
        {true, true, true});

    if (dtype == DataType::Half) {
      layer_norm_results.grad_input =
          castOp(dtype, layer_norm_results.grad_input);
      layer_norm_results.grad_bias =
          castOp(dtype, layer_norm_results.grad_bias);
      layer_norm_results.grad_weight =
          castOp(dtype, layer_norm_results.grad_weight);
    }

    switch (case_id) {
      case 0: {
        // tensor input is a produer of a consumer of the inner and outer
        // reduction results this a not allowed, expect segmented
        auto use_inner = add(layer_norm_results.grad_input, input);
        auto use_outer = add(layer_norm_results.grad_weight, input);
        fusion.addOutput(use_inner);
        fusion.addOutput(use_outer);
      } break;
      case 1: {
        // tensor bias is a producer of the inner reduction and also a
        // produer of a consumer of the outer reduction results this a not
        // allowed, expect segmented
        auto bias_broad = add(bias, mean);
        auto use_inner = sum(bias_broad, {-1});
        auto use_outer = add(layer_norm_results.grad_weight, bias);
        fusion.addOutput(use_inner);
        fusion.addOutput(use_outer);
      } break;
      case 2: {
        // tensor bias is a producer of the outer reduction and also a
        // produer of a consumer of the inner reduction results this a
        // allowed, becase the first part of outer reduction is computed
        // with inner reduction. expect unsegmented
        auto bias_broad = add(bias, mean);
        auto use_inner = add(layer_norm_results.grad_input, bias_broad);
        auto use_outer = sum(bias_broad, {0});
        fusion.addOutput(use_inner);
        fusion.addOutput(use_outer);
      } break;
      case 3: {
        // tensor bias is a producer of the two outer reductions' consumers,
        // expect segmented
        auto outer_1_consumer =
            add(layer_norm_results.grad_weight, IrBuilder::create<Val>(1.0));
        auto outer_2_consumer =
            add(layer_norm_results.grad_bias, IrBuilder::create<Val>(1.0));
        auto use_producer_1 = add(outer_1_consumer, bias);
        auto use_producer_2 = add(outer_2_consumer, bias);
        fusion.addOutput(use_producer_1);
        fusion.addOutput(use_producer_2);
      } break;
      default:
        NVF_THROW("Invalid case id");
    }

    fusion.addOutput(layer_norm_results.grad_input);
    fusion.addOutput(layer_norm_results.grad_weight);
    fusion.addOutput(layer_norm_results.grad_bias);

    auto maybe_fp16_options = at::TensorOptions()
                                  .dtype(data_type_to_aten(dtype))
                                  .device(at::kCUDA, 0);
    at::Tensor aten_grad_out = at::randn(input_shape, maybe_fp16_options);
    at::Tensor aten_input = at::randn(input_shape, maybe_fp16_options);
    at::Tensor aten_weight = at::randn(norm_shape, maybe_fp16_options);
    at::Tensor aten_bias = at::randn(norm_shape, maybe_fp16_options);

    constexpr float kEps = 1e-5;
    auto aten_results = at::native_layer_norm(
        aten_input, norm_shape, aten_weight, aten_bias, kEps);
    auto aten_output = std::get<0>(aten_results);
    auto aten_mean = std::get<1>(aten_results);
    auto aten_rstd = std::get<2>(aten_results);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    KernelArgumentHolder args = {
        aten_grad_out,
        aten_input,
        aten_mean,
        aten_rstd,
        aten_weight,
        aten_bias};
    auto cg_outputs = executor_cache.runFusionWithInputs(args);

    FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
    switch (case_id) {
      case 0:
      case 1:
      case 3:
        EXPECT_TRUE(runtime->isSegmented());
        break;
      case 2:
        EXPECT_FALSE(runtime->isSegmented());
        break;
      default:
        NVF_THROW("Invalid case id");
    }

    auto tolerance_overwrite = ValidationConstants();
    // bump tolerance, CI errors are higher than local
    std::array<std::array<double, 2>, 20> relaxed_sum_tol;
    for (auto& arr : relaxed_sum_tol) {
      arr = {128, 2e-5};
    }
    tolerance_overwrite.sum_tolerances_float = relaxed_sum_tol;

    testValidate(
        &fusion,
        cg_outputs,
        args,
        __LINE__,
        __FILE__,
        "",
        LaunchParams(),
        tolerance_overwrite);
  };

  DataType dtype = DataType::Float;
  // to test hasSharedConsumerNonReductionProducer, needs to use small sizes,
  // otherwise this fusion will be rejected due to register usage.
  std::vector<int64_t> batch_shape = {64};
  std::vector<int64_t> norm_shape = {32};
  for (int i = 0; i < 4; i++) {
    runTest(batch_shape, norm_shape, dtype, i);
  }
}

// Manual schedule of inner and outer reduction on the same tensor
TEST_F(CombinedSchedulerTest, CombinedReduction) {
  // https://github.com/csarofeen/pytorch/issues/2566
  // this case will fail, if using tidx = 8 and tidy = 64
  // for inner reduction, tidy is derived as 10240 / (tidx*vecx*nloadx) = 64
  // for outer reduction, tidy is derived as 216  / nloady = 54
  // the kernel will be launched with bdimy = 64
  // in the generated kernel, all these 64 threads are attending the block
  // reduction but only 54 of them have valid initial values. thus the result is
  // polluted by other 10 threads and can't pass the validation. to avoid this
  // issue, we can use one of the following methods: (1) make sure tidy derived
  // from inner reduction & outer reduction is same (when 216 % tidy == 0) or
  // (2) instead of split outer reduction tensor with nloady, split it with
  // bdimy. The current scheduler is using method-2.

  auto ceilDiv = [](const int a, const int b) { return (a + b - 1) / b; };
  constexpr bool verbose = false;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;
  const int dim0 = 2048;
  const int dim1 = 10240;
  const int tidx = 64;
  const int tidy = 8;
  const int bidy = 2 * device_multiprocessor_count; // 216
  const int vecx = 4;
  const int nloadx =
      ceilDiv(dim1, vecx * tidx * tidy); // 5, simulate persistent buffer
  const int nloady = ceilDiv(bidy, tidy); // 216/16=13.5 -> 14

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv0, {0});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  auto cached_inputs = scheduler_utils::cacheInputs(&fusion, true);
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(&fusion, true);
  auto reduction_tvs = scheduler_utils::getReductionTvs(&fusion);
  scheduler_utils::clearMemorySpace(&fusion);
  std::vector<TensorView*> inner_reduction_tvs, outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
    if (verbose)
      std::cout << "tv= " << tv->toString() << ", fastest_dim_reduction= "
                << scheduler_utils::isFastestDimReduction(tv) << std::endl;
  }
  TensorView* inner_reduction_tv = inner_reduction_tvs[0];
  TensorView* outer_reduction_tv = outer_reduction_tvs[0];

  inner_reduction_tv->split(-1, vecx);
  inner_reduction_tv->split(-2, tidx);
  inner_reduction_tv->split(-3, nloadx, false);
  inner_reduction_tv->split(0, bidy, false);
  inner_reduction_tv->axis(0)->parallelize(ParallelType::BIDy);
  inner_reduction_tv->axis(-3)->parallelize(ParallelType::TIDy);
  inner_reduction_tv->axis(-2)->parallelize(ParallelType::TIDx);
  inner_reduction_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  if (verbose)
    std::cout << "inner_reduction_tv " << inner_reduction_tv->toString()
              << std::endl;
  auto reference_tv_inner =
      reduction_scheduler_utils::sortAndRFactor(inner_reduction_tv);
  if (verbose)
    std::cout << "reference_tv_inner " << reference_tv_inner->toString()
              << std::endl;

  outer_reduction_tv->split(0, bidy, false);
  auto partialResult = outer_reduction_tv->rFactor({1});
  partialResult->cacheBefore();
  partialResult->setMemoryType(MemoryType::Global);
  auto partialResultReload = partialResult->cacheAfter();

  outer_reduction_tv->split(0, nloady, false);
  outer_reduction_tv->split(-1, tidx);
  outer_reduction_tv->split(-2, bidy);
  outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
  outer_reduction_tv->axis(-2)->parallelize(ParallelType::BIDy);
  outer_reduction_tv->axis(-1)->parallelize(ParallelType::TIDx);

  if (verbose)
    std::cout << "outer_reduction_tv " << outer_reduction_tv->toString()
              << std::endl;
  auto reference_tv_outer =
      reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
  if (verbose)
    std::cout << "reference_tv_outer " << reference_tv_outer->toString()
              << std::endl;

  reduction_scheduler_utils::propagateTransformation(
      reference_tv_inner, {partialResultReload});
  reduction_scheduler_utils::propagateTransformation(
      reference_tv_outer, {partialResultReload});

  std::vector<TensorView*> cached_gmem_temp{partialResult};
  // cached_gmem is float, may use a different vectorization factor
  for (auto tv : cached_gmem_temp) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  reduction_scheduler_utils::propagateParallelization(
      inner_reduction_tv,
      reference_tv_inner,
      true,
      false,
      inner_reduction_tvs,
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv_inner, true, cached_inputs, cached_outputs));
  reduction_scheduler_utils::propagateParallelization(
      outer_reduction_tv,
      reference_tv_outer,
      true,
      false,
      outer_reduction_tvs,
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv_outer, true, cached_inputs, cached_outputs));

  inlineMost();
  LaunchParams launch_constraints;
  constexpr int64_t maxrregcount = 64;
  CompileParams compile_params{DataType::Int, maxrregcount, true};
  if (verbose)
    fusion.print();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor tv_input = at::randn({dim0, dim1}, options);
  auto tv_aten_output = tv_input.to(at::kFloat).sum({1});
  at::Tensor tv_cg_output = at::empty({dim0}, options);

  at::Tensor qv_cg_output = at::empty({dim1}, options);
  auto qv_aten_output = tv_input.to(at::kFloat).sum({0});
  KernelExecutor ke;
  ke.compile(&fusion, {tv_input}, launch_constraints, compile_params);
  ke.run(
      {tv_input},
      {tv_cg_output, qv_cg_output},
      launch_constraints,
      compile_params);

  testValidate(
      &fusion,
      {tv_cg_output, qv_cg_output},
      {tv_input},
      {tv_aten_output, qv_aten_output},
      __LINE__,
      __FILE__);
}

// Manual schedule of inner and outer reduction on the same tensor. Each block
// will do multiple reductions.
TEST_F(CombinedSchedulerTest, CombinedReductionMultiPerBlock) {
  auto ceilDiv = [](const int a, const int b) { return (a + b - 1) / b; };
  constexpr bool verbose = false;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // avoid future architecture with too many SMs
  // then we don't have enough parallelism to split out.
  const int64_t device_multiprocessor_count =
      std::min(dev_prop->multiProcessorCount, 128);
  const int dim0 = 216;
  const int dim1 = 1024;
  const int bidy = 2 * device_multiprocessor_count;
  const int vecx = 4;
  const int nloadx = 8;
  const int tidx = dim1 / vecx / nloadx;
  const int tidy = ceilDiv(dim1, bidy);
  // https://github.com/csarofeen/pytorch/issues/2458
  const bool swap_xy = true;

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {1});
  TensorView* tv2 = sum(tv0, {0});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  auto cached_inputs = scheduler_utils::cacheInputs(&fusion, true);
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(&fusion, true);
  auto reduction_tvs = scheduler_utils::getReductionTvs(&fusion);
  scheduler_utils::clearMemorySpace(&fusion);
  std::vector<TensorView*> inner_reduction_tvs, outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
    if (verbose)
      std::cout << "tv= " << tv->toString() << ", fastest_dim_reduction= "
                << scheduler_utils::isFastestDimReduction(tv) << std::endl;
  }
  TensorView* inner_reduction_tv = inner_reduction_tvs[0];
  TensorView* outer_reduction_tv = outer_reduction_tvs[0];

  inner_reduction_tv->split(-1, vecx);
  inner_reduction_tv->split(-2, nloadx, false);
  inner_reduction_tv->split(0, tidy);
  inner_reduction_tv->split(0, bidy, false);
  // bidy, i0/tidy/bidy, tidy

  inner_reduction_tv->axis(0)->parallelize(ParallelType::BIDy);
  inner_reduction_tv->axis(1)->parallelize(ParallelType::Serial);
  inner_reduction_tv->axis(2)->parallelize(ParallelType::TIDy);
  inner_reduction_tv->axis(-2)->parallelize(ParallelType::TIDx);
  inner_reduction_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  if (verbose)
    std::cout << "inner_reduction_tv " << inner_reduction_tv->toString()
              << std::endl;
  auto reference_tv_inner =
      reduction_scheduler_utils::sortAndRFactor(inner_reduction_tv);
  if (verbose)
    std::cout << "reference_tv_inner " << reference_tv_inner->toString()
              << std::endl;

  // outer_reduction_tv->split(0, bidy, false);
  // auto partialResult = outer_reduction_tv->rFactor({1});
  std::vector<int32_t> rfactor_axis = {1, 2};

  outer_reduction_tv->split(0, tidy);
  outer_reduction_tv->split(0, bidy, false);
  outer_reduction_tv->rFactor({1});
  TensorView* partialResult = outer_reduction_tv->rFactor({1});

  if (verbose)
    std::cout << "outer_reduction_tv " << outer_reduction_tv->toString()
              << std::endl;

  partialResult->cacheBefore();
  partialResult->setMemoryType(MemoryType::Global);
  auto partialResultReload = partialResult->cacheAfter();

  if (swap_xy) {
    outer_reduction_tv->split(0, tidx);
    outer_reduction_tv->split(-1, tidy);
    outer_reduction_tv->split(-2, bidy);
    outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
    outer_reduction_tv->axis(-2)->parallelize(ParallelType::BIDy);
    outer_reduction_tv->axis(-1)->parallelize(ParallelType::TIDy);
  } else {
    outer_reduction_tv->split(0, tidy);
    outer_reduction_tv->split(-1, tidx);
    outer_reduction_tv->split(-2, bidy);
    outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
    outer_reduction_tv->axis(-2)->parallelize(ParallelType::BIDy);
    outer_reduction_tv->axis(-1)->parallelize(ParallelType::TIDx);
  }
  if (verbose)
    std::cout << "outer_reduction_tv " << outer_reduction_tv->toString()
              << std::endl;
  auto reference_tv_outer =
      reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
  if (verbose)
    std::cout << "reference_tv_outer " << reference_tv_outer->toString()
              << std::endl;

  // empty in this test
  std::vector<TensorView*> smem_consumers;

  reduction_scheduler_utils::propagateTransformation(
      reference_tv_inner, {partialResultReload});
  const auto& selected_tvs_inner = scheduler_utils::getAllTvsFrom(
      inner_reduction_tvs, {partialResultReload});
  reduction_scheduler_utils::propagateParallelization(
      inner_reduction_tv,
      reference_tv_inner,
      true,
      false,
      inner_reduction_tvs,
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv_inner, true, cached_inputs, cached_outputs),
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  const auto& selected_tvs_outer =
      scheduler_utils::getAllTvsFrom(outer_reduction_tvs, {partialResult});
  reduction_scheduler_utils::propagateTransformation(
      reference_tv_outer, {partialResultReload});
  reduction_scheduler_utils::propagateParallelization(
      outer_reduction_tv,
      reference_tv_outer,
      true,
      false,
      outer_reduction_tvs,
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv_outer, true, cached_inputs, cached_outputs),
      {selected_tvs_outer.begin(), selected_tvs_outer.end()});

  std::vector<TensorView*> cached_gmem_temp{partialResult};
  for (auto tv : cached_gmem_temp) {
    tv->split(-1, 4);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  inlineMost();
  LaunchParams launch_constraints;
  constexpr int64_t maxrregcount = 64;
  CompileParams compile_params{DataType::Int, maxrregcount, true};
  if (verbose)
    fusion.print();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor tv_input = at::ones({dim0, dim1}, options);
  auto tv_aten_output = tv_input.to(at::kFloat).sum({1});
  at::Tensor tv_cg_output = at::empty({dim0}, options);

  at::Tensor qv_cg_output = at::empty({dim1}, options);
  at::Tensor tv_input2 = at::ones({dim0, dim1}, options);
  auto qv_aten_output = tv_input2.to(at::kFloat).sum({0});
  KernelExecutor ke;
  ke.compile(&fusion, {tv_input}, launch_constraints, compile_params);
  ke.run(
      {tv_input},
      {tv_cg_output, qv_cg_output},
      launch_constraints,
      compile_params);

  testValidate(
      &fusion,
      {tv_cg_output, qv_cg_output},
      {tv_input},
      {tv_aten_output, qv_aten_output},
      __LINE__,
      __FILE__);
}

// Reproduce of issue 1023, where iteration axis in inner reduction tv doesn't
// match to reduction axis in outer reduction tv.
TEST_F(CombinedSchedulerTest, InnerOuterMismatch) {
  auto test = [](const std::vector<int64_t>& outer_reduction_axis) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    const int x = 8, y = 16, z = 32;
    auto tv0 = makeContigTensor(3);
    fusion.addInput(tv0);
    auto tv1 = sum(tv0, {-1});
    auto tv2 = broadcast(tv1, {false, false, true});
    auto tv3 = add(tv2, tv0);
    auto tv4 = sum(tv0, outer_reduction_axis);
    fusion.addOutput(tv3);
    fusion.addOutput(tv4);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({x, y, z}, options);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    bool is_segmented =
        executor_cache.getMostRecentKernelRuntime()->isSegmented();
    if (outer_reduction_axis.size() == 2) {
      NVF_ERROR(!is_segmented, "Fusion should NOT be segmented!");
    } else {
      NVF_ERROR(is_segmented, "Fusion should be segmented!");
    }

    auto t1 = t0.sum({-1});
    auto t2 = t1.unsqueeze(-1);
    auto t3 = t0 + t2;
    auto t4 = t0.sum(outer_reduction_axis);
    testValidate(&fusion, cg_outputs, {t0}, {t3, t4}, __LINE__, __FILE__);
  };

  // inner reduction is [I, I, R]
  // outer reduction is [R, R, I]
  // every iteration domain in inner reduction tv is a reduction domain in outer
  // reduction tv, matched.
  test({0, 1});

  // inner reduction is [I, I, R]
  // outer reduction is [R, I, I]
  // axis-1 is a iteration domain in inner reduction tv but it is not a
  // reduction domain in outer reduction tv, not matched.
  test({0});
}

// innerOuter scheduler projects buffer to inputs when there is one or more
// outer broadcast tvs, e.g. in layer norm backward and RMS norm backward.
// This test covers the branch where the outer broadcast tensor is not exist
// and data type is fp32, so the buffer is not projected to inputs.
TEST_F(CombinedSchedulerTest, InnerOuterNoOuterBroadcastTv) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int dim0 = 1024, dim1 = 2048;
  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  auto tv4 = sum(tv0, {0});
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);

  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::InnerOuterPersistent, {t0});

  auto persistent_params = cg_results.heuristic_params->as<ReductionParams>();
  NVF_CHECK(
      !persistent_params->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");

  auto t1 = t0.sum({1});
  auto t2 = t1.unsqueeze(-1);
  auto t3 = t0 + t2;
  auto t4 = t0.sum({0});
  testValidate(
      &fusion,
      cg_results.outputs,
      {t0},
      {t3, t4},
      __LINE__,
      __FILE__,
      "",
      persistent_params->lparams);
}

// Reproduce error found in:
// thunder/tests/test_torch_compile_executor.py::test_torch_compile_cat_nvfuser_phi2_tanh
// Only happens when shared memory persistent is used.
TEST_F(CombinedSchedulerTest, SharedMemoryPersistentVectFactor) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // When the input is float16, the vectorization factor is set to 8.
  // If the persistent buffer tv1 is stored in shared memory and is not
  // projected to inputs, the scheduler adds a cacheAfter to load tv1 from
  // shared memory to registers in a vectorized manner, avoiding bank conflicts.
  // However, since tv1 is float32, we can't directly use the vectorization
  // factor set for float16 inputs because the maximum allowed vectorization
  // width is 16 bytes.
  const int dim0 = 1024;
  const int dim1 = 4096;
  auto dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv3, tv1);
  auto tv5 = sum(tv1, {0});
  auto tv6 = castOp(DataType::Half, tv4);
  auto tv7 = castOp(DataType::Half, tv5);
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  Fusion fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);

  SchedulerRuntimeInfo runtime_info(&fusion, {t0});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerOuterPersistent, &fusion, runtime_info));
  auto scheduler = SchedulerEntry::makeSchedulerInstance(
      SchedulerType::InnerOuterPersistent);
  auto heuristic_params = scheduler->computeHeuristics(&fusion, runtime_info);

  // disable projection to inputs, so shared memory buffer is using float32
  heuristic_params->as<ReductionParams>()->project_persistent_buffers = false;
  // Set vectorization factor to 8, so the exent of the innermost dimension
  // exceed 16 bytes (8 x 4 = 32 bytes).
  heuristic_params->as<ReductionParams>()->unroll_factor_inner_reduction = 8;
  // when compute heuristics, the buffer is projected to inputs and the shared
  // memory persistent buffer is the input, tv0. Then, we modified the
  // heuristics to disable project to inputs, so needs to update the buffer
  // being stored in shared memory to the original unprojected buffer, tv1.
  heuristic_params->as<ReductionParams>()->smem_persistent_buffers =
      std::vector<TensorView*>{tv1};
  scheduler->schedule(&fusion, heuristic_params.get());
  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  for (auto tv : fusion.allTvs()) {
    if (tv->getMemoryType() == MemoryType::Shared) {
      for (auto consumer : ir_utils::consumerTvsOf(tv)) {
        EXPECT_TRUE(isVectorized(consumer));
      }
    }
  }
  auto cg_outputs =
      ke.run({t0}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&fusion_copy, cg_outputs, {t0}, __LINE__, __FILE__);
}

using InnerOuterReshapeTest = NVFuserFixtureParamTest<bool>;
INSTANTIATE_TEST_SUITE_P(
    ,
    InnerOuterReshapeTest,
    testing::Bool(),
    testing::PrintToStringParamName());
TEST_P(InnerOuterReshapeTest, ReshapeOuterDimTrueOrFalse) {
  auto reshape_outer_dim = GetParam();
  Fusion fusion;
  FusionGuard fg(&fusion);
  // reshape a 3D input tensor to 2D
  // [4, 1024, 4096] -> [4096, 4096]
  // [4096, 4, 1024] -> [4096, 4096]
  const int dim0 = reshape_outer_dim ? 4 : 4096;
  const int dim1 = reshape_outer_dim ? 1024 : 4;
  const int dim2 = reshape_outer_dim ? 4096 : 1024;
  auto dtype = DataType::Half;
  auto tv0 = makeContigTensor(3, dtype);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);

  auto tv4 = reshape(tv1, {dim0, dim1, dim2}, {4096, 4096});

  auto tv5 = sum(tv4, {1});
  auto tv6 = broadcast(tv5, {false, true});
  auto tv7 = add(tv6, tv4);
  auto tv8 = sum(tv4, {0});
  auto tv9 = castOp(DataType::Half, tv7);
  auto tv10 = castOp(DataType::Half, tv8);
  fusion.addOutput(tv9);
  fusion.addOutput(tv10);

  Fusion fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1, dim2}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::InnerOuterPersistent, {t0});
  auto persistent_params = cg_results.heuristic_params->as<ReductionParams>();
  ASSERT_FALSE(persistent_params->project_persistent_buffers);
  testValidate(&fusion_copy, cg_results.outputs, {t0}, __LINE__, __FILE__);
}

// contig, dtype, dim0, dim1
using TmaWarpSpecializedParams = std::tuple<bool, DataType, int64_t, int64_t>;
class TmaWarpSpecializedTest
    : public NVFuserFixtureParamTest<TmaWarpSpecializedParams> {
 public:
  void SetUp() override {
    opt_guard_ = std::make_unique<EnableOptionsGuard>();
    EnableOptionsGuard::getCurOptions().set(
        EnableOption::WarpSpecializedNormalization);
    NVFuserTest::SetUp();
  }

  void validateHeuristics(FusionKernelRuntime* runtime) {
    EXPECT_THAT(
        runtime->fusionSegments()->groups(),
        UnorderedElementsAre(HeuristicIs(SchedulerType::InnerOuterPersistent)));
    HeuristicParams* heur =
        runtime->schedulerHeuristics()->heuristicsList().at(0).get();
    ASSERT_NE(heur, nullptr);
    ASSERT_TRUE(heur->isA<ReductionParams>());
    ReductionParams* rparams = heur->as<ReductionParams>();
    EXPECT_TRUE(rparams->computation_warp_groups > 1);
  }

 protected:
  // This keeps the guard alive until all TmaWarpSpecializedTests are done.
  std::unique_ptr<EnableOptionsGuard> opt_guard_;
};

TEST_P(TmaWarpSpecializedTest, SimpleFusion) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto [contig, dtype, dim0, dim1] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, dim1}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  tv0 = maybeCastOp(DataType::Float, tv0);
  tv1 = maybeCastOp(DataType::Float, tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv2, tv4);
  auto tv6 = sum(tv1, {0});
  tv5 = maybeCastOp(dtype, tv5);
  fusion->addOutput(tv5);
  fusion->addOutput(tv6);
  auto fusion_copy = *fusion;

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = at::randn({dim0, dim1}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  validateHeuristics(runtime);
  testValidate(&fusion_copy, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_P(TmaWarpSpecializedTest, RMSNormBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto [contig, dtype, dim0, dim1] = GetParam();

  std::vector<int64_t> norm_shape{dim1};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto grad_out = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto input = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto rstd = contig ? makeContigConcreteTensor({dim0, 1})
                     : makeConcreteTensor({dim0, 1});
  auto weight = makeContigTensor(1, dtype);
  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(rstd);
  fusion->addInput(weight);

  grad_out = maybeCastOp(DataType::Float, grad_out);
  input = maybeCastOp(DataType::Float, input);
  weight = maybeCastOp(DataType::Float, weight);
  auto grads = rms_norm_backward(
      grad_out, input, norm_shape, rstd, weight, {true, true});
  grads.grad_input = maybeCastOp(dtype, grads.grad_input);
  grads.grad_weight = maybeCastOp(dtype, grads.grad_weight);
  fusion->addOutput(grads.grad_input);
  fusion->addOutput(grads.grad_weight);

  auto fusion_copy = *fusion;
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  std::vector<int64_t> shape{dim0, dim1};
  at::Tensor aten_grad_out = at::randn(shape, options);
  at::Tensor aten_input = at::randn(shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  const float kEps = 1e-6;
  auto pow2 = at::pow(aten_input.to(at::kFloat), 2);
  auto sum = at::sum(pow2, -1, true);
  auto var = at::mul(sum, 1.0 / dim1);
  auto aten_rstd = at::pow(at::add(var, kEps), -0.5);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {
      aten_grad_out, aten_input, aten_rstd, aten_weight};
  auto cg_outputs = executor_cache.runFusionWithInputs(args);
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::InnerOuterPersistent)));
  testValidate(
      &fusion_copy,
      cg_outputs,
      {aten_grad_out, aten_input, aten_rstd, aten_weight},
      __LINE__,
      __FILE__);
}

TEST_P(TmaWarpSpecializedTest, ThunderRMSNormBwd) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto [contig, dtype, dim0, dim1] = GetParam();

  std::vector<int64_t> norm_shape{dim1};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto grad_out = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto input = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto rms = contig ? makeContigConcreteTensor({dim0, 1})
                    : makeConcreteTensor({dim0, 1});
  auto weight = makeContigConcreteTensor({dim1}, dtype);
  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(rms);
  fusion->addInput(weight);

  grad_out = maybeCastOp(DataType::Float, grad_out);
  input = maybeCastOp(DataType::Float, input);
  weight = maybeCastOp(DataType::Float, weight);
  auto grads = thunder_rms_norm_backward(
      grad_out, input, norm_shape, rms, weight, {true, true});
  grads.grad_input = maybeCastOp(dtype, grads.grad_input);
  grads.grad_weight = maybeCastOp(dtype, grads.grad_weight);
  fusion->addOutput(grads.grad_input);
  fusion->addOutput(grads.grad_weight);

  auto fusion_copy = *fusion;
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  std::vector<int64_t> shape{dim0, dim1};
  at::Tensor aten_grad_out = at::randn(shape, options);
  at::Tensor aten_input = at::randn(shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  const float kEps = 1e-6;
  auto pow2 = at::pow(aten_input.to(at::kFloat), 2);
  auto sum = at::sum(pow2, -1, true);
  auto var = at::mul(sum, 1.0 / dim1);
  auto aten_rms = at::pow(at::add(var, kEps), 0.5);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {
      aten_grad_out, aten_input, aten_rms, aten_weight};
  auto cg_outputs = executor_cache.runFusionWithInputs(args);
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::InnerOuterPersistent)));
  testValidate(
      &fusion_copy,
      cg_outputs,
      {aten_grad_out, aten_input, aten_rms, aten_weight},
      __LINE__,
      __FILE__);
}
TEST_P(TmaWarpSpecializedTest, LayerNormBackward) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto [contig, dtype, dim0, dim1] = GetParam();

  std::vector<int64_t> norm_shape{dim1};
  std::vector<int64_t> input_shape{dim0, dim1};
  std::vector<int64_t> outer_shape{dim0, 1};
  auto grad_out = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto input = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto mean = contig ? makeContigConcreteTensor(outer_shape)
                     : makeConcreteTensor(outer_shape);
  auto rstd = contig ? makeContigConcreteTensor(outer_shape)
                     : makeConcreteTensor(outer_shape);
  auto weight = makeContigConcreteTensor(norm_shape, dtype);
  auto bias = makeContigConcreteTensor(norm_shape, dtype);
  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(mean);
  fusion->addInput(rstd);
  fusion->addInput(weight);
  fusion->addInput(bias);
  grad_out = maybeCastOp(DataType::Float, grad_out);
  input = maybeCastOp(DataType::Float, input);
  weight = maybeCastOp(DataType::Float, weight);
  bias = maybeCastOp(DataType::Float, bias);

  auto res = layer_norm_backward(
      grad_out,
      input,
      norm_shape,
      mean,
      rstd,
      weight,
      bias,
      {true, true, true});
  res.grad_input = maybeCastOp(dtype, res.grad_input);
  res.grad_weight = maybeCastOp(dtype, res.grad_weight);
  res.grad_bias = maybeCastOp(dtype, res.grad_bias);
  fusion->addOutput(res.grad_input);
  fusion->addOutput(res.grad_weight);
  fusion->addOutput(res.grad_bias);
  auto fusion_copy = *fusion;

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor aten_grad_out = at::randn(input_shape, options);
  at::Tensor aten_input = at::randn(input_shape, options);
  at::Tensor aten_weight = at::randn(norm_shape, options);
  at::Tensor aten_bias = at::randn(norm_shape, options);

  constexpr float kEps = 1e-5;
  auto aten_results = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);
  auto aten_output = std::get<0>(aten_results);
  auto aten_mean = std::get<1>(aten_results);
  auto aten_rstd = std::get<2>(aten_results);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {
      aten_grad_out, aten_input, aten_mean, aten_rstd, aten_weight, aten_bias};
  auto cg_outputs = executor_cache.runFusionWithInputs(args);
  testValidate(&fusion_copy, cg_outputs, args, __LINE__, __FILE__);
}
auto TmaWarpSpecializedTestParams() {
  std::vector<TmaWarpSpecializedParams> values;
  int64_t dim0 = 2048;
  for (int64_t dim1 = 1024; dim1 <= 8192; dim1 += 256) {
    for (bool contig : {true, false}) {
      // to save test time
      if (dim1 != 1024 && !contig) {
        continue;
      }
      for (auto dtype : {DataType::Float, DataType::BFloat16}) {
        values.emplace_back(contig, dtype, dim0, dim1);
      }
    }
  }
  return testing::ValuesIn(values);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    TmaWarpSpecializedTest,
    TmaWarpSpecializedTestParams(),
    [](const testing::TestParamInfo<TmaWarpSpecializedParams>& info)
        -> std::string {
      std::stringstream ss;
      ss << "contig_" << std::get<0>(info.param);
      ss << "_dtype_" << std::get<1>(info.param);
      ss << "_batch_" << std::get<2>(info.param);
      ss << "_hidden_" << std::get<3>(info.param);
      return sanitizeTestName(ss.str());
    });

TEST(StaticWarpReductionTest, StaticWarpReductionValidation) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  // Enable warp specialization explicitly
  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(
      EnableOption::WarpSpecializedNormalization);

  int64_t dim0 = 2048;
  int64_t dim1 = 8192;
  DataType dtype = DataType::Float;

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, dim1}, dtype);
  fusion_ptr->addInput(tv0);
  fusion_ptr->addInput(tv1);
  tv0 = maybeCastOp(DataType::Float, tv0);
  tv1 = maybeCastOp(DataType::Float, tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv2, tv4);
  auto tv6 = sum(tv1, {0});
  tv5 = maybeCastOp(dtype, tv5);
  fusion_ptr->addOutput(tv5);
  fusion_ptr->addOutput(tv6);
  auto fusion_copy = *fusion_ptr;

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dim0, dim1}, options);
  at::Tensor t1 = at::randn({dim0, dim1}, options);

  // Get default heuristics and revise unroll_factor_iter_dom to 1
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0, t1});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerOuterPersistent, fusion_ptr.get(), runtime_info));
  auto scheduler = SchedulerEntry::makeSchedulerInstance(
      SchedulerType::InnerOuterPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);

  // Revise unroll_factor_iter_dom to 1 to enable static warp reduction
  heuristic_params->as<ReductionParams>()->unroll_factor_iter_dom = 1;

  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto outputs =
      ke.run({t0, t1}, {}, heuristic_params->as<ReductionParams>()->lparams);

  // Validate that static warp reduction is used by checking the generated code
  std::string kernel_code = ke.compiledKernel()->kernelString();

  // Check for the specific static warp reduction function call
  EXPECT_TRUE(
      kernel_code.find("warp::staticWarpAllReduceTIDX<false, false") !=
      std::string::npos)
      << "Static warp reduction function 'warp::staticWarpAllReduceTIDX<false, "
         "false' not found in kernel code";

  testValidate(&fusion_copy, outputs, {t0, t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
