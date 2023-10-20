#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <inlining.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>

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

// This case is to test the correctness of the combined inner and outer
// scheduler used in layer norm backward. It can also be configured to test the
// performance using different data types.
TEST_F(NVFuserTest, CombinedSchedulerLayerNormBackward_CUDA) {
  auto runTest = [](const std::vector<int64_t>& batch_shape,
                    const std::vector<int64_t>& norm_shape,
                    DataType dtype) {
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
    for (const auto idx : c10::irange(kOuterNumDims)) {
      outer_shape.push_back(input_shape[idx]);
    }
    for (const auto idx : c10::irange(kOuterNumDims, kM)) {
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

    fusion.addOutput(layer_norm_results.grad_input);
    fusion.addOutput(layer_norm_results.grad_weight);
    fusion.addOutput(layer_norm_results.grad_bias);

    auto maybe_fp16_options = at::TensorOptions()
                                  .dtype(data_type_to_aten(dtype))
                                  .device(at::kCUDA, 0);

    // Reduce the scale to avoid fp16 overflow. In segmented scenarios,
    // intermediates across different segments are saved in fp16. Input down
    // scaling is essential to avert fp16 overflow. Refer to:
    // https://github.com/NVIDIA/Fuser/issues/704
    constexpr float scale_down_factor = 0.01;
    constexpr float scale_back_factor = 1.0 / scale_down_factor;
    at::Tensor aten_grad_out = at::randn(input_shape, maybe_fp16_options)
                                   .mul(scale_down_factor)
                                   .to(data_type_to_aten(dtype));
    at::Tensor aten_input = at::randn(input_shape, maybe_fp16_options)
                                .mul(scale_down_factor)
                                .to(data_type_to_aten(dtype));
    at::Tensor aten_weight = at::randn(norm_shape, maybe_fp16_options)
                                 .mul(scale_down_factor)
                                 .to(data_type_to_aten(dtype));
    at::Tensor aten_bias = at::randn(norm_shape, maybe_fp16_options)
                               .mul(scale_down_factor)
                               .to(data_type_to_aten(dtype));

    auto at_weight = c10::optional<at::Tensor>(aten_weight);
    auto at_bias = c10::optional<at::Tensor>(aten_bias);

    const float kEps = 1e-5;
    auto aten_results =
        at::native_layer_norm(aten_input, norm_shape, at_weight, at_bias, kEps);
    auto aten_output = std::get<0>(aten_results);
    auto aten_mean = std::get<1>(aten_results);
    auto aten_rstd = std::get<2>(aten_results);

    FusionExecutorCache fec(std::move(fusion_ptr));
    std::vector<c10::IValue> aten_inputs = {
        aten_grad_out,
        aten_input,
        aten_mean,
        aten_rstd,
        aten_weight,
        aten_bias};
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

    auto aten_gradients = at::native_layer_norm_backward(
        aten_grad_out,
        aten_input,
        norm_shape,
        aten_mean,
        aten_rstd,
        c10::optional<at::Tensor>(aten_weight),
        c10::optional<at::Tensor>(aten_bias),
        {true, true, true});

    testValidate(
        &fusion,
        {cg_outputs[0].mul(scale_back_factor),
         cg_outputs[1].mul(scale_back_factor),
         cg_outputs[2].mul(scale_back_factor)},
        aten_inputs,
        {std::get<0>(aten_gradients).mul(scale_back_factor),
         std::get<1>(aten_gradients).mul(scale_back_factor),
         std::get<2>(aten_gradients).mul(scale_back_factor)},
        __LINE__,
        __FILE__);

    //  should use shared memory if the register file is insufficient but there
    //  is ample space in shared memory.
    int64_t hidden_size = 1l;
    for (int64_t dim : norm_shape) {
      hidden_size *= dim;
    }

    int64_t persistent_buffer_size = hidden_size *
        (dtype == DataType::Half ? 14l : (dtype == DataType::Float ? 20l : 0l));
    ASSERT_TRUE(persistent_buffer_size) << "Unsupported data type!";

    if (persistent_buffer_size > register_file_size_combined) {
      auto dev_prop = at::cuda::getCurrentDeviceProperties();
      int64_t available_smem = (int64_t)dev_prop->sharedMemPerBlockOptin -
          scheduler_utils::getSharedMemoryOverheadPerBlock(
                                   &fusion,
                                   scheduler_utils::getReductionTvs(&fusion),
                                   max_threads_per_block_combined);

      if (available_smem >= persistent_buffer_size) {
        const auto& kernel_runtime = fec.getMostRecentKernelRuntime();
        ASSERT_TRUE(!kernel_runtime->isSegmented())
            << "Should not segment! hidden_size: " << hidden_size
            << ", dataTypeSize: " << dataTypeSize(dtype);
        auto heuristic_params = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .at(0)
                                    ->params();
        ASSERT_TRUE(heuristic_params->isA<ReductionParams>());
        auto rparams = heuristic_params->as<ReductionParams>();
        ASSERT_TRUE(rparams->shared_mem_persistent_buffer)
            << "Should use shared memory buffer! hidden_size: " << hidden_size
            << ", dataTypeSize: " << dataTypeSize(dtype);
      }
    }
  };

  std::vector<DataType> data_types = {DataType::Half, DataType::Float};
  std::vector<std::vector<int64_t>> batch_sizes = {{216}};
  std::vector<std::vector<int64_t>> hidden_sizes = {
      {3},
      {32},
      {96},
      {576},
      {768},
      {1024},
      {1280},
      {1600},
      {1984},
      {1987},
      {16384}, //! use shared memory for persistent
      {32768}, //! segment and the inner reduction part has 2 persistent tensors
      {65536}};
  for (auto dtype : data_types) {
    for (auto batch_shape : batch_sizes) {
      for (auto norm_shape : hidden_sizes) {
        runTest(batch_shape, norm_shape, dtype);
      }
    }
  }
}

// This case is to test the correctness of the combined inner and outer
// scheduler, if link_inner_outer = true, the inner and outer reductions are
// linked, otherwise the two outer reductions are linked. In either case, the
// fusion should be segmented since the current combined scheduler assumes there
// is no shared consumer between inter reductions and outer reductions and among
// tensors in outer reductions.
TEST_F(NVFuserTest, CombinedSchedulerSharedConsumer_CUDA) {
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
    for (const auto idx : c10::irange(kOuterNumDims)) {
      outer_shape.push_back(input_shape[idx]);
    }
    for (const auto idx : c10::irange(kOuterNumDims, kM)) {
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

    FusionExecutorCache fec(std::move(fusion_ptr));
    std::vector<c10::IValue> aten_inputs = {
        aten_grad_out,
        aten_input,
        aten_mean,
        aten_rstd,
        aten_weight,
        aten_bias};
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

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
    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    NVF_CHECK(is_segmented, "Fusion is not segmented");

    testValidate(
        &fusion,
        cg_outputs,
        aten_inputs,
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
TEST_F(NVFuserTest, CombinedSchedulerSharedProducer_CUDA) {
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
    for (const auto idx : c10::irange(kOuterNumDims)) {
      outer_shape.push_back(input_shape[idx]);
    }
    for (const auto idx : c10::irange(kOuterNumDims, kM)) {
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
        // tensor bias is a producer of the inner reduction and also a produer
        // of a consumer of the outer reduction results this a not allowed,
        // expect segmented
        auto bias_broad = add(bias, mean);
        auto use_inner = sum(bias_broad, {-1});
        auto use_outer = add(layer_norm_results.grad_weight, bias);
        fusion.addOutput(use_inner);
        fusion.addOutput(use_outer);
      } break;
      case 2: {
        // tensor bias is a producer of the outer reduction and also a produer
        // of a consumer of the inner reduction results this a allowed, becase
        // the first part of outer reduction is computed with inner reduction.
        // expect unsegmented
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
        NVF_ERROR(false, "Invalid case id");
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

    FusionExecutorCache fec(std::move(fusion_ptr));
    std::vector<c10::IValue> aten_inputs = {
        aten_grad_out,
        aten_input,
        aten_mean,
        aten_rstd,
        aten_weight,
        aten_bias};
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

    auto aten_gradients = at::native_layer_norm_backward(
        aten_grad_out,
        aten_input,
        norm_shape,
        aten_mean,
        aten_rstd,
        c10::optional<at::Tensor>(aten_weight),
        c10::optional<at::Tensor>(aten_bias),
        {true, true, true});

    // check the results depending on the case
    at::Tensor aten_use_inner, aten_use_outer;
    bool expected_segmented;
    switch (case_id) {
      case 0: {
        aten_use_inner = std::get<0>(aten_gradients) + aten_input;
        aten_use_outer = std::get<1>(aten_gradients) + aten_input;
        expected_segmented = true;
      } break;
      case 1: {
        aten_use_inner = (aten_bias + aten_mean).sum({-1});
        aten_use_outer = std::get<1>(aten_gradients) + aten_bias;
        expected_segmented = true;
      } break;
      case 2: {
        aten_use_inner = std::get<0>(aten_gradients) + (aten_bias + aten_mean);
        aten_use_outer = (aten_bias + aten_mean).sum({0});
        expected_segmented = false;
      } break;
      case 3: {
        aten_use_inner = std::get<1>(aten_gradients) + (aten_bias + 1.0);
        aten_use_outer = std::get<2>(aten_gradients) + (aten_bias + 1.0);
        expected_segmented = true;
      } break;
      default:
        NVF_ERROR(false, "Invalid case id");
    }
    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    NVF_CHECK(
        is_segmented == expected_segmented,
        expected_segmented ? "Fusion should be segmented!"
                           : "Fusion should not be segmented!");

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
        aten_inputs,
        {aten_use_inner,
         aten_use_outer,
         std::get<0>(aten_gradients),
         std::get<1>(aten_gradients),
         std::get<2>(aten_gradients)},
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
TEST_F(NVFuserTest, CombinedReduction_CUDA) {
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
      &fusion,
      inner_reduction_tv,
      reference_tv_inner,
      true,
      true,
      false,
      inner_reduction_tvs,
      cached_inputs,
      cached_outputs);
  reduction_scheduler_utils::propagateParallelization(
      &fusion,
      outer_reduction_tv,
      reference_tv_outer,
      true,
      true,
      false,
      outer_reduction_tvs,
      cached_inputs,
      cached_outputs);

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
  FusionExecutor fe;
  fe.compileFusion(&fusion, {tv_input}, launch_constraints, compile_params);
  fe.runFusion(
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
TEST_F(NVFuserTest, CombinedReductionMultiPerBlock_CUDA) {
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

  reduction_scheduler_utils::propagateTransformation(
      reference_tv_inner, {partialResultReload});
  const auto& selected_tvs_inner = scheduler_utils::getAllTvsFrom(
      inner_reduction_tvs, {partialResultReload});
  reduction_scheduler_utils::propagateParallelization(
      &fusion,
      inner_reduction_tv,
      reference_tv_inner,
      true,
      true,
      false,
      inner_reduction_tvs,
      cached_inputs,
      cached_outputs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  const auto& selected_tvs_outer =
      scheduler_utils::getAllTvsFrom(outer_reduction_tvs, {partialResult});
  reduction_scheduler_utils::propagateTransformation(
      reference_tv_outer, {partialResultReload});
  reduction_scheduler_utils::propagateParallelization(
      &fusion,
      outer_reduction_tv,
      reference_tv_outer,
      true,
      true,
      false,
      outer_reduction_tvs,
      cached_inputs,
      cached_outputs,
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
  FusionExecutor fe;
  fe.compileFusion(&fusion, {tv_input}, launch_constraints, compile_params);
  fe.runFusion(
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
TEST_F(NVFuserTest, CombinedSchedulerInnerOuterMismatch) {
  auto test = [](const std::vector<int>& outer_reduction_axis) {
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
    std::vector<c10::IValue> aten_inputs = {t0};

    FusionExecutorCache fec(std::move(fusion_ptr));
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    if (outer_reduction_axis.size() == 2) {
      NVF_ERROR(!is_segmented, "Fusion should NOT be segmented!");
    } else {
      NVF_ERROR(is_segmented, "Fusion should be segmented!");
    }

    std::vector<int64_t> vec64(
        outer_reduction_axis.begin(), outer_reduction_axis.end());
    auto t1 = t0.sum({-1});
    auto t2 = t1.unsqueeze(-1);
    auto t3 = t0 + t2;
    auto t4 = t0.sum(vec64);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {t3, t4}, __LINE__, __FILE__);
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

// In this test, all the 3 input tvs are persistent.
// tv2 is broadcasted in outer dim, it will be move to shared memory first to
// minimize gmem to smem traffic. tv1 has 1 direct consumer and tv2 has 2
// direct consumers, so tv1 has higher priority than tv2 to minimize smem to
// register traffic.
TEST_F(NVFuserTest, TMP) {
  auto test = [](int hidden_size) {
    auto dtype = DataType::Float;
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(2, dtype);
    auto tv1 = makeContigTensor(2, dtype);
    auto tv2 = makeContigTensor(1, dtype);
    fusion.addInput(tv0);
    fusion.addInput(tv1);
    fusion.addInput(tv2);
    auto tv3 = set(tv1);
    auto tv4 = broadcast(tv2, {true, false});
    auto tv5 = add(tv0, tv4);
    auto tv6 = sum(tv5, {-1});
    auto tv7 = add(tv3, tv5);
    auto tv8 = sum(tv7, {-1});
    auto tv9 = add(tv0, tv3);
    auto tv10 = sum(tv9, {0});
    auto tv11 = broadcast(tv6, {false, true});
    auto tv12 = add(tv11, tv5);
    auto tv13 = broadcast(tv8, {false, true});
    auto tv14 = add(tv13, tv7);
    fusion.addOutput(tv12);
    fusion.addOutput(tv14);
    fusion.addOutput(tv10);

    auto options = at::TensorOptions()
                       .dtype(data_type_to_aten(dtype))
                       .device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({528, hidden_size}, options);
    at::Tensor t1 = at::randn({528, hidden_size}, options);
    at::Tensor t2 = at::randn({hidden_size}, options);
    std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

    FusionExecutorCache fec(std::move(fusion_ptr));
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

    const auto& kernel_runtime = fec.getMostRecentKernelRuntime();
    if (!kernel_runtime->isSegmented()) {
      auto heuristic_params = kernel_runtime->schedulerHeuristics()
                                  ->heuristicsList()
                                  .at(0)
                                  ->params();
      ASSERT_TRUE(heuristic_params->isA<ReductionParams>());
      auto rparams = heuristic_params->as<ReductionParams>();
      // if shared memory persistent is being used, check which tv is stored in
      // shared memory. since the priority order should be tv2, tv1, tv0.
      // if there is only 1, it must be tv2; if there are 2, they must be tv2
      // and tv1.
      if (rparams->shared_mem_persistent_buffer) {
        const auto& sm_tvs = rparams->smem_persistent_tvs;
        if (sm_tvs.size() == 1) {
          NVF_ERROR(
              sm_tvs.at(0)->name() == tv2->name(),
              "tv2 should be shared memory persistent!");
        } else if (sm_tvs.size() == 2) {
          NVF_ERROR(
              sm_tvs.at(0)->name() != tv0->name() &&
                  sm_tvs.at(1)->name() != tv0->name(),
              "tv0 shouldn't be shared memory persistent!");
        }
      }
      auto t3 = t1;
      auto t4 = t2.unsqueeze(0);
      auto t5 = t0 + t4;
      auto t6 = t5.sum({-1});
      auto t7 = t3 + t5;
      auto t8 = t7.sum({-1});
      auto t9 = t0 + t3;
      auto t10 = t9.sum({0});
      auto t11 = t6.unsqueeze(-1);
      auto t12 = t11 + t5;
      auto t13 = t8.unsqueeze(-1);
      auto t14 = t13 + t7;
      testValidate(
          &fusion,
          cg_outputs,
          aten_inputs,
          {t12, t14, t10},
          __LINE__,
          __FILE__);
    }
  };

  // on H100, only tv2 is moved to shared memory, latency is 0.261 ms, will
  // increase to 0.311 ms if reverse order in sort_buffer_tvs. This was tested
  // on a pre-production card, only relative change matters.
  test(18 * 1024);

  // on H100, only tv2 and tv1 are moved to shared memory, latency is 0.436 ms,
  // will increase to 0.481 ms if reverse order in sort_buffer_tvs. This was
  // tested on a pre-production card, only relative change matters.
  test(27 * 1024);
}
} // namespace nvfuser
