#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <inlining.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
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

// mean & var
std::tuple<float, float> getMeanVar(const std::vector<float>& v) {
  const int nele = v.size();
  float mean = std::accumulate(v.begin(), v.end(), 0.0f) / nele;
  std::vector<float> sub_mean(nele);
  std::transform(v.begin(), v.end(), sub_mean.begin(), [mean](float x) {
    return x - mean;
  });
  float sq_sum = std::inner_product(
      sub_mean.begin(), sub_mean.end(), sub_mean.begin(), 0.0);
  float stdev = std::sqrt(sq_sum / nele);
  return {mean, stdev};
}

// This case is to test the correctness of the combined inner and outer
// scheduler used in layer norm backward. It can also be configured to test the
// performance using different data types.
TEST_F(NVFuserTest, CombinedSchedulerLayerNormBackward_CUDA) {
  auto runTest = [](const std::vector<int64_t>& batch_shape,
                    const std::vector<int64_t>& norm_shape,
                    DataType dtype,
                    bool isBenchmark,
                    int verbose) {
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

    testValidate(
        &fusion,
        cg_outputs,
        aten_inputs,
        {std::get<0>(aten_gradients),
         std::get<1>(aten_gradients),
         std::get<2>(aten_gradients)},
        __LINE__,
        __FILE__);

    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    TORCH_CHECK(!is_segmented, "Fusion is segmented");

    if (isBenchmark) {
      FusionKernelRuntime* fkr = fec.getMostRecentKernelRuntime();
      fkr->setMeasureKernelTime(true);

      constexpr int nwarm = 5;
      constexpr int niter = 10;
      std::vector<float> bw(niter, 0.f);
      std::vector<float> timeus(niter, 0.f);

      size_t read_write_bytes = 0;
      const std::vector<at::Tensor> aten_inputs_tmp = {
          aten_grad_out,
          aten_input,
          aten_mean,
          aten_rstd,
          aten_weight,
          aten_bias};
      const std::vector<at::Tensor> aten_output_tmp = {
          std::get<0>(aten_gradients),
          std::get<1>(aten_gradients),
          std::get<2>(aten_gradients)};
      for (auto input : aten_inputs_tmp) {
        read_write_bytes += input.numel() * input.element_size();
      }
      for (auto output : aten_output_tmp) {
        read_write_bytes += output.numel() * output.element_size();
      }

      for (int i = 0; i < nwarm + niter; i++) {
        clearL2Cache();
        // fe.runFusion(inputs, outputs, launch_constraints);
        auto cg_outputs = fec.runFusionWithInputs(aten_inputs);
        if (i >= nwarm) {
          float runTimeus = 0.0f;
          int num_kernels = fkr->executors().size();
          for (int i = 0; i < num_kernels; i++) {
            const FusionExecutor& fe = fkr->executors()[i];
            runTimeus += fe.kernelTimeMs() * 1e3;
          }
          float bandwidth = read_write_bytes / 1e9 / (runTimeus * 1e-6);
          timeus[i - nwarm] = runTimeus;
          bw[i - nwarm] = bandwidth;
          if (verbose == 2)
            std::cout << "iter= " << i << ", bandwidth= " << bandwidth << "GB/s"
                      << ", time= " << runTimeus << " us" << std::endl;
        }
      }
      return getMeanVar(timeus);
    } else {
      if (verbose == 1) {
        std::stringstream sdim0, sdim1;
        std::for_each(
            batch_shape.begin(), batch_shape.end(), [&sdim0](int64_t n) {
              sdim0 << n << " x ";
            });
        std::for_each(
            norm_shape.begin(), norm_shape.end(), [&sdim1](int64_t n) {
              sdim1 << n << " x ";
            });
        std::string str1 = sdim1.str();
        str1.erase(str1.end() - 2);
        std::cout << "passed, shape= " << sdim0.str() << str1 << std::endl;
      }
      return std::make_tuple(-1.0f, -1.0f);
    }
  };

  std::vector<DataType> data_types = {DataType::Half, DataType::Float};
  std::vector<std::vector<int64_t>> batch_sizes = {{216}};
  std::vector<std::vector<int64_t>> hidden_sizes = {
      {576}, {768}, {1024}, {1280}, {1600}};

  bool isBenchmark = false;
  bool onlyTestFirstCase = false;
  int verbose = 0;
  for (auto dtype : data_types) {
    for (auto batch_shape : batch_sizes) {
      for (auto norm_shape : hidden_sizes) {
        std::tuple<float, float> avg_var =
            runTest(batch_shape, norm_shape, dtype, isBenchmark, verbose);
        if (isBenchmark) {
          std::stringstream sdim0, sdim1;
          std::for_each(
              batch_shape.begin(), batch_shape.end(), [&sdim0](int64_t n) {
                sdim0 << n << " x ";
              });
          std::for_each(
              norm_shape.begin(), norm_shape.end(), [&sdim1](int64_t n) {
                sdim1 << n << " x ";
              });
          std::cout << "shape= " << sdim0.str() << sdim1.str()
                    << ", time_us mean(var)= " << std::get<0>(avg_var) << " ("
                    << std::get<1>(avg_var) << ")" << std::endl;
        }
        if (onlyTestFirstCase)
          break;
      }
      if (onlyTestFirstCase)
        break;
    }
    if (onlyTestFirstCase)
      break;
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

    fusion.addOutput(out_linked);
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

    auto aten_out_linked = link_inner_outer
        ? std::get<0>(aten_gradients) + std::get<1>(aten_gradients)
        : std::get<1>(aten_gradients) + std::get<2>(aten_gradients);

    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    TORCH_CHECK(is_segmented, "Fusion is not segmented");

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
            add(layer_norm_results.grad_weight, IrBuilder::create<Double>(1));
        auto outer_2_consumer =
            add(layer_norm_results.grad_bias, IrBuilder::create<Double>(1));
        auto use_producer_1 = add(outer_1_consumer, bias);
        auto use_producer_2 = add(outer_2_consumer, bias);
        fusion.addOutput(use_producer_1);
        fusion.addOutput(use_producer_2);
      } break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Invalid case id");
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
        TORCH_INTERNAL_ASSERT(false, "Invalid case id");
    }
    bool is_segmented = fec.getMostRecentKernelRuntime()->isSegmented();
    TORCH_CHECK(
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

} // namespace nvfuser
