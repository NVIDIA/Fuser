// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_kernel_runtime.h>
#include <tests/cpp/validator.h>
#include <validator_utils.h>

namespace nvfuser {

void testValidate(
    Fusion* fusion,
    const KernelArgumentHolder& fusion_outputs,
    const KernelArgumentHolder& aten_inputs,
    std::vector<at::Tensor> aten_outputs,
    int line_number,
    const char* file_name,
    std::string err_msg,
    const LaunchParams& lparams,
    const ValidationConstants& tolerances) {
  FusionGuard fg(fusion);

  std::vector<Val*> non_hidden_outputs;
  std::copy_if(
      fusion->outputs().begin(),
      fusion->outputs().end(),
      std::back_inserter(non_hidden_outputs),
      [fusion](Val* out) {
        // Returns true when `out` is **not** an aliased output that's hidden
        // from integration. Hidden outputs won't show up in `fusion_outputs`
        // for us to compare, so we skip them.
        return !fusion->getOutputAlias(out).hide_output;
      });

  auto expr_eval = bindInputsAndLaunchParams(fusion, aten_inputs, lparams);

  auto reduction_sizes =
      ReductionSizeMapper::computeReductionSizes(fusion, expr_eval);

  if (aten_outputs.empty()) {
    for (Val* out : non_hidden_outputs) {
      aten_outputs.push_back(expr_eval.evaluate(out).as<at::Tensor>());
    }
  }

  NVF_ERROR_EQ(
      fusion_outputs.size(),
      std::ssize(aten_outputs),
      "Number of outputs don't match.");

  NVF_ERROR(
      std::ssize(fusion->inputs()) == aten_inputs.size(),
      "Number of inputs don't match.");

  for (auto i : arange(fusion->inputs().size())) {
    if (fusion->inputs()[i]->isA<TensorView>()) {
      NVF_ERROR(aten_inputs[i].is<at::Tensor>(), "Mismatch of tensor inputs.");

      auto fusion_input_tv = fusion->inputs()[i]->as<TensorView>();
      auto at_tensor = aten_inputs[i].as<at::Tensor>();

      NVF_ERROR(
          at_tensor.dim() ==
              static_cast<int64_t>(TensorDomain::noReductions(
                                       fusion_input_tv->getLogicalDomain())
                                       .size()),
          "Dimensionality mismatch in inputs.");
    }
  }

  for (auto i : arange(non_hidden_outputs.size())) {
    Val* out = non_hidden_outputs[i];
    NVF_ERROR(out->isA<TensorView>());
    TensorView* out_tv = out->as<TensorView>();

    NVF_ERROR(
        fusion_outputs[i].is<at::Tensor>(),
        "Fusion output is not a tensor at index ",
        i);
    const at::Tensor& fusion_output_tensor = fusion_outputs[i].as<at::Tensor>();
    const at::Tensor& aten_output_tensor = aten_outputs[i];

    NVF_ERROR(
        reduction_sizes.count(out_tv),
        "Missed reduction size count on fusion output: ",
        out_tv->toString());

    int64_t reduction_size = reduction_sizes.at(out_tv);

    NVF_ERROR(
        aten_output_tensor.dim() == fusion_output_tensor.dim() &&
            fusion_output_tensor.dim() ==
                static_cast<int64_t>(
                    TensorDomain::noReductions(out_tv->getLogicalDomain())
                        .size()),
        "Dimensionality mismatch in outputs: ",
        aten_output_tensor.sizes(),
        " vs ",
        fusion_output_tensor.sizes());

    auto tolerance_values =
        getTolerance(out_tv->getDataType().value(), reduction_size, tolerances);

    if (aten_output_tensor.is_floating_point() ||
        aten_output_tensor.is_complex()) {
      auto common_dtype = aten_output_tensor.dtype();
      if (common_dtype == at::ScalarType::Float8_e4m3fn ||
          common_dtype == at::ScalarType::Float8_e5m2 ||
          common_dtype == at::ScalarType::Float8_e8m0fnu) {
        common_dtype = at::ScalarType::Float;
      }
      auto aten_output_in_common_dtype = aten_output_tensor.to(common_dtype);
      auto fusion_output_in_common_dtype = fusion_output_tensor.to(common_dtype);
      if (aten_output_tensor.dtype() == at::ScalarType::Float8_e8m0fnu ||
          fusion_output_tensor.dtype() == at::ScalarType::Float8_e8m0fnu) {
        // Unfortunately PyTorch's implementation of e8m0 casting mismatches with
        // the hardware implementation. So we can not check the equality of the
        // two tensors directly. Note that e8m0 can only represent 2^x, so we
        // check that the x for aten and fusion are off by at most 1.
        // e8m0 is always positive, however, other types can be zero, when
        // aten and fusion dtypes mismatch, we try our best to pick the one
        // that is not zero.
        at::Tensor numerator = aten_output_tensor.dtype() == at::ScalarType::Float8_e8m0fnu ?
            fusion_output_in_common_dtype :
            aten_output_in_common_dtype;
        at::Tensor denominator = aten_output_tensor.dtype() == at::ScalarType::Float8_e8m0fnu ?
            aten_output_in_common_dtype :
            fusion_output_in_common_dtype;
        at::Tensor ratio = (numerator / denominator).abs();
        NVF_ERROR(
            ratio.max().item<double>() <= 2.0,
            "\n",
            err_msg,
            ".\n  Validation error in output ",
            i,
            " on line ",
            line_number,
            " in file ",
            file_name,
            ".\n  Detected max ratio of: ",
            ratio.max().item<double>(),
            "\n   tolerance was set to 2");
        NVF_ERROR(
            ratio.min().item<double>() >= 0.5,
            "\n",
            err_msg,
            ".\n  Validation error in output ",
            i,
            " on line ",
            line_number,
            " in file ",
            file_name,
            ".\n  Detected min ratio of: ",
            ratio.min().item<double>(),
            "\n   tolerance was set to 0.5");
      } else {
        NVF_ERROR(
            aten_output_in_common_dtype.allclose(
                fusion_output_in_common_dtype,
                tolerance_values.second,
                tolerance_values.first,
                /*equal_nan=*/true),
            "\n",
            err_msg,
            "\nValidation error in output ",
            i,
            " on line ",
            line_number,
            " in file ",
            file_name,
            ".\n  Detected max abs error of: ",
            aten_output_in_common_dtype
                .sub(fusion_output_in_common_dtype)
                .abs()
                .max()
                .item()
                .to<double>(),
            "\n    absolute tolerance was set to ",
            tolerance_values.first,
            "\n    and relative tolerance set to ",
            tolerance_values.second);
      }
    } else {
      NVF_ERROR(
          aten_output_tensor.equal(
              fusion_output_tensor.to(aten_output_tensor.dtype())),
          "\n",
          err_msg,
          ".\n  Validation error in output ",
          i,
          " on line ",
          line_number,
          " in file ",
          file_name,
          ".\n Values are not equal and are not a floating type.");
    }
  }
}

void testValidate(
    Fusion* fusion,
    const KernelArgumentHolder& fusion_outputs,
    const KernelArgumentHolder& aten_inputs,
    int line_number,
    const char* file_name,
    std::string err_msg,
    const LaunchParams& lparams,
    const ValidationConstants& tolerances) {
  testValidate(
      fusion,
      fusion_outputs,
      aten_inputs,
      /*aten_outputs=*/{},
      line_number,
      file_name,
      err_msg,
      lparams,
      tolerances);
}

void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<SchedulerType>& expected_heuristics) {
  const auto& segment_groups = runtime->fusionSegments()->groups();

  NVF_CHECK(
      segment_groups.size() == expected_heuristics.size(),
      "Unexpected segments. Expected: ",
      expected_heuristics.size(),
      ". Actual: ",
      segment_groups.size());

  // Assumes up to two segments exist for simplicity
  NVF_ERROR(
      segment_groups.size() <= 2, "True segment order analysis is required");

  for (auto& group : segment_groups) {
    int64_t segment_order = group->producer_edges.empty() ? 0 : 1;
    NVF_CHECK(
        group->schedulerType() == expected_heuristics.at(segment_order),
        "Expected to use the ",
        expected_heuristics.at(segment_order),
        " scheduler but ",
        group->schedulerType(),
        " was used");
  }
}

} // namespace nvfuser
