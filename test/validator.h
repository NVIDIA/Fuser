// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/validator_utils.h>

namespace nvfuser {

// Validation will look through the fusion and figure out how many elements were
// reduced to create each output. It will then compute a tolernace to use for
// allclose based on experimental results. The experimental results were based
// on adding two tensors then summing them. This of course has an assumption
// that we're always summing values between -2 and 2. If we start summing values
// larger than that this approach might not hold.
// If aten_outputs is empty, then infer the expected outputs from the fusion
// using expr evaluator.
void testValidate(
    Fusion* fusion,
    const std::vector<at::Tensor>& fusion_outputs,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    std::vector<at::Tensor> aten_outputs,
    int line_number,
    const char* file_name,
    std::string err_msg = "",
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants()) {
  FusionGuard fg(fusion);

  auto expr_eval = bindInputsAndLaunchParams(fusion, aten_inputs, lparams);

  auto reduction_sizes =
      ReductionSizeMapper::computeReductionSizes(fusion, expr_eval);

  if (aten_outputs.empty()) {
    for (auto v : fusion->outputs()) {
      aten_outputs.emplace_back(expr_eval.evaluate(v).as<at::Tensor>());
    }
  }

  NVF_ERROR(
      fusion_outputs.size() == aten_outputs.size(),
      "Number of outputs don't match.");

  NVF_ERROR(
      fusion->inputs().size() == aten_inputs.size(),
      "Number of inputs don't match.");

  for (size_t i = 0; i < fusion->inputs().size(); i++) {
    if (fusion->inputs()[i]->isA<TensorView>()) {
      NVF_ERROR(aten_inputs[i].isTensor(), "Mismatch of tensor inputs.");

      auto fusion_input_tv = fusion->inputs()[i]->as<TensorView>();
      auto at_tensor = aten_inputs[i].toTensor();

      NVF_ERROR(
          at_tensor.dim() ==
              static_cast<int64_t>(TensorDomain::noReductions(
                                       fusion_input_tv->getMaybeRFactorDomain())
                                       .size()),
          "Dimensionality mismatch in inputs.");
    }
  }

  const auto& io_alias = fusion->ioAlias();
  auto should_remove = [&io_alias](Val* out_val) -> bool {
    if (auto alias_it = io_alias.find(out_val); alias_it != io_alias.end()) {
      return alias_it->second.second.hide_output;
    }
    return false;
  };

  for (size_t i = 0, j = 0; i < fusion->outputs().size(); i++) {
    NVF_ERROR(fusion->outputs()[i]->isA<TensorView>());
    if (should_remove(fusion->outputs()[i])) {
      // This is an aliased output that's hidden from integration.
      // Let's not check this.
      continue;
    }

    auto fusion_output_tensor = fusion_outputs[j];
    auto fusion_output_tv = fusion->outputs()[i]->as<TensorView>();
    auto aten_output_tensor = aten_outputs[j];

    NVF_ERROR(
        reduction_sizes.count(fusion_output_tv),
        "Missed reduction size count on fusion output at index: ",
        i);

    int64_t reduction_size = reduction_sizes.at(fusion_output_tv);

    NVF_ERROR(
        aten_output_tensor.dim() == fusion_output_tensor.dim() &&
            fusion_outputs[j].dim() ==
                static_cast<int64_t>(
                    TensorDomain::noReductions(
                        fusion_output_tv->getMaybeRFactorDomain())
                        .size()),
        "Dimensionality mismatch in outputs.");

    auto tolerance_values = getTolerance(
        fusion_output_tv->getDataType().value(), reduction_size, tolerances);

    if (aten_output_tensor.is_floating_point() ||
        aten_output_tensor.is_complex()) {
      NVF_ERROR(
          aten_output_tensor.allclose(
              fusion_output_tensor.to(aten_output_tensor.dtype()),
              tolerance_values.second,
              tolerance_values.first,
              /*equal_nan=*/true),
          "\n",
          err_msg,
          "\nValidation error in output ",
          j,
          " on line ",
          line_number,
          " in file ",
          file_name,
          ".\n  Detected abs error of: ",
          aten_output_tensor.sub(fusion_output_tensor)
              .abs()
              .max()
              .item()
              .to<double>(),
          "\n    absolute tolerance was set to ",
          tolerance_values.first,
          "\n    and relative tolerance set to ",
          tolerance_values.second);
    } else {
      NVF_ERROR(
          aten_output_tensor.equal(
              fusion_output_tensor.to(aten_output_tensor.dtype())),
          "\n",
          err_msg,
          ".\n  Validation error in output ",
          j,
          " on line ",
          line_number,
          " in file ",
          file_name,
          ".\n Values are not equal and are not a floating type.");
    }
    j++;
  }
}

// The variant with automatically inferred aten outputs. The `evaluate` method
// of the exprs in the fusion must be overriden to handle at::Tensor.
void testValidate(
    Fusion* fusion,
    const std::vector<at::Tensor>& fusion_outputs,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    int line_number,
    const char* file_name,
    std::string err_msg = "",
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants()) {
  testValidate(
      fusion,
      fusion_outputs,
      aten_inputs,
      {},
      line_number,
      file_name,
      err_msg,
      lparams,
      tolerances);
}

} // namespace nvfuser
