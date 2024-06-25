// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/utils.h>
#include <exceptions.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/iostream.h>

#include <ATen/cuda/CUDAContext.h>

#include <unordered_map>

namespace nvfuser {

namespace {

struct ValidationConstants {
  // Tolerances generated from randn + add + sum fusion
  // compared against double precision
  std::array<std::array<double, 2>, 20> sum_tolerances_float = {
      {{4, 1.68222e-06},      {8, 2.23704e-06},      {16, 2.95788e-06},
       {32, 4.4778e-06},      {64, 6.75395e-06},     {128, 8.57934e-06},
       {256, 1.30594e-05},    {512, 2.19122e-05},    {1024, 3.3451e-05},
       {2048, 5.78476e-05},   {4096, 0.000108292},   {8192, 0.00012207},
       {16384, 0.000136882},  {32768, 0.000248561},  {65536, 0.000407594},
       {131072, 0.000500901}, {262144, 0.000923019}, {524288, 0.00156909},
       {1048576, 0.00223107}, {2097152, 0.00343043}}};

  // Tolerances generated from randn + add + sum fusion
  // compared against double precision
  std::array<std::array<double, 2>, 20> sum_tolerances_half = {
      {{4, 0.00390625},    {8, 0.0078125},    {16, 0.0078125},
       {32, 0.0155334},    {64, 0.0156269},   {128, 0.0312042},
       {256, 0.0312548},   {512, 0.0619979},  {1024, 0.0625103},
       {2048, 0.124686},   {4096, 0.12501},   {8192, 0.24945},
       {16384, 0.250049},  {32768, 0.498946}, {65536, 0.500071},
       {131072, 0.985087}, {262144, 1.00006}, {524288, 1.99234},
       {1048576, 2.00032}, {2097152, 3.99073}}};

  double base_half_abs_tol = -1;
  double base_half_rel_tol = -1;
  double base_float_abs_tol = -1;
  double base_float_rel_tol = -1;
};

// Returns abs and relative values to use for validation
std::pair<double, double> getTolerance(
    DataType dtype,
    int64_t reduction_size,
    const ValidationConstants& tolerances) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  switch (std::get<PrimDataType>(dtype.type)) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
    case DataType::Float:
    // TODO: Pull new tolerances for Double, for now we will just use float
    // tolerances as it should be no worse.
    case DataType::Double: {
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_float;
      const auto& base_abs = tolerances.base_float_abs_tol;
      const auto& base_rel = tolerances.base_float_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs, base_rel};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while (entry < sum_tolerance_entry.size() &&
               (int64_t)sum_tolerance_entry[entry][0] < reduction_size) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol, abs_tol * 0.01};
      }
    }
    case DataType::Half: {
      // Copied from float case
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_half;
      const auto& base_abs = tolerances.base_half_abs_tol;
      const auto& base_rel = tolerances.base_half_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs, base_rel};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while ((int64_t)sum_tolerance_entry[entry][0] < reduction_size &&
               entry < sum_tolerance_entry.size()) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol, abs_tol * 0.01};
      }
    }
    // TODO: fp8 likely will need higher tolerance.
    case DataType::Float8_e4m3fn:
    case DataType::Float8_e5m2:
    case DataType::BFloat16: {
      // Copied from float case
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_half;
      const auto& base_abs = tolerances.base_half_abs_tol;
      const auto& base_rel = tolerances.base_half_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs * 10.0, base_rel * 10.0};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while ((int64_t)sum_tolerance_entry[entry][0] < reduction_size &&
               entry < sum_tolerance_entry.size()) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol * 10.0, abs_tol * 0.01 * 10.0};
      }
    }
    case DataType::Int:
    case DataType::Int32:
    case DataType::Index:
    case DataType::Bool:
      return {0.0, 0.0};
    default:
      NVF_ERROR(
          false, "Do not have tolerance computation for type ", dtype, ".");
  }
}

class ReductionSizeMapper : private IterVisitor {
 public:
  //! Runs through the fusion and determines how many reductions were performed
  //! to compute each tensorview.
  static std::unordered_map<TensorView*, int64_t> computeReductionSizes(
      Fusion* fusion,
      ExpressionEvaluator& expr_eval) {
    ReductionSizeMapper mapper(fusion, expr_eval);
    return mapper.reduction_map;
  }

 private:
  ReductionSizeMapper(Fusion* fusion, ExpressionEvaluator& expr_eval)
      : expr_eval_(expr_eval) {
    // Initialize input values
    for (auto inp : fusion->inputs()) {
      if (inp->isA<TensorView>()) {
        auto tv = inp->as<TensorView>();
        // Shouldn't have any reductions, but run it through analysis anyways.
        reduction_map[tv] = getReductionSize(tv);
      }
    }

    IterVisitor::traverse(fusion);

    // catch up with dangling outputs;
    for (auto out : fusion->outputs()) {
      if (out->isA<TensorView>()) {
        auto tv = out->as<TensorView>();
        // possible that we have a dangling output that's not generated by any
        // expression. e.g. 0 workspace or null tensor
        if (reduction_map.count(tv) == 0) {
          // Shouldn't have any reductions, but run it through analysis anyways.
          reduction_map[tv] = getReductionSize(tv);
        }
      }
    }
  }

  int64_t getReductionSize(const TensorView* tv) {
    int64_t reduction_elements = 1;
    for (auto id : tv->getLogicalDomain()) {
      if (id->isReduction()) {
        auto inferred_extent = expr_eval_.evaluate(id->extent());
        NVF_ERROR(
            inferred_extent.hasValue(),
            "Couldn't figure out what the dimensions of a tensorview is in evaluation for validation. ",
            id,
            " in ",
            tv);
        reduction_elements = reduction_elements * inferred_extent.as<int64_t>();
      }
    }
    return reduction_elements;
  }

  void dispatch(Expr* expr) override {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }

    int64_t inp_reduction_elements = 1;
    for (auto inp : expr->inputs()) {
      if (inp->isA<TensorView>()) {
        if (auto tv = inp->as<TensorView>()) {
          inp_reduction_elements =
              std::max(inp_reduction_elements, reduction_map.at(tv));
        }
      }
    }

    for (auto out : expr->outputs()) {
      if (out->isA<TensorView>()) {
        auto tv = out->as<TensorView>();
        reduction_map[tv] = getReductionSize(tv) * inp_reduction_elements;
      }
    }
  }

 private:
  using IterVisitor::handle;

  std::unordered_map<TensorView*, int64_t> reduction_map;
  ExpressionEvaluator& expr_eval_;
};

ExpressionEvaluator bindInputsAndLaunchParams(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    const LaunchParams& launch_constraints) {
  KernelArgumentHolder argument_holder;
  argument_holder.push(aten_inputs);

  auto expr_eval = executor_utils::bindInputs(argument_holder, fusion);
  for (auto val : fusion->vals()) {
    if (!val->isA<TensorView>()) {
      continue;
    }

    // Roughly taken from executor.cpp/computeLaunchParams
    auto tv = val->as<TensorView>();
    for (auto id : tv->getLoopDomain()) {
      if (!(id->isThread() && id->extent()->definition() == nullptr)) {
        continue;
      }

      if (id->isBroadcast()) {
        continue;
      }

      auto extent = id->extent();
      auto inferred_extent = expr_eval.evaluate(extent);
      auto p_type = id->getParallelType();

      if (inferred_extent.hasValue()) {
        // This value could have been inferred, make sure it was set right.
        NVF_CHECK(
            inferred_extent == launch_constraints.getDim(p_type) ||
                launch_constraints.getRawVal(p_type) == -1,
            "inferred that ",
            p_type,
            " should be set to ",
            inferred_extent,
            " but launch constraints specified ",
            launch_constraints.getRawVal(p_type));
      } else {
        // Bind the launch constraint into our evaluation context
        if (launch_constraints.hasDim(id->getParallelType())) {
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
        }
      }
    }
  }
  return expr_eval;
}

std::vector<std::pair<double, double>> get_val_constants(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants()) {
  FusionGuard fg(fusion);
  auto expr_eval = bindInputsAndLaunchParams(fusion, aten_inputs, lparams);
  auto reduction_sizes =
      ReductionSizeMapper::computeReductionSizes(fusion, expr_eval);

  std::vector<std::pair<double, double>> tolerance_values;
  for (size_t i = 0; i < fusion->outputs().size(); i++) {
    auto fusion_output_tv = fusion->outputs()[i]->as<TensorView>();
    NVF_ERROR(
        reduction_sizes.count(fusion_output_tv),
        "Missed reduction size count on fusion output at index: ",
        i);

    int64_t reduction_size = reduction_sizes.at(fusion_output_tv);

    auto tolerance_value = getTolerance(
        fusion_output_tv->getDataType().value(), reduction_size, tolerances);
    tolerance_values.push_back(tolerance_value);
  }
  return tolerance_values;
}

} // namespace
} // namespace nvfuser
