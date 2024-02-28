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
      {{4, 8.29392e-06},       {8, 9.80065e-06},      {16, 1.22630e-05},
       {32, 1.71170e-05},      {64, 2.12105e-05},     {128, 2.86866e-05},
       {256, 3.92410e-05},     {512, 5.95525e-05},    {1024, 8.62084e-05},
       {2048, 1.22814e-04},    {4096, 1.63618e-04},   {8192, 2.47255e-04},
       {16384, 3.37849e-04},   {32768, 5.31622e-04},  {65536, 7.87915e-04},
       {131072, 1.00538e-03},  {262144, 1.41515e-03}, {524288, 2.22404e-03},
       {1048576, 3.08768e-03}, {2097152, 4.72822e-03}}};

  // Tolerances generated from randn + add + sum fusion
  // compared against fp32 precision
  std::array<std::array<double, 2>, 20> sum_tolerances_half = {
      {{4, 2.55661e-02},       {8, 3.88184e-02},      {16, 3.88241e-02},
       {32, 4.61884e-02},      {64, 6.26907e-02},     {128, 7.64923e-02},
       {256, 1.16180e-01},     {512, 1.48727e-01},    {1024, 2.35977e-01},
       {2048, 2.71042e-01},    {4096, 4.51538e-01},   {8192, 5.76965e-01},
       {16384, 8.43750e-01},   {32768, 1.16052e+00},  {65536, 1.85815e+00},
       {131072, 2.52466e+00},  {262144, 3.62988e+00}, {524288, 4.48608e+00},
       {1048576, 7.91895e+00}, {2097152, 9.35449e+00}}};

  // Tolerances generated from randn + add + sum fusion
  // compared against fp32 precision
  std::array<std::array<double, 2>, 20> sum_tolerances_bfloat = {
      {{4, 2.29774e-01},       {8, 2.38712e-01},      {16, 3.20366e-01},
       {32, 3.81409e-01},      {64, 5.09972e-01},     {128, 6.98776e-01},
       {256, 8.89732e-01},     {512, 1.13058e+00},    {1024, 1.76106e+00},
       {2048, 2.31541e+00},    {4096, 3.72748e+00},   {8192, 4.65298e+00},
       {16384, 7.46301e+00},   {32768, 8.82568e+00},  {65536, 1.45876e+01},
       {131072, 1.71610e+01},  {262144, 2.96763e+01}, {524288, 3.55269e+01},
       {1048576, 5.56523e+01}, {2097152, 7.13672e+01}}};

  double base_half_abs_tol = -1;
  double base_half_rel_tol = -1;
  double base_float_abs_tol = -1;
  double base_float_rel_tol = -1;
  double base_bfloat_abs_tol = -1;
  double base_bfloat_rel_tol = -1;
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
    case DataType::BFloat16: {
      // Copied from float case
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_bfloat;
      const auto& base_abs = tolerances.base_bfloat_abs_tol;
      const auto& base_rel = tolerances.base_bfloat_rel_tol;

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
    for (auto id : tv->getMaybeRFactorDomain()) {
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
    for (auto id : tv->getLeafDomain()) {
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
