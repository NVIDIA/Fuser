// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <validator_utils.h>

#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>

#include <device_lower/utils.h>
#include <exceptions.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/iostream.h>
#include <runtime/executor_utils.h>

namespace nvfuser {

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
    case DataType::Char:
    case DataType::Short:
    case DataType::Int32:
    case DataType::Int:
    case DataType::Byte:
    case DataType::UInt16:
    case DataType::UInt32:
    case DataType::UInt64:
    case DataType::Index:
    case DataType::Bool:
      return {0.0, 0.0};
    default:
      NVF_THROW("Do not have tolerance computation for type ", dtype, ".");
  }
}

/*static*/ std::unordered_map<TensorView*, int64_t> ReductionSizeMapper::
    computeReductionSizes(Fusion* fusion, ExpressionEvaluator& expr_eval) {
  ReductionSizeMapper mapper(fusion, expr_eval);
  return mapper.reduction_map;
}

ReductionSizeMapper::ReductionSizeMapper(
    Fusion* fusion,
    ExpressionEvaluator& expr_eval)
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

int64_t ReductionSizeMapper::getReductionSize(const TensorView* tv) {
  int64_t reduction_elements = 1;
  for (auto id : tv->getLogicalDomain()) {
    if (id->isReduction()) {
      auto inferred_extent = expr_eval_.evaluate(id->extent());
      NVF_ERROR(
          inferred_extent.hasValue(),
          "Couldn't figure out what the dimensions of a tensorview is in "
          "evaluation for validation. ",
          id,
          " in ",
          tv);
      reduction_elements = reduction_elements * inferred_extent.as<int64_t>();
    }
  }
  return reduction_elements;
}

void ReductionSizeMapper::dispatch(Expr* expr) {
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

ExpressionEvaluator bindInputsAndLaunchParams(
    Fusion* fusion,
    const KernelArgumentHolder& aten_inputs,
    const LaunchParams& launch_constraints) {
  auto expr_eval = executor_utils::bindInputs(aten_inputs, fusion);
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
    const KernelArgumentHolder& aten_inputs,
    const LaunchParams& lparams,
    const ValidationConstants& tolerances) {
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

} // namespace nvfuser
