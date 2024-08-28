// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <array>
#include <unordered_map>
#include <utility>

#include <ATen/ArrayRef.h>

#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <iter_visitor.h>
#include <type.h>

namespace nvfuser {

struct ValidationConstants {
  // Tolerances generated from randn + add + sum fusion
  // compared against double precision
  std::array<std::array<double, 2>, 20> sum_tolerances_float = {
      {{4, 5.04710e-06},       {8, 6.18093e-06},      {16, 1.12203e-05},
       {32, 1.29464e-05},      {64, 1.98594e-05},     {128, 2.46788e-05},
       {256, 3.41311e-05},     {512, 4.69429e-05},    {1024, 7.67564e-05},
       {2048, 1.11040e-04},    {4096, 1.66449e-04},   {8192, 2.04933e-04},
       {16384, 3.09691e-04},   {32768, 5.43202e-04},  {65536, 7.27045e-04},
       {131072, 8.75371e-04},  {262144, 1.25448e-03}, {524288, 2.12646e-03},
       {1048576, 2.54961e-03}, {2097152, 3.79751e-03}}};

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

// Returns abs and relative values to use for validation.
std::pair<double, double> getTolerance(
    DataType dtype,
    int64_t reduction_size,
    const ValidationConstants& tolerances);

class ReductionSizeMapper : private IterVisitor {
 public:
  //! Runs through the fusion and determines how many reductions were performed
  //! to compute each tensorview.
  static std::unordered_map<TensorView*, int64_t> computeReductionSizes(
      Fusion* fusion,
      ExpressionEvaluator& expr_eval);

 private:
  ReductionSizeMapper(Fusion* fusion, ExpressionEvaluator& expr_eval);

  int64_t getReductionSize(const TensorView* tv);

  void dispatch(Expr* expr) override;

  using IterVisitor::handle;

  std::unordered_map<TensorView*, int64_t> reduction_map;
  ExpressionEvaluator& expr_eval_;
};

ExpressionEvaluator bindInputsAndLaunchParams(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    const LaunchParams& launch_constraints);

std::vector<std::pair<double, double>> get_val_constants(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs,
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants());

} // namespace nvfuser
