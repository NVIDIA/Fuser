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

#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <iter_visitor.h>
#include <runtime/executor_params.h>
#include <type.h>

namespace nvfuser {

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
    const KernelArgumentHolder& aten_inputs,
    const LaunchParams& launch_constraints);

std::vector<std::pair<double, double>> get_val_constants(
    Fusion* fusion,
    const KernelArgumentHolder& aten_inputs,
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants());

} // namespace nvfuser
