// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <expr_simplifier.h>
#include <interval_analysis.h>
#include <iter_visitor.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <cctype>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace nvfuser {

class IntervalAnalysisTest : public NVFuserTest {
  std::unique_ptr<Fusion> fusion_ptr;
  std::unique_ptr<FusionGuard> fusion_guard_ptr;

  void SetUp() override {
    NVFuserTest::SetUp();
    fusion_ptr = std::make_unique<Fusion>();
    fusion_guard_ptr = std::make_unique<FusionGuard>(fusion_ptr.get());
  }
};

namespace {

class RangeChecker {
 public:
  static void check(
      Val* output_val,
      const std::vector<BoundedInt>& input_bounds,
      const BoundedInt& expected_range,
      const bool bound_is_tight = true,
      const LaunchParams& launch_params = LaunchParams()) {
    RangeChecker checker(
        output_val,
        input_bounds,
        expected_range,
        bound_is_tight,
        launch_params);
    if (input_bounds.size() == 1) {
      checker.checkOneInput();
    } else {
      NVF_THROW("Unhandled number of inputs: ", input_bounds.size());
    }
  }

 private:
  RangeChecker(
      Val* output_val,
      const std::vector<BoundedInt>& input_bounds,
      const BoundedInt& expected_range,
      const bool bound_is_tight,
      const LaunchParams& launch_params)
      : output_val_(output_val),
        input_vals_(InputsOf::output(output_val)),
        input_bounds_(input_bounds),
        expected_range_(expected_range),
        bound_is_tight_(bound_is_tight) {
    NVF_ERROR(input_vals_.size() == input_bounds_.size());

    ExpressionEvaluator expr_eval;
    // Compute the range using ScalarBoundsCalculator and check that it matches
    // expected
    ScalarBoundsCalculator calc(/*kernel=*/nullptr, expr_eval, launch_params);
    calc.setBounds(input_vals_.at(0), input_bounds_.at(0));
    // Check that the computed range is correct
    calc.dispatch(output_val_);
    auto bound_opt = calc.maybeGetBounds(output_val_);
    // Cannot use ASSERT_* in constructor
    NVF_ERROR(
        bound_opt.has_value(),
        "Expected bounds to be computed following call to dispatch");
    EXPECT_EQ(bound_opt.value(), expected_range);
  }

  void checkOneInput() {
    ASSERT_EQ(input_bounds_.size(), 1);
    for (int64_t i0 = input_bounds_.at(0).min; i0 <= input_bounds_.at(0).max;
         i0++) {
      ExpressionEvaluator expr_eval;
      // bind input vals
      expr_eval.bind(input_vals_.at(0), i0);

      PolymorphicValue pv = expr_eval.evaluate(output_val_);
      ASSERT_TRUE(pv.hasValue());
      ASSERT_TRUE(pv.is<int64_t>());
      int64_t eval = pv.as<int64_t>();
      EXPECT_GE(eval, expected_range_.min);
      EXPECT_LE(eval, expected_range_.max);

      eval_min_ = std::min(eval_min_, eval);
      eval_max_ = std::max(eval_max_, eval);
    }

    if (bound_is_tight_) {
      EXPECT_EQ(eval_min_, expected_range_.min);
      EXPECT_EQ(eval_max_, expected_range_.max);
    }
  }

 private:
  Val* output_val_;
  const std::vector<Val*> input_vals_;
  const std::vector<BoundedInt>& input_bounds_;
  const BoundedInt& expected_range_;
  bool bound_is_tight_;

  int64_t eval_min_ = std::numeric_limits<int64_t>::max();
  int64_t eval_max_ = std::numeric_limits<int64_t>::min();
};

} // namespace

TEST_F(IntervalAnalysisTest, UnaryOps) {
  Val* x = IrBuilder::create<Val>(DataType::Index);
  RangeChecker::check(
      x, /*input_bounds=*/{{-1, 5}}, /*expected_range=*/{-1, 5});
  RangeChecker::check(
      neg(x), /*input_bounds=*/{{-1, 5}}, /*expected_range=*/{-5, 1});
  // TODO: fix evaluate function for BitwiseNot. Currently it returns uint64_t
  // RangeChecker::check(bitwise_not(x), /*input_bounds=*/{{-1, 5}}, {-5, 1});
}

} // namespace nvfuser
