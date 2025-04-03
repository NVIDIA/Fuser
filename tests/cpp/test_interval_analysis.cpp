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

#include <algorithm>
#include <exception>
#include <unordered_map>

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

// This class lets us test that our computed ranges match our expectation and
// that they are correct. We provide a val, a mapping from input vals to
// bounds, and the expected range. Then the actual range is computed by
// exhaustively checking all valid input combinations. This is checked against
// the expected range, and the range computed by ScalarBoundsCalculator.
class RangeChecker {
 public:
  static void check(
      Val* output_val,
      const std::unordered_map<Val*, BoundedInt>& input_bounds,
      const BoundedInt& expected_range,
      const bool bound_is_tight = true,
      const LaunchParams& launch_params = LaunchParams()) {
    RangeChecker checker(
        output_val,
        input_bounds,
        expected_range,
        bound_is_tight,
        launch_params);
    checker.checkAllInputs();
  }

 private:
  RangeChecker(
      Val* output_val,
      const std::unordered_map<Val*, BoundedInt>& input_bounds,
      const BoundedInt& expected_range,
      const bool bound_is_tight,
      const LaunchParams& launch_params)
      : output_val_(output_val),
        input_bounds_(input_bounds),
        expected_range_(expected_range),
        bound_is_tight_(bound_is_tight) {
    ExpressionEvaluator expr_eval;
    // Compute the range using ScalarBoundsCalculator and check that it matches
    // expected
    ScalarBoundsCalculator calc(/*kernel=*/nullptr, expr_eval, launch_params);
    for (auto& [v, b] : input_bounds_) {
      calc.setBounds(v, b);
    }
    // Check that the computed range is correct
    calc.dispatch(output_val_);
    auto bound_opt = calc.maybeGetBounds(output_val_);
    // Cannot use ASSERT_* in constructor
    NVF_ERROR(
        bound_opt.has_value(),
        "Expected bounds to be computed following call to dispatch");
    EXPECT_EQ(bound_opt.value(), expected_range);
  }

  // Evaluate output_val_ exhaustively for every possible combination of inputs
  void checkAllInputs() {
    std::vector<Val*> inputs;
    inputs.reserve(input_bounds_.size());
    // Number of valid combinations of input values
    int64_t num_combos = 1;
    for (auto& [v, b] : input_bounds_) {
      inputs.push_back(v);
      NVF_ERROR(b.max >= b.min);
      num_combos *= b.max - b.min + 1;
    }
    // Sort inputs by name so that test deterministically traverses inputs
    std::stable_sort(inputs.begin(), inputs.end(), [](Val* v1, Val* v2) {
      return v1->name() < v2->name();
    });

    // Iterate over all input combinations
    for (size_t i : arange(num_combos)) {
      ExpressionEvaluator expr_eval;

      // All the input combinations are enumerated
      // For example if there are three inputs with the following bounds:
      //  x: [min_x, max_x]
      //  y: [min_y, max_y]
      //  z: [min_z, max_z]
      // Then there are nx*ny*nz=(max_x-min_x+1)*(max_y-min_y+1)*(max_z-min_z+1)
      // combinations of valid inputs. The jth input is determined by
      //  x = j / (ny*nz) + min_x
      //  y = (j % (ny*nz)) / nz + min_y
      //  z = j % nz + min_z
      int64_t num_inner_combos = num_combos;
      for (size_t inp_num : arange(inputs.size())) {
        const BoundedInt& inp_bound = input_bounds_.at(inputs.at(inp_num));
        int64_t next_offset = i % num_inner_combos;
        num_inner_combos /= inp_bound.max - inp_bound.min + 1L;
        int64_t this_input_value =
            inp_bound.min + (next_offset / num_inner_combos);
        expr_eval.bind(inputs.at(inp_num), this_input_value);
      }

      PolymorphicValue pv;
      try {
        pv = expr_eval.evaluate(output_val_);
      } catch (const std::exception& ex) {
        // Floating point exception due to division or modulo by zero avoided
        if (std::string(ex.what()).find("zero detected") != std::string::npos) {
          continue;
        } else {
          throw;
        }
      }
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
  const std::unordered_map<Val*, BoundedInt>& input_bounds_;
  const BoundedInt& expected_range_;
  bool bound_is_tight_;

  int64_t eval_min_ = std::numeric_limits<int64_t>::max();
  int64_t eval_max_ = std::numeric_limits<int64_t>::min();
};

} // namespace

TEST_F(IntervalAnalysisTest, UnaryOps) {
  Val* x = IrBuilder::create<Val>(DataType::Index);
  RangeChecker::check(
      x, /*input_bounds=*/{{x, {-1, 5}}}, /*expected_range=*/{-1, 5});
  RangeChecker::check(
      neg(x), /*input_bounds=*/{{x, {-1, 5}}}, /*expected_range=*/{-5, 1});
  // TODO: fix evaluate function for BitwiseNot
  // RangeChecker::check(bitwise_not(x), /*input_bounds=*/{{-1, 5}}, {-5, 1});
}

TEST_F(IntervalAnalysisTest, BinaryOps) {
  Val* x = IrBuilder::create<Val>(DataType::Index);
  Val* y = IrBuilder::create<Val>(DataType::Index);
  RangeChecker::check(
      x,
      /*input_bounds=*/{{x, {-1, 5}}, {y, {-3, 2}}},
      /*expected_range=*/{-1, 5});
  RangeChecker::check(
      y,
      /*input_bounds=*/{{x, {-1, 5}}, {y, {-3, 2}}},
      /*expected_range=*/{-3, 2});
  RangeChecker::check(
      add(x, y),
      /*input_bounds=*/{{x, {-1, 5}}, {y, {-3, 2}}},
      /*expected_range=*/{-4, 7});
  RangeChecker::check(
      sub(x, y),
      /*input_bounds=*/{{x, {-1, 5}}, {y, {-3, 2}}},
      /*expected_range=*/{-3, 8});

  // Check multiple scenarios for mul
  RangeChecker::check(
      mul(x, y),
      /*input_bounds=*/{{x, {3, 5}}, {y, {4, 6}}},
      /*expected_range=*/{12, 30});
  RangeChecker::check(
      mul(x, y),
      /*input_bounds=*/{{x, {-1, 5}}, {y, {-3, 2}}},
      /*expected_range=*/{-15, 10});
  RangeChecker::check(
      mul(x, y),
      /*input_bounds=*/{{x, {0, 1}}, {y, {-2, 1}}},
      /*expected_range=*/{-2, 1});
  RangeChecker::check(
      mul(x, y),
      /*input_bounds=*/{{x, {-2, 1}}, {y, {-2, 3}}},
      /*expected_range=*/{-6, 4});

  // Check scenarios for div and ceilDiv where each input contains zero, is
  // only positive, or is only negative
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {1, 4}}, {y, {3, 3}}},
      /*expected_range=*/{0, 1});
  RangeChecker::check(
      ceilDiv(x, y),
      /*input_bounds=*/{{x, {1, 4}}, {y, {3, 3}}},
      /*expected_range=*/{1, 2});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {-3, 4}}, {y, {1, 3}}},
      /*expected_range=*/{-3, 4});
  RangeChecker::check(
      ceilDiv(x, y),
      /*input_bounds=*/{{x, {-3, 4}}, {y, {1, 3}}},
      /*expected_range=*/{-3, 4});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {-3, -1}}, {y, {1, 3}}},
      /*expected_range=*/{-3, 0});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {-3, -1}}, {y, {-3, -1}}},
      /*expected_range=*/{0, 3});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {-3, -1}}, {y, {-3, 2}}},
      /*expected_range=*/{-3, 3});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {-2, 1}}, {y, {-1, 2}}},
      /*expected_range=*/{-2, 2});
  RangeChecker::check(
      ceilDiv(x, y),
      /*input_bounds=*/{{x, {-3, -1}}, {y, {-3, 2}}},
      /*expected_range=*/{-3, 5},
      // NOTE: ceilDiv(-3, -1) = (-3 + (-1) - 1) / (-1) = 5 is what is computed
      // in-kernel, but ExpressionEvaluator computes (numer + denom + 1) / denom
      // when denom < 0. The bound above is actually tight for the in-kernel
      // code but that does not currently match our ExpressionEvaluator
      /*bound_is_tight=*/false);
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {0, 0}}, {y, {2, 3}}},
      /*expected_range=*/{0, 0});
  RangeChecker::check(
      div(x, y),
      /*input_bounds=*/{{x, {0, 1}}, {y, {1, 1}}},
      /*expected_range=*/{0, 1});
  RangeChecker::check(
      ceilDiv(x, y),
      /*input_bounds=*/{{x, {0, 1}}, {y, {2, 3}}},
      /*expected_range=*/{0, 1});

  RangeChecker::check(
      mod(x, y),
      /*input_bounds=*/{{x, {2, 3}}, {y, {3, 3}}},
      /*expected_range=*/{0, 2});
  RangeChecker::check(
      mod(x, y),
      /*input_bounds=*/{{x, {2, 4}}, {y, {7, 8}}},
      /*expected_range=*/{2, 4});
  RangeChecker::check(
      mod(x, y),
      /*input_bounds=*/{{x, {2, 4}}, {y, {2, 5}}},
      /*expected_range=*/{0, 4});
  RangeChecker::check(
      mod(x, y),
      /*input_bounds=*/{{x, {2, 4}}, {y, {-8, -7}}},
      /*expected_range=*/{2, 4});

  // We do not generally place the tightest bounds on bitwise ops because it is
  // difficult to do without exhaustively trying input combinations.
  RangeChecker::check(
      bitwise_and(x, y),
      /*input_bounds=*/{{x, {0b1001, 0b1011}}, {y, {0b1010, 0b1100}}},
      // NOTE: this bound is not tight because we assume all variable bits can
      // take any combination of values, but since there is only one y value
      // with high third bit, the highest we can actually get is 0b1011=11
      /*expected_range=*/{0b1000, 0b1111},
      /*bound_is_tight=*/false);
  RangeChecker::check(
      bitwise_or(x, y),
      /*input_bounds=*/{{x, {0b1001, 0b1011}}, {y, {0b1010, 0b1100}}},
      /*expected_range=*/{0b1000, 0b1111},
      // NOTE: this bound is not tight because we assume all variable bits can
      // take any combination of values, but since there is only one y value
      // with high third bit, the lowest we can actually get is 0b1010=10, not
      // 0b1000=8
      /*bound_is_tight=*/false);
  RangeChecker::check(
      bitwise_xor(x, y),
      /*input_bounds=*/{{x, {0b1001, 0b1011}}, {y, {0b1010, 0b1100}}},
      /*expected_range=*/{0b0000, 0b0111});

  RangeChecker::check(
      bitwise_left_shift(x, y),
      /*input_bounds=*/{{x, {0b1001, 0b1011}}, {y, {1, 5}}},
      /*expected_range=*/{0b10010, 0b101100000});
  RangeChecker::check(
      bitwise_right_shift(x, y),
      /*input_bounds=*/{{x, {0b100100, 0b101100}}, {y, {1, 5}}},
      /*expected_range=*/{0b1, 0b10110});
}

// Test that loop indices are properly bounded, as are expressions derived from
// them
TEST_F(IntervalAnalysisTest, SerialLoops) {
  kir::Kernel kernel(FusionGuard::getCurFusion());
  FusionGuard fg(&kernel);

  Val* ext = IrBuilder::create<Val>(DataType::Index);
  Val* start = kernel.zeroVal();
  auto* id = IterDomainBuilder(start, ext).extent(ext).build();
  Val* index = IrBuilder::create<Val>(DataType::Index);
  auto* loop = IrBuilder::create<ForLoop>(
      id,
      index,
      /*circular_buffer_loop_stage=*/CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);
  Val* offset = IrBuilder::create<Val>(DataType::Index);
  Val* index_plus_offset = add(index, offset);
  // Compute index + offset inside the "for index in id" loop
  loop->body().push_back(index_plus_offset->definition());

  ExpressionEvaluator expr_eval;
  LaunchParams launch_params;
  ScalarBoundsCalculator calc(/*kernel=*/nullptr, expr_eval, launch_params);
  calc.setBounds(ext, {4, 7});
  calc.setBounds(offset, {2, 5});
  calc.dispatch(loop);
  calc.dispatch(index_plus_offset);
  auto bound_opt = calc.maybeGetBounds(index_plus_offset);
  NVF_ERROR(bound_opt.has_value());
  BoundedInt true_bound{2, 11};
  EXPECT_EQ(bound_opt.value(), true_bound);
}

// Test that parallelized loop indices are properly bounded, as are expressions
// derived from them
TEST_F(IntervalAnalysisTest, ParallelLoops) {
  kir::Kernel kernel(FusionGuard::getCurFusion());
  FusionGuard fg(&kernel);

  Val* ext = NamedScalar::getParallelDim(ParallelType::TIDx);
  Val* start = kernel.zeroVal();
  auto* id = IterDomainBuilder(start, ext)
                 .extent(ext)
                 .parallel_type(ParallelType::TIDx)
                 .build();
  Val* index = IrBuilder::create<Val>(DataType::Index);
  auto* loop = IrBuilder::create<ForLoop>(
      id,
      index,
      /*circular_buffer_loop_stage=*/CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);
  Val* offset = IrBuilder::create<Val>(DataType::Index);
  Val* index_plus_offset = add(index, offset);
  // Compute index + offset inside the "for index in id" loop
  loop->body().push_back(index_plus_offset->definition());

  ExpressionEvaluator expr_eval;
  LaunchParams launch_params;
  launch_params.bind(128, ParallelType::TIDx);
  ScalarBoundsCalculator calc(/*kernel=*/nullptr, expr_eval, launch_params);
  calc.setBounds(offset, {2, 5});
  calc.dispatch(loop);
  calc.dispatch(index_plus_offset);
  auto bound_opt = calc.maybeGetBounds(index_plus_offset);
  NVF_ERROR(bound_opt.has_value());
  BoundedInt true_bound{2, 132};
  EXPECT_EQ(bound_opt.value(), true_bound);
}

} // namespace nvfuser
