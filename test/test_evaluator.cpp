// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <expr_evaluator.h>
#include <fusion.h>
#include <ops/all_ops.h>

namespace nvfuser {

class ExprEvalTest : public NVFuserTest {};

// Evaluate basic scalar operations with constant values
TEST_F(ExprEvalTest, Constants) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Int>(7);
  auto* b = IrBuilder::create<Int>(3);

  // Avoid div operation because it casts int operands to float
  checkIntValue(evaluator, neg(a), -7);
  checkIntValue(evaluator, add(a, b), 10);
  checkIntValue(evaluator, neg(mul(sub(a, b), add(a, b))), -40);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
}

TEST_F(ExprEvalTest, Double) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto ten = IrBuilder::create<Double>(10);
  auto two = IrBuilder::create<Double>(2);
  auto three = IrBuilder::create<Double>(3);
  auto val = castOp(DataType::Int, ceilDiv(sub(ten, two), three));
  auto reference = static_cast<int64_t>(std::ceil((10.0 - 2.0) / 3.0));
  TORCH_CHECK(reference == val->evaluateInt());
}

// Evaluate basic scalar operations with bound values
TEST_F(ExprEvalTest, Bindings) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Int>();
  auto* b = IrBuilder::create<Int>();
  auto* c = add(a, b);
  auto* d = neg(ceilDiv(c, b));
  auto* e = IrBuilder::create<Int>(0);

  // trying to evaluate before binding should give empty results
  TORCH_CHECK(!evaluator.evaluate(a).hasValue());
  TORCH_CHECK(!evaluator.evaluate(d).hasValue());

  evaluator.bind(a, 7L);
  evaluator.bind(b, 3L);

  // can't bind to the results of expressions
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(c, 100L));

  // can't bind to concrete values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(e, 100L));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, sub(a, b), 4);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset evaluation context
  evaluator = ExpressionEvaluator();

  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, sub(a, b), -3);
  checkIntValue(evaluator, mod(a, b), 2);
  checkIntValue(evaluator, ceilDiv(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

// Evaluate expressions in a simple IR
TEST_F(ExprEvalTest, Basic) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator;

  // 2. Bind values
  //
  // IMPORTANT:
  // a. The bindings are only as stable as the Vals are in the fusion graph
  // b. You must use the original (rootDomain) extents
  //  (ex. `tv0->getRootDomain()[0]->extent()`
  //   instead of `tv0->axis(0)->extent()`)
  //
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 6L);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 128L);
  evaluator.bind(tv1->getRootDomain()[0]->extent(), 6L);
  evaluator.bind(tv1->getRootDomain()[1]->extent(), 128L);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);
}

// Evaluate expressions in a more complex IR
TEST_F(ExprEvalTest, Complex) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Double>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Double>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Double>(2.0));
  TensorView* tv4 = add(tv2, tv1);
  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv0, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv5->reorder({{-1, 0}});

  tv6->split(0, 5);
  tv5->merge(0);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator;

  // 2. Bind values
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 129L);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 127L);

  // Evaluate and check extent values
  TORCH_CHECK(tv0->domain()->nDims() == 2);
  checkIntValue(evaluator, tv0->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv0->axis(1)->extent(), 127);

  TORCH_CHECK(tv3->domain()->nDims() == 2);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 127);

  TORCH_CHECK(tv4->domain()->nDims() == 2);
  checkIntValue(evaluator, tv4->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv4->axis(1)->extent(), 127);

  TORCH_CHECK(tv5->domain()->nDims() == 1);
  checkIntValue(evaluator, tv5->axis(0)->extent(), 16383);

  TORCH_CHECK(tv6->domain()->nDims() == 3);
  checkIntValue(evaluator, tv6->axis(0)->extent(), 26);
  checkIntValue(evaluator, tv6->axis(1)->extent(), 5);
  checkIntValue(evaluator, tv6->axis(2)->extent(), 127);
}

// Evaluate expressions post lowering
TEST_F(ExprEvalTest, PostLower) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto* bid_x = add(tv3->axis(0)->extent(), IrBuilder::create<Int>(0));
  auto* tid_x = add(tv3->axis(-1)->extent(), IrBuilder::create<Int>(0));

  // Lower
  GpuLower gpulw(&fusion);

  // 1. Create an evaluation context
  ExpressionEvaluator evaluator;

  // 2. Bind values
  evaluator.bind(tv0->getRootDomain()[0]->extent(), 6L);
  evaluator.bind(tv0->getRootDomain()[1]->extent(), 128L);
  evaluator.bind(tv1->getRootDomain()[0]->extent(), 6L);
  evaluator.bind(tv1->getRootDomain()[1]->extent(), 128L);

  // 3. Evaluate and check result values
  TORCH_CHECK(tv2->domain()->nDims() == 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  TORCH_CHECK(tv3->domain()->nDims() == 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);

  checkIntValue(evaluator, bid_x, 2);
  checkIntValue(evaluator, tid_x, 128);
}

// Kernel IR: Evaluate basic scalar operations with constant values
TEST_F(ExprEvalTest, KernelConstants) {
  Fusion fusion;
  kir::Kernel kernel(&fusion);
  FusionGuard fg((&kernel)->as<Fusion>());

  auto a = IrBuilder::create<Int>(7);
  auto b = IrBuilder::create<Int>(3);
  auto c = IrBuilder::subExpr(a, b);
  auto d = IrBuilder::divExpr(a, b);
  auto e = IrBuilder::mulExpr(c, d);

  ExpressionEvaluator evaluator;

  checkIntValue(evaluator, IrBuilder::negExpr(a), -7);
  checkIntValue(evaluator, IrBuilder::addExpr(a, b), 10);
  checkIntValue(evaluator, IrBuilder::negExpr(e), -8);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 1);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 3);
}

// Kernel IR: Evaluate basic scalar operations with bound values
TEST_F(ExprEvalTest, KernelBindings) {
  Fusion fusion;
  kir::Kernel kernel(&fusion);
  FusionGuard fg((&kernel)->as<Fusion>());

  ExpressionEvaluator evaluator;

  auto a = IrBuilder::create<Int>(std::nullopt);
  auto b = IrBuilder::create<Int>(std::nullopt);
  auto c = IrBuilder::addExpr(a, b);
  auto d = IrBuilder::negExpr(IrBuilder::ceilDivExpr(c, b));
  auto e = IrBuilder::create<Int>(0);

  // trying to evaluate before binding should give empty results
  TORCH_CHECK(!evaluator.evaluate(a).hasValue());
  TORCH_CHECK(!evaluator.evaluate(d).hasValue());

  evaluator.bind(a, 7L);
  evaluator.bind(b, 3L);

  // can't bind to the results of expressions
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(c, 100L));

  // can't bind to concrete values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(evaluator.bind(e, 100L));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, IrBuilder::subExpr(a, b), 4);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 1);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset the evaluation context
  evaluator = ExpressionEvaluator();

  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, IrBuilder::subExpr(a, b), -3);
  checkIntValue(evaluator, IrBuilder::modExpr(a, b), 2);
  checkIntValue(evaluator, IrBuilder::ceilDivExpr(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

TEST_F(ExprEvalTest, Array) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* a = IrBuilder::create<Int>();
  auto* b = IrBuilder::create<Int>();

  auto arr = IrBuilder::arrayExpr(std::vector<Val*>{a, b});

  auto aa = IrBuilder::getItemExpr(arr, fusion.zeroVal());
  auto bb = IrBuilder::getItemExpr(arr, fusion.oneVal());

  ExpressionEvaluator evaluator;
  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  auto arr_val = evaluator.evaluate(arr);
  std::vector<ScalarValue> arr_vec = {2, 5};
  EXPECT_EQ(arr_val, arr_vec);

  checkIntValue(evaluator, aa, 2L);
  checkIntValue(evaluator, bb, 5L);
}

TEST_F(ExprEvalTest, Tensor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeSymbolicTensor(2);
  auto data = IrBuilder::getAttrExpr(tv, "data");
  auto sizes = IrBuilder::getAttrExpr(tv, "sizes");
  auto strides = IrBuilder::getAttrExpr(tv, "strides");
  auto size0 = IrBuilder::getItemExpr(sizes, fusion.zeroVal());
  auto size1 = IrBuilder::getItemExpr(sizes, fusion.oneVal());
  auto stride0 = IrBuilder::getItemExpr(strides, fusion.zeroVal());
  auto stride1 = IrBuilder::getItemExpr(strides, fusion.oneVal());

  float a;
  std::vector<int64_t> sizes_vec = {6, 128};
  std::vector<int64_t> strides_vec = {128, 1};

  ExpressionEvaluator evaluator;
  evaluator.bind(tv, &a, sizes_vec, strides_vec);

  // TODO: why the following don't compile?
  // EXPECT_EQ(evaluator.evaluate(data), &a);
  EXPECT_EQ(evaluator.evaluate(data), Pointer(&a));
  EXPECT_EQ((std::vector<int64_t>)evaluator.evaluate(sizes), sizes_vec);
  EXPECT_EQ((std::vector<int64_t>)evaluator.evaluate(strides), strides_vec);

  checkIntValue(evaluator, size0, 6L);
  checkIntValue(evaluator, size1, 128L);
  checkIntValue(evaluator, stride0, 128L);
  checkIntValue(evaluator, stride1, 1L);
}

} // namespace nvfuser
