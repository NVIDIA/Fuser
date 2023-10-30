// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <executor.h>
#include <fusion.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class ScalarHoistTest : public NVFuserTest {};

TEST_F(ScalarHoistTest, IndexHoist1) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv1->split(-1, 4);
  tv2->split(-1, 4);
  tv3->merge(0, 1);
  tv3->split(0, 8);
  tv5->merge(0, 1);
  tv5->split(0, 8);
  tv4->computeAt(tv5, -1);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);

  // Use Int32 as the index type to verify Int32 is used as the type
  // of hoisted indices
  GpuLower gpulw(&fusion, {DataType::Int32});
  auto kernel = gpulw.kernel();

  auto is_index_times_ns = [](Val* val, Val* index, std::string name) -> bool {
    auto def = dynamic_cast<BinaryOp*>(val->definition());
    if (def == nullptr) {
      return false;
    }
    return def->getBinaryOpType() == BinaryOpType::Mul &&
        def->rhs()->isA<NamedScalar>() &&
        def->rhs()->as<NamedScalar>()->name() == name && def->lhs() == index;
  };

  // Validate indices in the kernel are hoisted as
  // intended. Validation could be also done by just string comparison
  // as the parser test, but updating such tests would be tedious.
  for (auto top_level_loop :
       ir_utils::filterByType<kir::ForLoop>(kernel->topLevelExprs())) {
    auto innermost_loop = top_level_loop;
    while (auto first_expr_loop = dynamic_cast<kir::ForLoop*>(
               innermost_loop->body().exprs().at(0))) {
      innermost_loop = first_expr_loop;
    }
    const auto& exprs = innermost_loop->body().exprs();
    NVF_CHECK(!exprs.empty(), "No expression found");
    NVF_CHECK(
        exprs.at(0)->isA<kir::Allocate>(),
        "Invalid expression: ",
        exprs.at(0)->toString());
    auto hoisted_index = exprs.at(0)->as<kir::Allocate>()->buffer();
    NVF_CHECK(
        hoisted_index->dtype() == DataType::Index,
        "Invalid data type of hoisted indices. Should be nvfuser_index_t but: ",
        hoisted_index->dtype());
    kir::Predicate* pred = nullptr;
    for (auto expr : exprs) {
      if (expr->isA<kir::IfThenElse>()) {
        pred = expr->as<kir::IfThenElse>()->predicate();
        auto arith_expr = expr->as<kir::IfThenElse>()->thenBody().exprs().at(0);
        auto out_ti = arith_expr->outputs()[0]->as<kir::TensorIndex>();
        if (out_ti->view()->name() == 1) {
          // Ref: T1[*, hoisted_index] = T0[*, hoisted_index * T0.stride];
          auto t1_index =
              out_ti->index()->definition()->as<BinaryOp>()->input(1);
          NVF_CHECK(
              t1_index == hoisted_index,
              "Invalid index: ",
              t1_index->toInlineString());
          // Pred: hoisted_index < T0.size[1]
          NVF_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          NVF_CHECK(in0->view()->name() == 0);
          // hoisted_index * T0.stride[1]
          auto t0_index = in0->index()->definition()->as<BinaryOp>()->input(1);
          NVF_CHECK(
              is_index_times_ns(t0_index, hoisted_index, "T0.stride[1]"),
              "Invalid index: ",
              t0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 2) {
          // Ref: T3[*, hoisted_index] = T2[*, hoisted_index];
          auto out_index =
              out_ti->index()->definition()->as<BinaryOp>()->input(1);
          NVF_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          NVF_CHECK(in0->view()->name() == 1);
          auto in0_index = in0->index()->definition()->as<BinaryOp>()->input(1);
          NVF_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 3) {
          // Ref: T3[hoisted_index] = T2[hoisted_index];
          auto out_index = out_ti->index();
          NVF_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          NVF_CHECK(in0->view()->name() == 2);
          auto in0_index = in0->index();
          NVF_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 4) {
          // Ref: T4[0] = T3[hoisted_index];
          NVF_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          NVF_CHECK(in0->view()->name() == 3);
          auto in0_index = in0->index();
          NVF_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 5) {
          // Ref: T5[hoisted_index] = T4[0]
          auto out_index = out_ti->index();
          NVF_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          NVF_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
        }
      }
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({15, 17}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Hoist indices for vectorized tensors
TEST_F(ScalarHoistTest, IndexHoist2) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv5->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  tv4->split(-1, 3);

  tv0->computeAt(tv5, 1);
  tv1->computeAt(tv5, 1);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16}, options);
  auto t1 = at::randn({16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(ScalarHoistTest, IndexHoist3) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto input = makeContigTensor(2);
  fusion->addInput(input);
  auto sin_input = sin(input);
  auto numel = mul(input->axis(0)->extent(), input->axis(1)->extent());
  auto output = add(sin_input, numel);
  fusion->addOutput(output);

  for (auto tv : {output, sin_input}) {
    tv->merge(0);
    tv->split(0, 256);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
  inlineMost();

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(10000, options).view({100, 100});
  at::Tensor t1 = t0.sin() + 10000;

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T2) {
  nvfuser_index_t i0;
  i0 = ((nvfuser_index_t)threadIdx.x) + (256LL * ((nvfuser_index_t)blockIdx.x));
  Tensor<float, 2, 2> s1;
  s1.data = T0.data;
  s1.logical_size = T0.logical_size;
  s1.alloc_stride = T0.alloc_stride;
  Array<nvfuser_index_t, 2, 1> a2;
  a2 = s1.logical_size;
  nvfuser_index_t i3;
  i3 = a2[0LL];
  Array<nvfuser_index_t, 2, 1> a4;
  a4 = s1.logical_size;
  nvfuser_index_t i5;
  i5 = a4[1LL];
  nvfuser_index_t i6;
  i6 = i3 * i5;
  bool b7;
  b7 = i0 < i6;
  float f8;
  f8 = (float)(i6);
  float T1[1LL];
  if (b7) {
    T1[0LL]
       = sinf(T0[i0]);
  }
  if (b7) {
    T2[i0]
      = T1[0LL]
      + f8;
  }
}
)";

  assertCUDAKernel(fusion.get(), expected_kernel);

  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(ScalarHoistTest, ARange) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* start_int = IrBuilder::create<Val>(DataType::Int);
  Val* end_int = IrBuilder::create<Val>(DataType::Int);
  Val* step_int = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(start_int);
  fusion->addInput(end_int);
  fusion->addInput(step_int);
  auto output1 = arange(start_int, end_int, step_int, DataType::Int);
  auto output2 = full_like(output1, output1->axis(0)->extent(), DataType::Int);
  fusion->addOutput(output1);
  fusion->addOutput(output2);

  int64_t start = 0, end = 100, step = 1;

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {start, end, step});
  auto cg_outputs = fe.runFusion({start, end, step});

  const auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(start, end, step, options);
  at::Tensor t1 = at::full_like(t0, end - start, options);

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(int64_t i0, int64_t i1, int64_t i2, Tensor<int64_t, 1, 1> T0, Tensor<int64_t, 1, 1> T1) {
  int64_t i3;
  i3 = i1 - i0;
  int64_t i4;
  i4 = abs(i3);
  int64_t i5;
  i5 = abs(i2);
  int64_t i6;
  i6 = ceilDiv(i4, i5);
  nvfuser_index_t i7;
  i7 = (nvfuser_index_t)(i6);
  int64_t i8;
  i8 = (int64_t)(i7);
  #pragma unroll 1
  for(nvfuser_index_t i9 = 0; i9 < i7; ++i9) {
    T0[i9] = (i0 + (i2 * i9));
  }
  #pragma unroll 1
  for(nvfuser_index_t i10 = 0; i10 < i7; ++i10) {
    T1[i10] = i8;
  }
}
)";

  assertCUDAKernel(fusion.get(), expected_kernel);

  testValidate(
      fusion.get(),
      cg_outputs,
      {start, end, step},
      {t0, t1},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
