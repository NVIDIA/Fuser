// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <swizzle.h>
#include <test/utils.h>
#include <test/validator.h>
#include <transform_iter.h>

namespace nvfuser {

class SwizzleTest : public NVFuserTest {};

// Test a basic swizzle pattern
TEST_F(SwizzleTest, SimpleSwizzle0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  // Make a 2x8 Zshape tile
  tv1->split(-1, 16);
  tv1->split(-1, 8);
  // [O, 2, 8]

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv1->computeAt(tv2, 1);
  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  GpuLower gpulw(&fusion);
  auto exprs = gpulw.kernel()->topLevelExprs();
  auto str = ir_utils::toString(exprs);
  TORCH_CHECK(str.find("where") != std::string::npos);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test swizzle inlining
TEST_F(SwizzleTest, SimpleSwizzle1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t3 = t0 + 3.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t3}, __LINE__, __FILE__);
}

// Test sync insertion and memory check in parallelized swizzles.
//  In this test, data is parallel written into smem in zcurve
//   pattern and then read out and output to global mem unswizzled.
TEST_F(SwizzleTest, SimpleSwizzle2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv1->swizzle(Swizzle2DType::ZShape, -2, -1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);

  // Validation should fail since TV1 is not in shared
  //  memory as required by sync info pass.
  ASSERT_ANY_THROW(GpuLower gpulw_throw(&fusion));

  tv1->setMemoryType(MemoryType::Shared);

  // Make sure that a sync is inserted:
  bool sync_found = false;
  GpuLower gpu_lw(&fusion);
  auto flattened_exps =
      ir_utils::flattenScopedExprs(gpu_lw.kernel()->topLevelExprs());

  for (auto expr : flattened_exps) {
    if (expr->isA<kir::BlockSync>()) {
      sync_found = true;
    }
    // Will require a sync thread before any shared memory read.
    for (auto inp_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (inp_tv->getMemoryType() == MemoryType::Shared) {
        TORCH_INTERNAL_ASSERT(
            sync_found, "Block sync required but not inserted");
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test BestEffortReplay behavior with swizzle op
TEST_F(SwizzleTest, SwizzleMapping) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  // Make a 2x8 Zshape tile
  tv2->split(-1, 16);
  tv2->split(-1, 8);
  // [O, 2, 8]

  tv3->split(-1, 16);
  tv3->split(-1, 4);
  //[O, 4, 4]

  tv2->computeAt(tv3, 1);
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Inlining a producer into a swizzled consumer is ok
  tv1->computeAt(tv2, -1);

  // Check BestEffortReplay behavior with skip swizzles option on.
  PairwiseRootDomainMap root_map(tv1, tv2);

  // Check producer to consumer map,
  //  i.e. unswizzled tensor to swizzled tensor map
  //----------------------------------------------------------
  auto p2c_disjoint_id_map =
      BestEffortReplay::replayCasP(tv2, tv1, -1, root_map)
          .getIterDomainEquivalence();
  // P2C map should exist and both the x and y map should
  //  map to the output of the swizzle op.
  TORCH_INTERNAL_ASSERT(
      p2c_disjoint_id_map.mappingExists(tv1->axis(-2)) &&
      p2c_disjoint_id_map.mappingExists(tv1->axis(-1)));

  TORCH_INTERNAL_ASSERT(
      p2c_disjoint_id_map.strictAreMapped(tv1->axis(-2), tv2->axis(-2)) &&
      p2c_disjoint_id_map.strictAreMapped(tv1->axis(-1), tv2->axis(-1)));

  // Check consumer to producer map,
  //  i.e. swizzled tensor to unswizzled tensor map
  //----------------------------------------------------------
  auto c2p_disjoint_id_map =
      BestEffortReplay::replayPasC(tv1, tv2, -1, root_map)
          .getIterDomainEquivalence();

  auto swizzle_op = tv2->axis(-1)->definition()->as<Swizzle2D>();

  // Input of swizzle ops will not be mapped to any
  //  by BestEffortReplay, as BestEffortReplay has to be
  //  one to one. IdGraph will further map them together.
  TORCH_INTERNAL_ASSERT(
      !p2c_disjoint_id_map.mappingExists(swizzle_op->inX()) &&
      !p2c_disjoint_id_map.mappingExists(swizzle_op->inY()));

  // Mapping for swizzle outputs should be mapped and should
  //  also map to the corresponding axes on the unswizzled tensor.
  TORCH_INTERNAL_ASSERT(
      p2c_disjoint_id_map.mappingExists(swizzle_op->outX()) &&
      p2c_disjoint_id_map.mappingExists(swizzle_op->outY()));

  TORCH_INTERNAL_ASSERT(
      p2c_disjoint_id_map.strictAreMapped(swizzle_op->outX(), tv1->axis(-2)) &&
      p2c_disjoint_id_map.strictAreMapped(swizzle_op->outY(), tv1->axis(-1)));

  // Check id graph behavior
  //----------------------------------------------------------
  ComputeAtMap ca_map(&fusion);
  // Corresponding inputs and outputs of swizzle ops are
  //  map through by exact and permissive map.
  TORCH_INTERNAL_ASSERT(
      ca_map.areMapped(tv1->axis(-2), swizzle_op->inX(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(
      ca_map.areMapped(tv1->axis(-1), swizzle_op->inY(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::EXACT));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::EXACT));

  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->inX(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->inY(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-2), swizzle_op->outX(), IdMappingMode::PERMISSIVE));
  TORCH_INTERNAL_ASSERT(ca_map.areMapped(
      tv1->axis(-1), swizzle_op->outY(), IdMappingMode::PERMISSIVE));
}

// Test a basic loop swizzle pattern
TEST_F(SwizzleTest, LoopSwizzle0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  tv0->computeAt(tv2, -1);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Outer block zshape pattern
TEST_F(SwizzleTest, LoopSwizzle1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-2, 8);
  tv2->split(-1, 4);
  //[I0o, I0i, I1o, I1i]
  tv2->reorder({{1, 2}, {2, 1}});
  //[I0o, I1o, I0i, I1i]

  tv2->swizzle(Swizzle2DType::ZShape, 0, 1, SwizzleMode::Loop);
  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({45, 77}, options);
  auto t2 = t0 + 2.0;
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

// Test assertion in unsupported pattern: non-leaf loop swizzle.
TEST_F(SwizzleTest, LoopSwizzleCheck0) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv2);

  tv2->split(-1, 16);
  tv2->split(-1, 4);
  //[O, 4, 4]

  // Swizzle the inner tile.
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make swizzle output not a leaf domain.
  tv2->merge(-2);

  tv0->computeAt(tv2, -1);

  FusionExecutor fe;
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

// Test assertion in unsupported pattern: half-inlined loop swizzle.
TEST_F(SwizzleTest, LoopSwizzleCheck1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  //[O, 4, 4]
  tv2->split(-1, 16);
  tv2->split(-1, 4);

  //[O, 4, 4]
  tv3->split(-1, 16);
  tv3->split(-1, 4);

  // Swizzle inner tile of tv2
  tv2->swizzle(Swizzle2DType::ZShape, -2, -1, SwizzleMode::Loop);

  // Make tv2 swizzled and partially-inlined (unsupported).
  tv0->computeAt(tv3, -2);

  FusionExecutor fe;
  ASSERT_ANY_THROW(fe.compileFusion(&fusion));
}

TEST_F(SwizzleTest, SwizzleVectorize) {
  // When there is a swizzle, non of the involved dimensions are contiguous, so
  // unable to vectorize.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4, 4});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->swizzle(Swizzle2DType::XOR, 0, 1);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  ASSERT_ANY_THROW(GpuLower lower(&fusion));
}

TEST_F(SwizzleTest, TransposeBankConflictSwizzle1) {
  // Both Xor and CyclicShift swizzling should fully remove bank confliction of
  // a 32x32 non-vectorized transpose.
  std::vector<Swizzle2DType> swizzles{
      Swizzle2DType::XOR, Swizzle2DType::CyclicShift};
  for (auto swizzle_type : swizzles) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeConcreteTensor({32, 32});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = transpose(tv1, 0, 1);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->axis(0)->parallelize(ParallelType::TIDy);
    tv1->axis(1)->parallelize(ParallelType::TIDx);
    tv2->axis(0)->parallelize(ParallelType::TIDy);
    tv2->axis(1)->parallelize(ParallelType::TIDx);
    tv3->axis(0)->parallelize(ParallelType::TIDy);
    tv3->axis(1)->parallelize(ParallelType::TIDx);

    // 32-way bank confliction
    auto bank_conflict_info = fusion.bankConflictInfo();
    ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{32});

    // no bank confliction after swizzle
    tv1->swizzle(swizzle_type, 0, 1);
    bank_conflict_info = fusion.bankConflictInfo();
    TORCH_CHECK(
        bank_conflict_info.empty(),
        "Expecting no bank conflict after swizzle, but got ",
        bank_conflict_info.size(),
        "bank conflicting expressions.",
        ". Something in our lowering or bank conflict checker must have changed, ",
        "please update them or this test consistently.");
  }
}

TEST_F(SwizzleTest, TransposeBankConflictSwizzle2) {
  // ZShape should remove half of the bank confliction of a 32x32 non-vectorized
  // transpose.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  // 32-way bank confliction
  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{32});

  // 16-way bank confliction
  tv1->swizzle(Swizzle2DType::ZShape, 0, 1);
  bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{16});
}

TEST_F(SwizzleTest, DataSwizzleGlobal) {
  // Data swizzle is ignored in global indexing, so we should just throw an
  // error if someone wants to do so.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);
  ASSERT_ANY_THROW(tv1->swizzle(Swizzle2DType::XOR, 0, 1));
}

namespace {

// Get the swizzled tensor from input. For example, for ZShape swizzle, if the
// input is
//    1 2 3
//    4 5 6
//    7 8 9
// Then the output will be:
//    1 2 3
//    6 5 4
//    7 8 9
at::Tensor getSwizzledTensor(
    at::Tensor input,
    Swizzle2DType type,
    bool is_unswizzle = false) {
  auto size_x = input.size(0);
  auto size_y = input.size(1);

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  Val* size_x_input = IrBuilder::create<Val>(DataType::Int);
  Val* size_y_input = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(size_x_input);
  fusion.addInput(size_y_input);
  auto x = arange(size_x_input);
  auto xx = broadcast(x, {false, true});
  auto y = arange(size_y_input);
  auto yy = broadcast(y, {true, false});
  std::pair<Val*, Val*> swizzle;
  if (is_unswizzle) {
    swizzle = dispatchUnSwizzle(type, xx, yy, size_x_input, size_y_input);
  } else {
    swizzle = dispatchSwizzle(type, xx, yy, size_x_input, size_y_input);
  }
  fusion.addOutput(swizzle.first);
  fusion.addOutput(swizzle.second);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs({size_x, size_y});

  return input.index_put({outputs[0], outputs[1]}, input);
}

} // namespace

TEST_F(SwizzleTest, SwizzleExampleZShape) {
  //    1 2 3      1 2 3
  //    4 5 6  =>  6 5 4
  //    7 8 9      7 8 9
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, options);
  auto expect = torch::tensor({{1, 2, 3}, {6, 5, 4}, {7, 8, 9}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::ZShape);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::ZShape, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

TEST_F(SwizzleTest, SwizzleExampleXor) {
  //    1   2  3  4       1   2   3  4
  //    5   6  7  8       6   5   8  7
  //    9  10 11 12  =>   11  12  9 10
  //    13 14 15 16       16  15 14 13
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {6, 5, 8, 7}, {11, 12, 9, 10}, {16, 15, 14, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::XOR);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::XOR, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

TEST_F(SwizzleTest, SwizzleExampleCyclicShift) {
  //    1   2  3  4       1   2   3   4
  //    5   6  7  8       8   5   6   7
  //    9  10 11 12  =>   11  12  9  10
  //    13 14 15 16       14  15  16 13
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, options);
  auto expect = torch::tensor(
      {{1, 2, 3, 4}, {8, 5, 6, 7}, {11, 12, 9, 10}, {14, 15, 16, 13}}, options);
  auto output = getSwizzledTensor(input, Swizzle2DType::CyclicShift);
  auto unswizzled = getSwizzledTensor(output, Swizzle2DType::CyclicShift, true);
  TORCH_CHECK(at::equal(expect, output));
  TORCH_CHECK(at::equal(input, unswizzled));
}

TEST_F(SwizzleTest, SwizzleIndexing170) {
  // https://github.com/NVIDIA/Fuser/issues/170
  GTEST_SKIP() << "Repro for an unfixed bug";
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({64, 64});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv1->split(1, 8);
  tv1->split(1, 4);
  tv1->split(0, 8);
  tv1->split(0, 4);
  // [2 4 8 2 4 8]
  tv1->swizzle(Swizzle2DType::XOR, 1, 4);
  tv1->merge(0);
  tv1->merge(0);
  tv1->merge(1);
  tv1->merge(1);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->split(0, 256);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t = at::randn({64, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({t});

  testValidate(&fusion, outputs, {t}, {t}, __LINE__, __FILE__);
}

TEST_F(SwizzleTest, TransformPropagatorSkipSwizzleOnTarget) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeConcreteTensor({64, 64});
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv0->split(1, 8);
  tv0->split(0, 8);
  tv0->merge(0);
  tv0->merge(1);

  tv1->split(1, 8);
  tv1->split(0, 8);
  tv1->swizzle(Swizzle2DType::XOR, 0, 2);
  tv1->merge(0);
  tv1->merge(1);

  tv0->merge(0);

  TransformPropagatorWithCheck propagator(tv0);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);

  auto exprs = StmtSort::getExprsBetween(
      tv1->fusion(),
      {tv1->getRootDomain().begin(), tv1->getRootDomain().end()},
      {tv1->getLeafDomain().begin(), tv1->getLeafDomain().end()});
  EXPECT_TRUE(std::any_of(exprs.begin(), exprs.end(), [](Expr* expr) {
    return expr->isA<Swizzle2D>();
  }));
}

} // namespace nvfuser
