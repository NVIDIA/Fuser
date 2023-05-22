#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <executor_utils.h>
#include <fusion.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

// Simple pad test
TEST_F(NVFuserTest, FusionResizePad1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// pad + split
TEST_F(NVFuserTest, FusionResizePad2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// pad, merge + split, inlineMost
TEST_F(NVFuserTest, FusionResizePad3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9, 11});
  std::vector<int64_t> padded_shape({9, 11 + 2});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = pad(tv2, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  // TransformPropagator and inlineMost do not inline tv2, so it can't
  // be on Local memory. It should be possible to expand tv2 such that
  // it has the same extent as tv3, allowing it to be inlined.
  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t3 = at::pad(t0, {1, 1});
  auto ref = t3 + t1;

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// pad + parallelization
TEST_F(NVFuserTest, FusionResizePad4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// pad + parallelization + RAW sync
TEST_F(NVFuserTest, FusionResizePad5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  TORCH_CHECK(
      tv1->getMemoryType() == MemoryType::Shared,
      "tv1 should be on shared memory: ",
      tv1->getMemoryType());

  GpuLower gpulw(&fusion);
  auto all_lowered_exprs = KernelExprVisitor::getAllExprs(gpulw.kernel());
  TORCH_CHECK(
      std::find_if(
          all_lowered_exprs.begin(),
          all_lowered_exprs.end(),
          [](Expr* expr) { return expr->isA<kir::BlockSync>(); }) !=
          all_lowered_exprs.end(),
      "Block sync not found");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// pad + merge + split parallelization
TEST_F(NVFuserTest, FusionResizePad6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({99, 111});
  std::vector<int64_t> padded_shape({shape[0], shape[1] + 2});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(padded_shape);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = pad(tv2, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t2 = t0 + 1;
  auto t3 = at::pad(t2, {1, 1});
  auto ref = t3 + t1;

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// pad + unswitch. Having different extents in an unswitched loop nest
// needs a special care (see UnrollPass::canOmitElseClause)
TEST_F(NVFuserTest, FusionResizePad7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9, 11});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv3->split(0, 1);
  tv3->split(-1, 4);
  tv3->reorder({{1, 2}});

  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-2)->parallelize(ParallelType::Unswitch);

  scheduler_utils::parallelizeAllLike(tv3);

  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Disable for now. Unclear what would be the best way to handle
// when a tensor is resized multiple times. It would likely need a
// different transform propagator.
#if 0
// Stencil-like pattern
TEST_F(NVFuserTest, FusionResizePad8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Sort of shift(tv1, {-1});
  auto tv2 = pad(tv1, {IrBuilder::create<Int>(0), IrBuilder::create<Int>(1)});
  // Sort of shift(tv1, {1});
  auto tv3 = pad(tv1, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(0)});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 128);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  scheduler_utils::promoteProducerMemoryTypesOfResizedTensors(&fusion, {});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(999, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {0, 1}) + at::pad(t0, {1, 0});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}
#endif

TEST_F(NVFuserTest, FusionResizePadScheduler1_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  fusion->addOutput(tv1);

  std::vector<int64_t> shape({99, 111});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionResizePadScheduler2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({9, 11});
  std::vector<int64_t> padded_shape({9, 11 + 2});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = pad(tv2, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t3 = at::pad(t0, {1, 1});
  auto ref = t3 + t1;

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Disabled due to the same reason as Pad8
#if 0
// Auto scheduled version of Pad8
TEST_F(NVFuserTest, FusionResizePadScheduler3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Int>(0), IrBuilder::create<Int>(1)});
  auto tv3 = pad(tv1, {IrBuilder::create<Int>(1), IrBuilder::create<Int>(0)});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(999, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {0, 1}) + at::pad(t0, {1, 0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}
#endif

// Two pad exprs, both using the same symbolic pad widths, segmented
// into two kernels. Make sure the symbolic inputs are available to
// both of the segmented kernels.
TEST_F(NVFuserTest, FusionResizePadScheduler4_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto left_pad = IrBuilder::create<Int>();
  fusion->addInput(left_pad);
  auto right_pad = IrBuilder::create<Int>();
  fusion->addInput(right_pad);

  auto tv1 = pad(tv0, {left_pad, right_pad});
  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto tv3 = pad(tv0, {left_pad, right_pad});
  auto tv4 = sum(tv3, {1});
  fusion->addOutput(tv4);

  std::vector<int64_t> shape({99, 111});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<int64_t> pad_extents{1, 1};
  std::vector<c10::IValue> aten_inputs({t0, 1, 1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t0_double = t0.to(at::kDouble);
  auto t2 = at::pad(t0_double, {1, 1}).sum({0});
  auto t4 = at::pad(t0_double, {1, 1}).sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2, t4},
      __LINE__,
      __FILE__);
}

// Trivial cat
TEST_F(NVFuserTest, FusionResizeCat1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2});
  std::vector<int64_t> shape1({3});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial 2D inner cat
TEST_F(NVFuserTest, FusionResizeCat2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial 2D outer cat
TEST_F(NVFuserTest, FusionResizeCat3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({4, 2});
  std::vector<int64_t> shape1({4, 3});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Cat + merge + split + parallelization + inlineMost
TEST_F(NVFuserTest, FusionResizeCat4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({11, 13});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 128);

  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Cat + arith op
TEST_F(NVFuserTest, FusionResizeCat5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor({11, 12});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({11, 13});
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor({11, 25});
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1}, 1);
  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 128);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});
  std::vector<int64_t> shape2({shape0[0], shape0[1] + shape1[1]});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1) + t2;

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Cat 3 tensors
TEST_F(NVFuserTest, FusionResizeCat6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({5, 4});
  std::vector<int64_t> shape2({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor(shape2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(0, 4);
  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1, t2}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Cat many tensors
TEST_F(NVFuserTest, FusionResizeCat7_CUDA) {
  int num_tensors_to_concat = 10;
  std::vector<int64_t> base_shape({11, 13});

  for (int concat_dim : {0, 1}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    std::vector<TensorView*> inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      (void)i;
      // concrete shapes to avoid dynamic Fusion
      auto shape = base_shape;
      shape[concat_dim] = 10 + (i % 5);
      auto tv = makeConcreteTensor(shape);
      fusion.addInput(tv);
      inputs.push_back(tv);
    }

    auto concat_tv = cat(inputs, concat_dim);
    fusion.addOutput(concat_tv);

    concat_tv->merge(0);
    concat_tv->split(0, 128);

    TransformPropagator propagator(concat_tv);
    MaxRootDomainInfoSpanningTree(concat_tv).traverse(&propagator);

    inlineMost();

    concat_tv->axis(0)->parallelize(ParallelType::BIDx);
    concat_tv->axis(1)->parallelize(ParallelType::TIDx);
    scheduler_utils::parallelizeAllLike(concat_tv);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    std::vector<at::Tensor> aten_inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      auto shape = base_shape;
      shape[concat_dim] = 10 + (i % 5);
      aten_inputs.emplace_back(at::randn(shape, options));
    }

    std::vector<c10::IValue> aten_inputs_ivalue(
        {aten_inputs.begin(), aten_inputs.end()});

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs_ivalue);
    auto cg_outputs = fe.runFusion(aten_inputs_ivalue);

    auto ref = at::cat(aten_inputs, concat_dim);

    TORCH_CHECK(ref.equal(cg_outputs[0]));
  }
}

// Auto scheduled version of Cat1
TEST_F(NVFuserTest, FusionResizeCatScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({2});
  std::vector<int64_t> shape1({3});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Auto scheduled version of Cat5
TEST_F(NVFuserTest, FusionResizeCatScheduler2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1}, 1);
  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});
  std::vector<int64_t> shape2({shape0[0], shape0[1] + shape1[1]});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1}, 1) + t2;

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Auto scheduled version of Cat6
TEST_F(NVFuserTest, FusionResizeCatScheduler3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({5, 4});
  std::vector<int64_t> shape2({3, 4});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1, t2}, 0);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial slice
TEST_F(NVFuserTest, FusionResizeSlice1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(1),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Int>(1))}});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Split a tensor to half and add them up
TEST_F(NVFuserTest, FusionResizeSlice2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({11, 30});

  TORCH_CHECK(shape[1] % 2 == 0);

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {Slice(),
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(shape[1] / 2)}});
  auto tv2 = slice(tv0, {Slice(), {IrBuilder::create<Int>(shape[1] / 2)}});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(0, shape[1] / 2)});
  auto t2 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(shape[1] / 2)});
  auto ref = t1 + t2;

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// "Trivial" slice is converted to Set
TEST_F(NVFuserTest, FusionResizeSlice3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // These should result in unary set op
  auto tv1 = slice(tv0, {{nullptr, tv0->axis(0)->extent()}});
  auto tv2 = slice(tv0, {Slice()});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  TORCH_CHECK(tv1->definition()->isA<LoadStoreOp>());
  TORCH_CHECK(tv2->definition()->isA<LoadStoreOp>());
}

// Partition an input, reduce each and concatenate them
TEST_F(NVFuserTest, FusionResizeSlice4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({5, 100});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // Consider a fusion of:
  // auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  // auto tv2 = sum(tv1, {1});

  // Reproduce the above fusion with split tensors

  // Split the input to [0:2, :] and [2:, :]
  auto tv1 = slice(
      tv0, {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(2)}, Slice()});
  auto tv2 = slice(tv0, {{IrBuilder::create<Int>(2)}, Slice()});

  auto tv3 = add(tv1, IrBuilder::create<Double>(1));
  auto tv4 = add(tv2, IrBuilder::create<Double>(1));

  auto tv5 = sum(tv3, {1});
  auto tv6 = sum(tv4, {1});
  auto tv7 = cat({tv5, tv6}, 0);
  fusion.addOutput(tv7);

  // Schedule the two reductions separately
  tv5->split(-1, 32);
  auto tv5_rf = tv5->rFactor({-2});
  tv5_rf->reorder({{-1, -2}});
  auto tv5_cache = tv5->cacheBefore();
  tv5->setMemoryType(MemoryType::Global);
  SetSelector tv5_rf_selector({tv1, tv3, tv5, tv5_cache});
  TransformPropagator tv5_rf_tp(tv5_rf);
  MaxRootDomainInfoSpanningTree(tv5_rf, &tv5_rf_selector).traverse(&tv5_rf_tp);
  inlineMost(std::vector<TensorView*>{tv1, tv3, tv5_rf});
  tv5_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv5_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5_rf, {tv1, tv3, tv5, tv5_cache});

  tv6->split(-1, 32);
  auto tv6_rf = tv6->rFactor({-2});
  tv6_rf->reorder({{-1, -2}});
  auto tv6_cache = tv6->cacheBefore();
  tv6->setMemoryType(MemoryType::Global);
  SetSelector tv6_rf_selector({tv2, tv4, tv6, tv6_cache});
  TransformPropagator tv6_rf_tp(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf, &tv6_rf_selector).traverse(&tv6_rf_tp);
  inlineMost(std::vector<TensorView*>{tv2, tv4, tv6_rf});
  tv6_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv6_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv6_rf, {tv2, tv4, tv6, tv6_cache});

  // cat consits of a PadOp and a CatOp. Fully inline the PadOp
  for (auto tv7_inp :
       ir_utils::filterByType<TensorView>(tv7->definition()->inputs())) {
    tv7_inp->inlineAt(-1);
  }

  // Use just one block to concat the two results
  tv7->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = (t0 + 1).to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Multiple slices of the same tensor with the same arguments
TEST_F(NVFuserTest, FusionResizeSlice5_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({11, 1000});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {Slice(),
       {IrBuilder::create<Int>(1),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Int>(1))}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(
      tv0,
      {Slice(),
       {IrBuilder::create<Int>(1),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Int>(1))}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  tv2->split(1, 128);

  // tv1 and tv3 are both slice outputs. Propagation should occur from
  // tv1 to tv3 through tv0, which should work as both tensors are
  // sliced in the same way.
  TransformPropagator propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[1] - 1)});
  auto t2 = t1.to(at::kDouble).sum({1});
  auto t3 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[1] - 1)});
  auto t4 = t3.to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, aten_inputs, {t2, t4}, __LINE__, __FILE__);
}

// Auto scheduled version of Slice1
TEST_F(NVFuserTest, FusionResizeSliceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(1),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Int>(1))}});
  fusion.addOutput(tv1);

  std::vector<int64_t> shape({9});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionResizePadReduceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto left_pad0 = IrBuilder::create<Int>();
  fusion.addInput(left_pad0);
  auto right_pad0 = IrBuilder::create<Int>();
  fusion.addInput(right_pad0);
  auto left_pad1 = IrBuilder::create<Int>();
  fusion.addInput(left_pad1);
  auto right_pad1 = IrBuilder::create<Int>();
  fusion.addInput(right_pad1);

  auto tv1 = pad(tv0, {left_pad0, right_pad0, left_pad1, right_pad1});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> pad_extents{1, 2, 2, 1};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::transform(
      pad_extents.begin(),
      pad_extents.end(),
      std::back_inserter(aten_inputs),
      [](auto pad_extent) { return pad_extent; });

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, pad_extents).sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionResizeSliceReduceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Int>();
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Int>();
  fusion.addInput(end0);
  auto start1 = IrBuilder::create<Int>();
  fusion.addInput(start1);
  auto end1 = IrBuilder::create<Int>();
  fusion.addInput(end1);

  auto tv1 = slice(tv0, {{start0, end0}, {start1, end1}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> slice_inputs({1, shape[0] - 2, 3, shape[1] - 4});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(slice_inputs[0], slice_inputs[1]),
       at::indexing::Slice(slice_inputs[2], slice_inputs[3])});
  auto ref = t1.sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Multiple slice+reduction. Different slices.
TEST_F(NVFuserTest, FusionResizeSliceReduceScheduler2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Int>();
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Int>();
  fusion.addInput(end0);
  auto start1 = IrBuilder::create<Int>();
  fusion.addInput(start1);
  auto end1 = IrBuilder::create<Int>();
  fusion.addInput(end1);

  auto tv1 = slice(tv0, {Slice(), {start0, end0}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(tv0, {Slice(), {start1, end1}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 1024});
  std::vector<int64_t> slice_inputs({1, shape[0] - 2, 3, shape[1] - 4});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(slice_inputs[0], slice_inputs[1])});
  auto t2 = t1.sum({1});
  auto t3 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(slice_inputs[2], slice_inputs[3])});
  auto t4 = t3.sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2, t4},
      __LINE__,
      __FILE__);
}

// Multiple slice+reduction. Same slices. Should be segmented at the moment.
TEST_F(NVFuserTest, FusionSliceReduceScheduler3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Int>();
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Int>();
  fusion.addInput(end0);

  auto tv1 = slice(tv0, {Slice(), {start0, end0}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(tv0, {Slice(), {start0, end0}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> slice_inputs({1, shape[1] - 2});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(slice_inputs[0], slice_inputs[1])});
  auto t2 = t1.to(at::kDouble).sum({1});
  auto t3 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(slice_inputs[0], slice_inputs[1])});
  auto t4 = t3.to(at::kDouble).sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2, t4},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionResizeCatReduceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  auto tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1}, 1).sum({1});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionResizeCatSoftmaxScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  auto tv3 = softmax(tv2, 1);
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({11, 99});
  std::vector<int64_t> shape1({shape0[0], 100});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t2 = at::cat({t0, t1}, 1);
  auto ref = at::_softmax(t2.to(at::kDouble), -1, false);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionResizeReductionSliceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Int>(1),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Int>(2))}});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({10, 1234});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.to(at::kDouble).sum({1});
  auto t2 = t1.index({at::indexing::Slice(1, shape0[0] - 2)});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2},
      __LINE__,
      __FILE__);
}

// Softmax followed by slicing of a non-normalized dimension
TEST_F(NVFuserTest, FusionResizeSoftmaxSliceScheduler1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = softmax(tv0, 1);
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Int>(1),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Int>(2))},
       Slice()});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({13, 1234});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = at::_softmax(t0.to(at::kDouble), -1, false);
  auto t2 = t1.index(
      {at::indexing::Slice(1, shape0[0] - 2),
       at::indexing::Slice(0, at::indexing::None)});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2},
      __LINE__,
      __FILE__);
}

// Softmax followed by slicing of a normalized dimension
TEST_F(NVFuserTest, FusionResizeSoftmaxSliceScheduler2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = softmax(tv0, 1);
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Int>(1),
        sub(tv1->axis(1)->extent(), IrBuilder::create<Int>(2))}});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({110, 12345});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = at::_softmax(t0.to(at::kDouble), -1, false);
  auto t2 = t1.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape0[1] - 2)});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {t2},
      __LINE__,
      __FILE__);
}

// Same as Pad1 but pad by specified value
TEST_F(NVFuserTest, FusionResizePadWithValue_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)},
          IrBuilder::create<Int>(2));
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Same as above but try to pad an int tensor with a double value
TEST_F(NVFuserTest, FusionResizePadIntWithDoubleValue_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)},
          IrBuilder::create<Double>(2.5));
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  auto t0 = at::ones(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2.5);

  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

// Test that padding Half tensor by Double does not promote output
TEST_F(NVFuserTest, FusionResizePadHalfWithDoubleValue_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Int>(1), IrBuilder::create<Int>(1)},
          IrBuilder::create<Double>(2.5));
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t0 = at::ones(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2.5);

  TORCH_CHECK(ref.dtype() == cg_outputs[0].dtype());
  TORCH_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionSliceForNanoGPT1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> input_shape0{1, 1, 1024, 1024};
  std::vector<int64_t> input_shape1{32, 16, 128, 128};

  auto tv0 = makeContigConcreteTensor({1, 1, -1, -1});
  auto tv1 = makeContigConcreteTensor({-1, -1, -1, -1});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  Slice dim0{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(1),
      IrBuilder::create<Int>(1)};
  Slice dim1{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(1),
      IrBuilder::create<Int>(1)};
  Slice dim2{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(128),
      IrBuilder::create<Int>(1)};
  Slice dim3{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(128),
      IrBuilder::create<Int>(1)};
  auto tv2 = slice(tv0, {dim0, dim1, dim2, dim3});

  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  auto tv4 = slice(tv0, {dim0, dim1, dim2, dim3});

  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape0, options);
  auto t1 = at::randn(input_shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto kernel =
      executor_cache.getMostRecentKernelRuntime()->executors().at(0).kernel();
  TORCH_CHECK(
      !kernel->summary().has_cooperative_grid_reduction,
      "Grid sync should not be used as slicing input should avoid input caching");

  auto aten_t0_slice = t0.index(
      {at::indexing::Slice(0, 1, 1),
       at::indexing::Slice(0, 1, 1),
       at::indexing::Slice(0, 128, 1),
       at::indexing::Slice(0, 128, 1)});
  auto aten_t3 = torch::add(aten_t0_slice, t1);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {aten_t3, aten_t3},
      __LINE__,
      __FILE__);
}

// Similar to FusionSliceForNanoGPT1 but the input to slice is an
// intermediate tensor
TEST_F(NVFuserTest, FusionSliceForNanoGPT2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> input_shape0{100, 100};
  std::vector<int64_t> input_shape1{32, 32};

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));

  Slice dim0{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(32),
      IrBuilder::create<Int>(1)};
  Slice dim1{
      IrBuilder::create<Int>(0),
      IrBuilder::create<Int>(32),
      IrBuilder::create<Int>(1)};

  auto tv3 = slice(tv2, {dim0, dim1});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto tv5 = slice(tv2, {dim0, dim1});
  auto tv6 = add(tv5, tv1);
  fusion.addOutput(tv6);

  // Another use of tv2. Unlike the above two slice ops, this should
  // not use the copy of tv2
  auto tv7 = add(tv2, IrBuilder::create<Double>(1));
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape0, options);
  auto t1 = at::randn(input_shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto kernel =
      executor_cache.getMostRecentKernelRuntime()->executors().at(0).kernel();

  // Make sure the slices ops use the same producer
  TensorView* known_slice_producer = nullptr;
  for (auto expr : KernelExprVisitor::getAllExprs(kernel)) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    auto out_tv = ir_utils::getTvOutput(expr);
    if (out_tv->name() == tv3->name() || out_tv->name() == tv5->name()) {
      TORCH_CHECK(
          expr->isA<LoadStoreOp>(),
          "Unexpected defintion of slice output tensor: ",
          out_tv->toString(),
          ", ",
          expr->toString());
      auto producer =
          dynamic_cast<kir::TensorIndex*>(expr->as<LoadStoreOp>()->in());
      if (producer == nullptr) {
        // this could be a default initialization
        continue;
      }
      if (known_slice_producer == nullptr) {
        known_slice_producer = producer->view();
      } else {
        TORCH_CHECK(
            known_slice_producer == producer->view(),
            "Expected to have the same tensor is used for the two slice ops. ",
            "Previously found producer: ",
            known_slice_producer->toString(),
            ", new producer: ",
            producer->view()->toString());
      }
    } else if (auto binary_op = dynamic_cast<BinaryOp*>(expr)) {
      // If this is a binary op producing tv7, make sure its producer
      // is tv2
      if (binary_op->getBinaryOpType() == BinaryOpType::Add &&
          binary_op->out()->isA<kir::TensorIndex>() &&
          binary_op->out()->as<kir::TensorIndex>()->view()->name() ==
              tv7->name()) {
        TORCH_CHECK(
            binary_op->lhs()->as<kir::TensorIndex>()->view()->name() ==
                tv2->name(),
            "Unexpected tv7 definition: ",
            binary_op->toString());
      }
    }
  }

  TORCH_CHECK(known_slice_producer != nullptr, "Slice producer not found");

  // The slice producer must be a copy of tv2
  TORCH_CHECK(
      known_slice_producer->definition() &&
          known_slice_producer->definition()->isA<LoadStoreOp>() &&
          known_slice_producer->definition()->as<LoadStoreOp>()->in()->name() ==
              tv2->name(),
      "Unexpected slice producer: ",
      known_slice_producer->toString());

  auto aten_t2 = t0 + 1;
  auto aten_slice = aten_t2.index(
      {at::indexing::Slice(0, 32, 1), at::indexing::Slice(0, 32, 1)});
  auto aten_t4 = aten_slice + t1;

  auto aten_t7 = aten_t2 + 1;

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {aten_t4, aten_t4, aten_t7},
      __LINE__,
      __FILE__);
}

// C++ version of TestNvFuserFrontend.test_nanogpt_split_mha_linears
TEST_F(NVFuserTest, FusionSliceForNanoGPT3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> input_shape{16, 128, 3072};

  auto tv0 = makeSymbolicTensor(3);

  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(16)},
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(128)},
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(1024)}});
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(16)},
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(128)},
       {IrBuilder::create<Int>(1024), IrBuilder::create<Int>(2048)}});
  auto tv3 = slice(
      tv0,
      {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(16)},
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(128)},
       {IrBuilder::create<Int>(2048), IrBuilder::create<Int>(3072)}});

  auto tv4 = reshape(tv1, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv5 = reshape(tv2, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv6 = reshape(tv3, {16, 128, 1024}, {16, 128, 16, 64});

  // TODO: add permute
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto kernel = runtime->executors().at(0).kernel();
  TORCH_CHECK(
      !kernel->summary().has_cooperative_grid_reduction,
      "Grid sync should not be used as slicing input should avoid input caching");

  auto at_t1 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(0, 1024)});
  auto at_t2 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(1024, 2048)});
  auto at_t3 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(2048, 3072)});

  auto at_t4 = at_t1.reshape({16, 128, 16, 64});
  auto at_t5 = at_t2.reshape({16, 128, 16, 64});
  auto at_t6 = at_t3.reshape({16, 128, 16, 64});

  TORCH_CHECK(cg_outputs.at(0).equal(at_t4));
  TORCH_CHECK(cg_outputs.at(1).equal(at_t5));
  TORCH_CHECK(cg_outputs.at(2).equal(at_t6));
}

TEST_F(NVFuserTest, ResizeReshapeAndSlice_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {4, 8}, {8, 4});
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Int>(0), IrBuilder::create<Int>(2)},
       {IrBuilder::create<Int>(0), IrBuilder::create<Int>(2)}});
  fusion->addOutput(tv2);

  std::vector<int64_t> shape({4, 8});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto ref = t0.reshape({8, 4}).index(
      {at::indexing::Slice(0, 2), at::indexing::Slice(0, 2)});

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

// Make sure resize works with the transpose scheduler
TEST_F(NVFuserTest, ResizePermuteAndSlice_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Set the problem size so that it can trigger the transpose
  // scheduler. The scheduler selection is validated below.
  auto num_sms =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  std::vector<int64_t> shape({num_sms + 2, 32 * 32 + 10});

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Int>(1), IrBuilder::create<Int>(shape.at(0) - 1)},
       {IrBuilder::create<Int>(2), IrBuilder::create<Int>(shape.at(1) - 2)}});
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);
  auto tv4 = add(tv2, IrBuilder::create<Double>(1));
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto heuristic =
      runtime->schedulerHeuristics()->heuristicsList().at(0).get()->heuristic();
  TORCH_CHECK(
      heuristic == ScheduleHeuristic::Transpose,
      "Unexpected heuristic: ",
      heuristic);

  auto ref_t2 = (t0 + 1).index(
      {at::indexing::Slice(1, shape.at(0) - 1),
       at::indexing::Slice(2, shape.at(1) - 2)});
  auto ref_t3 = ref_t2.transpose(0, 1);
  auto ref_t4 = ref_t2 + 1;

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref_t3, ref_t4},
      __LINE__,
      __FILE__);
}

// When scheduling this test, the pointwise scheduler attempt to replay a Split
// transform on a size-0 dimension, which is not allowed.
TEST_F(NVFuserTest, FusionSizeZeroSliceSplitSchedule_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({8});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion->addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(0),
        IrBuilder::create<Int>(2),
        IrBuilder::create<Int>(1)}});
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Int>(2),
        IrBuilder::create<Int>(4),
        IrBuilder::create<Int>(1)}});
  auto tv3 = slice(
      tv0,
      {{IrBuilder::create<Int>(4),
        IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(1)}});
  auto tv4 = slice(
      tv0,
      {{IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(1)}});
  auto tv5 = slice(
      tv0,
      {{IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(1)}});
  auto tv6 = slice(
      tv0,
      {{IrBuilder::create<Int>(6),
        IrBuilder::create<Int>(8),
        IrBuilder::create<Int>(1)}});
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);
  fusion->addOutput(tv4);
  fusion->addOutput(tv5);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  FusionExecutor fe;

  auto ref0 = t0.index({at::indexing::Slice(0, 2)});
  auto ref1 = t0.index({at::indexing::Slice(2, 4)});
  auto ref2 = t0.index({at::indexing::Slice(4, 6)});
  auto ref3 = t0.index({at::indexing::Slice(6, 6)});
  auto ref4 = t0.index({at::indexing::Slice(6, 6)});
  auto ref5 = t0.index({at::indexing::Slice(6, 8)});

  TORCH_CHECK(ref0.equal(cg_outputs[0]));
  TORCH_CHECK(ref1.equal(cg_outputs[1]));
  TORCH_CHECK(ref2.equal(cg_outputs[2]));
  TORCH_CHECK(ref3.equal(cg_outputs[3]));
  TORCH_CHECK(ref4.equal(cg_outputs[4]));
  TORCH_CHECK(ref5.equal(cg_outputs[5]));
}

// In this test, we split and merge with size-zero dimensions directly.
TEST_F(NVFuserTest, FusionSizeZeroSliceSplit_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({4, 5});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion->addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(2),
        IrBuilder::create<Int>(2),
        IrBuilder::create<Int>(1)},
       {IrBuilder::create<Int>(0),
        IrBuilder::create<Int>(5),
        IrBuilder::create<Int>(1)}});
  // tv1 is of shape {0, 5}
  fusion->addOutput(tv1);

  tv1->merge(0, 1); // size 0*5 = 0
  tv1->split(0, 4); // sizes (0, 4)

  FusionExecutor fe;
  fe.compileFusion(fusion.get());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref0 = t0.index({at::indexing::Slice(2, 2), at::indexing::Slice(0, 5)});

  TORCH_CHECK(ref0.equal(cg_outputs[0]));
}

} // namespace nvfuser
