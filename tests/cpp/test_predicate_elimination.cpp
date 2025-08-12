// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>

namespace nvfuser {

class PredicateEliminationTest : public NVFuserTest {};

TEST_F(PredicateEliminationTest, 1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(3.0));

  fusion.addOutput(tv3);

  tv3->split(0, 32);
  tv0->computeAt(tv3, 1);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  {
    GpuLower gpulw(&fusion);
    gpulw.run();
    NVF_CHECK(!PredicatedChecker::isPredicated(tv2, gpulw));
  }

  tv2->axis(1)->parallelize(ParallelType::Serial);
  tv2->split(1, 5);

  {
    GpuLower gpulw(&fusion);
    gpulw.run();
    NVF_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));
  }
}

// Repro of issue #1571
TEST_F(PredicateEliminationTest, 2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  tv1->split(1, 4);
  tv1->split(0, 4);
  tv2->split(1, 4);
  tv2->split(0, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = (t0 + 1).sum({1}) + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(PredicateEliminationTest, 3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv0->cacheAfter();

  tv1->split(0, 10);
  tv1->split(0, 33);
  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  auto tv4 = tv1->rFactor({-1});
  auto tv5 = tv1->rFactor({-1});

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  GpuLower gpulw(&fusion);
  gpulw.run();

  // The fusion has three reductions: one within each thread, one
  // within each block, and another with the whole grid. All of them
  // should not need to be predicated as they use the same init value
  // and same reduction op.
  NVF_CHECK(!PredicatedChecker::isPredicated(tv4, gpulw));
  NVF_CHECK(!PredicatedChecker::isPredicated(tv5, gpulw));
  NVF_CHECK(!PredicatedChecker::isPredicated(tv1, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  for (auto size : {1, 2, 999, 1001, 1234, 10000}) {
    auto t0 = at::randn({size}, options);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});

    auto ref = sum(t0) + 1;
    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(PredicateEliminationTest, 4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});

  auto tv2 = sum(tv1, {0});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  auto tv4 = max(tv1, {0});
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  tv1->split(1, 7);
  tv1->split(0, 11);
  tv1->reorder({{1, 2}, {2, 1}});
  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  GpuLower gpulw(&fusion);
  gpulw.run();

  // tv2 uses the same op and init with tv1, so tv2 should be fine
  // without a predicate. However, tv4, while it uses the tv1 as its
  // input, the reduction op and init value is different from those of
  // tv1, so tv4 needs to be predicated.
  NVF_CHECK(!PredicatedChecker::isPredicated(tv2, gpulw));
  NVF_CHECK(PredicatedChecker::isPredicated(tv4, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> sizes = {1, 2, 33, 34, 64, 99};
  for (auto s0 : sizes) {
    for (auto s1 : sizes) {
      auto t0 = at::randn({s0, s1}, options);

      KernelExecutor ke;
      ke.compile(&fusion, {t0});
      auto cg_outputs = ke.run({t0});

      auto t1 = t0.sum({1});
      auto t3 = t1.sum({0}) + 1;
      auto t5 = std::get<0>(t1.max(0)) + 1;

      testValidate(&fusion, cg_outputs, {t0}, {t3, t5}, __LINE__, __FILE__);
    }
  }
}

TEST_F(PredicateEliminationTest, 5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tvs2 = Welford(tv1, {0});
  auto tv3 = set(tvs2.avg);
  fusion.addOutput(tv3);

  tvs2.avg->split(0, 4);
  TransformPropagatorWithCheck propagator(tvs2.avg);
  MaxLogicalDomainInfoSpanningTree(tvs2.avg).traverse(&propagator);
  auto avg_rf = ir_utils::rFactorHelper(tvs2.avg, {1});

  avg_rf->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(avg_rf);

  GpuLower gpulw(&fusion);
  gpulw.run();

  // The first per-thread welford needs to be predicated as the N
  // input is different from its init value. The second welford op
  // does not need a predicate.
  NVF_CHECK(PredicatedChecker::isPredicated(avg_rf, gpulw));
  NVF_CHECK(!PredicatedChecker::isPredicated(tvs2.avg, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> sizes = {1, 2, 33, 34, 64, 99};
  for (auto s0 : sizes) {
    auto t0 = at::randn({s0}, options);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});

    auto ref = t0.mean({0});

    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(PredicateEliminationTest, 6) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv4);

  tv4->split(1, 5);
  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->reorder({{0, 1}, {1, 0}});
  tv3->computeAt(tv4, 1);

  GpuLower gpulw(&fusion);
  gpulw.run();

  // The expression for tv2 is a local-to-local expression. It
  // satisfies all the requirements of predicate elimination, except
  // for the on on split logical domains. As the second logical axis of tv2
  // is split, its index exceeds its extent (i.e., 3 in this case)
  // without its predicate.
  NVF_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));

  // Unlike tv2, tv3 is computed at tv4, so the second logical axis does
  // have a zero domain. Its index should look like "i * 5 + j", where
  // i comes from the first logical domain and j comes from the split
  // inner domain.
  NVF_CHECK(!PredicatedChecker::isPredicated(tv3, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PredicateEliminationTest, 7) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv3->split(-1, 5);
  tv3->split(-1, 4);
  tv3->split(-1, 3);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // The last split of tv2 is a non-divisible split, and omitting it
  // is invalid.
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(PredicatedChecker::isPredicated(tv2, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({123}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Repro of failing to eliminate predicates due to
// unarySetOpInserter. See PR #2136.
TEST_F(PredicateEliminationTest, 8) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t channel_size = 16;
  const int64_t batch_size = 8;
  const int64_t hw_size = 56;

  std::vector<int64_t> bcast_size = {batch_size, channel_size, 1, 1};
  std::vector<int64_t> full_size = {batch_size, channel_size, hw_size, hw_size};

  auto tv0 = makeContigConcreteTensor(bcast_size);
  auto tv1 = makeContigConcreteTensor(full_size);
  auto tv2 = makeContigConcreteTensor(bcast_size);
  auto tv3 = makeContigConcreteTensor(full_size);
  auto tv4 = makeContigConcreteTensor({channel_size});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addInput(tv4);

  std::vector<int64_t> reduction_axes = {0, 2, 3};

  Val* num_features = IrBuilder::create<Val>(1.0);
  for (const auto dim : reduction_axes) {
    num_features = mul(num_features, tv1->getLoopDomain()[dim]->extent());
  }

  auto tv5 = mul(tv1, tv0);
  auto tv6 = expand(
      tv2,
      {IrBuilder::create<Val>(batch_size),
       IrBuilder::create<Val>(channel_size),
       IrBuilder::create<Val>(hw_size),
       IrBuilder::create<Val>(hw_size)});
  auto tv7 = div(tv6, IrBuilder::create<Val>(3136.0));
  auto tv8 = add(tv5, tv7);
  auto tv9 = sum(tv8, reduction_axes);
  auto tv10 = broadcast(tv4, {true, false, true, true});
  auto tv11 = sub(tv3, tv10);
  auto tv12 = mul(tv8, tv11);
  auto tv13 = sum(tv12, reduction_axes);
  auto tv14 = mul(tv13, reciprocal(num_features));
  auto tv15 = sub(tv3, tv10);

  fusion.addOutput(tv9);
  fusion.addOutput(tv13);
  fusion.addOutput(tv8);
  fusion.addOutput(tv14);
  fusion.addOutput(tv15);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn(bcast_size, options); // tv8 - 0
  at::Tensor aten_t1 = at::randn(full_size, options); // tv7 - 1
  at::Tensor aten_t2 = at::randn(bcast_size, options); // tv6 - 2
  at::Tensor aten_t3 = at::randn(full_size, options); // tv0 - 3
  at::Tensor aten_t4 = at::randn({channel_size}, options); // tv4 - 4

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  KernelArgumentHolder args = {aten_t0, aten_t1, aten_t2, aten_t3, aten_t4};
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
  NVF_CHECK(
      !PredicatedChecker::isPredicated(tv6, ke->compiledKernel()->kernel()),
      "T6 should not be predicated");
}

TEST_F(PredicateEliminationTest, 9) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int M = 1024, split = 512;

  // Algorithm
  auto tv0 = makeContigConcreteTensor({M}, DataType::Float);
  auto tv1 = set(tv0);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  // Schedule
  auto tv0c = tv0->cacheAfter();
  tv0c->setMemoryType(MemoryType::Shared);
  tv0c->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->split(-1, split);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Validation
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({M}, options);

  GpuLower gpulw(fusion.get());
  gpulw.run();
  // tv0c expectation: no predicate present as domain with TIDX parallel type
  //  has the same extend as max extend stored for TIDx type in parallel domain
  //  map
  EXPECT_FALSE(PredicatedChecker::isPredicated(tv0c, gpulw));
  // tv1 expectation: with a predicate, max extend for TIDx parallel type in
  //  parallel domain map is not the same as the extend of domain parallized
  //  with TIDx in this tensor
  EXPECT_TRUE(PredicatedChecker::isPredicated(tv1, gpulw));

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});
  auto cg_outputs = ke.run({t0});
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PredicateEliminationTest, ExtentEqualToMaxParallelTypeExtent) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  std::vector<TensorView*> tvs = {tv1, tv2};
  for (auto tv : tvs) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }
  fusion.manage("interested_tvs", tvs);

  auto validate_smem_predicate_elimination =
      [](const std::vector<Expr*>& exprs) -> std::vector<Expr*> {
    kir::Kernel* kernel = GpuLower::current()->kernel();
    auto kernel_tvs =
        kernel->getManaged<std::vector<TensorView*>>("interested_tvs");
    for (auto tv : kernel_tvs) {
      EXPECT_TRUE(
          lower_utils::isExtentEqualToMaxParallelTypeExtent(tv->axis(0)));
    }
    return exprs;
  };

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10 * 32}, options);
  KernelExecutor ke;
  ke.registerLoweringHook([&](GpuLower* lower) {
    lower->passes().insert(
        lower->passes().begin(),
        {"validate_smem_predicate_elimination",
         validate_smem_predicate_elimination});
  });
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
