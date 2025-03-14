// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using IndexSelectTest = NVFuserTest;

TEST_F(IndexSelectTest, Simple1) {
  for (int i = 0; i < 5; ++i) {
    // fix seed
    std::srand(i);

    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();

    FusionGuard fg(&fusion);
    // dimensionality of the problem
    int nDims = 2;
    int nElem = std::rand() % 1023 + 1;
    int nElem_select = nElem + 115;
    int nFeat = std::rand() % 128 + 1;

    // Set up your input tensor views
    TensorView* tv0 = makeContigTensor(nDims);
    TensorView* tv_idx = makeContigTensor(1, DataType::Int);

    fusion.addInput(tv0);
    fusion.addInput(tv_idx);
    TensorView* tv_sel = indexSelect(tv0, 0, tv_idx);
    fusion.addOutput(tv_sel);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

    at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
    at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});
    testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
  }
}

TEST_F(IndexSelectTest, Simple2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor t1 = at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

// Test 1D schedule
// If (n_elems * 2 > device_multiprocessor_count * kThreadX), just use 1D
// scheduler or use 2D scheduler
TEST_F(IndexSelectTest, 1DSchedule) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 13;
  int nElem_select = nElem + 1;
  int nFeat = 7;

  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor t1 = at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

TEST_F(IndexSelectTest, 3DTensor) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 89;
  int nElem_select = nElem + 35;
  int nFeat0 = 255;
  int nFeat1 = 63;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(27.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat0, nFeat1}, options); // lookup
  at::Tensor t1 =
      at::randn({nElem_select, nFeat0, nFeat1}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

TEST_F(IndexSelectTest, CanSchedule) {
  // fix seed
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 31;
  int nElem_select = nElem + 15;
  int nFeat = 64;

  // Negative Case I
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_fail;
  FusionGuard fg(&fusion_fail);
  TensorView* tv_pre = makeContigTensor(nDims);
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_fail.addInput(tv_pre);
  fusion_fail.addInput(tv1);
  fusion_fail.addInput(tv0);
  fusion_fail.addInput(tv_idx);
  TensorView* tv_t = mul(tv0, tv_pre);
  TensorView* tv_sel = indexSelect(tv_t, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  // Register your outputs
  fusion_fail.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor input_pre = at::rand_like(t0);

  at::Tensor t1 = at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor output = at::zeros({nElem_select, nFeat}, options);
  KernelArgumentHolder inputs({input_pre, t1, t0, idx});

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion_fail, inputs);
  auto sch_fail = Schedule::canSchedule(
      SchedulerType::PointWise, &fusion_fail, runtime_info);

  // Negative Case II
  // lookup tv of index select cannot become conumser of other OP
  // Set up your input tensor views
  Fusion fusion_sum_fail;
  FusionGuard fg_sum(&fusion_sum_fail);
  TensorView* tv_sum_pre = makeContigTensor(nDims);
  TensorView* tv_sum_0 = makeContigTensor(nDims);
  TensorView* tv_sum_1 = makeContigTensor(nDims);
  TensorView* tv_sum_idx = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_sum_fail.addInput(tv_sum_pre);
  fusion_sum_fail.addInput(tv_sum_1);
  fusion_sum_fail.addInput(tv_sum_0);
  fusion_sum_fail.addInput(tv_sum_idx);
  TensorView* tv_sum_t = mul(tv_sum_0, tv_sum_pre);
  TensorView* tv_sum_sel = indexSelect(tv_sum_t, 0, tv_sum_idx);
  TensorView* tv_sum_2 = mul(tv_sum_1, tv_sum_sel);
  TensorView* tv_sum_add = add(IrBuilder::create<Val>(17.0), tv_sum_2);
  auto tv_sum_3 = sum(tv_sum_add, {1});
  // Register your outputs
  fusion_sum_fail.addOutput(tv_sum_3);
  KernelArgumentHolder sum_inputs({input_pre, t1, t0, idx});
  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_sum_info(&fusion_sum_fail, sum_inputs);
  auto sch_sum_fail = Schedule::canSchedule(
      SchedulerType::Reduction, &fusion_sum_fail, runtime_sum_info);

  // Positive  Case I
  Fusion fusion_pass;
  FusionGuard fg_p(&fusion_pass);
  TensorView* tv0_p = makeContigTensor(nDims);
  TensorView* tv1_p = makeContigTensor(nDims);
  TensorView* tv_idx_p = makeContigTensor(1, DataType::Int);
  // Register your inputs
  fusion_pass.addInput(tv1_p);
  fusion_pass.addInput(tv0_p);
  fusion_pass.addInput(tv_idx_p);
  TensorView* tv_sel_p = indexSelect(tv0_p, 0, tv_idx_p);
  TensorView* tv2_p = mul(tv1_p, tv_sel_p);
  TensorView* tv3_p = add(IrBuilder::create<Val>(17.0), tv2_p);
  // Register your outputs
  fusion_pass.addOutput(tv3_p);
  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info_pass(&fusion_pass, {t1, t0, idx});
  auto sch_pass = Schedule::canSchedule(
      SchedulerType::PointWise, &fusion_pass, runtime_info_pass);

  NVF_CHECK(sch_pass == true && sch_fail == false && sch_sum_fail == false);
}

TEST_F(IndexSelectTest, Sum) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 0, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv_add = add(IrBuilder::create<Val>(17.0), tv2);
  auto tv3 = sum(tv_add, {1});
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor t1 = at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);
  at::Tensor cg_output = at::zeros({nElem_select}, options);

  auto heuristic_params = SchedulerEntry::scheduleWith(
      &fusion, SchedulerType::Reduction, {t1, t0, idx});
  KernelExecutor ke;
  ke.compile(&fusion, {t1, t0, idx}, heuristic_params->lparams);
  ke.run({t1, t0, idx}, {cg_output}, heuristic_params->lparams);

  auto tv0_ref = at::index_select(t0, 0, idx);
  at::Tensor tv2_ref = tv0_ref * t1;
  at::Tensor output_add = tv2_ref + 17.0;
  at::Tensor output_ref = output_add.sum({1});

  NVF_CHECK(output_ref.allclose(cg_output));
}

TEST_F(IndexSelectTest, IdxTvFuseable) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 2;
  int nElem = 23;
  int nElem_select = nElem + 15;
  int nFeat = 32;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  TensorView* tv_idx_pre = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_idx_pre);
  TensorView* tv_idx_ret = add(tv_idx, tv_idx_pre);
  TensorView* tv_sel = indexSelect(tv0, 0, tv_idx_ret);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nElem, nFeat}, options); // lookup
  at::Tensor t1 = at::randn({nElem_select, nFeat}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);
  auto idx_pre = at::zeros({nElem_select}, options_i);

  KernelArgumentHolder args = {t1, t0, idx, idx_pre};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);
  testValidate(&fusion, cg_outputs, args, __LINE__, __FILE__);
}

TEST_F(IndexSelectTest, Dim1InRank2) {
  for (int i = 0; i < 5; ++i) {
    // fix seed
    std::srand(i);

    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // dimensionality of the problem
    int nDims = 2;
    int nElem = std::rand() % 15 + 1;
    int nElem_select = std::rand() % 10 + 1;
    int nFeat = std::rand() % 7 + 1;

    // Set up your input tensor views
    TensorView* tv0 = makeContigTensor(nDims);
    TensorView* tv1 = makeContigTensor(nDims);
    TensorView* tv_idx = makeContigTensor(1, DataType::Int);
    fusion.addInput(tv1);
    fusion.addInput(tv0);
    fusion.addInput(tv_idx);
    TensorView* tv_sel = indexSelect(tv0, 1, tv_idx);
    TensorView* tv2 = mul(tv1, tv_sel);
    TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({nFeat, nElem}, options); // lookup
    at::Tensor t1 =
        at::randn({nFeat, nElem_select}, options); // output&elemwise
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
    testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
  }
}

TEST_F(IndexSelectTest, Dim2InRank3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 2, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nFeat0, nFeat1, nElem}, options); // lookup
  at::Tensor t1 =
      at::randn({nFeat0, nFeat1, nElem_select}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

TEST_F(IndexSelectTest, Dim1InRank3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;
  int nElem = 4;
  int nElem_select = nElem - 2;
  int nFeat0 = 5;
  int nFeat1 = 7;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nFeat0, nElem, nFeat1}, options); // lookup
  at::Tensor t1 =
      at::randn({nFeat0, nElem_select, nFeat1}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

TEST_F(IndexSelectTest, Dim2InRank4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 4;
  int nElem = 4;
  int nElem_select = nElem + 15;
  int nFeat0 = 5;
  int nFeat1 = 7;
  int nFeat2 = 25;

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(nDims);
  TensorView* tv1 = makeContigTensor(nDims);
  TensorView* tv_idx = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv0);
  fusion.addInput(tv_idx);
  TensorView* tv_sel = indexSelect(tv0, 1, tv_idx);
  TensorView* tv2 = mul(tv1, tv_sel);
  TensorView* tv3 = add(IrBuilder::create<Val>(17.0), tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nFeat0, nElem, nFeat1, nFeat2}, options); // lookup
  at::Tensor t1 = at::randn(
      {nFeat0, nElem_select, nFeat1, nFeat2}, options); // output&elemwise
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor idx = at::randint(0, nElem, (nElem_select), options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0, idx});
  testValidate(&fusion, cg_outputs, {t1, t0, idx}, __LINE__, __FILE__);
}

// Repro of issue #961
TEST_F(IndexSelectTest, BroadcastIndex) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1}, DataType::Int);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = indexSelect(tv1, 0, tv0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(100, {1}, options_long);
  auto t1 = at::randn({100, 100}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = at::index_select(t1, 0, t0);

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().equal(ref));
}

// See #1049
TEST_F(IndexSelectTest, MultipleIndexSelectIssue) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1, DataType::Int);
  fusion.addInput(tv2);

  auto tv3 = indexSelect(tv0, 0, tv2);
  auto tv4 = indexSelect(tv1, 0, tv2);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  std::vector<int64_t> shape1({17, 19});
  std::vector<int64_t> shape2({3});
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randint(0, shape1[0], shape2, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  ASSERT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented())
      << "Should not segmented";

  auto ref = at::index_select(t0, 0, t2) + at::index_select(t1, 0, t2);

  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

} // namespace nvfuser
