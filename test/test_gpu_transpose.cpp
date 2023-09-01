// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

namespace {

TensorView* transposeMaybeInplace(
    TensorView* inp,
    int64_t dim1,
    int64_t dim2,
    bool inplace) {
  if (!inplace) {
    return transpose(inp, dim1, dim2);
  } else {
    inp->reorder({{dim1, dim2}, {dim2, dim1}});
    inp->commitLeafToRFactor();
    return inp;
  }
}

} // namespace

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSimple_CUDA) {
  for (auto inplace : {true, false}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(3);
    fusion.addInput(tv0);
    auto tv1 = sin(tv0);
    auto tv2 = transposeMaybeInplace(tv1, 1, 2, inplace);
    auto tv3 = cos(tv2);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::randn({256, 1024, 1024}, options);

    auto lparams = scheduleTranspose(&fusion, {input});

    FusionExecutor fe;
    fe.compileFusion(&fusion, {input}, lparams);
    auto outputs = fe.runFusion({input}, lparams);

    auto tv_ref = input.sin().transpose(1, 2).cos();

    testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
  }
}

// x->tanspose->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSinTransposeCos_CUDA) {
  for (auto inplace : {true, false}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(3);
    fusion.addInput(tv0);
    // fusion input can not be transposed inplace
    auto tv1 = transpose(tv0, 0, 2);
    auto tv2 = sin(tv1);
    auto tv3 = transposeMaybeInplace(tv2, 1, 2, inplace);
    auto tv4 = cos(tv3);
    fusion.addOutput(tv4);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::randn({256, 1024, 1024}, options);

    auto lparams = scheduleTranspose(&fusion, {input});

    FusionExecutor fe;
    fe.compileFusion(&fusion, {input}, lparams);
    auto outputs = fe.runFusion({input}, lparams);

    auto tv_ref = input.transpose(0, 2).sin().transpose(1, 2).cos();

    testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
  }
}

/*
 * t0->transpose--.
 *                 \
 * t1->transpose---add-->sin->t5
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleInput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 0, 2);
  auto tv3 = transpose(tv1, 0, 2);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sin(tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 1024, 1024}, options);
  at::Tensor input1 = at::randn({256, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = (input0.transpose(0, 2) + input1.transpose(0, 2)).sin();

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// t0->sin->transpose->t5
//  `->cos->transpose->t6
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleOutput_CUDA) {
  for (auto inplace : {true, false}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(3);
    fusion.addInput(tv0);
    auto tv2 = sin(tv0);
    auto tv3 = cos(tv0);
    auto tv5 = transposeMaybeInplace(tv2, 0, 2, inplace);
    auto tv6 = transposeMaybeInplace(tv3, 0, 2, inplace);
    fusion.addOutput(tv5);
    fusion.addOutput(tv6);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::randn({32, 1024, 1024}, options);

    auto lparams = scheduleTranspose(&fusion, {input});

    FusionExecutor fe;
    fe.compileFusion(&fusion, {input}, lparams);
    auto outputs = fe.runFusion({input}, lparams);

    auto tv_ref1 = input.sin().transpose(0, 2);
    auto tv_ref2 = input.cos().transpose(0, 2);

    testValidate(
        &fusion, outputs, {input}, {tv_ref1, tv_ref2}, __LINE__, __FILE__);
  }
}

/*
 * t0->transpose->sin->t3
 *   \_.-->cos->t5
 *   /
 * t1
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMultipleInputOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 0, 2);
  auto tv3 = sin(tv2);
  fusion.addOutput(tv3);
  auto tv4 = add(tv0, tv1);
  auto tv5 = cos(tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({32, 1024, 1024}, options);
  at::Tensor input1 = at::randn({32, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref1 = input0.transpose(0, 2).sin();
  auto tv_ref2 = (input0 + input1).cos();

  testValidate(
      &fusion,
      outputs,
      {input0, input1},
      {tv_ref1, tv_ref2},
      __LINE__,
      __FILE__);
}

/*
 *             .------>sin------>z
 * x->transpose->transpose->add->y
 *  \_______________________/
 */
TEST_F(NVFuserTest, FusionScheduleTransposeMatchingSkipConnection_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 2);
  auto tv2 = transpose(tv1, 0, 2);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);
  auto tv4 = sin(tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({32, 1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref1 = input.transpose(0, 2).transpose(0, 2) + input;
  auto tv_ref2 = input.transpose(0, 2).sin();

  testValidate(
      &fusion, outputs, {input}, {tv_ref1, tv_ref2}, __LINE__, __FILE__);
}

// x->transpose--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleTransposeBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = transpose(tv0, 1, 2);
  auto tv3 = broadcast(tv1, {false, false, true});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1024, 256, 1024}, options);
  at::Tensor input1 = at::randn({1024, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = input0.transpose(1, 2) + input1.unsqueeze(2);

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// x->broadcast--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleTransposeNoReference_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true, false});
  auto tv3 = broadcast(tv1, {false, false, true});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 1024}, options);

  EXPECT_THAT(
      [&]() {
        scheduleTranspose(&fusion, {input0, input1});
      },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("reference tensor")));
}

// x->broadcast--add->z
// y->broadcast-/
TEST_F(NVFuserTest, FusionScheduleBroadcastOnly_CUDA) {
  for (bool contig0 : {true, false}) {
    for (bool contig1 : {true, false}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto tv0 = contig0 ? makeContigConcreteTensor({-1, 1, -1})
                         : makeConcreteTensor({-1, 1, -1});
      auto tv1 = contig1 ? makeContigConcreteTensor({-1, -1, 1})
                         : makeConcreteTensor({-1, -1, 1});
      fusion.addInput(tv0);
      fusion.addInput(tv1);
      auto tv2 = add(tv0, tv1);
      fusion.addOutput(tv2);

      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor input0 = at::randn({1024, 1, 256}, options);
      at::Tensor input1 = at::randn({1024, 1024, 1}, options);

      auto lparams = scheduleTranspose(&fusion, {input0, input1});

      FusionExecutor fe;
      fe.compileFusion(&fusion, {input0, input1}, lparams);
      auto outputs = fe.runFusion({input0, input1}, lparams);

      auto tv_ref = input0 + input1;

      testValidate(
          &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
    }
  }
}

// mermaid graph:
// ```mermaid
// %%{
//   init: {
//     'theme': 'base',
//     'themeVariables': { 'fontSize': '30px', 'fontFamily': 'times'}}
// }%%
// graph TD
//   T0("T0(M, N, K)")
//   T1("T1(N, M, K)")
//   T2("T2(M, K, N)")
//   T0 --> A("transpose(1, 2)") --> T3("T3(M, K, N)")
//   T1 ---> sigmoid --> T5("T5(N, M, K)")
//   T5 --> B("transpose(0, 2)") --> T7("T7(K, M, N)")
//   T2 ----> C("add")
//   T3 --> C --> T6("T6(M, K, N)")
//   T6 --> D("transpose(0, 1)") --> T11("T11(K, M, N)")
//   T11 --> E("add") -->T12("T12(K, M, N)")
//   T7 --> E
//   T1 ---> F("transpose(0, 1)") --> T4("T4(M, N, K)")
//   T0 --> G("add") --> T8("T8(M, N, K)") --> relu ---> T9("T9(M, N, K)")
//   T4 --> G
//   T6 ---> sin ---> T10("T10(M, K, N)")
//   style T0 fill:lightgreen
//   style T1 fill:lightgreen
//   style T2 fill:lightgreen
//   style T12 fill:lightblue
//   style T9 fill:lightblue
//   style T10 fill:lightblue
// ```
TEST_F(NVFuserTest, FusionScheduleTransposeComplexDAG1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  auto tv2 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = transpose(tv0, 1, 2);
  auto tv4 = transpose(tv1, 0, 1);
  auto tv5 = sigmoid(tv1);
  auto tv6 = add(tv2, tv3);
  auto tv7 = transpose(tv5, 0, 2);
  auto tv8 = add(tv4, tv0);
  auto tv9 = relu(tv8);
  fusion.addOutput(tv9);
  auto tv10 = sin(tv6);
  fusion.addOutput(tv10);
  auto tv11 = transpose(tv6, 0, 1);
  auto tv12 = add(tv7, tv11);
  fusion.addOutput(tv12);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({512, 1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 512, 256}, options);
  at::Tensor input2 = at::randn({512, 256, 1024}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1, input2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2}, lparams);
  auto outputs = fe.runFusion({input0, input1, input2}, lparams);

  auto t3 = input0.transpose(1, 2);
  auto t4 = input1.transpose(0, 1);
  auto t5 = input1.sigmoid();
  auto t6 = input2 + t3;
  auto t7 = t5.transpose(0, 2);
  auto t8 = t4 + input0;
  auto t9 = t8.relu();
  auto t10 = t6.sin();
  auto t11 = t6.transpose(0, 1);
  auto t12 = t7 + t11;

  testValidate(
      &fusion,
      outputs,
      {input0, input1, input2},
      {t9, t10, t12},
      __LINE__,
      __FILE__);
}

// mermaid graph:
// ```mermaid
// %%{
//   init: {
//     'theme': 'base',
//     'themeVariables': { 'fontSize': '30px', 'fontFamily': 'times'}}
// }%%
// graph TD
//   T0("T0(M, N, K)")
//   T1("T1(N, M, K)")
//   T2("T2(M, K, N)")
//   T0 --> A("transpose(1, 2)") --> T3("T3(M, K, N)")
//   T1 ---> sigmoid --> T5("T5(N, M, K)")
//   T5 --> B("transpose(0, 2)") --> T7("T7(K, M, N)")
//   T2 ----> C("add")
//   T3 --> C --> T6("T6(M, K, N)")
//   T6 --> D("transpose(0, 1)") --> T11("T11(K, M, N)")
//   T11 --> E("add") -->T12("T12(K, M, N)")
//   T7 --> E
//   T1 ---> F("transpose(0, 1)") --> T4("T4(M, N, K)")
//   T0 --> G("add") --> T8("T8(M, N, K)") --> relu ---> T9("T9(M, N, K)")
//   T4 --> G
//   T6 ---> sin ---> T10("T10(M, K, N)")
//   style T0 fill:lightgreen
//   style T1 fill:lightgreen
//   style T2 fill:lightgreen
//   style T12 fill:lightblue
//   style T9 fill:lightblue
//   style T10 fill:lightblue
// ```
TEST_F(NVFuserTest, FusionManualScheduleTransposeComplexDAG1_CUDA) {
  // achieved: 833.526 GB/s on RTX 3090 (theoretical bandwidth: 936 GB/s)
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(3);
  auto tv2 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = transpose(tv0, 1, 2);
  auto tv4 = transpose(tv1, 0, 1);
  auto tv5 = sigmoid(tv1);
  auto tv6 = add(tv2, tv3);
  auto tv7 = transpose(tv5, 0, 2);
  auto tv8 = add(tv4, tv0);
  auto tv9 = relu(tv8);
  fusion.addOutput(tv9);
  auto tv10 = sin(tv6);
  fusion.addOutput(tv10);
  auto tv11 = transpose(tv6, 0, 1);
  auto tv12 = add(tv7, tv11);
  fusion.addOutput(tv12);

  // group 1: tv0, tv1, *tv9, innermost dim K
  // group 2: tv2, *tv10, tv12, innermost dim N

  // cache inputs and outputs
  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();
  auto tv2_cache = tv2->cacheAfter();
  tv9->cacheBefore();
  auto tv10_cache = tv10->cacheBefore();
  auto tv12_cache = tv12->cacheBefore();

  // Step 1: Make 32x32 tiles, schedule outer dimensions
  {
    // Pick an arbitrary tensor as a reference tensor for this step. There is no
    // requirement on which group this reference tensor should belong to. Here
    // we pick tv9, which belongs to group 1.

    // Make 32x32 tile:
    // [M, N, K]
    tv9->split(1, 32);
    tv9->reorder({{2, -1}});
    tv9->split(2, 32);
    tv9->reorder({{3, -1}});
    // [M, N/32, K/32, 32(N), 32(K)]

    // merge outer dims, parallelize on BIDx, and unswitch
    tv9->merge(0);
    tv9->merge(0);
    tv9->split(0, 1);
    // [M * N/32 * K/32, 1, 32(N), 32(K)]
    tv9->axis(0)->parallelize(ParallelType::BIDx);
    tv9->axis(1)->parallelize(ParallelType::Unswitch);
    // [BIDx, Unswitch, 32(N), 32(K)]

    // propagate to the entire DAG
    MaxRootDomainInfoSpanningTree entire_dag(tv9);
    TransformPropagator tp(tv9);
    entire_dag.traverse(&tp);
    scheduler_utils::parallelizeAllLike(tv9);
  }

  constexpr int threads_per_block = 128;

  // Step 2, schedule group 2
  {
    // group 2: tv2, *tv10, tv12, innermost dim N

    tv2_cache->setMemoryType(MemoryType::Shared);
    tv10_cache->setMemoryType(MemoryType::Shared);
    tv12_cache->setMemoryType(MemoryType::Shared);

    // pick tv10 as reference tensor for group 2
    // [BIDx, Unswitch, 32(N), 32(K)]
    tv10->reorder({{-1, -2}});
    // [BIDx, Unswitch, 32(K), 32(N)]
    tv10->merge(2);
    tv10->split(2, 4);
    tv10->split(2, threads_per_block);
    tv10->axis(-1)->parallelize(ParallelType::Vectorize);
    tv10->axis(-2)->parallelize(ParallelType::TIDx);
    tv10->axis(-3)->parallelize(ParallelType::Unroll);
    // [BIDx, Unswitch, Unroll, TIDx, Vectorize]

    // Propagate to group 2 and its cache. Note that group 2 and its cache are
    // not connected, so we need to borrow other tensors of the DAG to be able
    // to propagate. The transformations on borrowed tensors will be overwritten
    // in the next step. We can not borrow the reference tensor of group 1.
    auto all_tvs_except_ref1 = ir_utils::allTvsExcept(&fusion, {tv9});
    auto all_tvs_except_ref1_set = std::unordered_set<TensorView*>(
        all_tvs_except_ref1.begin(), all_tvs_except_ref1.end());
    SetSelector selector(all_tvs_except_ref1_set);
    MaxRootDomainInfoSpanningTree tree(tv10, &selector);
    TransformPropagator tp(tv10);
    tree.traverse(&tp);
    scheduler_utils::parallelizeAllLike(
        tv10, {tv2_cache, tv10, tv12}, {ParallelType::TIDx});
    scheduler_utils::parallelizeAllLike(
        tv10,
        {tv2_cache, tv10, tv12},
        {ParallelType::Vectorize, ParallelType::Unroll});
  }

  // Step 3, schedule group 1
  {
    // group 1: tv0, tv1, *tv9, innermost dim K
    // [BIDx, Unswitch, 32(N), 32(K)]
    tv9->merge(2);
    tv9->split(2, 4);
    tv9->split(2, threads_per_block);
    tv9->axis(-1)->parallelize(ParallelType::Vectorize);
    tv9->axis(-2)->parallelize(ParallelType::TIDx);
    tv9->axis(-3)->parallelize(ParallelType::Unroll);
    // [BIDx, Unswitch, Unroll, TIDx, Vectorize]

    // Propagate to the entire DAG except for group 2 and its cached inputs
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(&fusion, {tv2, tv2_cache, tv10, tv12});
    auto all_tvs_except2_set = std::unordered_set<TensorView*>(
        all_tvs_except2.begin(), all_tvs_except2.end());
    SetSelector selector(all_tvs_except2_set);
    MaxRootDomainInfoSpanningTree tree(tv9, &selector);
    TransformPropagator tp(tv9);
    tree.traverse(&tp);
    scheduler_utils::parallelizeAllLike(
        tv9, all_tvs_except2, {ParallelType::TIDx});
    scheduler_utils::parallelizeAllLike(
        tv9,
        {tv0_cache, tv1_cache, tv9},
        {ParallelType::Vectorize, ParallelType::Unroll});
  }

  // inline
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({512, 1024, 256}, options);
  at::Tensor input1 = at::randn({1024, 512, 256}, options);
  at::Tensor input2 = at::randn({512, 256, 1024}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2});
  auto outputs = fe.runFusion({input0, input1, input2});

  auto t3 = input0.transpose(1, 2);
  auto t4 = input1.transpose(0, 1);
  auto t5 = input1.sigmoid();
  auto t6 = input2 + t3;
  auto t7 = t5.transpose(0, 2);
  auto t8 = t4 + input0;
  auto t9 = t8.relu();
  auto t10 = t6.sin();
  auto t11 = t6.transpose(0, 1);
  auto t12 = t7 + t11;

  testValidate(
      &fusion,
      outputs,
      {input0, input1, input2},
      {t9, t10, t12},
      __LINE__,
      __FILE__);
}

// x->view->y
TEST_F(NVFuserTest, FusionViewNoTranspose_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = flatten(tv0, 1, 2);
  fusion.addOutput(tv1);

  TORCH_CHECK(!hasAtLeastTwoValidGroups(&fusion));
}

TEST_F(NVFuserTest, FusionTransposeSelfMapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  EXPECT_THAT(
      [&]() { IterDomainGraph(fusion_ptr.get()); },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Unsupported domain mapping detected")));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({5, 5}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.transpose(0, 1) + t0;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

#if 0
// silent wrong result
TEST_F(NVFuserTest, FusionTransposeViewSelfMapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = transpose(tv0, 0, 1);
  auto tv2 = view(tv0, {2, 3}, {3, 2});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.transpose(0, 1) + t0.view({3, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}
#endif

// t0------------.
// t2->broadcast->sub->mul->relu->t6
// t1------------------'
TEST_F(NVFuserTest, FusionScheduleTransposeMissingDim_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigConcreteTensor({1, -1, 1});
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  auto tv3 = broadcast(tv2, {true, false, true});
  auto tv4 = sub(tv0, tv3);
  auto tv5 = mul(tv4, tv1);
  auto tv6 = relu(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 512, 256}, options);
  at::Tensor input1 = at::randn({1, 512, 1}, options);
  at::Tensor input2 = at::randn({512}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1, input2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1, input2}, lparams);
  auto outputs = fe.runFusion({input0, input1, input2}, lparams);

  auto t3 = input2.unsqueeze(0).unsqueeze(-1);
  auto t4 = input0 - t3;
  auto t5 = t4 * input1;
  auto t6 = at::relu(t5);

  testValidate(
      &fusion, outputs, {input0, input1, input2}, {t6}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmall_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1024, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({64 * 1024 * 1024, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(1, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 0, 2);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 64 * 1024 * 1024, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(0, 2).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTransposeSmallInnerSize3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(8);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = transpose(tv1, 4, 7);
  auto tv3 = cos(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1024 * 1024, 2, 2, 2, 2, 2, 2, 2}, options);

  auto lparams = scheduleTranspose(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.sin().transpose(4, 7).cos();

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// x->sin->transpose->cos->y
TEST_F(NVFuserTest, FusionScheduleTranspose2DSmallInnerSize_CUDA) {
  std::array<std::vector<int64_t>, 2> shapes{
      std::vector<int64_t>{1024 * 1024 * 128, 2},
      std::vector<int64_t>{2, 1024 * 1024 * 128}};
  for (const auto& shape : shapes) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(2);
    fusion.addInput(tv0);
    auto tv1 = sin(tv0);
    auto tv2 = transpose(tv1, 0, 1);
    auto tv3 = cos(tv2);
    fusion.addOutput(tv3);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::randn(shape, options);

    auto lparams = scheduleTranspose(&fusion, {input});

    FusionExecutor fe;
    fe.compileFusion(&fusion, {input}, lparams);
    auto outputs = fe.runFusion({input}, lparams);

    auto tv_ref = input.sin().transpose(0, 1).cos();

    testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionTransposeBankConflict1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{32});
}

TEST_F(NVFuserTest, FusionTransposeBankConflict2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).second, std::vector<int>(2, 32));
}

TEST_F(NVFuserTest, FusionTransposeBankConflict3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32, 32}, DataType::Bool);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{8});
}

TEST_F(NVFuserTest, FusionTransposeBankConflict4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 8);
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  // T1 [TIDx(32), 8, V(4)]

  tv2->setMemoryType(MemoryType::Shared);
  tv2->merge(0);
  tv2->split(0, 4);
  tv2->split(0, 32);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  // T2 [8, TIDx(32), 4]

  tv3->merge(0);
  tv3->split(0, 2);
  tv3->split(0, 32);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  // T3 [16, TIDx(32), 2]

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{8});
  ASSERT_EQ(bank_conflict_info.at(tv1).second, std::vector<int>(2, 8));
  ASSERT_EQ(bank_conflict_info.at(tv2).first, std::vector<int>{2});
  ASSERT_EQ(bank_conflict_info.at(tv2).second, std::vector<int>{4});
}

TEST_F(NVFuserTest, FusionTransposeBankConflict5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);
  tv3->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{32});
}

TEST_F(NVFuserTest, FusionTransposeBankConflict6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{32});
}

TEST_F(NVFuserTest, FusionTransposeBankConflict7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 8, 8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).second, std::vector<int>(2, 2));
}

TEST_F(NVFuserTest, FusionTransposeBankConflict8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1024, 8, 8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDy);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto bank_conflict_info = fusion.bankConflictInfo();

  // no bank confliction
  TORCH_CHECK(bank_conflict_info.empty());
}

TEST_F(NVFuserTest, FusionTransposeBankConflict9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv1->merge(0);
  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 32);
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);

  for (auto tv : {tv2, tv3}) {
    tv->merge(1);
    tv->split(1, 2);
    tv->split(1, 32);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
    tv->axis(-2)->parallelize(ParallelType::TIDx);
  }

  auto bank_conflict_info = fusion.bankConflictInfo();
  ASSERT_EQ(bank_conflict_info.at(tv1).first, std::vector<int>{16});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({32, 32, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input});

  auto tv_ref = input.transpose(0, 1);

  testValidate(&fusion, outputs, {input}, {tv_ref}, __LINE__, __FILE__);
}

// small transpose dimension with merge and split. See issue #667
TEST_F(NVFuserTest, UnswitchPredicateIssueRepro667_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(5);
  fusion->addInput(tv0);

  auto tv1 = transpose(tv0, 1, 4);
  auto tv2 = transpose(tv1, 0, 3);
  fusion->addOutput(tv2);

  std::vector<int64_t> shape({2, 7, 102400, 4, 5});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto ref = t0.transpose(1, 4).transpose(0, 3);

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

// small transpose dimension with merge but no split
TEST_F(NVFuserTest, TransposeAggregatedVectorizationWidth_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(5);
  fusion->addInput(tv0);

  auto tv1 = transpose(tv0, 0, 4);
  auto tv2 = transpose(tv1, 1, 3);
  fusion->addOutput(tv2);

  std::vector<int64_t> shape({2, 7, 102400, 4, 9});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");
  auto scheduler = runtime->schedulerHeuristics()->heuristicsList().at(0).get();
  auto heuristic = scheduler->heuristic();
  TORCH_CHECK(
      heuristic == ScheduleHeuristic::Transpose,
      "Unexpected heuristic: ",
      heuristic);
  TORCH_CHECK(
      scheduler->transposeParams().vectorize_factor1 == 4,
      "expecting vectorization for group 1 to be 4");
  TORCH_CHECK(
      scheduler->transposeParams().vectorize_factor2 == 4,
      "expecting vectorization for group 2 to be 4");

  auto ref = t0.transpose(0, 4).transpose(1, 3);

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

TEST_F(NVFuserTest, ViewTransposeReshape_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(3);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {1024, 2, 6}, {1024, 2, 2, 3});
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = reshape(tv2, {1024, 2, 2, 3}, {1024, 2, 6});
  fusion->addOutput(tv3);

  std::vector<int64_t> shape({1024, 2, 6});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto t1 = at::native::view(t0, {1024, 2, 2, 3});
  auto t2 = t1.transpose(1, 2);
  auto ref = at::reshape(t2, {1024, 2, 6});

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

TEST_F(NVFuserTest, ReshapePermuteTransposeScheduler_CUDA) {
  // This is extracted from CSA in nanogpt, where we want transpose scheduler
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({8, 1024, 1024});

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {8, 1024, 1024}, {8, 1024, 16, 64});
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = transpose(tv2, 2, 3);
  fusion->addOutput(tv3);

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

  auto at_out = t0.reshape({8, 1024, 16, 64}).transpose(1, 2).transpose(2, 3);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {at_out},
      __LINE__,
      __FILE__);
}

TEST_F(
    NVFuserTest,
    ReshapePermuteTransposeSchedulerRejectByTransposeViewPropagator_CUDA) {
  // This example sets transpose scheduler that requires P2C transform
  // propagation across a reshape op, which is not currently supported yet.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({8, 1024, 1024});

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {8, 1024, 1024}, {8, 1024, 16, 64});
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = transpose(tv2, 2, 3);
  fusion->addOutput(tv3);
  auto tv4 = add(tv0, IrBuilder::create<Val>(1.0));
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
      heuristic != ScheduleHeuristic::Transpose,
      "Unexpected heuristic: ",
      heuristic);

  auto at_out = t0.reshape({8, 1024, 16, 64}).transpose(1, 2).transpose(2, 3);
  auto at_out_add = t0.add(1.0);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {at_out, at_out_add},
      __LINE__,
      __FILE__);
}

// Test reshape with small transpose dimension
// This introduces an incoherent transformation that can't currently be
// replayed. Transpose scheduler should have rejected this
TEST_F(NVFuserTest, FusionReshapeSmallTransposeDimensionSchedule_CUDA) {
  int x = 2, y = 1024, z = 128, w = 2;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {x, y, z, w}, {x, y * z, w});
  auto tv2 = transpose(tv1, 0, 2);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z, w}, options);
  auto t1 = at::native::view(t0, {x, y * z, w});

  auto t2 = t1.transpose(0, 2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  // NOTE: Aggressive check. If a transpose scheduler can handle this, we should
  // just let it handle this
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<PointwiseParams>());

  testValidate(&fusion, cg_outputs, {t0}, {t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, ViewTransposeMergedInnermostOnGroupTwo_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(3);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {8, 64, 1024}, {8, 64, 64, 16});
  auto tv2 = transpose(tv1, 1, 2);
  auto tv3 = reshape(tv2, {8, 64, 64, 16}, {8, 64, 1024});
  fusion->addOutput(tv3);

  auto tv4 = transpose(tv1, 0, 3);
  fusion->addOutput(tv4);

  std::vector<int64_t> shape({8, 64, 1024});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto t1 = at::native::view(t0, {8, 64, 64, 16});
  auto t2 = t1.transpose(1, 2);
  auto t3 = at::reshape(t2, {8, 64, 1024});
  auto t4 = t1.transpose(0, 3);

  TORCH_CHECK(t3.equal(cg_outputs.at(0)));
  TORCH_CHECK(t4.equal(cg_outputs.at(1)));
}

// TODO: we don't yet support vectorization on split dimension
// https://github.com/NVIDIA/Fuser/pull/690#issue-1837392331
TEST_F(NVFuserTest, TransposeSplitAggregatedVectorizationWidth_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(3);
  fusion->addInput(tv0);

  auto tv1 = transpose(tv0, 0, 2);
  fusion->addOutput(tv1);

  // vectorization on input should be exploited.
  std::vector<int64_t> shape({7, 102400, 9});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");
  // TODO: check on vectorization!
  auto heuristic =
      runtime->schedulerHeuristics()->heuristicsList().at(0).get()->heuristic();
  TORCH_CHECK(
      heuristic == ScheduleHeuristic::Transpose,
      "Unexpected heuristic: ",
      heuristic);

  auto ref = t0.transpose(0, 2);

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

} // namespace nvfuser
