// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <device_lower/analysis/bank_conflict.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include "ops/arith.h"
#include "type.h"
namespace nvfuser {

using testing::Contains;
using testing::UnorderedElementsAre;
using PersistentBufferTest = NVFuserTest;

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  EXPECT_EQ(buffers.size(), 1);
  EXPECT_EQ(resolution.size(), 1);
  EXPECT_EQ(resolution.at(0).size(), 1) << toDelimitedString(resolution.at(0));
  EXPECT_EQ(projectable.size(), 1);
  EXPECT_EQ(projectable_inputs.size(), 1);

  EXPECT_TRUE(isTvWithinVec(buffers, tv1));
  EXPECT_TRUE(isTvWithinVec(projectable, tv1));
  EXPECT_TRUE(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  EXPECT_TRUE(tv1_resolution_it != resolution.end());

  EXPECT_TRUE(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);

  EXPECT_EQ(
      persistent_buffer_size_bit.persistent_buffer_size_bit,
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Float)));
  EXPECT_EQ(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit,
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Float)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = castOp(DataType::Half, tv5);
  fusion.addOutput(tv6);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 1);
  NVF_ERROR(resolution.size() == 1 && resolution[0].size() == 1);
  NVF_ERROR(projectable.size() == 1);
  NVF_ERROR(projectable_inputs.size() == 1);

  NVF_ERROR(isTvWithinVec(buffers, tv1));
  NVF_ERROR(isTvWithinVec(projectable, tv1));
  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())

  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);

  NVF_ERROR(
      persistent_buffer_size_bit.persistent_buffer_size_bit ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Float)));
  NVF_ERROR(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Half)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});

  auto tv5 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv5);

  auto tv6 = castOp(DataType::Float, tv5);

  auto tv7 = add(tv6, tv4);
  auto tv8 = set(tv1);
  auto tv9 = add(tv7, tv8);
  auto tv10 = sum(tv9, {1});
  auto tv11 = broadcast(tv10, {false, true});
  auto tv12 = set(tv7);
  auto tv13 = add(tv12, tv11);

  fusion.addOutput(tv13);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 2);
  NVF_ERROR(
      resolution.size() == 2 && resolution[0].size() == 1 &&
      resolution[1].size() == 1);
  NVF_ERROR(projectable.size() == 2);
  NVF_ERROR(projectable_inputs.size() == 2);

  NVF_ERROR(isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv7));
  NVF_ERROR(isTvWithinVec(projectable, tv1) && isTvWithinVec(projectable, tv7));

  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv9));

  auto tv7_resolution_it = tvEntryInVecVec(resolution, buffers, tv7);
  NVF_ERROR(tv7_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv7_resolution_it, tv13));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);
  at::Tensor aten_t5 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0, aten_t5});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);

  NVF_ERROR(
      persistent_buffer_size_bit.persistent_buffer_size_bit ==
      static_cast<int64_t>(
          aten_t0.size(1) * dataTypeSizeBit(DataType::Float) * 2));
  NVF_ERROR(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit ==
      static_cast<int64_t>(
          aten_t0.size(1) * dataTypeSizeBit(DataType::Half) * 2));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  EXPECT_EQ(buffers.size(), 2);
  ASSERT_EQ(resolution.size(), 2);
  EXPECT_EQ(resolution[0].size(), 1);
  EXPECT_EQ(resolution[1].size(), 1);

  EXPECT_EQ(projectable.size(), 2);
  EXPECT_EQ(projectable_inputs.size(), 1);

  NVF_ERROR(isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv2));
  NVF_ERROR(isTvWithinVec(projectable, tv1) && isTvWithinVec(projectable, tv2));

  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv6));

  auto tv2_resolution_it = tvEntryInVecVec(resolution, buffers, tv2);
  NVF_ERROR(tv2_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv2_resolution_it, tv8));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);

  // T1 and T2 are persistent buffers, but T2 can be projected to T1.
  // So, the actual buffer size is just the size to save T1.
  NVF_ERROR(
      persistent_buffer_size_bit.persistent_buffer_size_bit ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Float)));

  NVF_ERROR(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSizeBit(DataType::Half)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferProjection_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  reduction_scheduler_utils::projectPersistentBuffers(
      &fusion, scheduler_utils::persistentBuffers(&fusion), true);

  auto tv5_producers = ir_utils::producerTvsOf(tv5);
  auto tv7_producers = ir_utils::producerTvsOf(tv7);

  // Projection should have broken these dependencies

  NVF_ERROR(
      std::find(tv5_producers.begin(), tv5_producers.end(), tv1) ==
      tv5_producers.end());
  NVF_ERROR(
      std::find(tv7_producers.begin(), tv7_producers.end(), tv2) ==
      tv7_producers.end());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({aten_t0});

  testValidate(&fusion, cg_outputs, {aten_t0}, __LINE__, __FILE__);
}

// Repro of issue #2381
TEST_F(PersistentBufferTest, FusionPersistentBufferProjection2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv1);

  auto tv2 = castOp(DataType::Float, tv0);
  auto tv3 = castOp(DataType::Float, tv1);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sum(tv4, {1});
  auto tv6 = broadcast(tv5, {false, true});
  // Cast tv1 again
  auto tv7 = castOp(DataType::Float, tv1);
  // No error if this is done with tv3 rather than tv7
  auto tv8 = sub(tv6, tv7);
  auto tv9 = sub(tv8, tv4);
  auto tv10 = castOp(DataType::Half, tv9);
  fusion.addOutput(tv10);

  std::vector<int64_t> shape({10, 11});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape, options);

  // Persistent buffers: tv1, tv4
  // Projectable buffer: tv4
  // Projectable buffer inputs: tv0, tv1

  // tv1 is both a persistent buffer and an input to the projected
  // buffer of tv4. It is NOT considered as projectable.

  auto persistent_info = scheduler_utils::persistentBuffers(&fusion);

  NVF_CHECK(persistent_info.persistent_buffers.size() == 2);
  for (auto tv : persistent_info.persistent_buffers) {
    NVF_CHECK(
        tv == tv4 || tv == tv1,
        "Unexpected persistent buffer: ",
        tv->toString());
  }

  NVF_CHECK(persistent_info.projectable_persistent_buffers.size() == 1);
  for (auto tv : persistent_info.projectable_persistent_buffers) {
    NVF_CHECK(
        tv == tv4,
        "Unexpected projectable persistent buffer: ",
        tv->toString());
  }

  for (auto tv : persistent_info.projectable_buffer_inputs) {
    NVF_CHECK(
        tv == tv0 || tv == tv1,
        "Unexpected projectable buffer input: ",
        tv->toString());
  }

  SchedulerRuntimeInfo runtime_info(&fusion, {t0, t1});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_info);

  // Since tv1 is not projectable, it is included in the active mask
  // of projected buffers, even though it is also included in the
  // projectable buffer inputs. Thus, the buffer size would be
  // calculated as the sum of tv1, tv0 and tv1.
  auto projected_size =
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit;
  auto expected_size =
      static_cast<int64_t>(shape[1] * 2 * dataTypeSizeBit(DataType::Half));
  NVF_CHECK(
      projected_size == expected_size,
      "Buffer projection failure. Expected size: ",
      expected_size,
      ". Actual: ",
      projected_size);
}

// https://github.com/csarofeen/pytorch/issues/2321
TEST_F(
    PersistentBufferTest,
    FusionPersistentBufferProjectionAfterWelfordTranslate_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);

  DataType dtype = DataType::Half;
  constexpr int64_t dim0 = 2048;
  constexpr int64_t dim1 = 10240;
  std::vector<int64_t> input_shape{dim0, dim1};
  std::vector<int64_t> norm_shape{dim1};
  auto input_half = makeContigTensor(2, dtype);
  auto weight_half = makeContigTensor(1, dtype);
  auto bias_half = makeContigTensor(1, dtype);
  fusion.addInput(input_half);
  fusion.addInput(weight_half);
  fusion.addInput(bias_half);
  auto input = castOp(DataType::Float, input_half);
  auto weight = castOp(DataType::Float, weight_half);
  auto bias = castOp(DataType::Float, bias_half);
  auto result = layer_norm(input, norm_shape, weight, bias, eps_ptr);
  auto result_output = castOp(dtype, result.output);
  fusion.addOutput(result_output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = at::randn({input_shape[1]}, options);
  c10::optional<at::Tensor> aten_bias = at::randn({input_shape[1]}, options);
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  // welford translate
  KernelArgumentHolder runtime_inputs({aten_input, aten_weight, aten_bias});
  bool isTranslated =
      SegmentCandidateFinder::translateWelfordInFusion(&fusion, runtime_inputs);
  NVF_ERROR(isTranslated);

  // persistent buffer should be projected to input
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  NVF_CHECK(
      persistent_buffer_info.projectable_persistent_buffers.size() == 1,
      "should have only one projectable_persistent_buffer!");
  NVF_CHECK(
      persistent_buffer_info.projectable_buffer_inputs.size() == 1,
      "should have only one projectable_buffer_inputs!");
  NVF_CHECK(
      persistent_buffer_info.projectable_buffer_inputs[0] == input_half,
      "persistent buffer should be projected to input!");

  auto cg_outputs = scheduleAndRun(
                        &fusion,
                        SchedulerType::InnerPersistent,
                        {aten_input, aten_weight, aten_bias})
                        .outputs;
  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, aten_weight, aten_bias},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__,
      "");
}

// https://github.com/NVIDIA/Fuser/issues/335
// This test is to make sure the benchmark in layer_norm_fused.cpp is correctly
// implemented.
TEST_F(PersistentBufferTest, FusionLayerNormFusedOpsRedundantCast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  const int batch_size = 2048 * 8;
  const int hidden_size = 20480;
  DataType dtype = DataType::Half;
  {
    auto tv0 = makeContigTensor(1, dtype);
    auto tv1 = makeContigTensor(2, dtype);
    auto tv2 = makeContigTensor(1, dtype);
    auto tv3 = makeContigTensor(1, dtype);
    auto tv4 = makeContigTensor(1, dtype);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addInput(tv3);
    fusion->addInput(tv4);
    auto tv5 = broadcast(tv0, {true, false});
    auto tv6 = castOp(DataType::Float, tv1);
    auto tv7 = castOp(DataType::Float, tv5);
    auto tv8 = add(tv6, tv7);
    auto tv9 = castOp(DataType::Half, tv8);
    auto tv10 = broadcast(tv2, {true, false});
    auto tv11 = castOp(DataType::Float, tv9);
    auto tv12 = castOp(DataType::Float, tv10);
    auto tv13 = add(tv11, tv12);
    auto tv14 = castOp(DataType::Half, tv13);
    auto tv15 = castOp(DataType::Float, tv14);
    auto tv16 = variance(tv15, {1}, false, false);
    auto tv17 = broadcast(tv16, {false, true});
    auto tv18 = sum(tv15, {1}, false);
    auto tv19 = broadcast(tv18, {false, true});

    nvfuser::Val* num_features = IrBuilder::create<Val>(1.0);
    num_features = mul(num_features, tv0->getLoopDomain()[0]->extent());
    auto s20 = num_features;

    auto s21 = reciprocal(s20);
    auto tv22 = mul(tv19, s21);
    auto s23 = IrBuilder::create<Val>(kEps);
    auto tv24 = add(tv17, s23);
    auto tv25 = rsqrt(tv24);
    auto tv26 = broadcast(tv22, {false, false});
    auto tv27 = castOp(DataType::Float, tv14);
    auto tv28 = sub(tv27, tv26);
    auto tv29 = broadcast(tv25, {false, false});
    auto tv30 = mul(tv28, tv29);
    auto tv31 = broadcast(tv4, {true, false});
    auto tv32 = castOp(DataType::Float, tv31);
    auto tv33 = mul(tv30, tv32);
    auto tv34 = broadcast(tv3, {true, false});
    auto tv35 = castOp(DataType::Float, tv34);
    auto tv36 = add(tv33, tv35);
    auto tv37 = castOp(DataType::Half, tv36);
    fusion->addOutput(tv37);
  }

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t0 = at::randn({hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({hidden_size}, options);
  auto t3 = at::randn({hidden_size}, options);
  auto t4 = at::randn({hidden_size}, options);

  auto t5 = t0.unsqueeze(0).expand({batch_size, hidden_size});
  auto t6 = t1.to(at::kFloat);
  auto t7 = t5.to(at::kFloat);
  auto t8 = at::add(t6, t7);
  auto t9 = t8.to(at::kHalf);
  auto t10 = t2.unsqueeze(0).expand({batch_size, hidden_size});
  auto t11 = t9.to(at::kFloat);
  auto t12 = t10.to(at::kFloat);
  auto t13 = at::add(t11, t12);
  auto t14 = t13.to(at::kHalf);
  auto aten_outputs = at::native_layer_norm(t14, {hidden_size}, t4, t3, kEps);
  auto t33 = std::get<0>(aten_outputs);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info.persistent_buffers.size() == 2,
      "Before project to other buffers, should have two persistent buffers!");

  // The buffer size should only count 1 buffer because the other one is
  // projected to its producer.
  SchedulerRuntimeInfo runtime_info(fusion, {t0, t1, t2, t3, t4});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(fusion, runtime_info, persistent_buffer_info);
  NVF_CHECK(
      persistent_buffer_size_bit.persistent_buffer_size_bit ==
          hidden_size * dataTypeSizeBit(dtype),
      "Persistent buffer size is not correct!");

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});
  testValidate(
      fusion, cg_outputs, {t0, t1, t2, t3, t4}, {t33}, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, FusionRecomputePersistentBuffer_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 1024;
  const int hidden_size = 2048;
  {
    DataType dtype = DataType::Float;
    auto tv0 = makeContigTensor(2, dtype);
    auto tv1 = makeContigTensor(2, dtype);
    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);
    auto tv3 = castOp(DataType::Half, tv2);

    auto tv4 = castOp(DataType::Float, tv3);
    auto tv5 = sum(tv4, {1});
    auto tv6 = broadcast(tv5, {false, true});
    auto tv7 = add(tv4, tv6);

    auto tv8 = castOp(DataType::Float, tv3);
    auto tv9 = add(tv6, tv8);

    fusion->addOutput(tv7);
    fusion->addOutput(tv9);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({batch_size, hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);

  auto t2 = t0.add(t1);
  auto t3 = t2.to(at::kHalf);
  auto t4 = t3.to(at::kFloat);
  auto t5 = t4.sum({1});
  auto t6 = t5.unsqueeze(1).expand({batch_size, hidden_size});
  auto t7 = t4.add(t6);
  auto t8 = t3.to(at::kFloat);
  auto t9 = t8.add(t6);

  auto persistent_buffer_info1 = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info1.persistent_buffers.size() == 2,
      "Before project to other buffers, should have two persistent buffers!");

  reduction_scheduler_utils::projectPersistentBuffers(
      fusion, scheduler_utils::persistentBuffers(fusion), false);
  auto persistent_buffer_info2 = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info2.persistent_buffers.size() == 1,
      "After project to other buffers, should have one persistent buffer!");

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(fusion, cg_outputs, {t0, t1}, {t7, t9}, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, ProjectPersistentBufferMultiScopes) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 2048;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, input_dtype);
  auto tv2 = makeContigTensor(2, input_dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = add(tv0, tv0);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = add(tv3, tv5);

  auto tv7 = add(tv3, tv3);
  auto tv8 = sum(tv7, {1});
  auto tv9 = broadcast(tv8, {false, true});
  auto tv10 = add(tv7, tv9);

  auto tv11 = add(tv0, tv1);
  auto tv12 = mul(tv11, tv11);
  auto tv13 = sum(tv12, {1});
  auto tv14 = broadcast(tv13, {false, true});
  auto tv15 = add(tv12, tv14);

  auto tv16 = add(tv12, tv2);
  auto tv17 = mul(tv16, tv16);
  auto tv18 = sum(tv17, {1});
  auto tv19 = broadcast(tv18, {false, true});
  auto tv20 = add(tv17, tv19);

  fusion->addOutput(tv6);
  fusion->addOutput(tv10);
  fusion->addOutput(tv15);
  fusion->addOutput(tv20);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({batch_size, hidden_size}, options);

  // The persistent buffers in this fusion are: tv3, tv7, tv12, and tv17. Note
  // that tv7 can be projected back to its producer, tv3. When calculating the
  // total size of persistent buffers ([persistent_buffer_size]), it's important
  // to consider the active scopes of these buffers. Simply subtracting the
  // buffer size of tv7 from the max buffer size may lead to an underestimation.
  // This is because there are two distinct scopes in this computation: (1)
  // During the calculation of tv10, the active persistent buffers are tv3 and
  // tv7. (2) For the calculation of tv20, the active persistent buffers are
  // tv12 and tv17. The max buffer size is based on tv12 and tv17. There is no
  // projectable buffer needs to be deducted in this scope.
  auto persistent_info = scheduler_utils::persistentBuffers(fusion);
  SchedulerRuntimeInfo runtime_info(fusion, {t0, t1, t2});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(fusion, runtime_info, persistent_info);
  auto calculated_size = persistent_buffer_size_bit.persistent_buffer_size_bit;
  auto expected_size =
      static_cast<int64_t>(hidden_size * 2 * dataTypeSizeBit(input_dtype));
  EXPECT_EQ(calculated_size, expected_size)
      << "Buffer size calculation failure";
  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion, SchedulerType::InnerPersistent, {t0, t1, t2});
  auto rparams = heuristic_params->as<ReductionParams>();
  NVF_CHECK(
      !rparams->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");
}

TEST_F(PersistentBufferTest, ChainProjectionToPersistentProducer) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 2048;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, input_dtype);
  auto tv2 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Float, tv0);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = castOp(DataType::Float, tv2);

  // tv7 is persistent
  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv6, tv5);
  auto tv8 = sum(tv7, {1});
  auto tv9 = broadcast(tv8, {false, true});
  auto tv10 = add(tv7, tv9);

  // tv11 is persistent, and can be projected to tv7
  auto tv11 = add(tv7, tv7);
  auto tv12 = sum(tv11, {1});
  auto tv13 = broadcast(tv12, {false, true});
  auto tv14 = add(tv11, tv13);

  // tv15 is persistent, and can be projected to tv11
  auto tv15 = add(tv11, tv11);
  auto tv16 = sum(tv15, {1});
  auto tv17 = broadcast(tv16, {false, true});
  auto tv18 = add(tv17, tv15);

  fusion->addOutput(tv10);
  fusion->addOutput(tv14);
  fusion->addOutput(tv18);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({batch_size, hidden_size}, options);
  auto t3 = t0.to(at::kFloat) + t1.to(at::kFloat) + t2.to(at::kFloat);
  auto t4 = at::sum(t3, {1}, true);
  auto t5 = t3 + t4;
  auto t6 = t3 + t3;
  auto t7 = at::sum(t6, {1}, true);
  auto t8 = t6 + t7;
  auto t9 = t6 + t6;
  auto t10 = at::sum(t9, {1}, true);
  auto t11 = t9 + t10;

  // There are 3 persistent buffers: tv7, tv11, and tv15.
  // The PersistentBufferProjector should firstly project
  // tv15 to tv11, then project tv11 to tv7.
  // After projection, tv7 is the only buffer.
  auto persistent_info = scheduler_utils::persistentBuffers(fusion);
  SchedulerRuntimeInfo runtime_info(fusion, {t0, t1, t2});
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(fusion, runtime_info, persistent_info);
  auto calculated_size = persistent_buffer_size_bit.persistent_buffer_size_bit;
  auto expected_size =
      static_cast<int64_t>(hidden_size * dataTypeSizeBit(DataType::Float));
  NVF_CHECK(
      calculated_size == expected_size,
      "Buffer size calculation failure. Expected size: ",
      expected_size,
      ". Actual: ",
      calculated_size);

  // If project to inputs, there are 3 fp16 tvs, which is larger than 1 fp32.
  // So, shouldn't project to inputs.
  auto cg_results =
      scheduleAndRun(fusion, SchedulerType::InnerPersistent, {t0, t1, t2});
  auto rparams = cg_results.heuristic_params->as<ReductionParams>();

  NVF_CHECK(
      !rparams->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");
  testValidate(
      fusion,
      cg_results.outputs,
      {t0, t1, t2},
      {t5, t8, t11},
      __LINE__,
      __FILE__);
}

// Test the persistent buffers in softmax are projected back to inputs.
TEST_F(PersistentBufferTest, SoftmaxProjectToInput) {
  auto test_softmax = [](int batch, int feature, DataType dtype) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    const int kReductionAxis = 1;
    std::vector<int64_t> input_shape{batch, feature};
    TensorView* input = makeContigTensor(input_shape.size(), dtype);
    fusion.addInput(input);
    if (dtype == DataType::Half) {
      input = castOp(DataType::Float, input);
    }
    auto output = softmax(input, kReductionAxis);
    if (dtype == DataType::Half) {
      output = castOp(DataType::Half, output);
    }
    fusion.addOutput(output);

    // There should be 2 projectable persistent buffers.
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
    auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
    NVF_ERROR(projectable.size() == 2);

    auto options = at::TensorOptions()
                       .dtype(data_type_to_aten(dtype))
                       .device(at::kCUDA, 0);
    at::Tensor aten_input = at::randn(input_shape, options);
    auto aten_output =
        at::_softmax(aten_input.to(at::kDouble), kReductionAxis, false);

    auto cg_results =
        scheduleAndRun(&fusion, SchedulerType::InnerPersistent, {aten_input});
    auto rparams = cg_results.heuristic_params->as<ReductionParams>();

    // Threshold to project to inputs
    int64_t buffer_threshold_bit = scheduler_utils::isHighBandwidthFlopsRatio()
        ? 24 * 1024 * 4 * 8
        : 6 * 1024 * 4 * 8;
    bool should_project_to_input =
        feature * dataTypeSizeBit(DataType::Float) > buffer_threshold_bit;
    NVF_CHECK(
        rparams->project_persistent_buffers == should_project_to_input,
        should_project_to_input ? "Should project to inputs!"
                                : "Shouldn't project to inputs!");
    testValidate(
        &fusion,
        cg_results.outputs,
        {aten_input},
        {aten_output},
        __LINE__,
        __FILE__,
        "",
        rparams->lparams);
  };
  const int batch = 2048;
  std::vector<int> features = {6 * 1024, 10240};
  for (auto feature : features) {
    test_softmax(batch, feature, DataType::Half);
  }
}

// Test projection to inputs when there are three persistent buffers.
TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs1) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 128;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = add(tv1, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = div(tv2, tv4);

  auto tv6 = add(tv5, tv5);
  auto tv7 = sum(tv6, {1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 = div(tv6, tv8);

  auto tv10 = add(tv9, tv9);
  auto tv11 = sum(tv10, {1});
  auto tv12 = broadcast(tv11, {false, true});
  auto tv13 = div(tv10, tv12);

  fusion->addOutput(tv5);
  fusion->addOutput(tv9);
  fusion->addOutput(tv13);

  // The persistent buffers in this fusion are: tv2, tv6, and tv10.
  // tv2 is projected to input.
  // tv6 is projected to input and tv4 which is a broadcast tv.
  // tv10 is projected to input, tv4 and tv8 which are broadcast tvs.
  // The only actual persisent buffer is the cached input.
  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto aten_input = at::randn({batch_size, hidden_size}, options);

  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion, SchedulerType::InnerPersistent, {aten_input});
  auto rparams = heuristic_params->as<ReductionParams>();

  NVF_CHECK(
      rparams->project_persistent_buffers,
      "Should project persistent buffers to inputs!");
}

// Test projection to inputs when the persistent buffer is a broadcast tv.
TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs2) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 128;
  const int hidden_size = 8192;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = exp(tv1);
  auto tv3 = sum(tv2, {-1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv2, tv4);
  fusion->addOutput(tv5);

  auto tv6 = broadcast(tv5, {true, false, false});
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, false, true});
  auto tv9 = add(tv6, tv8);
  fusion->addOutput(tv9);

  // In this fusion, tv6 is a persistent buffer with a broadcast dim.
  // Between reduction tv2 and tv6, there are two broadcast tvs: tv4 and tv6.
  // Only tv4 is a valid broadcast tv to project to.
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  const auto& [can_project, broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv6);
  NVF_CHECK(
      can_project, "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      broadcast_tvs.size() == 1,
      "Expect one target broadcast_tv!, Got: ",
      broadcast_tvs.size());
  NVF_CHECK(
      broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      broadcast_tvs.at(0)->toString());

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);

  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion, SchedulerType::InnerPersistent, {t0});
  auto rparams = heuristic_params->as<ReductionParams>();
  if (scheduler_utils::isHighBandwidthFlopsRatio()) {
    NVF_CHECK(
        !rparams->project_persistent_buffers,
        "Should not project persistent buffers to inputs!");
  } else {
    NVF_CHECK(
        rparams->project_persistent_buffers,
        "Should project persistent buffers to inputs!");
  }
}

TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs3) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int dim0 = 128;
  const int dim1 = 32;
  const int dim2 = 256;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(3, input_dtype);
  fusion->addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1, 2});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = broadcast(tv3, {false, false, true});
  auto tv5 = add(tv1, tv4);
  fusion->addOutput(tv5);

  // Ensure there is no exp op in the fusion, otherwise
  // project to inputs depends on the buffer size and bandwidth flops ratio of
  // the hardware.
  auto tv6 = mul(tv5, tv5);
  auto tv7 = sum(tv6, {1, 2});
  auto tv8 = broadcast(tv7, {false, true, true});
  auto tv9 = add(tv6, tv8);
  fusion->addOutput(tv9);

  auto tv10 = add(tv5, tv9);
  auto tv11 = sum(tv10, {1, 2});
  auto tv12 = broadcast(tv11, {false, true, true});
  auto tv13 = add(tv10, tv12);
  fusion->addOutput(tv13);

  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  // (1) Test projection to inputs when there are two broadcast tvs (tv3 and
  // tv4) between the reduction tv (tv2) and the persistent buffer (tv6). Should
  // only project to tv4.
  const auto& [can_project, broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv6);
  NVF_CHECK(
      can_project, "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      broadcast_tvs.size() == 1,
      "Expect one target broadcast_tv!, Got: ",
      broadcast_tvs.size());
  NVF_CHECK(
      broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      broadcast_tvs.at(0)->toString());

  // (2) Test projection to inputs when the persistent buffer (tv10) depends on
  // two reduction tvs (tv2 and tv7). Should project to tv4 and tv8.
  const auto& [tv10_can_project, tv10_broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv10);
  NVF_CHECK(
      tv10_can_project,
      "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      tv10_broadcast_tvs.size() == 2,
      "Expect two target broadcast_tv!, Got: ",
      tv10_broadcast_tvs.size());
  NVF_CHECK(
      tv10_broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      tv10_broadcast_tvs.at(0)->toString());
  NVF_CHECK(
      tv10_broadcast_tvs.at(1) == tv8,
      "Expect target tv8!, Got: ",
      tv10_broadcast_tvs.at(1)->toString());

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1, dim2}, options);

  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion, SchedulerType::InnerPersistent, {t0});
  auto rparams = heuristic_params->as<ReductionParams>();
  NVF_CHECK(
      rparams->project_persistent_buffers,
      "Should project persistent buffers to inputs!");
}

TEST_F(NVFuserTest, AvoidProjectingToInputsIfRecomputeHasDropout) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);

  // tv2 is a persistent buffer.
  // compute tv2 from inputs requires dropout, which is very expensive.
  // should not project tv2 to inputs to avoid recomputation.
  const int64_t hidden_size = 10240;
  std::vector<int64_t> norm_shape{hidden_size};
  auto tv1 = castOp(DataType::Float, tv0);
  auto dropout_res = dropout(tv1, IrBuilder::create<Val>(0.9));
  auto tv2 = dropout_res.output;
  auto ln_res = layer_norm(
      tv2, norm_shape, nullptr, nullptr, IrBuilder::create<Val>(1e-5));
  fusion->addOutput(ln_res.output);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({1024, hidden_size}, options);
  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion.get(), SchedulerType::InnerPersistent, {aten_input});
  auto rparams = heuristic_params->as<ReductionParams>();
  NVF_CHECK(
      !rparams->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");
}

// Reproduce of issue-2146
// hasNonNormalizePostReductionBCast() checks whether the post reduction
// broadcast ID is mapped to a reduction input ID. In this fuion,
// T6[I1,I2] = T4[I1,B] + T5[I1,I2]
// before this fix, the check backwards from T6 and can't find the
// corresponding reduction input ID. This fix moves forward from T6
// to the output tensor T7, where
// T7[I1,I2] = T6[I1,I2] + T2[I1,I2]
// From T7, the backward search can find the corresponding reduction
// input ID, which is {I2} in T2.
TEST_F(PersistentBufferTest, PostReductionBroadcastCheck) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int dim0 = 128;
  const int dim1 = 256;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, input_dtype);
  auto tv1 = makeContigConcreteTensor({dim0, dim1}, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = add(tv6, tv2);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0, dim1}, options);
  auto t2 = at::sum(t0, {1}).unsqueeze(1) + t0;
  auto t4 = t2 + t1;
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "unexpected segmentation!");

  testValidate(fusion, cg_outputs, {t0, t1}, {t4}, __LINE__, __FILE__);
}

// Cases with two broadcast IDs
TEST_F(PersistentBufferTest, PostReductionBroadcastCheckMultiBcastDims) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int dim0 = 16;
  const int dim1 = 32;
  const int dim2 = 64;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigConcreteTensor({dim0, dim1, dim2}, input_dtype);
  auto tv1 = makeContigConcreteTensor({dim0, dim1, dim2}, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = sum(tv2, {1, 2});
  auto tv4 = broadcast(tv3, {false, true, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = add(tv6, tv2);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1, dim2}, options);
  auto t1 = at::randn({dim0, dim1, dim2}, options);
  auto t2 = at::sum(t0, {1, 2}).unsqueeze(-1).unsqueeze(-1) + t0;
  auto t4 = t2 + t1;
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "unexpected segmentation!");

  testValidate(fusion, cg_outputs, {t0, t1}, {t4}, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, SmemPersistentNotSupportedIn3DReduction) {
  // 1024 elements is added to ensure the buffer size is larger than
  // max allowed register file size to trigger the use of smem persistent buffer
  // or segmentation.
  const int64_t max_element_for_reg_persistent =
      scheduler_utils::register_file_size_bit /
      scheduler_utils::bits_per_register;
  DataType input_dtype = DataType::Float;
  const int64_t total_elements = max_element_for_reg_persistent + 1024;
  const std::vector<int64_t> input_shape = {2, 64, 2, total_elements / (2 * 2)};
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(input_shape.size(), input_dtype);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0, 2, 3});
  auto tv2 = broadcast(tv1, std::vector<bool>{true, false, true, true});
  auto tv7 = div(tv0, tv2);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  // should be segmented since buffer size is larger than 32K and smem
  // persistent is not supported yet for 3D reduction.
  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, SmemPersistent2DReduction) {
  // 1024 elements is added to ensure the buffer size is larger than
  // max allowed register file size to trigger the use of smem persistent buffer
  // or segmentation.
  const int64_t max_element_for_reg_persistent =
      scheduler_utils::register_file_size_bit /
      scheduler_utils::bits_per_register;
  DataType input_dtype = DataType::Float;
  const int64_t total_elements = max_element_for_reg_persistent + 1024;
  const std::vector<int64_t> input_shape = {64, 2, 2, total_elements / (2 * 2)};
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(input_shape.size(), input_dtype);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {1, 2, 3});
  auto tv2 = broadcast(tv1, std::vector<bool>{false, true, true, true});
  auto tv7 = div(tv0, tv2);
  fusion->addOutput(tv7);

  // If device doesn't have enough shared memory, skip this test
  int64_t smem_overhead_bit = scheduler_utils::getReductionSmemWorkspaceBit(
      fusion.get(), scheduler_utils::getReductionTvs(fusion.get()));
  const size_t required_smem_size_bit =
      smem_overhead_bit + total_elements * dataTypeSizeBit(input_dtype);
  REQUIRE_DEVICE_SMEM_SIZE(required_smem_size_bit / 8, 0);

  // Schedule through magic scheduler and test the use of smem persistent buffer
  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion.get(), runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion.get(), runtime_info);
  EXPECT_FALSE(
      heuristic_params->as<ReductionParams>()->smem_persistent_buffers.empty());
  scheduler->schedule(fusion.get(), heuristic_params.get());

  // Run the fusion and validate the results
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});
  // Shared memory access should be vectorized.
  // getBankConflictInfo(ke.compiledKernel()->kernel()) triggers error
  // "std::get: wrong index for variant" when trying to evaluate index with:
  // `expr_eval.evaluate(ti->index()).as<int64_t>();`
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() == MemoryType::Shared) {
      // check self
      EXPECT_TRUE(isVectorized(tv));
      // check consumers
      for (auto consumer : ir_utils::consumerTvsOf(tv)) {
        EXPECT_TRUE(isVectorized(consumer));
      }
    }
  }
  auto cg_outputs =
      ke.run({t0}, {}, heuristic_params->as<ReductionParams>()->lparams);
  auto t1 = t0 / t0.sum({1, 2, 3}, true);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

// C++ version of the simplified repro of issue #1123
TEST_F(PersistentBufferTest, GetResolutionIssue1123) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv0, tv1);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = set(tv5);
  auto tv7 = add(tv6, tv2);
  fusion.addOutput(tv7);
  auto tv9 = add(tv3, tv2);
  fusion.addOutput(tv9);

  // tv3 is the persistent tensor. The resolution point is tv8.
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  EXPECT_EQ(
      persistent_buffer_info.persistent_buffers, std::vector<TensorView*>{tv3});
  EXPECT_EQ(
      persistent_buffer_info.persistent_buffer_resolution_points.size(), 1);
  EXPECT_EQ(
      persistent_buffer_info.persistent_buffer_resolution_points.at(0),
      std::vector<TensorView*>{tv7});
}

TEST_F(PersistentBufferTest, InnerPersistentNotEnoughSharedMemory) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1, DataType::Half);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1, DataType::Half);
  fusion.addInput(tv2);

  auto tv3 = castOp(DataType::Float, tv0);
  auto tvs = Welford(tv3, {1});
  auto tv6 = tvs.avg;
  auto tv7 = tvs.var_sum;
  auto tv9 = broadcast(tv6, {false, true});
  TensorView* tv10 = nullptr;
  auto tv21 = castOp(DataType::Float, tv0);
  tv10 = sub(tv21, tv9);
  auto tv11 = broadcast(tv7, {false, true});
  auto tv13 = add(tv11, IrBuilder::create<Val>(0.001));
  auto tv14 = rsqrt(tv13);
  auto tv15 = mul(tv10, tv14);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv16 = broadcast(tv4, {true, false});
  auto tv17 = mul(tv15, tv16);
  auto tv5 = castOp(DataType::Float, tv2);
  auto tv18 = broadcast(tv5, {true, false});
  auto tv19 = add(tv17, tv18);
  auto tv20 = castOp(DataType::Half, tv19);

  fusion.addOutput(tv20);
  fusion.addOutput(tv9);
  fusion.addOutput(tv14);

  std::vector<int64_t> input_shape{2048, 80 * 1024};

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({input_shape[1]}, options);
  auto t2 = at::randn({input_shape[1]}, options);

  // The logic size of the persistent buffer in this fusion is 80 * 1024 * 2
  // bytes. Inner persistent scheduler allows 32 * 1024 * 4 bytes for register
  // persistent, so it should use shared memory persistent buffer if there are
  // enough shared memory. Otherwise, it will be segmented.
  SchedulerRuntimeInfo runtime_info(&fusion, {t0, t1, t2});
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);
  int64_t logic_buffer_size_bit = 80 * 1024 * dataTypeSizeBit(DataType::Half);
  EXPECT_EQ(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit,
      logic_buffer_size_bit);

  // If total shared memory on device is less than logic buffer size, should
  // segment. Otherwise, further calculate available shared memory size by
  // removing overhead due to reduction broadcast workspace and non-divisible
  // split.
  bool is_segmented = false;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  if ((int64_t)dev_prop->sharedMemPerBlockOptin * 8 < logic_buffer_size_bit) {
    is_segmented = true;
  } else {
    int64_t available_buffer_size_bit = normalization_scheduler_utils::
        getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
            &fusion,
            runtime_info,
            scheduler_utils::getReductionTvs(&fusion),
            persistent_buffer_info,
            /*can_use_smem_persistent*/ true,
            /*project_to_inputs*/ true);
    is_segmented = logic_buffer_size_bit >= available_buffer_size_bit;
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  // check segmentation, if not segmented, further check shared memory
  // persistence
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  ASSERT_EQ(is_segmented, runtime->isSegmented());
  if (!is_segmented) {
    auto& params = runtime->schedulerHeuristics()->heuristicsList().at(0);
    ASSERT_TRUE(params->isA<ReductionParams>());
    ASSERT_TRUE(
        params->as<ReductionParams>()->smem_persistent_buffers.size() > 0);
  }
  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

using TestParam = std::tuple<DataType, int64_t>;
using LayerNormSharedMemoryTest = NVFuserFixtureParamTest<TestParam>;
TEST_P(LayerNormSharedMemoryTest, FusionLayerNormSharedMemoryBuffer_CUDA) {
  auto [dtype, hidden_size] = GetParam();

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);
  constexpr int64_t dim0 = 2048;
  std::vector<int64_t> input_shape{dim0, hidden_size};
  std::vector<int64_t> norm_shape{hidden_size};

  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  fusion.addInput(input);
  fusion.addInput(weight);
  fusion.addInput(bias);
  input = maybeCastOp(DataType::Float, input);
  weight = maybeCastOp(DataType::Float, weight);
  bias = maybeCastOp(DataType::Float, bias);
  auto result = layer_norm(input, norm_shape, weight, bias, eps_ptr);
  result.output = maybeCastOp(dtype, result.output);
  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = at::randn({input_shape[1]}, options);
  c10::optional<at::Tensor> aten_bias = at::randn({input_shape[1]}, options);

  // try translate Welford in fusion
  KernelArgumentHolder runtime_inputs =
      KernelArgumentHolder({aten_input, aten_weight, aten_bias});
  SegmentCandidateFinder::translateWelfordInFusion(&fusion, runtime_inputs);
  auto fusion_copy = fusion;

  // check persistent buffer size
  SchedulerRuntimeInfo runtime_info(&fusion, runtime_inputs);
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  auto persistent_buffer_size_bit =
      persistentBufferSizeBit(&fusion, runtime_info, persistent_buffer_info);
  int64_t logic_buffer_size_bit = hidden_size * dataTypeSizeBit(dtype);
  EXPECT_EQ(
      persistent_buffer_size_bit.projected_persistent_buffer_size_bit,
      logic_buffer_size_bit);

  // expect segmentation?
  bool has_enough_regs_smem = true;
  if (logic_buffer_size_bit > scheduler_utils::register_file_size_bit) {
    const auto dev_prop = at::cuda::getCurrentDeviceProperties();
    if ((int64_t)dev_prop->sharedMemPerBlockOptin * 8 < logic_buffer_size_bit) {
      has_enough_regs_smem = false;
    } else {
      int64_t available_buffer_size_bit = normalization_scheduler_utils::
          getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
              &fusion,
              runtime_info,
              scheduler_utils::getReductionTvs(&fusion),
              persistent_buffer_info,
              /*can_use_smem_persistent*/ true,
              /*project_to_inputs*/ true);
      has_enough_regs_smem = available_buffer_size_bit >= logic_buffer_size_bit;
    }
  }

  // check segmentation and smem usage
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs =
      executor_cache.runFusionWithInputs({aten_input, aten_weight, aten_bias});
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  if (has_enough_regs_smem) {
    EXPECT_THAT(
        runtime->fusionSegments()->groups(),
        UnorderedElementsAre(HeuristicIs(SchedulerType::InnerPersistent)));
    Fusion* scheduled_fusion = runtime->executors()
                                   .back()
                                   ->as<KernelExecutor>()
                                   ->compiledKernel()
                                   ->kernel();

    if (logic_buffer_size_bit > scheduler_utils::register_file_size_bit) {
      bool has_smem_tv = false;
      for (auto tv : scheduled_fusion->allTvs()) {
        if (tv->getMemoryType() == MemoryType::Shared) {
          has_smem_tv = true;
          break;
        }
      }
      EXPECT_TRUE(has_smem_tv);
    }
  } else {
    EXPECT_THAT(
        runtime->fusionSegments()->groups(),
        Contains(HeuristicIs(SchedulerType::Reduction)));
  }
  testValidate(
      &fusion_copy,
      cg_outputs,
      {aten_input, aten_weight, aten_bias},
      __LINE__,
      __FILE__,
      "");
}
INSTANTIATE_TEST_SUITE_P(
    PersistentBufferTest,
    LayerNormSharedMemoryTest,
    ::testing::Combine(
        ::testing::Values(DataType::Half, DataType::Float),
        ::testing::Range((int64_t)32768, (int64_t)81921, (int64_t)4096)),
    [](const testing::TestParamInfo<TestParam>& info) {
      std::stringstream ss;
      ss << "dtype_" << std::get<0>(info.param);
      ss << "_hidden_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

// If the persistent buffer is the output of an upcast op, project
// it back to the input to save register usage. This is similar to
// project to inputs, not abosutely necessary but we always do it to
// save register usage.
TEST_F(PersistentBufferTest, ProjectToUpcastInput) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int64_t dim0 = 128, dim1 = 1024;
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = gt(tv1, IrBuilder::create<Val>(0.5));
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = add(tv5, tv3);
  fusion.addOutput(tv6);
  auto fusion_copy = fusion;

  // tv3 is the persistent tensor
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  EXPECT_EQ(
      persistent_buffer_info.persistent_buffers, std::vector<TensorView*>{tv3});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({dim0, dim1}, options);

  // The persistent buffer size is dim1 * sizeof(bool) not sizeof(float)
  // becase tv3 is the output of an upcast op, the scheduler will project it
  // back to the input which is tv2 and its data type is bool.
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_input});
  auto persistent_buffer_size_bit = scheduler_utils::persistentBufferSizeBit(
      &fusion, runtime_info, persistent_buffer_info);
  EXPECT_EQ(
      persistent_buffer_size_bit.persistent_buffer_size_bit,
      dim1 * dataTypeSizeBit(DataType::Bool));

  // Check the compute position of the bool tensor, tv2, is at the top of the
  // kernel.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({aten_input});
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  Fusion* scheduled_fusion = runtime->executors()
                                 .back()
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();
  for (auto tv : scheduled_fusion->allTvs()) {
    if (tv->getDataType() == DataType::Bool) {
      EXPECT_EQ(tv->getComputeAtPosition(), 1);
    }
  }
  testValidate(&fusion_copy, cg_outputs, {aten_input}, __LINE__, __FILE__, "");
}

TEST_F(NVFuserTest, FalsePersistentBuffer) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeConcreteTensor({320});
  fusion.addInput(tv0);

  // Extracted from issue #4020
  auto tv1 = reshape(tv0, {320}, {10, 32});
  auto tv2 = reshape(tv0, {320}, {10, 32});
  auto tv3 = add(tv1, tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  auto tv5 = sum(tv3, {1});
  fusion.addOutput(tv5);
  auto tv6 = sum(tv2, {1});
  auto tv7 = set(tv6);
  fusion.addOutput(tv7);

  // In this fusion, ComputeAtLogicalDomainMap tells
  // tv0 has an unmappable consumer IDs, making
  // tv0 be a candidate of persistent buffers, even though it doesn't
  // need to be persistent. As a result, before PR #4083,
  // persistentBuffers(Fusion*) tries to find a resolution point, but it fails
  // there's no such tensor.
  scheduler_utils::PersistentBufferInfo info =
      scheduler_utils::persistentBuffers(&fusion);
  EXPECT_TRUE(info.persistent_buffers.empty());
}

// Repro of issue #4052 (https://github.com/NVIDIA/Fuser/issues/4052)
// without the input projection. Note that the original repro is triggered
// by the input projection.
TEST_F(PersistentBufferTest, BroadcastSync1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = sum(tv5, {0, 1});
  auto tv7 = broadcast(tv6, {true, true});

  {
    // Same sequence of the ops above
    auto tv2 = broadcast(tv0, {false, true});
    auto tv3 = castOp(DataType::Float, tv2);
    auto tv4 = castOp(DataType::Float, tv1);
    auto tv5 = add(tv3, tv4);

    auto tv9 = add(tv5, tv7);
    auto tv10 = castOp(DataType::BFloat16, tv9);
    fusion.addOutput(tv10);
  }

  auto unscheduled_fusion_copy = fusion;

  auto pb_info = scheduler_utils::persistentBuffers(&fusion);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({64}, options);
  auto t1 = at::randn({64, 16}, options);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0, t1});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion_ptr.get(), runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);
  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());

  // Lowering should succeed. Prior to the fix of the issue, the sync
  // analysis raises an exception as there's mismatched
  // parallelization between the cache of tv0 and its consumer
  GpuLower gpulw(&fusion);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto outputs =
      ke.run({t0, t1}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&unscheduled_fusion_copy, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Similar to BroadcastSync1 but just one of the reduction IDs is
// resolved with the input tensor
TEST_F(PersistentBufferTest, BroadcastSync2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = sum(tv5, {0, 1});
  auto tv7 = broadcast(tv6, {true});

  {
    auto tv3 = castOp(DataType::Float, tv0);

    auto tv9 = add(tv3, tv7);
    auto tv10 = castOp(DataType::BFloat16, tv9);
    fusion.addOutput(tv10);
  }

  // In this case, there's no persistent buffer, so it should not be
  // scheduled as a persistent kernel. Note that it's possible to
  // schedule it like a persistent kernel without segmentation, which
  // may be more favorable.

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({64}, options);
  auto t1 = at::randn({64, 16}, options);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0, t1});
  EXPECT_FALSE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion_ptr.get(), runtime_info));
}

// Make sure isCacheableUnmappableTv does not falsely claim not
// cacheable when an unmappable tensor is reduced through a reshape
TEST_F(PersistentBufferTest, BroadcastSyncReshape) {
  GTEST_SKIP() << "Disabled for now";
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeConcreteTensor({2, 32});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 = reshape(tv1, {IrBuilder::create<Val>(-1)});
  auto tv3 = sum(tv2, {0});
  auto tv4 = broadcast(tv3, {true});

  auto tv5 = reshape(tv1, {IrBuilder::create<Val>(-1)});
  auto tv6 = add(tv4, tv5);
  fusion.addOutput(tv6);

  fusion.printMath();

  auto unscheduled_fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 32}, options);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0});

  // This fusion has only one unmappable tensor. If it's falsely
  // detected as non cacheable, it will not be scheduled as a
  // persistent kernel.
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion_ptr.get(), runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);
  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto outputs =
      ke.run({t0}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&unscheduled_fusion_copy, outputs, {t0}, __LINE__, __FILE__);
}

// Simplified version to reproduce the issue #4052
TEST_F(PersistentBufferTest, BroadcastSyncProjectToInputs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = sum(tv5, {0, 1});
  auto tv7 = broadcast(tv6, {true, true});
  auto tv8 = add(tv5, tv7);
  fusion.addOutput(tv8);
  auto unscheduled_fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({64}, options);
  auto t1 = at::randn({64, 16}, options);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0, t1});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion_ptr.get(), runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);
  EXPECT_FALSE(
      heuristic_params->as<ReductionParams>()->project_persistent_buffers);
  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto outputs =
      ke.run({t0, t1}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&unscheduled_fusion_copy, outputs, {t0, t1}, __LINE__, __FILE__);
}

namespace {

inline TensorView* makeBroadcastTensor(
    const std::vector<bool>& is_broadcast_dim,
    DataType input_dtype = DataType::Float) {
  std::vector<IterDomain*> out_domain;
  out_domain.reserve(is_broadcast_dim.size());
  for (auto is_broadcast : is_broadcast_dim) {
    out_domain.push_back(
        IterDomainBuilder(
            FusionGuard::getCurFusion()->zeroVal(),
            is_broadcast ? FusionGuard::getCurFusion()->oneVal()
                         : IrBuilder::create<Val>(DataType::Index))
            .iter_type(is_broadcast ? IterType::Broadcast : IterType::Iteration)
            .build());
  }
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      input_dtype);
}

} // namespace

// Different from BroadcastSyncProjectToInputs, this test has a broadcast
// domain in one of the inputs. The scheduler should project the persistent
// to inputs.
TEST_F(PersistentBufferTest, BroadcastSyncInputsHasBcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeBroadcastTensor({false, true}, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = sum(tv5, {1});
  auto tv7 = broadcast(tv6, {false, true});
  auto tv8 = add(tv5, tv7);
  fusion.addOutput(tv8);
  auto unscheduled_fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({64}, options).unsqueeze(-1);
  auto t1 = at::randn({64, 16}, options);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0, t1});
  ASSERT_TRUE(Schedule::canSchedule(
      SchedulerType::InnerPersistent, fusion_ptr.get(), runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);
  EXPECT_TRUE(
      heuristic_params->as<ReductionParams>()->project_persistent_buffers);
  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto outputs =
      ke.run({t0, t1}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&unscheduled_fusion_copy, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test cluster reduction with different models, dtype, and cluster size
// is_softmax: true for softmax, false for simple norm
// softmax uses cluster reduction twice in a single kernel with op max and add
using ClusterReductionTestParams =
    std::tuple</*is_softmax*/ bool, DataType, /*blocks_per_cluster=*/int64_t>;
using ClusterReductionTest =
    NVFuserFixtureParamTest<ClusterReductionTestParams>;
TEST_P(ClusterReductionTest, SoftmaxDtypeClusterSize) {
  // reduction domain is scheduled as:
  // [blocks_per_cluster, batches_per_block, threads_per_block, vect_factor]
  auto [is_softmax, dtype, blocks_per_cluster] = GetParam();
  constexpr int batches_per_block = 2;
  constexpr int threads_per_block = 256;
  const int vect_factor = 128 / dataTypeSizeBit(dtype);
  int y =
      threads_per_block * blocks_per_cluster * vect_factor * batches_per_block;
  // use two waves
  int x = (deviceSMCount() * 2 + blocks_per_cluster - 1) / blocks_per_cluster;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  auto tv0 = makeContigConcreteTensor({x, y}, dtype);
  fusion.addInput(tv0);
  auto tv1 = maybeCastOp(DataType::Float, tv0);
  if (is_softmax) {
    auto tv2 = softmax(tv1, {1});
    auto tv3 = maybeCastOp(DataType::BFloat16, tv2);
    fusion.addOutput(tv3);
  } else {
    auto tv2 = sum(tv1, {1});
    auto tv3 = broadcast(tv2, {false, true});
    auto tv4 = add(tv1, tv3);
    auto tv5 = maybeCastOp(DataType::BFloat16, tv4);
    fusion.addOutput(tv5);
  }
  auto unscheduled_fusion_copy = fusion;

  torch::cuda::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({x, y}, options).clamp(-2, 2);
  SchedulerRuntimeInfo runtime_info(fusion_ptr.get(), {t0});
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params =
      scheduler->computeHeuristics(fusion_ptr.get(), runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();
  rparams->cross_cluster_reduction = true;
  rparams->cross_grid_inner_reduction = true;
  rparams->grid_dim_inner_reduction = ParallelType::BIDx;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->batches_per_block_inner_reduction = batches_per_block;
  rparams->static_bdimx = true;
  rparams->static_gdimx = true;
  rparams->lparams = LaunchParams(
      blocks_per_cluster,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      threads_per_block,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  scheduler->schedule(fusion_ptr.get(), heuristic_params.get());
  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto outputs =
      ke.run({t0}, {}, heuristic_params->as<ReductionParams>()->lparams);
  testValidate(&unscheduled_fusion_copy, outputs, {t0});
}
INSTANTIATE_TEST_SUITE_P(
    PersistentBufferTest,
    ClusterReductionTest,
    ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(DataType::BFloat16, DataType::Float),
        ::testing::Values(2, 3, 4, 5, 6, 7, 8)),
    [](const testing::TestParamInfo<ClusterReductionTestParams>& info) {
      std::stringstream ss;
      ss << "is_softmax_" << std::get<0>(info.param);
      ss << "_dtype_" << std::get<1>(info.param);
      ss << "_cluster_" << std::get<2>(info.param);
      return sanitizeTestName(ss.str());
    });
} // namespace nvfuser
