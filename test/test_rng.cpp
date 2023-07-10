// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

at::Tensor generate_uniform(int64_t size, at::ScalarType dtype);

at::Tensor generate_normal(int64_t size, at::ScalarType dtype);

class RNGTest : public NVFuserTest {};

TEST_F(RNGTest, ValidateWithCURand) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  Int* size_val = IrBuilder::create<Int>();
  fusion->addInput(size_val);
  TensorView* tv0 = rand({size_val}, DataType::Float);
  TensorView* tv1 = rand({size_val}, DataType::Double);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));

  for (int64_t size : {16, 1024, 10001, 10002, 10003, 100000, 10000001}) {
    at::manual_seed(0);
    auto cg_outputs = fec.runFusionWithInputs({size});

    at::manual_seed(0);
    auto ref0 = generate_uniform(size, at::kFloat);
    auto ref1 = generate_uniform(size, at::kDouble);

    testValidate(
        fec.fusion(), cg_outputs, {size}, {ref0, ref1}, __LINE__, __FILE__);
  }
}

TEST_F(RNGTest, ManualScheduleValidateWithCURand) {
  int64_t size = 128;
  auto dtype = at::kFloat;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(1, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = rand_like(tv0);
  auto tv2 = set(tv1);
  fusion->addOutput(tv2);

  tv2->split(0, 8);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  tv1->computeAt(tv2, 1);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({size}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0});

  at::manual_seed(0);
  auto cg_outputs = fe.runFusion({t0});
  auto out = cg_outputs[0];

  at::manual_seed(0);
  auto ref = generate_uniform(size, dtype);

  testValidate(fusion, {out}, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(RNGTest, ManualScheduleValidateWithCURand2) {
  auto dtype = at::kFloat;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  Int* size1 = IrBuilder::create<Int>();
  Int* size2 = IrBuilder::create<Int>();
  Int* size3 = IrBuilder::create<Int>();
  Int* size4 = IrBuilder::create<Int>();
  fusion->addInput(size1);
  fusion->addInput(size2);
  fusion->addInput(size3);
  fusion->addInput(size4);
  TensorView* tv0 = rand({size1, size2, size3, size4}, DataType::Float);
  fusion->addOutput(tv0);

  FusionExecutor fe;
  fe.compileFusion(fusion, {10, 10, 10, 10});

  at::manual_seed(0);
  auto cg_outputs = fe.runFusion({10, 10, 10, 10});
  auto out = cg_outputs[0];

  at::manual_seed(0);
  auto ref = generate_uniform(10000, dtype).view({10, 10, 10, 10});

  testValidate(fusion, {out}, {10, 10, 10, 10}, {ref}, __LINE__, __FILE__);
}

TEST_F(RNGTest, BroadcastingRNG) {
  for (auto dtype : {at::kFloat, at::kDouble}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    auto fusion = fusion_ptr.get();
    FusionGuard fg(fusion);

    TensorView* tv0 = makeConcreteTensor({5, 1}, aten_to_data_type(dtype));
    TensorView* tv1 = makeConcreteTensor({5, 5}, aten_to_data_type(dtype));
    fusion->addInput(tv0);
    fusion->addInput(tv1);
    auto tv2 = rand_like(tv0);
    auto tv3 = add(tv1, tv2);
    auto tv4 = add(tv0, tv3);
    fusion->addOutput(tv4);

    FusionExecutorCache fec(std::move(fusion_ptr));

    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
    at::Tensor t0 = at::zeros({5, 1}, options);
    at::Tensor t1 = at::zeros({5, 5}, options);

    auto cg_outputs = fec.runFusionWithInputs({t0, t1});
    auto out = cg_outputs[0];
    TORCH_CHECK((out.select(1, 0) == out.select(1, 1)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 2)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 3)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 4)).all().item<bool>())
  }
}

TEST_F(RNGTest, BroadcastingRNG2) {
  for (int64_t size : {16, 1024, 10001, 10002, 10003, 100000, 10000001}) {
    for (auto dtype : {at::kFloat, at::kDouble}) {
      std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
      auto fusion = fusion_ptr.get();
      FusionGuard fg(fusion);

      TensorView* tv0 = makeConcreteTensor({1}, aten_to_data_type(dtype));
      TensorView* tv1 = makeSymbolicTensor(1, aten_to_data_type(dtype));
      fusion->addInput(tv0);
      fusion->addInput(tv1);
      auto tv2 = rand_like(tv0);
      auto tv3 = add(tv1, tv2);
      fusion->addOutput(tv3);

      FusionExecutorCache fec(std::move(fusion_ptr));

      auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      at::Tensor t0 = at::zeros({1}, options);
      at::Tensor t1 = at::zeros({size}, options);

      at::manual_seed(0);
      auto cg_outputs = fec.runFusionWithInputs({t0, t1});
      auto out = cg_outputs[0];

      at::manual_seed(0);
      auto ref = generate_uniform(1, dtype).expand_as(t1);

      testValidate(fec.fusion(), {out}, {t0, t1}, {ref}, __LINE__, __FILE__);
    }
  }
}

TEST_F(RNGTest, BroadcastingRNGSmem) {
  for (auto dtype : {at::kFloat, at::kDouble}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    auto fusion = fusion_ptr.get();
    FusionGuard fg(fusion);

    TensorView* tv0 = makeConcreteTensor({5, 1}, aten_to_data_type(dtype));
    TensorView* tv1 = makeConcreteTensor({5, 5}, aten_to_data_type(dtype));
    fusion->addInput(tv0);
    fusion->addInput(tv1);
    auto tv2 = rand_like(tv0);
    auto tv3 = add(tv1, tv2);
    auto tv4 = add(tv0, tv3);
    fusion->addOutput(tv4);

    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
    at::Tensor t0 = at::zeros({5, 1}, options);
    at::Tensor t1 = at::zeros({5, 5}, options);

    auto lparams = scheduleTranspose(fusion, {t0, t1});

    FusionExecutor fe;
    fe.compileFusion(fusion, {t0, t1}, lparams);
    auto cg_outputs = fe.runFusion({t0, t1}, lparams);
    auto out = cg_outputs[0];

    TORCH_CHECK((out.select(1, 0) == out.select(1, 1)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 2)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 3)).all().item<bool>())
    TORCH_CHECK((out.select(1, 0) == out.select(1, 4)).all().item<bool>())
  }
}

TEST_F(RNGTest, BroadcastingRNGSmemNonSquareTile) {
  // https://github.com/csarofeen/pytorch/issues/1926
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({5, 1});
  TensorView* tv1 = makeConcreteTensor({5, 5});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = rand_like(tv0);
  auto tv3 = add(tv1, tv2);
  auto tv4 = add(tv0, tv3);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({5, 1}, options);
  at::Tensor t1 = at::zeros({5, 5}, options);

  TransposeParams heuristics;
  heuristics.tile_size1 = 8;
  heuristics.tile_size2 = 4;
  scheduleTranspose(fusion, heuristics);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  TORCH_CHECK((out.select(1, 0) == out.select(1, 1)).all().item<bool>());
  TORCH_CHECK((out.select(1, 0) == out.select(1, 2)).all().item<bool>());
  TORCH_CHECK((out.select(1, 0) == out.select(1, 3)).all().item<bool>());
  TORCH_CHECK((out.select(1, 0) == out.select(1, 4)).all().item<bool>());
}

TEST_F(RNGTest, Uniform) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  Int* size_val = IrBuilder::create<Int>();
  Double* low = IrBuilder::create<Double>();
  Double* high = IrBuilder::create<Double>();
  fusion->addInput(size_val);
  fusion->addInput(low);
  fusion->addInput(high);
  TensorView* tv0 = uniform({size_val}, low, high, DataType::Float);
  TensorView* tv1 = uniform({size_val}, low, high, DataType::Double);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));

  for (int64_t size : {16, 1024, 10001, 10002, 10003, 100000, 10000001}) {
    at::manual_seed(0);
    auto cg_outputs = fec.runFusionWithInputs({size, -1.0, 1.0});

    at::manual_seed(0);
    auto ref0 = generate_uniform(size, at::kFloat) * 2 - 1;
    auto ref1 = generate_uniform(size, at::kDouble) * 2 - 1;

    testValidate(
        fec.fusion(),
        cg_outputs,
        {size, -1.0, 1.0},
        {ref0, ref1},
        __LINE__,
        __FILE__);
  }
}

TEST_F(RNGTest, Normal) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  Int* size_val = IrBuilder::create<Int>();
  Double* mean = IrBuilder::create<Double>();
  Double* std = IrBuilder::create<Double>();
  fusion->addInput(size_val);
  fusion->addInput(mean);
  fusion->addInput(std);
  TensorView* tv0 = normal({size_val}, mean, std, DataType::Float);
  TensorView* tv1 = normal({size_val}, mean, std, DataType::Double);
  TensorView* tv2 = randn({size_val}, DataType::Double);
  TensorView* tv3 = randn_like(tv2);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  FusionExecutorCache fec(std::move(fusion_ptr));

  for (int64_t size : {16, 1024, 10001, 10002, 10003, 100000, 10000001}) {
    at::manual_seed(0);
    auto cg_outputs = fec.runFusionWithInputs({size, 1.0, 0.5});

    at::manual_seed(0);
    auto ref0 = generate_normal(size, at::kFloat) * 0.5f + 1.0f;
    auto ref1 = generate_normal(size, at::kDouble) * 0.5 + 1.0;
    auto ref2 = generate_normal(size, at::kDouble);
    auto ref3 = generate_normal(size, at::kDouble);

    testValidate(
        fec.fusion(),
        cg_outputs,
        {size, 1.0, 0.5},
        {ref0, ref1, ref2, ref3},
        __LINE__,
        __FILE__);
  }
}

TEST_F(RNGTest, RandLikeReduction) {
  auto dtype = at::kFloat;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0});
  auto tv2 = rand_like(tv1);
  auto tv3 = add(tv1, tv2);
  fusion->addOutput(tv3);

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({2, 3}, options);

  at::manual_seed(0);
  auto cg_outputs = fec.runFusionWithInputs({t0});
  auto out = cg_outputs[0];

  at::manual_seed(0);
  auto t1 = t0.sum(0);
  auto t2 = generate_uniform(3, dtype).expand_as(t1);
  auto t3 = t1.add(t2);

  testValidate(fec.fusion(), {out}, {t0}, {t3}, __LINE__, __FILE__);
}

//! This is the same as the Uniform test, but we compare against
//! functional_uniform in which we provide a seed and offset.
TEST_F(RNGTest, FunctionalUniform) {
  for (bool do_stochastic : {false, true}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    auto fusion = fusion_ptr.get();
    FusionGuard fg(fusion);

    Int* size_val = IrBuilder::create<Int>();
    Double* low = IrBuilder::create<Double>();
    Double* high = IrBuilder::create<Double>();
    Int* seed = IrBuilder::create<Int>();
    Int* first_offset = IrBuilder::create<Int>();
    fusion->addInput(size_val);
    fusion->addInput(low);
    fusion->addInput(high);
    fusion->addInput(seed);
    fusion->addInput(first_offset);

    if (do_stochastic) {
      // We test both with and without stochastic RNG ops. Testing with
      // stochastic ops allows us to ensure that the output is consistent.
      // Testing without them ensures that we are able to compile and run these
      // kernels without a stochastic op present, which means we do not rely on
      // any external philox seed info being passed to the kernel.
      TensorView* tv0 = uniform({size_val}, low, high, DataType::Float);
      TensorView* tv1 = uniform({size_val}, low, high, DataType::Double);
      fusion->addOutput(tv0);
      fusion->addOutput(tv1);
    }

    auto second_offset = add(first_offset, IrBuilder::create<Int>(4));

    TensorView* tv2 =
        uniform({size_val}, low, high, DataType::Float, seed, first_offset);
    TensorView* tv3 =
        uniform({size_val}, low, high, DataType::Double, seed, second_offset);

    fusion->addOutput(tv2);
    fusion->addOutput(tv3);

    FusionExecutorCache fec(std::move(fusion_ptr));

    for (int64_t size : {16, 1024, 10001, 10002, 10003, 100000, 10000001}) {
      at::manual_seed(0);
      auto ref0 = generate_uniform(size, at::kFloat) * 2 - 1;
      // Observe updated seed after first reference is generated.
      {
        auto gen = at::check_generator<at::CUDAGeneratorImpl>(
            at::cuda::detail::getDefaultCUDAGenerator());
        EXPECT_EQ(gen->current_seed(), 0);
        EXPECT_EQ(gen->get_offset(), 4);
      }

      auto ref1 = generate_uniform(size, at::kDouble) * 2 - 1;

      std::vector<c10::IValue> aten_inputs({size, -1.0, 1.0, 0, 0});

      at::manual_seed(0);
      auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

      std::vector<at::Tensor> aten_outputs;
      if (do_stochastic) {
        aten_outputs = {ref0, ref1, ref0, ref1};
      } else {
        aten_outputs = {ref0, ref1};
      }

      testValidate(
          fec.fusion(),
          cg_outputs,
          aten_inputs,
          aten_outputs,
          __LINE__,
          __FILE__);
    }
  }
}

} // namespace nvfuser
