
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
#include <ir/interface_nodes.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using VectorizationTest = NVFuserTest;

TEST_F(VectorizationTest, Reduction_fp16_32_64) {
  auto dtype = DataType::Half;
  const std::vector<int64_t> input_shape = {256, 1024};
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeContigConcreteTensor(input_shape, dtype);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {-1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv1, tv3);
  // output is fp32, max vectorization factor is 4
  fusion.addOutput(tv4);

  // output is fp16, max vectorization factor is 8
  auto tv5 = add(tv1, tv3);
  auto tv6 = castOp(DataType::Half, tv5);
  fusion.addOutput(tv6);

  // output is fp64, max vectorization factor is 2
  auto tv7 = add(tv5, tv5);
  auto tv8 = castOp(DataType::Double, tv7);
  fusion.addOutput(tv8);



  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::ones(input_shape, options);
  std::vector<c10::IValue> aten_inputs = {at_x};

  // FusionExecutorCache fec(std::move(fusion_ptr));
  // auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  const int expected_vect_factor = 8;
  auto hp = getInnerPersistentHeuristics(&fusion, aten_inputs);
  NVF_CHECK(hp, "getInnerPersistentHeuristics failed!");
  EXPECT_EQ(hp->unroll_factor_inner_reduction, expected_vect_factor);
  scheduleInnerPersistentKernel(&fusion, *hp);
  for (auto tv : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    int64_t expected_val = 16L / dataTypeSize(tv->getDataType().value());
    EXPECT_TRUE(tv->axis(-1)->getParallelType() == ParallelType::Vectorize);
    EXPECT_TRUE(tv->axis(-1)->extent()->isConst());
    EXPECT_EQ(tv->axis(-1)->extent()->value(), expected_val);
    std::cout << "tv: " << tv->toString() << std::endl;
  }
  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, hp->lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, hp->lparams);
  auto t1 = at_x.to(at::kFloat);
  auto t3 = t1.sum(-1).unsqueeze(-1);
  auto t4 = t1 / t3;
  auto t5 = t1 + t3;
  auto t6 = t5.to(at::kHalf);
  auto t8 = (t5 + t5).to(at::kDouble);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {t4, t6, t8}, __LINE__, __FILE__);
}
} // namespace nvfuser