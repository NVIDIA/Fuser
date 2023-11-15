// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/lower_resharding_expr.h>
#include <multidevice/pipeline.h>
#include <ops/all_ops.h>
#include <test/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, Pipeline_CUDA) {
  // Fusion definition
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  TensorView* tv4 = makeContigTensor(3);
  fusion.addInput(tv4);
  TensorView* tv5 = sum(tv4, {0});

  TensorView* tv6 = set(tv5);
  TensorView* tv7 = set(tv6);
  TensorView* tv8 = sum(tv7, {0});

  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv9);
  TensorView* tv11 = sum(tv10, {0});

  TensorView* tv12 = set(tv7);
  TensorView* tv13 = sum(tv12, {0});
  fusion.addOutput(tv13);

  TensorView* tv14 = set(tv8);
  TensorView* tv15 = set(tv11);
  TensorView* tv16 = set(tv13);
  TensorView* tv17 = add(tv14, tv15);
  TensorView* tv18 = add(tv17, tv16);
  TensorView* tv19 = sum(tv18, {0});
  fusion.addOutput(tv19);

  // Pipeline scheduling
  PipelineStageDescriptor stage0, stage1, stage2, stage3, stage4, stage5,
      stage6;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage2.addVal({tv4, tv5});
  stage3.addVal({tv6, tv7, tv8});
  stage4.addVal({tv9, tv10, tv11});
  stage5.addVal({tv12, tv13});
  stage6.addVal({tv14, tv15, tv16, tv17, tv18, tv19});

  PipelineDescriptor descriptor{.stage_descriptors{
      stage0, stage1, stage2, stage3, stage4, stage5, stage6}}; // the order
                                                                // doesnt matter
  Pipeline pipeline(&fusion, std::move(descriptor));

  // Validation
  std::string obtained_string = pipeline.toString();
  std::string ref_string{
      "Pipeline's inputs{:\n"
      " PipelineVal representing Val T0_g[ iS0{i0}, iS1{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      "\n"
      " PipelineVal representing Val T4_g[ iS6{i13}, iS7{i14}, iS8{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      "\n"
      "}\n"
      "Pipeline's Traversal inputs --> outputs {\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage0.unique_id) +
      ".Inputs={T0_g[ iS0{i0}, iS1{i2} ], }. Outputs={T1_l[ rS2{i0}, iS3{i2} ], }.\n"
      "  PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      " to PipelineVal representing Val T2_l[ iS4{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      "  PipelineVal representing Val T2_l[ iS4{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage1.unique_id) +
      ".Inputs={T2_l[ iS4{i2} ], }. Outputs={T3_g[ rS5{i2} ], }.\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage2.unique_id) +
      ".Inputs={T4_g[ iS6{i13}, iS7{i14}, iS8{i15} ], }. Outputs={T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ], }.\n"
      "  PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T6_l[ iS12{i14}, iS13{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineVal representing Val T6_l[ iS12{i14}, iS13{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage3.unique_id) +
      ".Inputs={T6_l[ iS12{i14}, iS13{i15} ], }. Outputs={T7_l[ iS14{i14}, iS15{i15} ], T8_l[ rS16{i14}, iS17{i15} ], }.\n"
      "  PipelineVal representing Val T7_l[ iS14{i14}, iS15{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T7_l[ iS14{i14}, iS15{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T12_l[ iS24{i14}, iS25{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineVal representing Val T12_l[ iS24{i14}, iS25{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage5.unique_id) +
      ".Inputs={T12_l[ iS24{i14}, iS25{i15} ], }. Outputs={T13_g[ rS26{i14}, iS27{i15} ], }.\n"
      "  PipelineVal representing Val T8_l[ rS16{i14}, iS17{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T8_l[ rS16{i14}, iS17{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T14_l[ iS28{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T14_l[ iS28{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T9_l[ iS18{i14}, iS19{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineVal representing Val T9_l[ iS18{i14}, iS19{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage4.unique_id) +
      ".Inputs={T9_l[ iS18{i14}, iS19{i15} ], }. Outputs={T11_l[ rS22{i14}, iS23{i15} ], }.\n"
      "  PipelineVal representing Val T11_l[ rS22{i14}, iS23{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T11_l[ rS22{i14}, iS23{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      " to PipelineVal representing Val T15_l[ iS29{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T15_l[ iS29{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T13_g[ rS26{i14}, iS27{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      " to PipelineVal representing Val T16_l[ iS30{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T16_l[ iS30{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage6.unique_id) +
      ".Inputs={T14_l[ iS28{i15} ], T15_l[ iS29{i15} ], T16_l[ iS30{i15} ], }. Outputs={T19_g[ rS33{i15} ], }.\n"
      "}\n"
      "Pipeline's outputs:{\n"
      " PipelineVal representing Val T3_g[ rS5{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      " PipelineVal representing Val T13_g[ rS26{i14}, iS27{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      " PipelineVal representing Val T19_g[ rS33{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "}"};

  // We sort the string so it doesn't depend on the order of the Pipeline's DAG
  // traversal

  // TODO: we should sort on lines, not on characters
  std::sort(obtained_string.begin(), obtained_string.end());
  std::sort(ref_string.begin(), ref_string.end());

  EXPECT_EQ(obtained_string, ref_string);
}


TEST_F(NVFuserTest, ReshardingDetection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  DeviceMesh mesh0,mesh1, mesh2;
  mesh0 = {0,1};
  mesh1 = {0,2};
  mesh2 = {0,1,2};

  TensorView* tv0 = makeContigTensor(3);
  tv0->setDeviceMesh(&mesh0);

  TensorView* tv1 = set(tv0);
  tv1->setDeviceMesh(&mesh0);

  TensorView* tv2 = set(tv1); // resharding
  tv2->setDeviceMesh(&mesh1);

  TensorView* tv3 = set(tv2);
  tv3->setDeviceMesh(&mesh1);

  TensorView* tv4 = set(tv3); // resharding
  tv4->setDeviceMesh(&mesh2);

  TensorView* tv5 = set(tv4); // resharding
  tv5->setDeviceMesh(&mesh2);
  tv5->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv6 = set(tv5);
  tv6->setDeviceMesh(&mesh2);
  tv6->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv7 = set(tv6); // resharding
  tv7->setDeviceMesh(&mesh2);

  TensorView* tv8 = sum(tv0, {0});
  tv8->setDeviceMesh(&mesh0);

  TensorView* tv9 = sum(tv0, {0}); //resharding, but seems invalid
  tv9->setDeviceMesh(&mesh0);
  tv9->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv10 = sum(tv0, {0}); //resharding,
  tv10->setDeviceMesh(&mesh0);
  tv10->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv11 = sum(tv0, {0}); //resharding,
  tv11->setDeviceMesh(&mesh1);

  TensorView* tv12 = sum(tv5, {0}); // resharding
  tv12->setDeviceMesh(&mesh2);
  tv12->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv13 = sum(tv5, {0}); // resharding
  tv13->setDeviceMesh(&mesh2);
  tv13->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv14 = sum(tv5, {0}); // resharding
  tv14->setDeviceMesh(&mesh2);
  tv14->axis(0)->parallelize(ParallelType::DIDx);
  tv14->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv15 = add(tv0, tv1);
  tv15->setDeviceMesh(&mesh0);

  TensorView* tv16 = add(tv0, tv1); //resharding
  tv16->setDeviceMesh(&mesh1);

  TensorView* tv17 = add(tv0, tv1); //resharding
  tv17->setDeviceMesh(&mesh0);
  tv17->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv18 = add(tv5, tv6);
  tv18->setDeviceMesh(&mesh2);
  tv18->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv19 = add(tv5, tv7); // resharding
  tv19->setDeviceMesh(&mesh2);
  tv19->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv20 = add(tv5, tv7); // resharding
  tv20->setDeviceMesh(&mesh2);

  TensorView* tv21 = add(tv0, tv7); // resharding
  tv21->setDeviceMesh(&mesh2);

  TensorView* tv22 = sum(tv5, {1});
  tv22->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv23 = sum(tv5, {1}); //resharding
  tv23->setDeviceMesh(&mesh2);

  TensorView* tv24 = sum(tv7, {0});
  tv24->setDeviceMesh(&mesh2);

  TensorView* tv25 = sum(tv7, {0}); //not resharding but invalid
  tv25->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv26 = add(tv5, tv6); // resharding
  tv26->setDeviceMesh(&mesh2);

  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv1->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv2->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv3->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv4->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv5->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv6->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv7->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv8->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv9->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv10->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv11->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv12->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv13->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv14->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv15->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv16->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv17->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv18->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv19->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv20->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv21->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv22->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv23->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv24->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv25->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv26->definition()));
}


using automaticSetInsertionTestParams =
    std::tuple<
    DeviceMesh,
    DeviceMesh,
    DeviceMesh,
    DeviceMesh,
    bool,
    bool,
    bool,
    bool>;

class automaticSetInsertionTest :
  public NVFuserTest,
  public ::testing::WithParamInterface<automaticSetInsertionTestParams> {
protected:
  void SetUp() override {
    fusion = std::make_unique<Fusion>();
    fg = std::make_unique<FusionGuard>(fusion.get());
  }
  void validate() {
    for (auto expr: fusion->exprs()) {
      GTEST_EXPECT_TRUE(!ir_utils::isResharding(expr) || isLowerableToCommunication(expr));
    }
  }

  std::unique_ptr<Fusion> fusion;
  std::unique_ptr<FusionGuard> fg;
};

TEST_P(automaticSetInsertionTest, ops) {
  auto [
  mesh0,
  mesh1,
  mesh2,
  mesh3,
  is_tv0_sharded,
  is_tv1_sharded,
  is_tv2_sharded,
  is_tv3_sharded] = GetParam();

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = unaryOp(UnaryOpType::Exp, tv0);
  TensorView* tv2 = binaryOp(BinaryOpType::Add, tv0, tv1);
  TensorView* tv3 = sum(tv2, {0});

  tv0->setDeviceMesh(&mesh0);
  tv1->setDeviceMesh(&mesh1);
  tv2->setDeviceMesh(&mesh2);
  tv3->setDeviceMesh(&mesh3);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv3);

  if (is_tv0_sharded) {
    tv0->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_tv1_sharded) {
    tv1->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_tv2_sharded) {
    tv2->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_tv3_sharded) {
    tv3->axis(0)->parallelize(ParallelType::DIDx);
  }

  insertSetBeforeReshardingExpr(fusion.get());
  validate();
}

namespace {

DeviceMesh mesh0({0});
DeviceMesh mesh1({1,2});
DeviceMesh mesh2({0,2});
DeviceMesh mesh3({0, 1, 2, 3});

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    automaticSetInsertionTest,
    ::testing::Combine(
        ::testing::Values(mesh0, mesh3),
        ::testing::Values(mesh1, mesh3),
        ::testing::Values(mesh2, mesh3),
        ::testing::Values(mesh0, mesh3),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool()));

TEST_F(NVFuserTest, 2DMesh) {
  auto fusion = std::make_unique<Fusion>();
  auto fg = std::make_unique<FusionGuard>(fusion.get());

  auto tv0 = makeContigTensor(3);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv0->axis(1)->parallelize(ParallelType::DIDy);
  DeviceMesh mesh ({0,1,2,3,4,5});
  mesh.reshape({2,3});
  tv0->setDeviceMesh(&mesh);
  std::cout << tv0 << std::endl;
}

TEST_F(NVFuserTest, select_bug) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = select(tv0, 0, IrBuilder::create<Val>(0));
  fusion.addOutput(tv1);
  fusion.print();
}

} // namespace nvfuser
